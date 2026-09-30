import argparse
import json
from pathlib import Path
import sys

import torch

from medimage.queue import enqueue, list_queue, review
from medimage.report import render
from medimage.synthetic import write_series


def main():
    torch.set_num_threads(2)
    parser = argparse.ArgumentParser(description="Synthetic-only imaging research workflow; no clinical use")
    commands = parser.add_subparsers(dest="command", required=True)
    demo = commands.add_parser("demo", help="Train, evaluate, ingest phantoms and export dashboard")
    demo.add_argument("--out", type=Path, default=Path("work/demo"))
    demo.add_argument("--epochs", type=int, default=30)
    fit = commands.add_parser("train")
    fit.add_argument("--out", type=Path, required=True)
    fit.add_argument("--epochs", type=int, default=30)
    ingest = commands.add_parser("ingest")
    ingest.add_argument("series", type=Path)
    ingest.add_argument("--model", type=Path, required=True)
    ingest.add_argument("--db", type=Path, required=True)
    queue = commands.add_parser("queue")
    queue.add_argument("--db", type=Path, required=True)
    done = commands.add_parser("review")
    done.add_argument("id")
    done.add_argument("--db", type=Path, required=True)
    report = commands.add_parser("report")
    report.add_argument("--db", type=Path, required=True)
    report.add_argument("--model", type=Path, required=True)
    report.add_argument("--series-root", type=Path, required=True)
    report.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    try:
        if args.command == "demo":
            from medimage.training import train

            if args.out.exists() and any(args.out.iterdir()):
                raise ValueError("Demo output must be empty; choose a new --out directory")
            metrics = train(args.out, args.epochs)
            folders = [write_series(args.out / "dicom" / f"phantom-{s}", s) for s in range(400, 406)]
            for folder in folders:
                enqueue(args.out / "queue.sqlite", folder, args.out / "model.pt")
            render(
                args.out / "queue.sqlite",
                args.out / "model.pt",
                folders,
                args.out / "dashboard.html",
                metrics,
            )
            print(
                json.dumps(
                    {k: v for k, v in metrics.items() if k not in ("history", "test_cases", "splits")},
                    indent=2,
                )
            )
            print(f"Dashboard: {args.out / 'dashboard.html'}")
        elif args.command == "train":
            from medimage.training import train

            train(args.out, args.epochs)
        elif args.command == "ingest":
            print(json.dumps(enqueue(args.db, args.series, args.model), indent=2))
        elif args.command == "queue":
            print(json.dumps(list_queue(args.db), indent=2))
        elif args.command == "review":
            review(args.db, args.id)
            print("Review event recorded")
        elif args.command == "report":
            render(args.db, args.model, sorted(p for p in args.series_root.iterdir() if p.is_dir()), args.out)
    except (ValueError, OSError, RuntimeError) as exc:
        print(f"Unable to complete workflow: {exc}", file=sys.stderr)
        return 2
    return 0


if __name__ == "__main__":
    sys.exit(main())
