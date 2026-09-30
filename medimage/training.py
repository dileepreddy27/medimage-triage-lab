"""Fixed split, validation checkpoint selection, single held-out evaluation."""

import hashlib
import json
from pathlib import Path
import platform
import time

import monai
from monai.losses import DiceCELoss
import numpy as np
import torch

from medimage.model import TinySegmenter, baseline, dice, normalize, predict
from medimage.synthetic import SPLITS, phantom


def train(destination: Path, epochs: int = 30) -> dict:
    if epochs < 1:
        raise ValueError("Epochs must be positive")
    destination.mkdir(parents=True, exist_ok=True)
    torch.manual_seed(17)
    torch.set_num_threads(2)
    torch.use_deterministic_algorithms(True)
    model = TinySegmenter()
    optimizer = torch.optim.Adam(model.parameters(), lr=0.003)
    loss_fn = DiceCELoss(sigmoid=True)
    pairs = [phantom(s) for s in SPLITS["train"]]
    x = torch.cat([normalize(a) for a, _ in pairs])
    y = torch.from_numpy(np.stack([b for _, b in pairs]))[:, None]
    best, history = -1.0, []
    started = time.perf_counter()
    for epoch in range(epochs):
        model.train()
        losses = []
        for ids in torch.randperm(len(x)).split(8):
            optimizer.zero_grad()
            loss = loss_fn(model(x[ids]), y[ids])
            loss.backward()
            optimizer.step()
            losses.append(loss.item())
        model.eval()
        val = np.mean(
            [dice(predict(model, a) >= 0.5, b) for a, b in (phantom(s) for s in SPLITS["validation"])]
        )
        history.append({"epoch": epoch + 1, "loss": float(np.mean(losses)), "validation_dice": float(val)})
        if val > best:
            best = val
            torch.save(
                {"format": "medimage-v1", "state_dict": model.state_dict(), "epoch": epoch + 1},
                destination / "model.pt",
            )
        if epoch == 0 or (epoch + 1) % 5 == 0 or epoch + 1 == epochs:
            print(f"Epoch {epoch + 1}/{epochs}: validation Dice {val:.4f}", flush=True)
    from medimage.model import load_model

    model = load_model(destination / "model.pt")
    rows = []
    for seed in SPLITS["test"]:
        hu, truth = phantom(seed)
        t = time.perf_counter()
        mask = predict(model, hu) >= 0.5
        latency = (time.perf_counter() - t) * 1000
        rows.append(
            {
                "seed": seed,
                "positive": bool(truth.any()),
                "cnn_dice": dice(mask, truth),
                "baseline_dice": dice(baseline(hu), truth),
                "cnn_voxels": int(mask.sum()),
                "truth_voxels": int(truth.sum()),
                "inference_ms": latency,
            }
        )
    metrics = {
        "scope": "Synthetic ellipsoids only; no diagnostic validation",
        "seed": 17,
        "checkpoint_sha256": hashlib.sha256((destination / "model.pt").read_bytes()).hexdigest(),
        "environment": {
            "os": platform.system(),
            "python": platform.python_version(),
            "torch": str(torch.__version__),
            "monai": monai.__version__,
            "numpy": np.__version__,
            "cpu_threads": 2,
        },
        "selected_epoch": torch.load(destination / "model.pt", weights_only=True)["epoch"],
        "splits": SPLITS,
        "epochs": epochs,
        "threshold": 0.5,
        "parameters": sum(p.numel() for p in model.parameters()),
        "training_and_evaluation_seconds": time.perf_counter() - started,
        "cnn_mean_dice": float(np.mean([r["cnn_dice"] for r in rows])),
        "cnn_positive_mean_dice": float(np.mean([r["cnn_dice"] for r in rows if r["positive"]])),
        "baseline_mean_dice": float(np.mean([r["baseline_dice"] for r in rows])),
        "negative_false_positive_cases": sum(r["cnn_voxels"] > 0 for r in rows if not r["positive"]),
        "inference_median_ms": float(np.median([r["inference_ms"] for r in rows])),
        "history": history,
        "test_cases": rows,
    }
    (destination / "metrics.json").write_text(json.dumps(metrics, indent=2), encoding="utf-8")
    return metrics
