"""Self-contained HTML research dashboard; all previews derive from actual inference."""

import base64
import hashlib
import html
import io
import json
from pathlib import Path

import numpy as np
from PIL import Image
import torch
from torchvision.utils import make_grid

from medimage.dicom import read_series
from medimage.model import load_model, predict
from medimage.queue import list_queue


def preview(hu, probability) -> bytes:
    indices = np.linspace(4, len(hu) - 5, 6).astype(int)
    gray = np.clip((hu[indices] + 200) / 400, 0, 1)
    rgb = np.repeat(gray[:, None], 3, axis=1)
    mask = probability[indices] >= 0.5
    rgb[:, 0] = np.where(mask, 0.95, rgb[:, 0])
    rgb[:, 1] = np.where(mask, 0.3, rgb[:, 1])
    rgb[:, 2] = np.where(mask, 0.1, rgb[:, 2])
    grid = make_grid(torch.from_numpy(rgb), nrow=6, padding=2)
    pixels = (grid.permute(1, 2, 0).numpy() * 255).astype(np.uint8)
    buffer = io.BytesIO()
    Image.fromarray(pixels).resize((816, 144), Image.Resampling.NEAREST).save(buffer, format="PNG")
    return buffer.getvalue()


def render(db_path: Path, checkpoint: Path, folders: list[Path], destination: Path, metrics=None):
    checkpoint_bytes = checkpoint.read_bytes()
    model = load_model(io.BytesIO(checkpoint_bytes))
    previews = {}
    for folder in folders:
        volume = read_series(folder)
        previews[volume.digest] = base64.b64encode(preview(volume.hu, predict(model, volume.hu))).decode()
    rows = list_queue(db_path)
    model_sha = hashlib.sha256(checkpoint_bytes).hexdigest()
    if any(row["model_sha"] != model_sha for row in rows):
        raise ValueError("Report checkpoint does not match queue provenance")
    if any(row["id"] not in previews for row in rows):
        raise ValueError("Missing source series for one or more queued scans")
    cards = []
    for index, row in enumerate(rows):
        picture = previews.get(row["id"], "")
        cards.append(f'''<article data-status="{html.escape(row["status"])}">
        <div class="row"><b>#{index + 1:02d} &nbsp; Phantom {row["id"][:10]}</b>
        <span class="pill">{html.escape(row["status"])}</span></div>
        <div class="stats"><strong>{row["score"]:.1f}<small>burden score / 100</small></strong>
        <strong>{row["volume_ml"]:.3f}<small>segmented mL</small></strong>
        <span>3D CNN · CPU<br>Threshold 0.50</span></div>
        <img alt="Six axial slices with model mask overlay in orange" src="data:image/png;base64,{picture}">
        <p class="muted">Six axial slices · orange = predicted synthetic foreground · model {row["model_sha"][:12]}</p>
        </article>''')
    summary = ""
    if metrics:
        summary = f"""<section class="metrics"><div><b>{metrics["cnn_mean_dice"]:.3f}</b>Held-out CNN Dice</div>
        <div><b>{metrics["baseline_mean_dice"]:.3f}</b>Threshold baseline Dice</div>
        <div><b>48 / 12 / 16</b>Train / validation / test</div></section>"""
    document = """<!doctype html><html lang="en"><meta charset="utf-8"><meta name="viewport" content="width=device-width">
    <title>MedImage | Research queue</title><style>
    *{box-sizing:border-box}body{margin:0;background:#0b1220;color:#e5eefb;font:16px system-ui,sans-serif}
    main{max-width:1060px;margin:auto;padding:42px 24px}header{border-bottom:1px solid #26354c;padding-bottom:24px}
    .eyebrow{color:#60d6c5;letter-spacing:3px;font-size:12px;font-weight:700}h1{font-size:42px;margin:12px 0}
    .muted,small{color:#94a8bf}aside{padding:16px;background:#263027;border:1px solid #536446;border-radius:10px;margin:24px 0}
    article{background:#121f32;border:1px solid #2c3d55;border-radius:14px;padding:24px;margin:18px 0}
    .row,.stats,.metrics{display:flex;align-items:center;justify-content:space-between;gap:18px}.stats{justify-content:start;margin:20px 0;gap:48px}
    strong{font-size:27px}small{display:block;font-size:12px;font-weight:400}.pill{background:#224744;color:#8ae2d2;padding:5px 12px;border-radius:20px;font-size:12px}
    img{width:100%;image-rendering:pixelated;border-radius:6px}.muted{font-size:12px}.metrics{margin:24px 0}.metrics b{display:block;font-size:28px;color:#60d6c5}
    button{border:1px solid #38536d;background:#152c41;color:#e5eefb;padding:10px 18px;border-radius:8px;cursor:pointer}button[aria-pressed=true]{border-color:#60d6c5}
    @media(max-width:650px){h1{font-size:30px}.metrics,.stats{flex-wrap:wrap;gap:20px}}
    </style><main><header><div class="eyebrow">MEDIMAGE / TRIAGE LAB</div><h1>A visible path from volume to queue.</h1>
    <p>Synthetic DICOM → validated geometry → 3D segmentation → auditable ranking</p></header>
    <aside>RESEARCH ONLY · Procedural phantoms, not patients. Scores measure synthetic foreground burden;
    they are not urgency probabilities or diagnostic findings. No clinical use.</aside>"""
    document += (
        summary
        + """<h2>Reading queue</h2><p class="muted">Descending synthetic burden · deterministic tie breaks · snapshot export</p>
    <button aria-pressed="true" onclick="filter('all',this)">All scans</button>
    <button aria-pressed="false" onclick="filter('pending',this)">Pending</button>
    <button aria-pressed="false" onclick="filter('reviewed',this)">Reviewed</button>"""
    )
    document += (
        "".join(cards)
        + """<footer class="muted">Local research snapshot. Review state is changed through the CLI and then re-exported.
    Source data are deliberately limited to a validated synthetic DICOM contract.</footer></main>
    <script>function filter(status,button){document.querySelectorAll('article').forEach(x=>x.hidden=status!=='all'&&x.dataset.status!==status);
    document.querySelectorAll('button').forEach(x=>x.setAttribute('aria-pressed',x===button?'true':'false'))}</script></html>"""
    )
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_text(document, encoding="utf-8")
    destination.with_suffix(".json").write_text(json.dumps(rows, indent=2), encoding="utf-8")
