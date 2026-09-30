"""SQLite queue with content deduplication and append-only application audit events."""

from contextlib import contextmanager
import hashlib
import io
import json
from pathlib import Path
import sqlite3

import numpy as np

from medimage.dicom import read_series
from medimage.model import load_model, predict


@contextmanager
def connect(path: Path):
    path.parent.mkdir(parents=True, exist_ok=True)
    db = sqlite3.connect(path, timeout=10)
    db.row_factory = sqlite3.Row
    db.executescript("""
    CREATE TABLE IF NOT EXISTS scans (
      id TEXT PRIMARY KEY, model_sha TEXT NOT NULL, volume_ml REAL NOT NULL,
      score REAL NOT NULL, status TEXT NOT NULL DEFAULT 'pending',
      created_at TEXT NOT NULL DEFAULT (strftime('%Y-%m-%dT%H:%M:%fZ','now')));
    CREATE TABLE IF NOT EXISTS events (
      sequence INTEGER PRIMARY KEY AUTOINCREMENT, scan_id TEXT NOT NULL,
      kind TEXT NOT NULL, detail TEXT NOT NULL,
      created_at TEXT NOT NULL DEFAULT (strftime('%Y-%m-%dT%H:%M:%fZ','now')));
    """)
    try:
        with db:
            yield db
    finally:
        db.close()


def enqueue(db_path: Path, folder: Path, checkpoint: Path) -> dict:
    volume = read_series(folder)
    checkpoint_bytes = checkpoint.read_bytes()
    model_sha = hashlib.sha256(checkpoint_bytes).hexdigest()
    probability = predict(load_model(io.BytesIO(checkpoint_bytes)), volume.hu)
    mask = probability >= 0.5
    ml = float(mask.sum() * np.prod(volume.spacing) / 1000)
    # Saturating burden score, not a probability or medically derived urgency.
    score = float(100 * ml / (ml + 1))
    with connect(db_path) as db:
        db.execute("BEGIN IMMEDIATE")
        prior = db.execute("SELECT * FROM scans WHERE id=?", (volume.digest,)).fetchone()
        if prior:
            if prior["model_sha"] != model_sha:
                raise ValueError("Already processed with a different model; use a new queue for comparisons")
            return {**dict(prior), "inserted": False}
        db.execute(
            "INSERT INTO scans(id,model_sha,volume_ml,score) VALUES(?,?,?,?)",
            (volume.digest, model_sha, ml, score),
        )
        db.execute(
            "INSERT INTO events(scan_id,kind,detail) VALUES(?,?,?)",
            (volume.digest, "enqueued", json.dumps({"threshold": 0.5, "model_sha": model_sha})),
        )
        return {
            **dict(db.execute("SELECT * FROM scans WHERE id=?", (volume.digest,)).fetchone()),
            "inserted": True,
        }


def list_queue(db_path: Path) -> list[dict]:
    with connect(db_path) as db:
        return [dict(r) for r in db.execute("SELECT * FROM scans ORDER BY score DESC, created_at, id")]


def review(db_path: Path, scan_id: str) -> None:
    with connect(db_path) as db:
        result = db.execute("UPDATE scans SET status='reviewed' WHERE id=? AND status='pending'", (scan_id,))
        if result.rowcount != 1:
            raise ValueError("Scan not found or already reviewed")
        db.execute("INSERT INTO events(scan_id,kind,detail) VALUES(?, 'reviewed', '{}')", (scan_id,))
