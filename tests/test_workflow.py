import json

import numpy as np
import pydicom
import pytest
import torch

from medimage.cli import main
from medimage.dicom import read_series
from medimage.model import TinySegmenter, baseline, dice, load_model, normalize, predict
from medimage.queue import connect, enqueue, list_queue, review
from medimage.report import render
from medimage.synthetic import SHAPE, SPLITS, phantom, write_series
from medimage.training import train


@pytest.fixture(scope="session", autouse=True)
def bounded_cpu_threads():
    torch.set_num_threads(2)


@pytest.fixture
def series(tmp_path):
    return write_series(tmp_path / "dicom", 401)


@pytest.fixture
def checkpoint(tmp_path):
    torch.manual_seed(17)
    model = TinySegmenter()
    # Deterministic constant-logit model isolates queue bookkeeping from training quality.
    for p in model.parameters():
        p.data.zero_()
    model.layers[-1].bias.data.fill_(-1)
    path = tmp_path / "test-model.pt"
    torch.save({"format": "medimage-v1", "state_dict": model.state_dict()}, path)
    return path


def mutate(folder, key, value):
    path = sorted(folder.glob("*.dcm"))[0]
    ds = pydicom.dcmread(path)
    setattr(ds, key, value)
    ds.save_as(path, enforce_file_format=True)


def test_round_trip_sorts_by_position_not_filename(series):
    volume = read_series(series)
    np.testing.assert_array_equal(volume.hu, phantom(401)[0])
    assert volume.spacing == (2, 1, 1)


def test_linear_rescale(series):
    for path in series.glob("*.dcm"):
        ds = pydicom.dcmread(path)
        ds.RescaleSlope, ds.RescaleIntercept = 2, -100
        ds.save_as(path, enforce_file_format=True)
    np.testing.assert_array_equal(read_series(series).hu, phantom(401)[0] * 2 - 100)


@pytest.mark.parametrize(
    "key,value",
    [
        ("PatientID", "REAL"),
        ("ImageType", ["ORIGINAL", "PRIMARY"]),
        ("Modality", "MR"),
        ("ImageOrientationPatient", [0, 1, 0, 1, 0, 0]),
        ("PixelSpacing", [2, 2]),
        ("ImagePositionPatient", [0, 0, 0]),
        ("SeriesInstanceUID", "1.2.3"),
        ("StudyInstanceUID", "1.2.3"),
        ("FrameOfReferenceUID", "1.2.3"),
        ("NumberOfFrames", 2),
        ("PhotometricInterpretation", "MONOCHROME1"),
        ("RescaleSlope", 0),
        ("Rows", 64),
        ("ImagePositionPatient", [1, 0, 46]),
    ],
)
def test_rejects_unsupported_input(series, key, value):
    mutate(series, key, value)
    with pytest.raises(ValueError):
        read_series(series)


def test_missing_slice(series):
    next(series.glob("*.dcm")).unlink()
    with pytest.raises(ValueError, match="exactly"):
        read_series(series)


def test_invalid_dicom_file(series):
    next(series.glob("*.dcm")).write_bytes(b"not a DICOM file")
    with pytest.raises(ValueError, match="attributes"):
        read_series(series)


def test_unsupported_transfer_syntax_rejected(series):
    path = next(series.glob("*.dcm"))
    ds = pydicom.dcmread(path)
    ds.file_meta.TransferSyntaxUID = pydicom.uid.ImplicitVRLittleEndian
    ds.save_as(path, enforce_file_format=True)
    with pytest.raises(ValueError, match="transfer syntax"):
        read_series(series)


def test_missing_attribute(series):
    path = next(series.glob("*.dcm"))
    ds = pydicom.dcmread(path)
    del ds.PixelSpacing
    ds.save_as(path, enforce_file_format=True)
    with pytest.raises(ValueError, match="attributes"):
        read_series(series)


def test_duplicate_sop(series):
    paths = sorted(series.glob("*.dcm"))
    mutate(series, "SOPInstanceUID", pydicom.dcmread(paths[1]).SOPInstanceUID)
    with pytest.raises(ValueError, match="Duplicate"):
        read_series(series)


def test_content_dedup_ignores_generated_uids(series, tmp_path):
    other = write_series(tmp_path / "second", 401)
    assert read_series(series).digest == read_series(other).digest


def test_splits_and_reproducibility():
    values = list(SPLITS.values())
    assert not set(values[0]) & set(values[1])
    assert not set(values[0]) & set(values[2])
    assert not set(values[1]) & set(values[2])
    np.testing.assert_array_equal(phantom(103)[0], phantom(103)[0])
    assert not phantom(100)[1].any()
    assert phantom(101)[1].any()


def test_dice_empty_and_disjoint():
    assert dice(np.zeros(2), np.zeros(2)) == 1
    assert dice(np.array([1, 0]), np.array([0, 1])) == 0
    assert dice(np.ones(2), np.ones(2)) == 1


def test_model_dimensions_and_baseline(checkpoint):
    hu, truth = phantom(401)
    assert normalize(hu).shape == (1, 1, *SHAPE)
    probability = predict(load_model(checkpoint), hu)
    assert probability.shape == SHAPE
    assert np.isfinite(probability).all()
    assert 0 <= probability.min() <= probability.max() <= 1
    assert dice(baseline(hu), truth) > 0.9


def test_idempotency_audit_and_review(series, checkpoint, tmp_path):
    db = tmp_path / "queue.sqlite"
    first = enqueue(db, series, checkpoint)
    assert first["inserted"]
    assert not enqueue(db, series, checkpoint)["inserted"]
    assert first["score"] == 0
    review(db, first["id"])
    with pytest.raises(ValueError, match="already reviewed"):
        review(db, first["id"])
    with pytest.raises(ValueError):
        review(db, "absent")
    with connect(db) as connection:
        assert connection.execute("SELECT count(*) FROM scans").fetchone()[0] == 1
        assert connection.execute("SELECT count(*) FROM events").fetchone()[0] == 2
    assert list_queue(db)[0]["status"] == "reviewed"


def test_queue_order_and_transaction_rollback(tmp_path):
    path = tmp_path / "queue.sqlite"
    with connect(path) as db:
        for id_, score in [("a", 1), ("b", 9), ("c", 4)]:
            db.execute(
                "INSERT INTO scans(id,model_sha,volume_ml,score) VALUES(?,?,?,?)", (id_, "x", 1, score)
            )
    assert [r["id"] for r in list_queue(path)] == ["b", "c", "a"]
    with pytest.raises(RuntimeError), connect(path) as db:
        db.execute("UPDATE scans SET score=99 WHERE id='a'")
        raise RuntimeError("Interrupted")
    assert list_queue(path)[0]["id"] == "b"


def test_model_version_conflict(series, checkpoint, tmp_path):
    path = tmp_path / "queue.sqlite"
    enqueue(path, series, checkpoint)
    model = load_model(checkpoint)
    model.layers[-1].bias.data.fill_(1)
    torch.save({"format": "medimage-v1", "state_dict": model.state_dict()}, checkpoint)
    with pytest.raises(ValueError, match="different model"):
        enqueue(path, series, checkpoint)


def test_report_from_actual_inference(series, checkpoint, tmp_path):
    db = tmp_path / "queue.sqlite"
    enqueue(db, series, checkpoint)
    output = tmp_path / "dashboard.html"
    render(db, checkpoint, [series], output)
    document = output.read_text(encoding="utf-8")
    assert "data:image/png;base64," in document
    assert "RESEARCH ONLY" in document
    assert "SYNTHETIC^PHANTOM" not in document
    assert len(json.loads(output.with_suffix(".json").read_text())) == 1


def test_report_rejects_missing_sources(series, checkpoint, tmp_path):
    db = tmp_path / "queue.sqlite"
    enqueue(db, series, checkpoint)
    with pytest.raises(ValueError, match="Missing source"):
        render(db, checkpoint, [], tmp_path / "bad.html")


def test_report_rejects_wrong_checkpoint(series, checkpoint, tmp_path):
    db = tmp_path / "queue.sqlite"
    enqueue(db, series, checkpoint)
    model = load_model(checkpoint)
    model.layers[-1].bias.data.fill_(1)
    torch.save({"format": "medimage-v1", "state_dict": model.state_dict()}, checkpoint)
    with pytest.raises(ValueError, match="provenance"):
        render(db, checkpoint, [series], tmp_path / "bad.html")


def test_failed_ingestion_leaves_no_scan(series, checkpoint, tmp_path):
    mutate(series, "PatientID", "NOT-SYNTHETIC")
    db = tmp_path / "queue.sqlite"
    with pytest.raises(ValueError):
        enqueue(db, series, checkpoint)
    assert list_queue(db) == []


def test_physical_volume_score(series, checkpoint, tmp_path):
    model = load_model(checkpoint)
    model.layers[-1].bias.data.fill_(1)
    torch.save({"format": "medimage-v1", "state_dict": model.state_dict()}, checkpoint)
    row = enqueue(tmp_path / "queue.sqlite", series, checkpoint)
    expected_ml = np.prod(SHAPE) * 2 / 1000
    assert row["volume_ml"] == pytest.approx(expected_ml)
    assert row["score"] == pytest.approx(100 * expected_ml / (expected_ml + 1))


def test_real_training_smoke(tmp_path):
    metrics = train(tmp_path / "run", epochs=1)
    assert len(metrics["test_cases"]) == 16
    assert np.isfinite(metrics["history"][0]["loss"])
    assert 0 <= metrics["cnn_mean_dice"] <= 1
    assert metrics["parameters"] == 1969
    load_model(tmp_path / "run" / "model.pt")


def test_cli_refuses_overwrite(tmp_path, monkeypatch):
    (tmp_path / "keep.txt").write_text("keep")
    monkeypatch.setattr("sys.argv", ["medimage", "demo", "--out", str(tmp_path)])
    assert main() == 2
    assert (tmp_path / "keep.txt").read_text() == "keep"
