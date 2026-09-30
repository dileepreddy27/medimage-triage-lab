"""Deliberately narrow DICOM contract; reject ambiguity before inference."""

from dataclasses import dataclass
import hashlib
from pathlib import Path

import numpy as np
import pydicom
from pydicom.errors import InvalidDicomError
from pydicom.uid import CTImageStorage, ExplicitVRLittleEndian

from medimage.synthetic import SHAPE, SPACING


@dataclass
class Volume:
    hu: np.ndarray
    spacing: tuple[float, float, float]
    digest: str


def read_series(folder: Path) -> Volume:
    paths = sorted(folder.glob("*.dcm"))
    if len(paths) != SHAPE[0]:
        raise ValueError(f"Expected exactly {SHAPE[0]} single-frame DICOM slices")
    if any(p.stat().st_size > 1_000_000 for p in paths):
        raise ValueError("Slice exceeds the 1 MB research input limit")
    try:
        slices = [pydicom.dcmread(p) for p in paths]
        for ds in slices:
            if ds.PatientID != "SYNTHETIC" or "SYNTHETIC" not in ds.ImageType:
                raise ValueError("Only explicitly marked synthetic phantoms are accepted")
            if ds.Modality != "CT" or ds.SOPClassUID != CTImageStorage:
                raise ValueError("Only CT Image Storage is supported")
            if ds.file_meta.TransferSyntaxUID != ExplicitVRLittleEndian:
                raise ValueError("Only uncompressed explicit little-endian transfer syntax is supported")
            if int(getattr(ds, "NumberOfFrames", 1)) != 1:
                raise ValueError("Multi-frame DICOM is unsupported")
            if (ds.Rows, ds.Columns) != SHAPE[1:] or ds.SamplesPerPixel != 1:
                raise ValueError("Expected 32 by 32 monochrome slices")
            if ds.PhotometricInterpretation != "MONOCHROME2":
                raise ValueError("Only MONOCHROME2 is supported")
            if not np.allclose(ds.ImageOrientationPatient, [1, 0, 0, 0, 1, 0], atol=1e-6):
                raise ValueError("Only axial orientation is supported")
            if not np.allclose(ds.PixelSpacing, SPACING[1:]):
                raise ValueError("Expected 1 mm in-plane spacing")
            if not np.isfinite(float(ds.RescaleSlope)) or float(ds.RescaleSlope) <= 0:
                raise ValueError("Rescale slope must be finite and positive")
            if not np.isfinite(float(ds.RescaleIntercept)) or "ModalityLUTSequence" in ds:
                raise ValueError("Expected finite linear intensity rescale")
        for key in ("SeriesInstanceUID", "StudyInstanceUID", "FrameOfReferenceUID"):
            if len({str(getattr(d, key)) for d in slices}) != 1:
                raise ValueError("Mixed series, study or reference frame")
        if len({str(d.SOPInstanceUID) for d in slices}) != len(slices):
            raise ValueError("Duplicate SOP instance")
        slices.sort(key=lambda d: float(d.ImagePositionPatient[2]))
        positions = np.asarray([d.ImagePositionPatient for d in slices], dtype=float)
        if not np.isfinite(positions).all() or not np.allclose(positions[:, :2], positions[0, :2]):
            raise ValueError("Invalid positions or gantry shift")
        if not np.allclose(np.diff(positions[:, 2]), SPACING[0], atol=1e-4):
            raise ValueError("Missing, duplicate or irregular slice positions")
        hu = np.stack(
            [
                d.pixel_array.astype(np.float32) * float(d.RescaleSlope) + float(d.RescaleIntercept)
                for d in slices
            ]
        )
        if not np.isfinite(hu).all():
            raise ValueError("Nonfinite intensities")
    except (AttributeError, KeyError, TypeError, OSError, InvalidDicomError) as exc:
        raise ValueError("Missing or invalid required DICOM attributes") from exc
    # Hash canonical voxel content and geometry, never persist patient identifiers.
    digest = hashlib.sha256(
        hu.astype("<f4").tobytes() + np.asarray(SPACING, dtype="<f4").tobytes()
    ).hexdigest()
    return Volume(hu, SPACING, digest)
