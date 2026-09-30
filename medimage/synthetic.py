"""Original procedurally generated phantoms: no patient or downloaded data."""

from pathlib import Path

import numpy as np
from pydicom.dataset import FileDataset, FileMetaDataset
from pydicom.uid import CTImageStorage, ExplicitVRLittleEndian, generate_uid

SHAPE = (24, 32, 32)
SPACING = (2.0, 1.0, 1.0)  # z, y, x millimetres
SPLITS = {"train": list(range(100, 148)), "validation": list(range(200, 212)), "test": list(range(300, 316))}


def phantom(seed: int) -> tuple[np.ndarray, np.ndarray]:
    rng = np.random.default_rng(seed)
    z, y, x = np.indices(SHAPE)
    body = ((z - 12) / 11) ** 2 + ((y - 16) / 14) ** 2 + ((x - 16) / 14) ** 2 < 1
    hu = rng.normal(-850, 12, SHAPE)
    hu[body] = rng.normal(35, 14, int(body.sum()))
    mask = np.zeros(SHAPE, dtype=bool)
    # Every fourth seed is a negative case. Split seeds never overlap.
    if seed % 4:
        center = rng.uniform([8, 10, 10], [16, 22, 22])
        radius = rng.uniform(2.0, 4.5)
        mask = (((z - center[0]) * 1.25) ** 2 + (y - center[1]) ** 2 + (x - center[2]) ** 2) < radius**2
        hu[mask] = rng.normal(145, 18, int(mask.sum()))
    return np.rint(hu).astype(np.int16), mask.astype(np.float32)


def write_series(folder: Path, seed: int) -> Path:
    folder.mkdir(parents=True, exist_ok=True)
    if any(folder.iterdir()):
        raise ValueError("Synthetic output directory must be empty")
    hu, _ = phantom(seed)
    series, study, frame = generate_uid(), generate_uid(), generate_uid()
    for z, pixels in enumerate(hu):
        sop = generate_uid()
        meta = FileMetaDataset()
        meta.TransferSyntaxUID = ExplicitVRLittleEndian
        meta.MediaStorageSOPClassUID = CTImageStorage
        meta.MediaStorageSOPInstanceUID = sop
        ds = FileDataset(None, {}, file_meta=meta, preamble=b"\0" * 128)
        ds.SOPClassUID, ds.SOPInstanceUID = CTImageStorage, sop
        ds.StudyInstanceUID, ds.SeriesInstanceUID = study, series
        ds.FrameOfReferenceUID = frame
        ds.PatientName, ds.PatientID = "SYNTHETIC^PHANTOM", "SYNTHETIC"
        ds.Modality = "CT"
        ds.SeriesDescription = "MedImage procedural research phantom"
        ds.ImageType = ["DERIVED", "SECONDARY", "SYNTHETIC"]
        ds.Rows, ds.Columns = pixels.shape
        ds.SamplesPerPixel, ds.PhotometricInterpretation = 1, "MONOCHROME2"
        ds.BitsAllocated, ds.BitsStored, ds.HighBit, ds.PixelRepresentation = 16, 16, 15, 1
        ds.PixelSpacing, ds.SliceThickness = [1, 1], 2
        ds.ImageOrientationPatient = [1, 0, 0, 0, 1, 0]
        ds.ImagePositionPatient = [0, 0, z * 2]
        ds.InstanceNumber = z + 1
        ds.RescaleSlope, ds.RescaleIntercept = 1, 0
        ds.PixelData = pixels.tobytes()
        ds.save_as(folder / f"slice-{23 - z:03d}.dcm", enforce_file_format=True)
    return folder
