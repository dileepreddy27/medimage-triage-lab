# Evidence-backed resume bullets

- Built a Python research pipeline that validates synthetic DICOM series, runs a custom PyTorch 3D segmentation CNN trained with MONAI, and exports ranked SQLite queues with model hashes, idempotent ingestion, review history, and actual inference overlays.
- Implemented reproducible synthetic train/validation/test splits and a fixed intensity-threshold baseline, with CPU workflow tests and Docker/GitHub Actions checks; explicitly scoped the system away from clinical diagnosis and unimplemented PACS integration.

Add numeric test or evaluation results only from the committed RESULTS.md and the corresponding verified Actions run. Do not describe this as clinically validated, HIPAA compliant, deployed to hospitals, or capable of detecting named diseases.
