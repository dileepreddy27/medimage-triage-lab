# Security and data boundary

This is a local synthetic-data research demo. Do not upload or process patient data. The `SYNTHETIC` metadata guard is not an anonymizer, an authenticity check, or a compliance control. Outputs omit patient identifiers, but input files themselves remain on disk.

Only load checkpoints you generated yourself. `torch.load(weights_only=True)` restricts checkpoint decoding; it does not make arbitrary third-party files trusted. Direct dependency versions are pinned for repeatability, not a guarantee that they are free of vulnerabilities. Keep this demo off public network services.

The DICOM reader limits slice count and per-file size, restricts supported encodings and geometry, and refuses mixed volumes. The SQLite API parameterizes queries and records workflow transitions atomically. Audit events are not cryptographically signed or protected against direct database modification. HTML report fields are escaped and images are embedded locally.

There is no authentication, encryption-at-rest layer, clinical retention policy, or production incident-response system. Report code defects through GitHub issues without including private data, scan files, credentials, or patient information.
