# Verification and measured results

Recorded September 29, 2026. These results concern procedural geometric phantoms, not clinical imaging or disease detection.

## Completed 30-epoch experiment

Run: `medimage demo --out outputs/demo --epochs 30`.

Environment: Windows, Python 3.11.9, PyTorch 2.6.0+cpu, MONAI 1.4.0, NumPy 1.26.4, two CPU threads. Training initialization seed: 17. Train/validation/test: 48/12/16 disjoint volumes. Four test cases are negative. Checkpoint selection used validation Dice; epoch 30 was selected. The fixed prediction threshold was 0.5.

| Measurement | Observed value |
|---|---:|
| CNN mean test Dice, 16 cases | 0.8850 |
| CNN mean Dice on 12 positive cases | 0.8467 |
| Fixed 100 HU threshold mean test Dice | 0.9973 |
| Negative cases with any predicted foreground | 0 / 4 |
| Selected checkpoint validation Dice | 0.9196 |
| Training plus evaluation, excluding imports and DICOM/report generation | 192.49 seconds |
| Median per-volume CPU inference, 16 observations | 94.01 ms |
| Trainable parameters | 1,969 |

**The threshold baseline outperformed the CNN.** The generator deliberately separates foreground from background by intensity. These measurements do not justify using a learned model for this distribution; the CNN demonstrates the training, evaluation and provenance path. Training remained at background-only validation predictions through epoch 25 before improving in the final epochs. No class-weight adjustment was applied. The committed history records every epoch.

Timings were collected on a shared local machine, include normalization and model inference in the per-volume timer, and exclude initial library imports. They are not a hardware throughput benchmark or a service latency guarantee. Empty/empty masks receive Dice 1, so positive-only Dice is reported separately.

Full per-case results, split seeds, history, environment and checkpoint hash: [metrics.json](metrics.json). The [HTML snapshot](demo.html) uses actual inference from that checkpoint on six additional synthetic display volumes.

## Verification scope

- Local pytest suite: 35 passing tests, including actual DICOM round-trips and one-epoch model training.
- Ruff static checks and dependency consistency (`pip check`): passed.
- CPU 30-epoch end-to-end demo: completed; generated six DICOM series, real checkpoint, queue, metrics and HTML/JSON report.
- Chrome browser checks: six loaded inference previews; All/Pending/Reviewed filters passed; no JavaScript errors; no horizontal overflow at a 390-pixel mobile viewport. Desktop and mobile screenshots were captured.
- Remote CI verification is recorded in the Actions history.
- Local Docker execution was unavailable because the Docker engine was stopped; the CI workflow includes a container build and execution job.

Not evaluated: patient data, external datasets, clinical diagnoses, medical urgency calibration, PACS interoperability, full CT IOD conformance, CUDA, distributed processing, hospital deployment, or regulatory compliance.
