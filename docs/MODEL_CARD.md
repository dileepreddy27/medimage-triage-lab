# Model card: TinySegmenter v1

## Intended use

Education, engineering demonstrations, and reproducible tests of a synthetic imaging queue. Not for diagnosis, treatment, prioritization of real patient scans, or medical research conclusions about any disease. No disease classes are supported.

## Architecture and training

Input N × 1 × 24 × 32 × 32. Clip synthetic HU-like intensities to [-200, 200] and divide by 200. Conv3d(1,8,3) → ReLU → Conv3d(8,8,3) → ReLU → Conv3d(8,1,1); spatial padding preserves the grid. 1,969 trainable parameters. MONAI DiceCELoss(sigmoid=True), Adam at 0.003, batches of eight, CPU, deterministic PyTorch algorithms, initialization seed 17. No pretrained weights. Best validation mean Dice selects the checkpoint; output sigmoid is thresholded at 0.5.

## Data provenance

Original NumPy procedural generator. One unique seed per volume; train, validation, test and display seed ranges are disjoint. There are no patient records, real anatomy, sensitive metadata, or downloaded datasets. Positive cases have simple bright ellipsoids. One quarter of each evaluation split is negative. Geometry, contrast and noise have limited variation and do not model clinical acquisition artifacts.

## Evaluation

Per-case Dice plus mean Dice, positive-only mean Dice, negative false-positive cases, segmented voxel counts, and CPU inference median. The fixed 100 HU baseline is evaluated on the same held-out volumes. Both methods benefit from the generator's deliberately simple intensity separation. Metrics are not sensitivity/specificity for any disease. No external validation, calibration, subgroup fairness assessment, prospective study, or real-world uncertainty model exists.

## Risks and limitations

The model could fail completely on real scans, different spacing, contrast, modalities or artifacts. The DICOM reader refuses unsupported inputs. Tag checks do not authenticate that a file is synthetic and do not remove PHI. The burden score has no clinical basis. Larger predicted masks mechanically increase the queue score, including false positives. Negative cases can therefore be misranked. Humans must not use these outputs for clinical decisions.

## Reproducibility

Run the 30-epoch demo to regenerate the checkpoint and evaluation. The checkpoint is intentionally excluded from Git: it is small and recreated locally. DICOM UIDs and event timestamps change between runs. Results can differ slightly across numerical libraries and platforms; direct versions are pinned and the measured environment is recorded in RESULTS.md.
