# Third-party dependencies and fixture provenance

The MIT license applies to original project code and original procedural fixtures. No third-party medical images or pretrained weights are bundled. Generated demo images contain only geometric phantoms and local model outputs.

Direct dependencies retain their upstream license terms:

| Dependency | License | Role |
|---|---|---|
| PyTorch | BSD-style | Tensor operations, 3D network and optimization |
| TorchVision | BSD-3-Clause | Slice montage visualization |
| MONAI | Apache-2.0 | Dice plus cross-entropy loss |
| pydicom | MIT | DICOM encoding and decoding |
| NumPy | BSD-3-Clause | Synthetic arrays and metrics |
| Pillow | HPND (PIL Software License) | PNG output |
| pytest | MIT | Tests |
| Ruff | MIT | Static checks |

Python and SQLite have their own distribution terms. Container base images, transitive dependencies and GitHub Actions retain upstream notices. No license grants clinical approval. Exact installed dependency metadata can be inspected with `python -m pip show PACKAGE`. The build does not copy third-party source into this repository.

Technical references:
- [pydicom pixel data documentation](https://pydicom.github.io/pydicom/stable/reference/generated/pydicom.pixels.pixel_array.html)
- [pydicom intensity rescaling](https://pydicom.github.io/pydicom/dev/reference/generated/pydicom.pixels.apply_rescale.html)
- [MONAI loss API](https://docs.monai.io/en/stable/losses.html)
