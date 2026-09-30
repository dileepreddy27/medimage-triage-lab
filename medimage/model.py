import numpy as np
import torch
from torch import nn


def normalize(hu: np.ndarray) -> torch.Tensor:
    return torch.from_numpy(np.clip(hu, -200, 200).astype(np.float32) / 200)[None, None]


class TinySegmenter(nn.Module):
    """Two spatial 3D convolutions plus a pointwise head; 1,969 parameters."""

    def __init__(self):
        super().__init__()
        self.layers = nn.Sequential(
            nn.Conv3d(1, 8, 3, padding=1),
            nn.ReLU(),
            nn.Conv3d(8, 8, 3, padding=1),
            nn.ReLU(),
            nn.Conv3d(8, 1, 1),
        )

    def forward(self, x):
        return self.layers(x)


def dice(pred: np.ndarray, target: np.ndarray) -> float:
    pred, target = pred.astype(bool), target.astype(bool)
    denominator = int(pred.sum()) + int(target.sum())
    return 1.0 if denominator == 0 else float(2 * np.logical_and(pred, target).sum() / denominator)


def baseline(hu: np.ndarray) -> np.ndarray:
    """Fixed 100 HU threshold; intentionally strong on this simple synthetic distribution."""
    return hu > 100


def load_model(path):
    model = TinySegmenter()
    checkpoint = torch.load(path, map_location="cpu", weights_only=True)
    if checkpoint.get("format") != "medimage-v1":
        raise ValueError("Unsupported checkpoint format")
    model.load_state_dict(checkpoint["state_dict"])
    return model.eval()


@torch.inference_mode()
def predict(model, hu):
    return torch.sigmoid(model(normalize(hu)))[0, 0].numpy()
