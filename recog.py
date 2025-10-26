"""
Inference utilities for the apparel classification convolutional network.
"""

from __future__ import annotations

import logging
from functools import lru_cache
from pathlib import Path
from typing import Final

import torch
import torch.nn as nn
from PIL import Image
from torchvision import transforms

logger = logging.getLogger(__name__)

MODEL_PATH: Final[Path] = Path("recognition-models/model_checkpoint_v2.pth")

CLASS_NAMES: Final[list[str]] = [
    "dress",
    "hoodie",
    "pants",
    "shirt",
    "shoes",
    "shorts",
    "skirt",
    "suit",
]


class ApparelModel(nn.Module):
    """Convolutional neural network used for garment classification."""

    def __init__(self) -> None:
        super().__init__()
        self.conv_layers = nn.Sequential(
            nn.Conv2d(3, 512, kernel_size=3),
            nn.ReLU(),
            nn.BatchNorm2d(512),
            nn.MaxPool2d(2),
            nn.Dropout(0.2),
            nn.Conv2d(512, 256, kernel_size=3),
            nn.ReLU(),
            nn.BatchNorm2d(256),
            nn.MaxPool2d(2),
            nn.Dropout(0.2),
            nn.Conv2d(256, 128, kernel_size=3),
            nn.ReLU(),
            nn.BatchNorm2d(128),
            nn.MaxPool2d(2),
            nn.Dropout(0.2),
            nn.Conv2d(128, 128, kernel_size=3),
            nn.ReLU(),
            nn.BatchNorm2d(128),
            nn.MaxPool2d(2),
        )
        self.fc_layers = nn.Sequential(
            nn.Flatten(),
            nn.Linear(128 * 14 * 14, 64),
            nn.ReLU(),
            nn.BatchNorm1d(64),
            nn.Dropout(0.2),
            nn.Linear(64, len(CLASS_NAMES)),
            nn.Softmax(dim=1),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.conv_layers(x)
        return self.fc_layers(x)


@lru_cache(maxsize=1)
def _load_model() -> ApparelModel:
    """Instantiate the network and load pretrained weights."""

    model = ApparelModel()
    if not MODEL_PATH.exists():
        raise FileNotFoundError(
            f"Model checkpoint not found at {MODEL_PATH.resolve()}. "
            "Ensure the training script has exported the weights."
        )

    state_dict = torch.load(MODEL_PATH, map_location=torch.device("cpu"))
    model.load_state_dict(state_dict)
    model.eval()
    logger.info("Loaded apparel classifier from %s", MODEL_PATH)
    return model


def _preprocess() -> transforms.Compose:
    """Build the deterministic preprocessing pipeline once."""

    return transforms.Compose(
        [
            transforms.Resize((256, 256)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ]
    )


@lru_cache(maxsize=1)
def _get_preprocess() -> transforms.Compose:
    return _preprocess()


def load_image(image_path: str) -> torch.Tensor:
    """Convert an image on disk into a model-ready tensor batch."""

    image = Image.open(image_path).convert("RGB")
    tensor = _get_preprocess()(image)
    return tensor.unsqueeze(0)


def predict(image_path: str) -> str:
    """Predict the garment category present in the supplied image."""

    model = _load_model()
    image_tensor = load_image(image_path)

    with torch.no_grad():
        outputs = model(image_tensor)
        _, predicted_class = torch.max(outputs, dim=1)

    predicted_label = CLASS_NAMES[predicted_class.item()]
    logger.debug("Predicted garment '%s' for %s", predicted_label, image_path)
    return predicted_label
