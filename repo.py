"""
Pipeline utilities for generating personalised outfit imagery.

The module coordinates garment recognition, colour analysis, prompt crafting,
and final image generation through OpenAI's APIs.
"""

from __future__ import annotations

import logging
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Sequence, Tuple

import cv2
import numpy as np
from sklearn.cluster import KMeans

import constants.colors as colors
from generate import generate_image_link
from recog import predict as predict_garment
from styling import get_styling_advice

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class OutfitGenerationResult:
    """Structured representation of an outfit generation request."""

    image_link: str
    prompt: str
    garment: str
    colour: str
    styling_advice: str


def generator(image_path: str, gender: str, height_cm: str) -> Tuple[str, str]:
    """
    Backwards compatible wrapper returning only the link and prompt.

    Existing callers expect this tuple, so we expose a thin adapter on top of
    the richer `generate_outfit` function implemented below.
    """

    result = generate_outfit(Path(image_path), gender, height_cm)
    return result.image_link, result.prompt


def generate_outfit(image_path: Path, gender: str, height_cm: str) -> OutfitGenerationResult:
    """
    Run the full inference pipeline and return all intermediary artefacts.

    Parameters
    ----------
    image_path:
        Path to the garment photo uploaded by the user.
    gender:
        User-provided gender label used to tailor styling advice.
    height_cm:
        User-provided height in centimetres, passed through to GPT.
    """

    if not image_path.exists():
        raise FileNotFoundError(f"Input image does not exist: {image_path}")

    garment = predict_garment(str(image_path))
    colour = detect_colour(image_path)
    raw_advice = get_styling_advice(gender, height_cm, garment)
    advice = sanitise_gpt_advice(raw_advice)
    prompt = build_image_prompt(gender, garment, colour, advice)
    image_link = generate_image_link(prompt)

    logger.info("Generated outfit: garment=%s colour=%s", garment, colour)

    return OutfitGenerationResult(
        image_link=image_link,
        prompt=prompt,
        garment=garment,
        colour=colour,
        styling_advice=advice,
    )


def build_image_prompt(gender: str, garment: str, colour: str, advice: str) -> str:
    """Combine all contextual data points into a single DALLE prompt."""

    return (
        f"A {gender} wearing {colour} {garment}, {advice}. "
        "Background is a city street. One person. Photorealistic."
    )


def sanitise_gpt_advice(advice: str) -> str:
    """Remove newlines and redundant whitespace from the GPT response."""

    return re.sub(r"\s+", " ", advice).strip()


def detect_colour(image_path: Path, crop_fraction: float = 0.5, clusters: int = 3) -> str:
    """
    Estimate the dominant colour inside the central crop of the image.

    Using a crop helps reduce the influence of the background on garments that
    occupy the centre of the frame.
    """

    image = cv2.imread(str(image_path))
    if image is None:
        raise ValueError(f"Unable to load image for colour detection: {image_path}")

    rgb_image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    cropped_image = crop_center(rgb_image, crop_fraction)
    representative_colour = dominant_colour(cropped_image, clusters)
    return closest_named_colour(representative_colour)


def crop_center(image: np.ndarray, crop_fraction: float) -> np.ndarray:
    """Return a central square crop of the image."""

    height, width, _ = image.shape
    start_x = int(width * (1 - crop_fraction) / 2)
    start_y = int(height * (1 - crop_fraction) / 2)
    crop_width = int(width * crop_fraction)
    crop_height = int(height * crop_fraction)
    return image[start_y : start_y + crop_height, start_x : start_x + crop_width]


def dominant_colour(image: np.ndarray, clusters: int) -> np.ndarray:
    """Cluster pixel colours and return the representative RGB triplet."""

    pixels = image.reshape(-1, 3)
    model = KMeans(n_clusters=clusters, n_init=10)
    model.fit(pixels)
    counts = np.bincount(model.labels_)
    return model.cluster_centers_[np.argmax(counts)].astype(int)


def closest_named_colour(candidate: Sequence[int]) -> str:
    """Map an arbitrary RGB value to the nearest named colour in our palette."""

    palette = colors.get_color()

    best_distance = float("inf")
    best_colour = "unknown"
    for (r_c, g_c, b_c), name in palette.items():
        distance = (r_c - candidate[0]) ** 2 + (g_c - candidate[1]) ** 2 + (b_c - candidate[2]) ** 2
        if distance < best_distance:
            best_distance = distance
            best_colour = name
    return best_colour
