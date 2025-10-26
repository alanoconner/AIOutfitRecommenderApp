"""
Helper functions for persisting qualitative user feedback.
"""

from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Any, Dict, Optional
from urllib.parse import unquote, urlparse

import requests

logger = logging.getLogger(__name__)

EVALUATION_FILE = Path("evaluationData.jsonl")


def download_generated_image(url: str, folder_path: str, file_name: Optional[str] = None) -> str:
    """
    Download the generated image referenced by the public CDN URL.

    Returns the local filepath where the asset was stored.
    """

    target_dir = Path(folder_path)
    target_dir.mkdir(parents=True, exist_ok=True)

    response = requests.get(url, stream=True, timeout=15)
    if response.status_code != 200:
        raise RuntimeError(f"Failed to download generated image: HTTP {response.status_code}")

    if not file_name:
        parsed_url = urlparse(url)
        file_name = Path(unquote(parsed_url.path)).name or "generated_image"

    output_path = target_dir / file_name
    with open(output_path, "wb") as file_pointer:
        for chunk in response.iter_content(chunk_size=1024):
            file_pointer.write(chunk)

    logger.info("Stored generated image at %s", output_path)
    return str(output_path)


def save_eval_results(results: Dict[str, Any], generated_filepath: str) -> None:
    """Append a single evaluation entry to the JSONL file."""

    enriched = dict(results)
    enriched["generatedImageLink"] = generated_filepath
    append_to_jsonl(EVALUATION_FILE, enriched)


def append_to_jsonl(file_path: Path | str, json_data: Dict[str, Any]) -> None:
    """Append a JSON serialisable dictionary as a line to the file."""

    file_path = Path(file_path)
    file_path.parent.mkdir(parents=True, exist_ok=True)
    with open(file_path, "a", encoding="utf-8") as handle:
        handle.write(json.dumps(json_data) + "\n")
