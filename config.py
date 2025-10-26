"""
Application-wide configuration helpers.

This module centralises how environment variables are retrieved so the rest of
the codebase can remain clean and explicit about the settings it depends on.
"""

from __future__ import annotations

import os
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path


BASE_DIR = Path(__file__).resolve().parent


@dataclass(frozen=True)
class Settings:
    """Container for immutable runtime configuration values."""

    openai_api_key: str
    uploads_dir: Path


@lru_cache(maxsize=1)
def get_settings() -> Settings:
    """
    Build the settings object from environment variables.

    The function is cached so repeated calls across modules are inexpensive,
    while still allowing unit tests to override environment variables between
    runs if needed.
    """

    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key:
        raise RuntimeError(
            "Environment variable OPENAI_API_KEY must be set before starting the application."
        )

    uploads_dir = Path(os.getenv("UPLOADS_DIR", BASE_DIR / "uploads")).resolve()
    uploads_dir.mkdir(parents=True, exist_ok=True)

    return Settings(openai_api_key=api_key, uploads_dir=uploads_dir)
