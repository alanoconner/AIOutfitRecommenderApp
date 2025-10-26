"""
Reusable OpenAI client factory.

The project communicates with both the Chat Completions and Images APIs, so a
shared client keeps authentication in one place.
"""

from __future__ import annotations

from functools import lru_cache

from openai import OpenAI

from config import get_settings


@lru_cache(maxsize=1)
def get_openai_client() -> OpenAI:
    """Return a lazily constructed and cached OpenAI client."""

    settings = get_settings()
    return OpenAI(api_key=settings.openai_api_key)
