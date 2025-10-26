"""
Outfit styling prompt generation powered by the OpenAI Chat Completions API.
"""

from __future__ import annotations

from typing import Final

from openai_client import get_openai_client

SYSTEM_PROMPT: Final[
    str
] = (
    "You are a helpful stylist crafting concise outfit descriptions to feed "
    "into an image generation model."
)


def build_styling_prompt(gender: str, height_cm: str, garment: str) -> str:
    """Format the user-provided attributes into a compact natural-language prompt."""

    return (
        f"I am a {gender}. My height is {height_cm} cm. Give me a short idea for "
        f"styling my {garment}. Respond with a comma-separated list of clothing "
        "pieces and nothing else."
    )


def get_styling_advice(gender: str, height_cm: str, garment: str) -> str:
    """
    Query GPT for a concise outfit suggestion tailored to the detected garment.

    Returns the raw response string so the caller can post-process it for the
    image generation pipeline.
    """

    client = get_openai_client()
    response = client.chat.completions.create(
        model="gpt-3.5-turbo",
        messages=[
            {"role": "system", "content": [{"type": "text", "text": SYSTEM_PROMPT}]},
            {"role": "user", "content": [{"type": "text", "text": build_styling_prompt(gender, height_cm, garment)}]},
        ],
        temperature=0.9,
        max_tokens=120,
        top_p=1,
        frequency_penalty=0,
        presence_penalty=0,
    )
    return response.choices[0].message.content or ""
