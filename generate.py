"""
Image generation helper built on top of the OpenAI Images API.
"""

from __future__ import annotations

from openai_client import get_openai_client


def generate_image_link(prompt: str) -> str:
    """
    Request a single DALL·E image generation and return its public URL.

    Parameters
    ----------
    prompt:
        A textual description of the outfit to generate.
    """

    client = get_openai_client()
    response = client.images.generate(
        model="dall-e-3",
        prompt=prompt,
        size="1024x1024",
        quality="standard",
        n=1,
    )

    return response.data[0].url
