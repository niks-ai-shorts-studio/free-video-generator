
"""OpenRouter free-only Chat API wrapper.

Preserves the AgnesChatAPI interface used by core/screenwriter.py.
All text and image-understanding requests use OpenRouter's free router.
No paid-model fallback is configured.
"""

import base64
import json
import logging
import mimetypes
import os
import re
from typing import List

import requests

logger = logging.getLogger(__name__)

OPENROUTER_URL = "https://openrouter.ai/api/v1/chat/completions"

# Fixed free-only router. Do not replace with a paid model ID.
FREE_MODEL = "openrouter/free"

_JSON_BLOCK_RE = re.compile(r"\{[\s\S]*\}")


def strip_code_fence(text: str) -> str:
    """Remove Markdown code fences from an LLM response."""
    text = (text or "").strip()

    if text.startswith("```"):
        first_newline = text.find("\n")
        text = text[first_newline + 1:] if first_newline >= 0 else ""

    if text.endswith("```"):
        text = text[:-3]

    return text.strip()


class AgnesChatAPI:
    """Compatibility wrapper using OpenRouter free models."""

    def __init__(self, api_key: str = "", model: str = "agnes-2.0-flash"):
        # Keep the old constructor signature for screenwriter.py.
        # Never use the Agnes key for OpenRouter requests.
        self.api_key = os.environ.get("OPENROUTER_API_KEY", "").strip()
        self.model = FREE_MODEL

    def _headers(self) -> dict:
        if not self.api_key:
            raise RuntimeError(
                "OPENROUTER_API_KEY is missing. Add it to the Render "
                "environment variables for niks-agnes-engine."
            )

        return {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json",
            "HTTP-Referer": "https://niks-agnes-engine.onrender.com",
            "X-Title": "Travel with Niks AI Studio",
        }

    def _image_to_b64_uri(self, path: str) -> str:
        with open(path, "rb") as image_file:
            b64 = base64.b64encode(image_file.read()).decode("utf-8")

        mime = mimetypes.guess_type(path)[0] or "image/png"
        return f"data:{mime};base64,{b64}"

    @staticmethod
    def _extract_content(data: dict) -> str:
        try:
            content = data["choices"][0]["message"]["content"]
        except (KeyError, IndexError, TypeError) as exc:
            raise RuntimeError(
                "OpenRouter returned an unexpected response format."
            ) from exc

        if isinstance(content, str):
            return content

        # Some compatible models return content as a list of text blocks.
        if isinstance(content, list):
            parts = []
            for item in content:
                if isinstance(item, dict) and item.get("text"):
                    parts.append(item["text"])
            if parts:
                return "\n".join(parts)

        raise RuntimeError("OpenRouter returned an empty text response.")

    def _request(self, messages: list, max_tokens: int = 4096,
                 timeout: int = 180) -> str:
        """Make one request through the free-only OpenRouter router.

        No paid fallback and no long retry delays. If free capacity is
        unavailable, return an error rather than spending money.
        """
        payload = {
            "model": FREE_MODEL,
            "messages": messages,
            "max_tokens": max_tokens,
        }

        logger.info(
            "[OpenRouter] Free-only request (%s), messages: %s",
            FREE_MODEL,
            len(messages),
        )

        try:
            response = requests.post(
                OPENROUTER_URL,
                headers=self._headers(),
                json=payload,
                timeout=timeout,
            )
            response.raise_for_status()
            data = response.json()
            return self._extract_content(data)

        except requests.HTTPError as exc:
            status = (
                exc.response.status_code
                if exc.response is not None
                else "unknown"
            )
            body = (
                exc.response.text[:700]
                if exc.response is not None
                else str(exc)
            )
            logger.error(
                "[OpenRouter] HTTP %s: %s",
                status,
                body,
            )
            raise RuntimeError(
                f"OpenRouter free model request failed (HTTP {status}). "
                f"Free models may be busy or rate-limited. Details: {body}"
            ) from exc

        except requests.Timeout as exc:
            logger.error("[OpenRouter] Request timed out.")
            raise RuntimeError(
                "OpenRouter free model request timed out. Please try again."
            ) from exc

        except requests.ConnectionError as exc:
            logger.error("[OpenRouter] Connection failed: %s", exc)
            raise RuntimeError(
                "Could not connect to OpenRouter. Please try again later."
            ) from exc

    def chat(self, system_prompt: str, user_prompt: str,
             max_tokens: int = 4096) -> str:
        """Text chat using free models only."""
        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt},
        ]

        return self._request(
            messages,
            max_tokens=max_tokens,
            timeout=180,
        )

    def chat_json(self, system_prompt: str, user_prompt: str,
                  max_tokens: int = 4096) -> dict:
        """Chat and parse a JSON response, with one format retry."""
        for attempt in range(2):
            content = self.chat(
                system_prompt,
                user_prompt,
                max_tokens=max_tokens,
            )

            cleaned = strip_code_fence(content)

            try:
                result = json.loads(cleaned)
                if isinstance(result, dict):
                    return result
            except (json.JSONDecodeError, ValueError):
                pass

            match = _JSON_BLOCK_RE.search(cleaned)
            if match:
                try:
                    result = json.loads(match.group())
                    if isinstance(result, dict):
                        return result
                except (json.JSONDecodeError, ValueError):
                    pass

            if attempt == 0:
                logger.warning(
                    "[OpenRouter] JSON parse failed; retrying once."
                )
                continue

            raise ValueError(
                "[OpenRouter] Could not parse JSON after 2 attempts. "
                f"Response preview: {content[:200]}"
            )

        raise ValueError("[OpenRouter] Unexpected JSON parsing flow.")

    def chat_multimodal(
        self,
        system_prompt: str,
        text_prompt: str,
        image_paths: List[str],
        max_tokens: int = 4096,
    ) -> str:
        """Image and text understanding using a free vision-capable model."""

        user_content = [
            {"type": "text", "text": text_prompt}
        ]

        for img_path in image_paths:
            if img_path.startswith(("http://", "https://")):
                image_url = img_path
            elif os.path.isfile(img_path):
                image_url = self._image_to_b64_uri(img_path)
            else:
                logger.warning(
                    "[OpenRouter] Skipping missing image: %s",
                    img_path,
                )
                continue

            user_content.append({
                "type": "image_url",
                "image_url": {"url": image_url},
            })

        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_content},
        ]

        logger.info(
            "[OpenRouter] Free multimodal request, %s image(s).",
            len(user_content) - 1,
        )

        return self._request(
            messages,
            max_tokens=max_tokens,
            timeout=300,
        )
