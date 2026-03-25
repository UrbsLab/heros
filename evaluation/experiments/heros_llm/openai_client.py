"""OpenAI-first client wrapper for generation and judge scoring."""

from __future__ import annotations

from datetime import datetime, timezone
import json
import os
from typing import Any, Dict

from .config import LLMConfig


class OpenAIClientWrapper:
    """Small wrapper around the OpenAI Chat Completions API."""

    def __init__(self, config: LLMConfig):
        self.config = config
        self._client = None

    def _get_client(self) -> Any:
        if self._client is not None:
            return self._client
        try:
            from openai import OpenAI
        except ImportError as exc:
            raise RuntimeError(
                "The 'openai' package is required for live generation. "
                "Install it in the runtime environment before running the experiment."
            ) from exc

        api_key = os.environ.get(self.config.api_key_env)
        if not api_key:
            raise RuntimeError(
                "Environment variable {0} is not set.".format(self.config.api_key_env)
            )

        timeout = getattr(self.config, "request_timeout_seconds", 60)
        self._client = OpenAI(api_key=api_key, timeout=timeout)
        return self._client

    @staticmethod
    def _response_to_dict(response: Any) -> Dict[str, Any]:
        if hasattr(response, "model_dump"):
            return response.model_dump()
        if hasattr(response, "json"):
            try:
                return json.loads(response.json())
            except Exception:
                pass
        return {"raw_response": str(response)}

    def generate_text(self, system_prompt: str, user_prompt: str) -> Dict[str, Any]:
        """Generate text from an OpenAI-compatible chat completion endpoint."""
        client = self._get_client()
        response = client.chat.completions.create(
            model=self.config.model,
            temperature=self.config.temperature,
            max_tokens=self.config.max_tokens,
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt},
            ],
        )
        text = ""
        if getattr(response, "choices", None):
            text = response.choices[0].message.content or ""
        return {
            "text": text.strip(),
            "created_at": datetime.now(timezone.utc).isoformat(),
            "raw_response": self._response_to_dict(response),
            "model_name": self.config.model,
            "temperature": self.config.temperature,
        }
