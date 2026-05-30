from __future__ import annotations

import json
import logging
from dataclasses import dataclass
from typing import Any
from urllib import error, request

from app.core.config import Settings

_LOG = logging.getLogger(__name__)


@dataclass
class GptClientState:
    loaded: bool
    model_name: str
    load_error: str | None = None


class GptStructuredClient:
    def __init__(self, settings: Settings) -> None:
        self._settings = settings
        self._state = GptClientState(
            loaded=bool(settings.gpt_api_key.strip()),
            model_name=settings.gpt_model_name.strip() or "gpt-4o-mini",
            load_error=None if settings.gpt_api_key.strip() else "GPT_API_KEY가 설정되지 않았습니다.",
        )

    @property
    def state(self) -> GptClientState:
        return self._state

    def ensure_loaded(self) -> None:
        if self._state.loaded:
            return
        self._state.load_error = self._state.load_error or "GPT_API_KEY가 설정되지 않았습니다."

    def generate_structured_text(self, system_prompt: str, user_prompt: str) -> str:
        self.ensure_loaded()
        if not self._state.loaded:
            raise RuntimeError(self._state.load_error or "GPT client is not ready")

        payload: dict[str, Any] = {
            "model": self._state.model_name,
            "messages": [
                {"role": "system", "content": system_prompt.strip()},
                {"role": "user", "content": user_prompt.strip()},
            ],
            "temperature": 0,
            "response_format": {"type": "json_object"},
        }
        if self._settings.max_new_tokens > 0:
            payload["max_tokens"] = self._settings.max_new_tokens

        endpoint = "https://api.openai.com/v1/chat/completions"
        body = json.dumps(payload).encode("utf-8")
        headers = {
            "Authorization": f"Bearer {self._settings.gpt_api_key.strip()}",
            "Content-Type": "application/json",
        }
        req = request.Request(endpoint, data=body, headers=headers, method="POST")

        try:
            with request.urlopen(req, timeout=self._settings.gpt_timeout_sec) as resp:
                raw = resp.read().decode("utf-8")
        except error.HTTPError as exc:
            detail = exc.read().decode("utf-8", errors="ignore")
            _LOG.warning("GPT structured call failed: %s %s", exc.code, detail)
            raise RuntimeError(f"GPT API 요청 실패: {exc.code} {detail}".strip()) from exc
        except Exception as exc:  # noqa: BLE001
            _LOG.warning("GPT structured call error: %s", exc)
            raise RuntimeError(f"GPT API 요청 실패: {exc}") from exc

        try:
            payload_obj = json.loads(raw)
            choices = payload_obj.get("choices") or []
            if not isinstance(choices, list) or not choices:
                raise RuntimeError("GPT 응답에 choices가 없습니다.")
            message = choices[0].get("message") or {}
            content = message.get("content")
            if isinstance(content, str) and content.strip():
                return content.strip()
        except Exception as exc:  # noqa: BLE001
            raise RuntimeError(f"GPT 응답 파싱 실패: {exc}") from exc

        raise RuntimeError("GPT 응답 본문이 비어 있습니다.")
