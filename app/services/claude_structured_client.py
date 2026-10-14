from __future__ import annotations

import logging
from dataclasses import dataclass

import anthropic

from app.core.config import Settings

_LOG = logging.getLogger(__name__)


@dataclass
class ClaudeClientState:
    loaded: bool
    model_name: str
    load_error: str | None = None


class ClaudeStructuredClient:
    """라우터·예약 추출처럼 "JSON 한 개만" 받는 짧은 호출용 Claude 클라이언트.

    형식은 시스템 프롬프트가 지정하고, 호출자가 extract_json_object로 꺼낸다(예전 GPT 클라이언트와 같은 계약).
    """

    def __init__(self, settings: Settings) -> None:
        self._settings = settings
        api_key = settings.anthropic_api_key.strip()
        self._state = ClaudeClientState(
            loaded=bool(api_key),
            model_name=settings.claude_model_name.strip() or "claude-opus-5-5",
            load_error=None if api_key else "ANTHROPIC_API_KEY가 설정되지 않았습니다.",
        )
        self._client = anthropic.Anthropic(api_key=api_key, timeout=float(settings.claude_timeout_sec)) if api_key else None

    @property
    def state(self) -> ClaudeClientState:
        return self._state

    def ensure_loaded(self) -> None:
        if not self._state.loaded:
            self._state.load_error = self._state.load_error or "ANTHROPIC_API_KEY가 설정되지 않았습니다."

    def generate_structured_text(self, system_prompt: str, user_prompt: str, schema: dict | None = None) -> str:
        """schema를 주면 output_config.format으로 그 JSON 스키마에 맞는 출력을 강제한다(파싱 실패 방지)."""
        self.ensure_loaded()
        if not self._state.loaded or self._client is None:
            raise RuntimeError(self._state.load_error or "Claude client is not ready")

        try:
            response = self._client.beta.messages.create(
                model=self._state.model_name,
                max_tokens=self._settings.max_new_tokens if self._settings.max_new_tokens > 0 else 2048,
                system=system_prompt.strip(),
                messages=[{"role": "user", "content": user_prompt.strip()}],
                # 분류·추출은 짧고 빠르게. 생각 깊이는 effort로만 조절한다(이 모델은 thinking을 끌 수 없다).
                output_config={"effort": "low", **({"format": {"type": "json_schema", "schema": schema}} if schema else {})},
                # 안전 분류기가 거절하면 서버가 대체 모델로 같은 요청을 다시 돌린다.
                betas=["server-side-fallback-2026-07-01"],
                fallbacks="default",
            )
        except anthropic.RateLimitError as exc:
            raise RuntimeError("Claude API 요청 한도 초과") from exc
        except anthropic.APIStatusError as exc:
            _LOG.warning("Claude structured call failed: %s %s", exc.status_code, exc.message)
            raise RuntimeError(f"Claude API 요청 실패: {exc.status_code} {exc.message}") from exc
        except anthropic.APIConnectionError as exc:
            raise RuntimeError(f"Claude API 연결 실패: {exc}") from exc

        if response.stop_reason == "refusal":
            raise RuntimeError("Claude가 요청을 거절했습니다.")
        text = "".join(block.text for block in response.content if block.type == "text").strip()
        if not text:
            raise RuntimeError("Claude 응답 본문이 비어 있습니다.")
        return text
