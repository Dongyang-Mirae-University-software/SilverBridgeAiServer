from __future__ import annotations

import logging
from dataclasses import dataclass

import anthropic

from app.core.config import Settings

_LOG = logging.getLogger(__name__)

DEFAULT_MODEL = "claude-opus-5-5"
# 서버 측 거절 대체(server-side fallback)가 없는 모델. 보내지 말라는 것이 공식 안내다(haiku 5.5).
_NO_SERVER_FALLBACK_MARKERS = ("haiku",)


class StructuredCallError(RuntimeError):
    """라우터·예약 추출 호출 실패. cause는 고정 코드(max_tokens·parse·unknown_route·api_<상태>·rate_limited·refusal·connection·
    not_ready·empty)라 로그에 그대로 남겨도 된다 - 입력·모델 출력·API 오류 본문은 담지 않는다."""

    def __init__(self, cause: str) -> None:
        super().__init__(cause)
        self.cause = cause


def failure_cause(exc: BaseException) -> str:
    """로그용 실패 원인 코드. 우리 쪽 예외가 아니면 클래스명만."""
    return exc.cause if isinstance(exc, StructuredCallError) else type(exc).__name__


def supports_server_fallback(model_name: str) -> bool:
    return not any(marker in model_name.lower() for marker in _NO_SERVER_FALLBACK_MARKERS)


@dataclass
class ClaudeClientState:
    loaded: bool
    model_name: str
    load_error: str | None = None


class ClaudeStructuredClient:
    """라우터·예약 추출처럼 "JSON 한 개만" 받는 짧은 호출용 Claude 클라이언트.

    형식은 시스템 프롬프트가 지정하고, 호출자가 extract_json_object로 꺼낸다(예전 GPT 클라이언트와 같은 계약).
    """

    def __init__(self, settings: Settings, model_name: str | None = None) -> None:
        """model_name을 주지 않으면 예약 추출용 모델(RESERVATION_EXTRACT_MODEL)을 쓴다. 라우터는 자기 모델을 넘긴다."""
        self._settings = settings
        api_key = settings.anthropic_api_key.strip()
        chosen = (model_name if model_name is not None else settings.reservation_extract_model).strip()
        self._state = ClaudeClientState(
            loaded=bool(api_key),
            model_name=chosen or DEFAULT_MODEL,
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
            raise StructuredCallError("not_ready")

        try:
            response = self._client.beta.messages.create(**self._request_kwargs(system_prompt, user_prompt, schema))
        except anthropic.RateLimitError as exc:
            raise StructuredCallError("rate_limited") from exc
        except anthropic.APIStatusError as exc:
            # 오류 본문(exc.message)에는 입력이 섞일 수 있어 상태 코드·모델 이름만 남긴다.
            _LOG.warning("Claude structured call failed: status=%s model=%s", exc.status_code, self._state.model_name)
            raise StructuredCallError(f"api_{exc.status_code}") from exc
        except anthropic.APIConnectionError as exc:
            raise StructuredCallError("connection") from exc

        if response.stop_reason == "refusal":
            raise StructuredCallError("refusal")
        if response.stop_reason == "max_tokens":
            # 잘린 JSON은 어차피 파싱되지 않는다 - 원인을 '파싱 실패'와 구분해 남긴다(생각 토큰도 상한에 포함된다).
            raise StructuredCallError("max_tokens")
        text = "".join(block.text for block in response.content if block.type == "text").strip()
        if not text:
            raise StructuredCallError("empty")
        return text

    def _request_kwargs(self, system_prompt: str, user_prompt: str, schema: dict | None) -> dict:
        model = self._state.model_name
        kwargs: dict = {
            "model": model,
            # MedGemma 의 MAX_NEW_TOKENS 와 따로 둔다(생각 토큰 포함 상한, CLAUDE_MAX_TOKENS).
            "max_tokens": self._settings.claude_max_tokens,
            "system": system_prompt.strip(),
            "messages": [{"role": "user", "content": user_prompt.strip()}],
            # 분류·추출은 짧고 빠르게. 생각 깊이는 effort로만 조절한다(opus 5.5는 thinking을 끌 수 없다). 후보 3종 모두 지원.
            "output_config": {"effort": "low", **({"format": {"type": "json_schema", "schema": schema}} if schema else {})},
        }
        if supports_server_fallback(model):
            # 안전 분류기가 거절하면 서버가 대체 모델로 같은 요청을 다시 돌린다.
            kwargs["betas"] = ["server-side-fallback-2026-07-01"]
            kwargs["fallbacks"] = "default"
        return kwargs
