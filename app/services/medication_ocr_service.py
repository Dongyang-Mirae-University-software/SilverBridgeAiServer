from __future__ import annotations

import base64
import json
import logging
from functools import lru_cache

import anthropic
from pydantic import ValidationError

from app.core.config import Settings, get_settings
from app.prompts.medication_ocr_prompt import SYSTEM_PROMPT, USER_PROMPT
from app.schemas.medication_ocr_schema import CLAUDE_OUTPUT_SCHEMA, SLOTS, MedicationOcrResponse, OcrDrug

_LOG = logging.getLogger(__name__)

DEFAULT_MODEL = "claude-sonnet-5-5"
CALL_TIMEOUT_SEC = 30.0
MAX_TOKENS = 4000
NAME_MAX_LEN = 100


class OcrNotConfiguredError(RuntimeError):
    """Claude 키가 없다."""


class OcrUpstreamError(RuntimeError):
    """Claude 호출 실패·거절·시간 초과·형식 불일치. 메시지는 고정 코드만 담는다(오류 원문 금지)."""


def resolve_model(configured: str) -> str:
    """haiku는 사진에 없는 약을 지어내 금지(2026-10-22 측정). 설정이 haiku면 sonnet으로 대체한다."""
    model = configured.strip() or DEFAULT_MODEL
    if "haiku" in model.lower():
        _LOG.warning("[MED-OCR] MEDICATION_OCR_MODEL 에 haiku 는 쓸 수 없어 %s 로 대체한다", DEFAULT_MODEL)
        return DEFAULT_MODEL
    return model


class MedicationOcrService:
    def __init__(self, settings: Settings) -> None:
        self._model = resolve_model(settings.medication_ocr_model)
        api_key = settings.anthropic_api_key.strip()
        # 재시도하면 30초 제한을 넘는다 - 한 번만 부른다.
        self._client = (
            anthropic.Anthropic(api_key=api_key, timeout=CALL_TIMEOUT_SEC, max_retries=0) if api_key else None
        )

    @property
    def model(self) -> str:
        return self._model

    @property
    def configured(self) -> bool:
        return self._client is not None

    def read(self, jpeg: bytes) -> MedicationOcrResponse:
        if self._client is None:
            raise OcrNotConfiguredError("not_configured")
        try:
            response = self._client.beta.messages.create(
                model=self._model,
                max_tokens=MAX_TOKENS,
                system=SYSTEM_PROMPT,
                messages=[
                    {
                        "role": "user",
                        "content": [
                            {
                                "type": "image",
                                "source": {
                                    "type": "base64",
                                    "media_type": "image/jpeg",
                                    "data": base64.b64encode(jpeg).decode("ascii"),
                                },
                            },
                            {"type": "text", "text": USER_PROMPT},
                        ],
                    }
                ],
                output_config={"effort": "low", "format": {"type": "json_schema", "schema": CLAUDE_OUTPUT_SCHEMA}},
                # 안전 분류기가 거절하면 서버가 대체 모델로 같은 요청을 다시 돌린다(claude_structured_client 와 같은 방식).
                betas=["server-side-fallback-2026-07-01"],
                fallbacks="default",
            )
        except anthropic.APITimeoutError as exc:
            raise OcrUpstreamError("timeout") from exc
        except anthropic.APIStatusError as exc:
            raise OcrUpstreamError(f"status_{exc.status_code}") from exc
        except anthropic.APIError as exc:
            raise OcrUpstreamError("connection") from exc

        if response.stop_reason == "refusal":
            raise OcrUpstreamError("refusal")
        if response.stop_reason == "max_tokens":
            raise OcrUpstreamError("max_tokens")
        text = "".join(block.text for block in response.content if block.type == "text").strip()
        return parse_output(text)


def parse_output(text: str) -> MedicationOcrResponse:
    """스키마를 다시 검증하고 값을 다듬는다. 형식이 어긋나면 OcrUpstreamError."""
    try:
        parsed = MedicationOcrResponse.model_validate(json.loads(text))
    except (ValueError, ValidationError) as exc:  # json.JSONDecodeError 는 ValueError
        raise OcrUpstreamError("invalid_output") from exc
    return clean(parsed)


def clean(result: MedicationOcrResponse) -> MedicationOcrResponse:
    if not result.readable:
        return MedicationOcrResponse(readable=False, reason=_strip_or_none(result.reason), drugs=[])
    drugs: list[OcrDrug] = []
    for drug in result.drugs:
        name = drug.name.strip()[:NAME_MAX_LEN].strip()
        if not name:
            continue
        drugs.append(
            OcrDrug(
                name=name,
                perDose=_positive_or_none(drug.perDose),
                timesPerDay=_positive_or_none(drug.timesPerDay),
                days=_positive_or_none(drug.days),
                # 중복 제거, 순서는 하루 순서(아침→취침 전)로 고정
                slots=[slot for slot in SLOTS if slot in set(drug.slots)],
                timing=_strip_or_none(drug.timing),
                confidence=drug.confidence,
            )
        )
    return MedicationOcrResponse(readable=True, reason=None, drugs=drugs)


def _positive_or_none(value):
    return value if value is not None and value > 0 else None


def _strip_or_none(value: str | None) -> str | None:
    if value is None:
        return None
    value = value.strip()
    return value or None


@lru_cache
def get_medication_ocr_service() -> MedicationOcrService:
    return MedicationOcrService(get_settings())
