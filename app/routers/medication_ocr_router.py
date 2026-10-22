from __future__ import annotations

import logging
import time

from fastapi import APIRouter, Depends, HTTPException, Request, status
from fastapi.concurrency import run_in_threadpool
from python_multipart.exceptions import MultipartParseError
from python_multipart.multipart import MultipartParser, parse_options_header

from app.core.config import Settings, get_settings
from app.core.response import error_response
from app.schemas.medication_ocr_schema import MedicationOcrResponse
from app.services.medication_ocr_image import InvalidImageError, normalize_image
from app.services.medication_ocr_service import (
    MedicationOcrService,
    OcrNotConfiguredError,
    OcrUpstreamError,
    get_medication_ocr_service,
)

router = APIRouter(prefix="/api/v1/medication", tags=["Medication OCR"])
_LOG = logging.getLogger(__name__)

MAX_IMAGE_BYTES = 8 * 1024 * 1024
# multipart 경계·헤더 여유분. 본문 전체가 이보다 크면 이미지도 8MB를 넘는다.
MAX_BODY_BYTES = MAX_IMAGE_BYTES + 64 * 1024
IMAGE_FIELD = "image"


def _error(status_code: int, message: str, code: str) -> HTTPException:
    return HTTPException(status_code=status_code, detail=error_response(message, code, None))


def _too_large() -> HTTPException:
    return _error(status.HTTP_413_CONTENT_TOO_LARGE, "이미지는 8MB 이하만 올릴 수 있습니다.", "OCR_IMAGE_TOO_LARGE")


def _invalid_image() -> HTTPException:
    return _error(status.HTTP_400_BAD_REQUEST, "jpeg·png·webp 이미지만 읽을 수 있습니다.", "OCR_INVALID_IMAGE")


async def _read_body(request: Request) -> bytes:
    """본문을 메모리로만 읽는다. UploadFile 은 1MB가 넘으면 임시 파일(디스크)에 쓰므로 쓰지 않는다."""
    declared = request.headers.get("content-length")
    if declared and declared.isdigit() and int(declared) > MAX_BODY_BYTES:
        raise _too_large()
    body = bytearray()
    async for chunk in request.stream():
        body.extend(chunk)
        if len(body) > MAX_BODY_BYTES:
            raise _too_large()
    return bytes(body)


def _extract_image(content_type: str, body: bytes) -> bytes:
    """multipart/form-data 에서 image 필드 하나를 메모리로 꺼낸다. 없거나 형식이 깨지면 400."""
    ctype, params = parse_options_header(content_type)
    boundary = params.get(b"boundary")
    if ctype != b"multipart/form-data" or not boundary:
        raise _invalid_image()

    state = {"field": b"", "value": b"", "name": None, "in_image": False, "found": False}
    headers: dict[bytes, bytes] = {}
    image = bytearray()

    def on_part_begin() -> None:
        headers.clear()
        state["in_image"] = False

    def on_header_field(data: bytes, start: int, end: int) -> None:
        state["field"] += data[start:end]

    def on_header_value(data: bytes, start: int, end: int) -> None:
        state["value"] += data[start:end]

    def on_header_end() -> None:
        headers[state["field"].lower()] = state["value"]
        state["field"], state["value"] = b"", b""

    def on_headers_finished() -> None:
        _, disposition = parse_options_header(headers.get(b"content-disposition", b""))
        # 같은 이름이 여러 번 오면 첫 번째만 쓴다.
        state["in_image"] = disposition.get(b"name") == IMAGE_FIELD.encode() and not state["found"]

    def on_part_data(data: bytes, start: int, end: int) -> None:
        if state["in_image"]:
            image.extend(data[start:end])
            if len(image) > MAX_IMAGE_BYTES:
                raise _too_large()

    def on_part_end() -> None:
        if state["in_image"]:
            state["found"] = True
            state["in_image"] = False

    parser = MultipartParser(
        boundary,
        callbacks={
            "on_part_begin": on_part_begin,
            "on_header_field": on_header_field,
            "on_header_value": on_header_value,
            "on_header_end": on_header_end,
            "on_headers_finished": on_headers_finished,
            "on_part_data": on_part_data,
            "on_part_end": on_part_end,
        },
    )
    try:
        parser.write(body)
        parser.finalize()
    except MultipartParseError:
        raise _invalid_image() from None
    if not state["found"] or not image:
        raise _invalid_image()
    return bytes(image)


@router.post("/ocr", response_model=MedicationOcrResponse)
async def read_medication_envelope(
    request: Request,
    settings: Settings = Depends(get_settings),
    service: MedicationOcrService = Depends(get_medication_ocr_service),
) -> MedicationOcrResponse:
    """약 봉투 사진 → 복약 정보 초안. 사진은 메모리에서만 다루고 저장·로그하지 않는다.

    로그에는 크기·처리 시간·약 개수·결과 코드만 남긴다(이미지·약 이름·Claude 응답·오류 문구 금지).
    """
    started = time.monotonic()
    if not settings.medication_ocr_enabled:
        _log("disabled", started)
        raise _error(status.HTTP_503_SERVICE_UNAVAILABLE, "약 봉투 인식이 꺼져 있습니다.", "OCR_DISABLED")
    if not service.configured:
        _log("not_configured", started)
        raise _error(status.HTTP_503_SERVICE_UNAVAILABLE, "약 봉투 인식을 사용할 수 없습니다.", "OCR_NOT_CONFIGURED")

    try:
        body = await _read_body(request)
        raw = _extract_image(request.headers.get("content-type", ""), body)
    except HTTPException as exc:
        _log("too_large" if exc.status_code == 413 else "invalid_image", started)
        raise
    del body

    try:
        image = await run_in_threadpool(normalize_image, raw)
    except InvalidImageError as exc:
        _log(f"invalid_image_{exc}", started, size=len(raw))
        raise _invalid_image() from None

    try:
        result = await run_in_threadpool(service.read, image.jpeg)
    except OcrNotConfiguredError:
        _log("not_configured", started)
        raise _error(status.HTTP_503_SERVICE_UNAVAILABLE, "약 봉투 인식을 사용할 수 없습니다.", "OCR_NOT_CONFIGURED") from None
    except OcrUpstreamError as exc:
        _log(f"upstream_{exc}", started, size=len(raw), px=f"{image.width}x{image.height}")
        raise _error(status.HTTP_502_BAD_GATEWAY, "약 봉투를 읽지 못했습니다. 잠시 후 다시 시도해 주세요.", "OCR_UPSTREAM_FAILED") from None

    _log(
        "ok" if result.readable else "unreadable",
        started,
        size=len(raw),
        px=f"{image.width}x{image.height}",
        drugs=len(result.drugs),
    )
    return result


def _log(result: str, started: float, size: int | None = None, px: str | None = None, drugs: int | None = None) -> None:
    _LOG.info(
        "[MED-OCR] result=%s bytes=%s px=%s ms=%d drugs=%s",
        result,
        size if size is not None else "-",
        px or "-",
        int((time.monotonic() - started) * 1000),
        drugs if drugs is not None else "-",
    )
