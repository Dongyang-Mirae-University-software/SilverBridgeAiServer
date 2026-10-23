from __future__ import annotations

import io
import json
import logging

import pytest
from fastapi import Depends, FastAPI, HTTPException, Request
from fastapi.responses import JSONResponse
from fastapi.testclient import TestClient
from PIL import Image

from app.core.config import get_settings
from app.core.response import error_response
from app.core.security import require_api_key
from app.routers import medication_ocr_router as router_module
from app.schemas.medication_ocr_schema import MedicationOcrResponse, OcrDrug
from app.services.drug_name_list import DrugNameList, get_drug_name_list
from app.services.medication_ocr_image import MAX_LONG_EDGE, InvalidImageError, normalize_image
from app.services.medication_ocr_service import (
    DEFAULT_MODEL,
    MedicationOcrService,
    OcrNotConfiguredError,
    OcrUpstreamError,
    clean,
    get_medication_ocr_service,
    parse_output,
    resolve_model,
)

API_KEY = "test-ocr-key"
HEADERS = {"X-API-Key": API_KEY}
URL = "/api/v1/medication/ocr"
DRUG_NAME = "슈다페드정"


class FakeService:
    def __init__(self, result=None, error: Exception | None = None, configured: bool = True) -> None:
        self.result = result
        self.error = error
        self.configured = configured
        self.calls: list[bytes] = []

    def read(self, jpeg: bytes):
        self.calls.append(jpeg)
        if self.error is not None:
            raise self.error
        return self.result


def _make_app(service: FakeService, drug_names: DrugNameList | None = None, **settings_overrides) -> FastAPI:
    # app.main 은 모델 로딩까지 끌고 와서, 라우터와 예외 핸들러(main.py 와 같은 규칙)만 붙인다.
    app = FastAPI()

    @app.exception_handler(HTTPException)
    async def _handler(_: Request, exc: HTTPException) -> JSONResponse:
        if isinstance(exc.detail, dict) and "success" in exc.detail:
            return JSONResponse(status_code=exc.status_code, content=exc.detail)
        return JSONResponse(status_code=exc.status_code, content=error_response(str(exc.detail), "HTTP_ERROR", None))

    app.include_router(router_module.router, dependencies=[Depends(require_api_key)])
    settings = get_settings().model_copy(update={"api_key": API_KEY, **settings_overrides})
    app.dependency_overrides[get_settings] = lambda: settings
    app.dependency_overrides[get_medication_ocr_service] = lambda: service
    names = drug_names if drug_names is not None else DrugNameList.from_lines([DRUG_NAME])
    app.dependency_overrides[get_drug_name_list] = lambda: names
    return app


def _jpeg(size=(400, 300), color=(200, 200, 200), fmt="JPEG", exif=None) -> bytes:
    img = Image.new("RGB", size, color)
    out = io.BytesIO()
    if exif is not None:
        img.save(out, format=fmt, exif=exif)
    else:
        img.save(out, format=fmt)
    return out.getvalue()


def _ok_result() -> MedicationOcrResponse:
    return MedicationOcrResponse(
        readable=True,
        reason=None,
        drugs=[OcrDrug(name=DRUG_NAME, perDose=0.5, timesPerDay=2, days=5, slots=["MORNING"], timing="식후 30분", confidence="high")],
    )


def _post(client: TestClient, data: bytes = None, field: str = "image", headers=HEADERS, filename="a.jpg", mime="image/jpeg"):
    files = {field: (filename, data if data is not None else _jpeg(), mime)}
    return client.post(URL, files=files, headers=headers)


# ---- 엔드포인트 -------------------------------------------------------------


def test_ok_returns_contract_without_wrapper():
    service = FakeService(result=_ok_result())
    with TestClient(_make_app(service)) as client:
        res = _post(client)
    assert res.status_code == 200
    body = res.json()
    assert set(body) == {"readable", "reason", "drugs"}  # 표준 래퍼({success, data}) 없음
    assert body["readable"] is True
    assert body["drugs"][0] == {
        "name": DRUG_NAME,
        "perDose": 0.5,
        "timesPerDay": 2,
        "days": 5,
        "slots": ["MORNING"],
        "timing": "식후 30분",
        "confidence": "high",
        "inList": True,
        "suggestions": [],
    }
    assert "ingredient" not in body["drugs"][0]
    # Claude 로는 정규화한 JPEG 가 간다
    assert service.calls and service.calls[0][:3] == b"\xff\xd8\xff"


def test_unreadable_is_200_with_empty_drugs():
    service = FakeService(result=MedicationOcrResponse(readable=False, reason="약 봉투가 아닙니다", drugs=[]))
    with TestClient(_make_app(service)) as client:
        res = _post(client, _jpeg(fmt="PNG"), filename="memo.png", mime="image/png")
    assert res.status_code == 200
    assert res.json() == {"readable": False, "reason": "약 봉투가 아닙니다", "drugs": []}


def test_webp_accepted():
    service = FakeService(result=_ok_result())
    with TestClient(_make_app(service)) as client:
        res = _post(client, _jpeg(fmt="WEBP"), filename="a.webp", mime="image/webp")
    assert res.status_code == 200


def test_missing_api_key_401():
    service = FakeService(result=_ok_result())
    with TestClient(_make_app(service)) as client:
        res = _post(client, headers={})
    assert res.status_code == 401
    assert res.json()["errorCode"] == "AUTH_INVALID_KEY"
    assert service.calls == []


def test_too_large_413():
    service = FakeService(result=_ok_result())
    with TestClient(_make_app(service)) as client:
        res = _post(client, b"\xff\xd8\xff" + b"0" * (router_module.MAX_IMAGE_BYTES + 1))
    assert res.status_code == 413
    assert res.json()["errorCode"] == "OCR_IMAGE_TOO_LARGE"
    assert service.calls == []


@pytest.mark.parametrize(
    "data,filename,mime",
    [
        (b"not an image at all", "a.jpg", "image/jpeg"),  # 디코드 실패
        (_jpeg()[:200], "a.jpg", "image/jpeg"),  # 잘린 파일
        (_jpeg(fmt="GIF"), "a.gif", "image/gif"),  # 허용하지 않는 형식
    ],
)
def test_invalid_image_400(data, filename, mime):
    service = FakeService(result=_ok_result())
    with TestClient(_make_app(service)) as client:
        res = _post(client, data, filename=filename, mime=mime)
    assert res.status_code == 400
    assert res.json()["errorCode"] == "OCR_INVALID_IMAGE"
    assert service.calls == []


def test_missing_image_field_400():
    service = FakeService(result=_ok_result())
    with TestClient(_make_app(service)) as client:
        res = _post(client, field="photo")
    assert res.status_code == 400
    assert res.json()["errorCode"] == "OCR_INVALID_IMAGE"


def test_not_multipart_400():
    service = FakeService(result=_ok_result())
    with TestClient(_make_app(service)) as client:
        res = client.post(URL, content=_jpeg(), headers={**HEADERS, "Content-Type": "image/jpeg"})
    assert res.status_code == 400


def test_kill_switch_503():
    service = FakeService(result=_ok_result())
    with TestClient(_make_app(service, medication_ocr_enabled=False)) as client:
        res = _post(client)
    assert res.status_code == 503
    assert res.json()["errorCode"] == "OCR_DISABLED"
    assert service.calls == []


def test_claude_key_missing_503():
    service = FakeService(result=_ok_result(), configured=False)
    with TestClient(_make_app(service)) as client:
        res = _post(client)
    assert res.status_code == 503
    assert res.json()["errorCode"] == "OCR_NOT_CONFIGURED"


def test_not_configured_raised_by_service_503():
    service = FakeService(error=OcrNotConfiguredError("not_configured"))
    with TestClient(_make_app(service)) as client:
        res = _post(client)
    assert res.status_code == 503


@pytest.mark.parametrize("code", ["timeout", "refusal", "status_500", "invalid_output", "connection"])
def test_claude_failure_502(code):
    service = FakeService(error=OcrUpstreamError(code))
    with TestClient(_make_app(service)) as client:
        res = _post(client)
    assert res.status_code == 502
    assert res.json()["errorCode"] == "OCR_UPSTREAM_FAILED"


def test_log_has_no_drug_name(caplog):
    service = FakeService(result=_ok_result())
    with caplog.at_level(logging.INFO, logger=router_module.__name__):
        with TestClient(_make_app(service)) as client:
            _post(client)
    text = "\n".join(r.getMessage() for r in caplog.records)
    assert "[MED-OCR] result=ok" in text
    assert "drugs=1" in text
    assert DRUG_NAME not in text


# ---- 이미지 정규화 ----------------------------------------------------------


def test_exif_orientation_applied_and_exif_stripped():
    exif = Image.Exif()
    exif[0x0112] = 6  # 시계 방향 90도로 돌려 봐야 하는 사진(휴대폰 세로 촬영)
    exif[0x8825] = {2: (37.0, 33.0, 0.0)}  # GPS
    data = _jpeg(size=(400, 200), exif=exif.tobytes())
    out = normalize_image(data)
    assert (out.width, out.height) == (200, 400)  # 바로 세워짐
    with Image.open(io.BytesIO(out.jpeg)) as img:
        assert img.format == "JPEG"
        assert len(img.getexif()) == 0  # 방향·GPS 모두 없음


def test_downscale_long_edge():
    out = normalize_image(_jpeg(size=(4000, 3000)))
    assert max(out.width, out.height) == MAX_LONG_EDGE
    assert (out.width, out.height) == (MAX_LONG_EDGE, 1176)


def test_no_upscale():
    out = normalize_image(_jpeg(size=(800, 600)))
    assert (out.width, out.height) == (800, 600)


def test_transparent_png_on_white():
    img = Image.new("RGBA", (10, 10), (0, 0, 0, 0))
    buf = io.BytesIO()
    img.save(buf, format="PNG")
    out = normalize_image(buf.getvalue())
    with Image.open(io.BytesIO(out.jpeg)) as decoded:
        assert decoded.getpixel((5, 5))[0] > 240


def test_pixel_limit_rejected(monkeypatch):
    from app.services import medication_ocr_image

    monkeypatch.setattr(medication_ocr_image, "MAX_PIXELS", 100)
    with pytest.raises(InvalidImageError):
        normalize_image(_jpeg(size=(20, 20)))


# ---- 응답 후처리 ------------------------------------------------------------


def _drug(**overrides) -> dict:
    base = {
        "name": DRUG_NAME,
        "perDose": 1,
        "timesPerDay": 2,
        "days": 3,
        "slots": [],
        "timing": None,
        "confidence": "high",
    }
    return {**base, **overrides}


def test_clean_trims_and_drops_and_nulls():
    raw = {
        "readable": True,
        "reason": "무시됨",
        "drugs": [
            _drug(name="  " + "가" * 150 + "  ", perDose=0, timesPerDay=0, days=-1),
            _drug(name="   "),
            _drug(slots=["DINNER", "MORNING", "DINNER"], timing="  ", perDose=-0.5),
        ],
    }
    result = parse_output(json.dumps(raw))
    assert result.reason is None
    assert len(result.drugs) == 2
    first, second = result.drugs
    assert first.name == "가" * 100
    assert (first.perDose, first.timesPerDay, first.days) == (None, None, None)
    assert second.slots == ["MORNING", "DINNER"]  # 중복 제거, 하루 순서
    assert second.timing is None
    assert second.perDose is None


def test_clean_unreadable_forces_empty_drugs():
    result = clean(MedicationOcrResponse(readable=False, reason=" 흐림 ", drugs=[OcrDrug(**_drug())]))
    assert result.drugs == []
    assert result.reason == "흐림"


@pytest.mark.parametrize(
    "text",
    [
        "not json",
        json.dumps({"drugs": []}),
        json.dumps({"readable": True, "reason": None, "drugs": [_drug(slots=["NOON"])]}),
        json.dumps({"readable": True, "reason": None, "drugs": [_drug(confidence="medium")]}),
    ],
)
def test_parse_output_invalid(text):
    with pytest.raises(OcrUpstreamError):
        parse_output(text)


# ---- 모델 선택 --------------------------------------------------------------


def test_haiku_replaced_with_sonnet(caplog):
    with caplog.at_level(logging.WARNING):
        assert resolve_model("claude-haiku-5-5") == DEFAULT_MODEL
    assert any("haiku" in r.getMessage() for r in caplog.records)
    assert resolve_model("") == DEFAULT_MODEL
    assert resolve_model("claude-opus-5-5") == "claude-opus-5-5"


def test_service_without_key_not_configured():
    service = MedicationOcrService(get_settings().model_copy(update={"anthropic_api_key": ""}))
    assert service.configured is False
    with pytest.raises(OcrNotConfiguredError):
        service.read(b"x")


class _FakeMessages:
    def __init__(self, response=None, error=None):
        self.response, self.error, self.kwargs = response, error, None

    def create(self, **kwargs):
        self.kwargs = kwargs
        if self.error:
            raise self.error
        return self.response


class _Block:
    def __init__(self, text):
        self.type, self.text = "text", text


class _Resp:
    def __init__(self, text, stop_reason="end_turn"):
        self.content, self.stop_reason = [_Block(text)], stop_reason


def _service_with(messages: _FakeMessages) -> MedicationOcrService:
    service = MedicationOcrService(get_settings().model_copy(update={"anthropic_api_key": "sk-test"}))

    class _Beta:
        pass

    beta = _Beta()
    beta.messages = messages
    service._client = type("C", (), {"beta": beta})()
    return service


def test_service_request_shape():
    messages = _FakeMessages(_Resp(json.dumps({"readable": True, "reason": None, "drugs": [_drug()]})))
    result = _service_with(messages).read(b"\xff\xd8jpeg")
    assert result.drugs[0].name == DRUG_NAME
    kw = messages.kwargs
    assert kw["model"] == DEFAULT_MODEL
    assert kw["max_tokens"] == 4000
    assert kw["output_config"]["effort"] == "low"
    assert kw["output_config"]["format"]["type"] == "json_schema"
    assert kw["betas"] == ["server-side-fallback-2026-07-01"]
    assert kw["fallbacks"] == "default"
    image = kw["messages"][0]["content"][0]
    assert image["type"] == "image" and image["source"]["media_type"] == "image/jpeg"
    assert "ingredient" not in json.dumps(kw["output_config"]["format"]["schema"])


@pytest.mark.parametrize("stop_reason", ["refusal", "max_tokens"])
def test_service_stop_reasons_fail(stop_reason):
    messages = _FakeMessages(_Resp("{}", stop_reason=stop_reason))
    with pytest.raises(OcrUpstreamError):
        _service_with(messages).read(b"x")


def test_service_api_errors_mapped():
    import anthropic
    import httpx

    req = httpx.Request("POST", "https://api.anthropic.com/v1/messages")
    for error in (
        anthropic.APITimeoutError(request=req),
        anthropic.APIConnectionError(request=req),
        anthropic.InternalServerError("boom", response=httpx.Response(500, request=req), body=None),
    ):
        with pytest.raises(OcrUpstreamError):
            _service_with(_FakeMessages(error=error)).read(b"x")


# ---- 정규화 메모리 방어(2026-10-22 기능 점검) --------------------------------


def test_large_jpeg_draft_keeps_orientation_and_size():
    exif = Image.Exif()
    exif[0x0112] = 6
    out = normalize_image(_jpeg(size=(4032, 3024), exif=exif.tobytes()))
    assert (out.width, out.height) == (1176, MAX_LONG_EDGE)  # 줄여 읽어도 회전·최종 크기는 같다


def test_non_jpeg_pixel_limit_lower_than_jpeg():
    from app.services.medication_ocr_image import MAX_PIXELS_NON_JPEG

    size = (4100, 4000)
    assert size[0] * size[1] > MAX_PIXELS_NON_JPEG
    with pytest.raises(InvalidImageError):
        normalize_image(_jpeg(size=size, fmt="PNG"))
    out = normalize_image(_jpeg(size=size))  # 같은 크기 JPEG 는 받는다
    assert max(out.width, out.height) == MAX_LONG_EDGE


def test_normalize_runs_at_most_two_at_once(monkeypatch):
    import threading
    import time

    from app.services import medication_ocr_image

    running, peak, lock = 0, 0, threading.Lock()

    def slow(data):
        nonlocal running, peak
        with lock:
            running += 1
            peak = max(peak, running)
        time.sleep(0.05)
        with lock:
            running -= 1

    monkeypatch.setattr(medication_ocr_image, "_normalize", slow)
    threads = [threading.Thread(target=normalize_image, args=(b"x",)) for _ in range(6)]
    for th in threads:
        th.start()
    for th in threads:
        th.join()
    assert peak == medication_ocr_image.MAX_CONCURRENT_NORMALIZE
