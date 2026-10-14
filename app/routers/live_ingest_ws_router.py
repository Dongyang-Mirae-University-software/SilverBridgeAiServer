"""H.264 WebSocket 송출 수신 - /api/v1/stream-sessions/{session_id}/ingest-ws.

접속자는 백엔드 한 곳이다(브라우저는 AI 에 직접 붙지 않는다). 인증은 기존 API 키(X-API-Key 헤더)이고 쿼리 키는 받지 않는다.
메시지 계약(백엔드·FE 와 합의, BE docs/(2026-10-14) design-video-transport-contract.md):
  - 첫 메시지(텍스트): {"type":"config","codec":"avc1...","width":W,"height":H,"framerate":F,"format":"annexb"}
  - 이후(바이너리): [8B 타임스탬프(us)][1B 플래그: bit0=키프레임][H.264 Annex B 액세스 유닛]
  - 서버 -> 클라이언트(텍스트): {"type":"keyframe_request"}
  - 타임스탬프는 참고용이다. 프레임 시각은 서버 수신 시각(set_frame)이다.
디코드한 JPEG 는 기존 /frame 경로와 같은 처리(frame_ingest.process_ingested_frame)를 탄다.
로그에는 sessionId·집계 수치·종료 코드만 남기고 프레임·키 원문은 남기지 않는다.
"""

from __future__ import annotations

import asyncio
import hmac
import json
import logging
import time
from collections import deque
from typing import Any

from fastapi import APIRouter, HTTPException, WebSocket

from app.core.config import Settings, get_settings
from app.database.session import SessionLocal
from app.services.frame_ingest import process_ingested_frame
from app.services.h264_decoder import DecoderUnavailable, H264ToJpegDecoder, ffmpeg_exe
from app.services.stream_session_service import StreamSessionService, frame_store

router = APIRouter(tags=["LiveIngestSocket"])
_LOG = logging.getLogger(__name__)

HEADER_BYTES = 9
_CONFIG_WAIT_SEC = 5.0
_MAX_TEXT_BYTES = 4096
_MAX_PENDING_INGEST = 8
_KEYFRAME_REQUEST_MIN_INTERVAL = 1.0
_KEYFRAME_REQUEST_JSON = json.dumps({"type": "keyframe_request"})
_ANNEXB_PREFIXES = (b"\x00\x00\x01", b"\x00\x00\x00\x01")


def parse_config(raw: str) -> dict[str, Any] | None:
    """config 메시지 검증. 올바르면 정규화한 값, 아니면 None."""
    try:
        payload = json.loads(raw)
    except ValueError:
        return None
    if not isinstance(payload, dict) or payload.get("type") != "config":
        return None
    codec = payload.get("codec")
    if not isinstance(codec, str) or not (codec.lower().startswith("avc1") or codec.lower() == "h264"):
        return None
    if payload.get("format") != "annexb":
        return None
    width, height, framerate = payload.get("width"), payload.get("height"), payload.get("framerate")
    for value in (width, height):
        if isinstance(value, bool) or not isinstance(value, int) or not 16 <= value <= 4096:
            return None
    if isinstance(framerate, bool) or not isinstance(framerate, (int, float)) or not 1 <= framerate <= 60:
        return None
    return {"codec": codec, "width": width, "height": height, "framerate": float(framerate)}


class _Stats:
    def __init__(self) -> None:
        self.received = 0
        self.fed = 0
        self.decoded = 0
        self.dropped_rate = 0
        self.dropped_resync = 0
        self.bad_payload = 0
        self.ingest_errors = 0
        self.keyframe_requests = 0
        self.delays_ms: list[float] = []
        self.max_gap_ms = 0.0

    def window(self) -> dict[str, Any]:
        delays = sorted(self.delays_ms)
        out = {
            "in": self.received, "fed": self.fed, "out": self.decoded,
            "dropRate": self.dropped_rate, "dropResync": self.dropped_resync,
            "bad": self.bad_payload, "ingestErr": self.ingest_errors, "kfReq": self.keyframe_requests,
            "decodeMsP50": round(delays[len(delays) // 2], 1) if delays else None,
            "decodeMsMax": round(delays[-1], 1) if delays else None,
            "maxGapMs": round(self.max_gap_ms, 1),
        }  # fmt: skip
        self.received = self.fed = self.decoded = 0
        self.dropped_rate = self.dropped_resync = self.bad_payload = 0
        self.ingest_errors = self.keyframe_requests = 0
        self.delays_ms = []
        self.max_gap_ms = 0.0
        return out


class _Connection:
    def __init__(self, websocket: WebSocket, session_id: str, settings: Settings) -> None:
        self.ws = websocket
        self.session_id = session_id
        self.settings = settings
        self.decoder: H264ToJpegDecoder | None = None
        self.stats = _Stats()
        self._fail: tuple[int, str] | None = None
        self._fail_event = asyncio.Event()
        self._tasks: set[asyncio.Task[None]] = set()
        self._reader: asyncio.Task[None] | None = None
        self._recv_times: deque[float] = deque(maxlen=64)
        self._first_key_seen = False
        self._need_resync = False
        self._last_key = 0.0
        self._started = time.monotonic()
        self._since_output = 0
        self._consecutive_bad = 0
        self._last_kf_request = 0.0
        self._last_output = 0.0
        self._tokens = 0.0
        self._rate = 15.0
        self._refilled = time.monotonic()
        self._closing = False

    # ---- 종료 사유 ----
    def fail(self, code: int, reason: str) -> None:
        if self._fail is None:
            self._fail = (code, reason)
        self._fail_event.set()

    # ---- 설정·디코더 ----
    async def apply_config(self, config: dict[str, Any]) -> None:
        # 해상도·회전 변경은 새 config 로 온다. 디코더를 새로 띄워 이전 상태를 끌고 가지 않는다.
        await self._stop_decoder()
        decoder = H264ToJpegDecoder(framerate=config["framerate"], quality=self.settings.ingest_ws_jpeg_quality)
        await decoder.start()
        self.decoder = decoder
        self._rate = min(config["framerate"] * 1.5, self.settings.ingest_ws_max_fps)
        self._tokens = max(self._rate, 2.0)
        self._refilled = time.monotonic()
        self._first_key_seen = False
        self._need_resync = False
        self._since_output = 0
        self._recv_times.clear()
        self._started = time.monotonic()
        self._reader = asyncio.ensure_future(self._read_loop(decoder))

    async def _stop_decoder(self) -> None:
        decoder, self.decoder = self.decoder, None
        reader, self._reader = self._reader, None
        if decoder is not None:
            await decoder.close()
        if reader is not None:
            reader.cancel()
            await asyncio.gather(reader, return_exceptions=True)

    async def _read_loop(self, decoder: H264ToJpegDecoder) -> None:
        async for jpeg in decoder.jpegs():
            now = time.monotonic()
            self._since_output = 0
            self.stats.decoded += 1
            if self._recv_times:
                self.stats.delays_ms.append((now - self._recv_times.popleft()) * 1000.0)
            if self._last_output:
                self.stats.max_gap_ms = max(self.stats.max_gap_ms, (now - self._last_output) * 1000.0)
            self._last_output = now
            self._spawn_ingest(jpeg)
        if decoder is self.decoder and not self._closing:
            self.fail(1011, "DECODER_EXITED")

    # ---- 프레임 처리 ----
    def _spawn_ingest(self, jpeg: bytes) -> None:
        if len(self._tasks) >= _MAX_PENDING_INGEST:
            self.stats.ingest_errors += 1
            return
        task = asyncio.ensure_future(self._ingest(jpeg))
        self._tasks.add(task)
        task.add_done_callback(self._tasks.discard)

    async def _ingest(self, jpeg: bytes) -> None:
        db = SessionLocal()
        try:
            service = StreamSessionService(db, frame_store)
            session = service.require_session(self.session_id)
            await process_ingested_frame(service, session, jpeg)
        except HTTPException as exc:
            if exc.status_code == 404:
                self.fail(1008, "SESSION_NOT_FOUND")
            else:
                self.stats.ingest_errors += 1
        except Exception as exc:  # noqa: BLE001
            self.stats.ingest_errors += 1
            _LOG.warning("[INGEST-WS] 프레임 처리 오류 sessionId=%s err=%s", self.session_id, type(exc).__name__)
        finally:
            db.close()

    # ---- 수신 ----
    async def _request_keyframe(self, now: float) -> None:
        if now - self._last_kf_request < _KEYFRAME_REQUEST_MIN_INTERVAL:
            return
        self._last_kf_request = now
        self.stats.keyframe_requests += 1
        try:
            await self.ws.send_text(_KEYFRAME_REQUEST_JSON)
        except Exception:  # noqa: BLE001
            pass

    def _take_token(self, now: float) -> bool:
        cap = max(self._rate, 2.0)
        self._tokens = min(cap, self._tokens + (now - self._refilled) * self._rate)
        self._refilled = now
        if self._tokens >= 1.0:
            self._tokens -= 1.0
            return True
        return False

    async def handle_binary(self, data: bytes) -> None:
        s = self.settings
        now = time.monotonic()
        self.stats.received += 1
        if len(data) > s.ingest_ws_max_message_bytes:
            self.fail(1009, "MESSAGE_TOO_LARGE")
            return
        payload = data[HEADER_BYTES:]
        if len(data) <= HEADER_BYTES or not payload.startswith(_ANNEXB_PREFIXES):
            self.stats.bad_payload += 1
            self._consecutive_bad += 1
            if self._consecutive_bad >= s.ingest_ws_max_consecutive_failures:
                self.fail(1003, "BAD_PAYLOAD")
            return
        self._consecutive_bad = 0
        is_key = bool(data[8] & 0x01)

        reference = self._last_key if self._first_key_seen else self._started
        if not is_key and now - reference > s.ingest_ws_keyframe_timeout_sec:
            self.fail(1011, "NO_KEYFRAME")
            return
        if is_key:
            self._first_key_seen = True
            self._need_resync = False
            self._last_key = now
            self._take_token(now)  # 키프레임은 버리지 않는다(버리면 뒤 프레임이 모두 깨진다)
        else:
            if not self._first_key_seen or self._need_resync:
                self.stats.dropped_resync += 1
                await self._request_keyframe(now)
                return
            if not self._take_token(now):
                # 델타 프레임을 하나라도 버리면 참조가 깨진다 - 다음 키프레임까지 델타를 모두 버리고 새 키프레임을 요청한다.
                self.stats.dropped_rate += 1
                self._need_resync = True
                await self._request_keyframe(now)
                return

        decoder = self.decoder
        if decoder is None:
            return
        try:
            await decoder.feed(payload)
        except Exception:  # noqa: BLE001
            self.fail(1011, "DECODER_STALLED")
            return
        self.stats.fed += 1
        self._recv_times.append(now)
        self._since_output += 1
        if self._since_output >= s.ingest_ws_max_consecutive_failures:
            self.fail(1011, "DECODE_FAILED")
        elif self._since_output == 10:
            await self._request_keyframe(now)

    async def handle_text(self, text: str) -> None:
        if len(text.encode("utf-8", "ignore")) > _MAX_TEXT_BYTES:
            self.fail(1009, "MESSAGE_TOO_LARGE")
            return
        config = parse_config(text)
        if config is None:
            return  # config 가 아닌 텍스트(향후 확장)는 무시한다.
        try:
            await self.apply_config(config)
        except DecoderUnavailable:
            self.fail(1011, "DECODER_UNAVAILABLE")

    async def run(self) -> None:
        s = self.settings
        stats_task = asyncio.ensure_future(self._stats_loop())
        fail_waiter = asyncio.ensure_future(self._fail_event.wait())
        recv: asyncio.Future[Any] | None = None
        try:
            # 첫 메시지는 config 여야 한다.
            first = await asyncio.wait_for(self.ws.receive(), _CONFIG_WAIT_SEC)
            if first.get("type") == "websocket.disconnect":
                return
            text = first.get("text")
            config = parse_config(text) if isinstance(text, str) else None
            if config is None:
                self.fail(1003, "CONFIG_REQUIRED")
                return
            try:
                await self.apply_config(config)
            except DecoderUnavailable:
                self.fail(1011, "DECODER_UNAVAILABLE")
                return
            _LOG.info(
                "[INGEST-WS] 수립 sessionId=%s fps=%.1f %dx%d",
                self.session_id, config["framerate"], config["width"], config["height"],
            )

            while self._fail is None:
                recv = asyncio.ensure_future(self.ws.receive())
                done, _ = await asyncio.wait(
                    {recv, fail_waiter}, timeout=s.ingest_ws_idle_timeout_sec, return_when=asyncio.FIRST_COMPLETED,
                )
                if recv not in done:
                    recv.cancel()
                    if self._fail is None:
                        self.fail(1000, "IDLE_TIMEOUT")
                    break
                message = recv.result()
                if message.get("type") == "websocket.disconnect":
                    break
                if message.get("bytes") is not None:
                    await self.handle_binary(message["bytes"])
                elif message.get("text") is not None:
                    await self.handle_text(message["text"])
        except TimeoutError:
            self.fail(1003, "CONFIG_REQUIRED")
        except Exception as exc:  # noqa: BLE001
            # 연결 끊김(RuntimeError 등) 포함. 사유 코드·예외 클래스명만 남긴다.
            if self._fail is None:
                _LOG.info("[INGEST-WS] 수신 종료 sessionId=%s err=%s", self.session_id, type(exc).__name__)
        finally:
            self._closing = True
            if recv is not None and not recv.done():
                recv.cancel()
            fail_waiter.cancel()
            stats_task.cancel()
            await asyncio.gather(fail_waiter, stats_task, return_exceptions=True)
            await self._stop_decoder()
            for task in list(self._tasks):
                task.cancel()
            if self._tasks:
                await asyncio.gather(*self._tasks, return_exceptions=True)

    async def _stats_loop(self) -> None:
        interval = self.settings.ingest_ws_log_interval_sec
        while True:
            await asyncio.sleep(interval)
            _LOG.info("[INGEST-WS] 집계 sessionId=%s %s", self.session_id, json.dumps(self.stats.window()))

    @property
    def close_reason(self) -> tuple[int, str]:
        return self._fail or (1000, "CLOSED")


# 단일 인스턴스 메모리. sessionId 당 1연결, 전체 동시 상한은 설정(기본 5).
_registry: dict[str, _Connection] = {}


def active_connection_count() -> int:
    return len(_registry)


async def _safe_close(websocket: WebSocket, code: int, reason: str) -> None:
    try:
        await websocket.close(code=code, reason=reason)
    except Exception:  # noqa: BLE001
        pass


def _session_exists(session_id: str) -> bool:
    db = SessionLocal()
    try:
        StreamSessionService(db, frame_store).require_session(session_id)
        return True
    except HTTPException:
        return False
    finally:
        db.close()


@router.websocket("/api/v1/stream-sessions/{session_id}/ingest-ws")
async def ingest_ws(websocket: WebSocket, session_id: str) -> None:
    settings = get_settings()
    await websocket.accept()

    # 인증은 헤더 키만(쿼리 키 금지 - URL 은 접속 로그에 남는다). 상수 시간 비교.
    token = websocket.headers.get("x-api-key") or ""
    if not settings.api_key or not hmac.compare_digest(token.encode(), settings.api_key.encode()):
        await _safe_close(websocket, 1008, "AUTH_INVALID_KEY")
        return
    if not settings.ingest_ws_enabled:
        await _safe_close(websocket, 1008, "INGEST_DISABLED")
        return
    try:
        ffmpeg_exe()
    except DecoderUnavailable:
        await _safe_close(websocket, 1011, "DECODER_UNAVAILABLE")
        return
    if not _session_exists(session_id):
        await _safe_close(websocket, 1008, "SESSION_NOT_FOUND")
        return

    old = _registry.get(session_id)
    if old is None and len(_registry) >= settings.ingest_ws_max_connections:
        await _safe_close(websocket, 1013, "BUSY")
        return
    conn = _Connection(websocket, session_id, settings)
    _registry[session_id] = conn
    if old is not None:
        # 같은 카메라에 두 송출이 섞이면 fps·클립이 깨진다 - 새 연결이 이긴다.
        old.fail(1013, "REPLACED")
    try:
        await conn.run()
    finally:
        if _registry.get(session_id) is conn:
            del _registry[session_id]
        code, reason = conn.close_reason
        await _safe_close(websocket, code, reason)
        _LOG.info("[INGEST-WS] 종료 sessionId=%s code=%d reason=%s", session_id, code, reason)
