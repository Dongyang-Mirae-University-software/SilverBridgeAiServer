from __future__ import annotations

import asyncio
import logging
import os
import shutil
import subprocess
import tempfile
import threading
import time
from concurrent.futures import Future, ThreadPoolExecutor
from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
from typing import Callable

from app.core.config import Settings, get_settings
from app.schemas.clip_schema import ClipRequest
from app.services.clip_buffer import ClipFrameBuffer, clip_frame_buffer

_LOG = logging.getLogger(__name__)

MIN_FPS = 1.0
MAX_FPS = 15.0
_POLL_SEC = 0.05


class ClipError(Exception):
    def __init__(self, status_code: int, error_code: str, message: str) -> None:
        super().__init__(error_code)
        self.status_code = status_code
        self.error_code = error_code
        self.message = message


class ClipEncodeError(Exception):
    """인코딩 실패. reason 은 로그용 고정 문자열(경로·stderr 원문 금지)."""

    def __init__(self, reason: str) -> None:
        super().__init__(reason)
        self.reason = reason


@dataclass
class ClipResult:
    data: bytes
    duration_ms: int
    frames: int
    width: int
    height: int
    started_at: datetime


Encoder = Callable[[list[bytes], float, int, int, str, float, str], bytes]


def parse_jpeg_size(data: bytes) -> tuple[int, int] | None:
    """JPEG SOF 마커에서 (가로, 세로)를 읽는다. 디코딩하지 않는다."""
    if len(data) < 4 or data[0:2] != b"\xff\xd8":
        return None
    i = 2
    n = len(data)
    while i + 4 <= n:
        if data[i] != 0xFF:
            i += 1
            continue
        marker = data[i + 1]
        if marker == 0xFF:
            i += 1
            continue
        if marker in (0x01, 0xD8) or 0xD0 <= marker <= 0xD7:
            i += 2
            continue
        if marker in (0xD9, 0xDA):
            return None
        seg_len = int.from_bytes(data[i + 2 : i + 4], "big")
        if 0xC0 <= marker <= 0xCF and marker not in (0xC4, 0xC8, 0xCC):
            if i + 9 > n:
                return None
            height = int.from_bytes(data[i + 5 : i + 7], "big")
            width = int.from_bytes(data[i + 7 : i + 9], "big")
            return (width, height) if width > 0 and height > 0 else None
        i += 2 + seg_len
    return None


def even_size(width: int, height: int) -> tuple[int, int]:
    # VP8 + yuv420p 는 짝수 해상도만 받는다.
    return max(2, width - width % 2), max(2, height - height % 2)


def select_frames(frames: list[bytes], span_sec: float) -> tuple[list[bytes], float]:
    """fps = 프레임 수 / 구간 길이, 1~15로 제한.

    15를 넘으면 프레임을 고르게 솎아 클립 길이를 구간 길이로 유지한다
    (fps만 15로 자르면 재생 길이가 실제보다 늘어난다).
    """
    n = len(frames)
    fps = n / span_sec
    if fps > MAX_FPS:
        keep = max(1, int(round(MAX_FPS * span_sec)))
        frames = [frames[int(i * n / keep)] for i in range(keep)]
        fps = len(frames) / span_sec
    return frames, min(MAX_FPS, max(MIN_FPS, fps))


_ffmpeg_checked: bool | None = None


def ffmpeg_available() -> bool:
    """imageio-ffmpeg 번들 바이너리 존재 여부(한 번만 확인해 캐시)."""
    global _ffmpeg_checked
    if _ffmpeg_checked is None:
        try:
            import imageio_ffmpeg

            _ffmpeg_checked = os.path.isfile(imageio_ffmpeg.get_ffmpeg_exe())
        except Exception:  # noqa: BLE001
            _ffmpeg_checked = False
    return _ffmpeg_checked


def encode_webm(
    jpegs: list[bytes],
    fps: float,
    width: int,
    height: int,
    bitrate: str,
    timeout_sec: float,
    tmp_dir: str,
) -> bytes:
    """JPEG 목록 → VP8 WebM. 임시 파일로 출력해야 duration·cue 정보가 들어간다(파이프 출력은 길이가 빠짐).

    성공·실패·타임아웃 모두 임시 파일과 ffmpeg 프로세스를 남기지 않는다.
    """
    try:
        import imageio_ffmpeg

        exe = imageio_ffmpeg.get_ffmpeg_exe()
    except Exception as exc:  # noqa: BLE001
        raise ClipEncodeError("ffmpeg_unavailable") from exc

    fd, out_path = tempfile.mkstemp(prefix="clip_", suffix=".webm", dir=tmp_dir or None)
    os.close(fd)
    nice = shutil.which("nice")
    cmd = ([nice, "-n", "19"] if nice else []) + [
        exe, "-hide_banner", "-loglevel", "error", "-y",
        "-f", "image2pipe", "-framerate", f"{fps:.4f}", "-c:v", "mjpeg", "-i", "-",
        # 첫 프레임 크기(짝수)로 고정 — 홀수 해상도와 클립 도중 해상도 변경(기기 회전)을 함께 처리한다.
        "-vf", f"scale={width}:{height}",
        "-c:v", "libvpx", "-b:v", bitrate, "-deadline", "realtime", "-cpu-used", "8", "-threads", "2",
        "-pix_fmt", "yuv420p", "-an", "-f", "webm", out_path,
    ]  # fmt: skip
    try:
        proc = subprocess.Popen(  # noqa: S603
            cmd, stdin=subprocess.PIPE, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL,
        )
        try:
            proc.communicate(input=b"".join(jpegs), timeout=max(0.1, timeout_sec))
        except subprocess.TimeoutExpired as exc:
            raise ClipEncodeError("timeout") from exc
        finally:
            # nice 는 ffmpeg 를 exec 하므로 pid 가 곧 ffmpeg 다.
            if proc.poll() is None:
                proc.kill()
                proc.wait()
        if proc.returncode != 0:
            raise ClipEncodeError(f"returncode={proc.returncode}")
        with open(out_path, "rb") as fh:
            data = fh.read()
        if not data:
            raise ClipEncodeError("empty_output")
        return data
    finally:
        try:
            os.unlink(out_path)
        except FileNotFoundError:
            pass


def _to_naive_utc(value: datetime | None) -> datetime:
    if value is None:
        return datetime.utcnow()
    if value.tzinfo is not None:
        return value.astimezone(timezone.utc).replace(tzinfo=None)
    return value


def _iso_utc(value: datetime) -> str:
    return value.isoformat(timespec="milliseconds") + "Z"


class ClipService:
    """클립 생성: 입장 제어 → 뒤 구간 대기 → 즉시 스냅샷 → 인코딩 큐(전용 스레드).

    - 입장 상한 = 동시 인코딩 + 대기열. 대기 중인 요청도 세어 메모리·코루틴 수를 묶는다(초과 시 429).
    - 인코딩 동시성은 전용 executor 스레드 수로 강제한다. 클라이언트가 끊겨도 스레드 작업은 끝까지
      가고 그때 슬롯을 돌려주므로, 취소된 요청 때문에 ffmpeg 가 상한을 넘어 뜨지 않는다.
    - 전체 처리 상한(timeout)은 요청 수신부터 잰다. 큐에서 다 써 버리면 인코딩 없이 실패한다.
    """

    def __init__(
        self,
        settings: Settings,
        buffer: ClipFrameBuffer,
        encoder: Encoder = encode_webm,
    ) -> None:
        self._settings = settings
        self._buffer = buffer
        self._encoder = encoder
        self._max_inflight = max(1, settings.clip_max_concurrency) + max(0, settings.clip_queue_max)
        self._executor = ThreadPoolExecutor(
            max_workers=max(1, settings.clip_max_concurrency),
            thread_name_prefix="clip-encode",
        )
        self._lock = threading.Lock()
        self._inflight = 0

    @property
    def inflight(self) -> int:
        with self._lock:
            return self._inflight

    def _try_admit(self) -> bool:
        with self._lock:
            if self._inflight >= self._max_inflight:
                return False
            self._inflight += 1
            return True

    def _release(self, *_: object) -> None:
        with self._lock:
            self._inflight = max(0, self._inflight - 1)

    async def create_clip(self, session_id: str, request: ClipRequest) -> ClipResult:
        started_mono = time.monotonic()
        deadline = started_mono + self._settings.clip_timeout_sec
        if not self._try_admit():
            _LOG.warning("clip busy session=%r", session_id)
            raise ClipError(429, "CLIP_BUSY", "클립 생성 요청이 많습니다. 잠시 후 다시 시도해 주세요.")

        submitted = False
        try:
            detected_at = _to_naive_utc(request.detectedAt)
            start = detected_at - timedelta(seconds=request.preSeconds)
            end = detected_at + timedelta(seconds=request.postSeconds)

            await self._wait_for_post_window(session_id, end, started_mono, request.postSeconds)

            # 대기가 끝나면 바로 스냅샷(바이트 참조) — 큐에서 기다리는 동안 링버퍼에서 밀려나 비는 일을 막는다.
            snapshot = self._buffer.snapshot(session_id, start, end)
            if len(snapshot) < max(1, self._settings.clip_min_frames):
                _LOG.info("clip not enough frames session=%r frames=%d", session_id, len(snapshot))
                raise ClipError(409, "CLIP_NOT_ENOUGH_FRAMES", "요청한 구간에 프레임이 없습니다.")

            span = request.preSeconds + request.postSeconds
            frames, fps = select_frames([item[1] for item in snapshot], span)
            size = parse_jpeg_size(frames[0])
            if size is None:
                _LOG.warning("clip encode failed session=%r frames=%d reason=bad_jpeg", session_id, len(frames))
                raise ClipError(500, "CLIP_ENCODE_FAILED", "클립 인코딩에 실패했습니다.")
            width, height = even_size(*size)

            future: Future[bytes] = self._executor.submit(
                self._encode_job, frames, fps, width, height, deadline,
            )
            submitted = True
            future.add_done_callback(self._release)
            try:
                data = await asyncio.wrap_future(future)
            except Exception as exc:  # noqa: BLE001 — 임시 디렉터리 없음·실행 실패(OSError) 등도 계약 형식으로
                reason = exc.reason if isinstance(exc, ClipEncodeError) else type(exc).__name__
                elapsed_ms = int((time.monotonic() - started_mono) * 1000)
                _LOG.warning(
                    "clip encode failed session=%r frames=%d elapsed_ms=%d reason=%s",
                    session_id, len(frames), elapsed_ms, reason,
                )
                raise ClipError(500, "CLIP_ENCODE_FAILED", "클립 인코딩에 실패했습니다.") from exc

            elapsed_ms = int((time.monotonic() - started_mono) * 1000)
            _LOG.info("clip created session=%r frames=%d elapsed_ms=%d", session_id, len(frames), elapsed_ms)
            return ClipResult(
                data=data,
                duration_ms=int(round(len(frames) / fps * 1000)),
                frames=len(frames),
                width=width,
                height=height,
                started_at=snapshot[0][0],
            )
        finally:
            if not submitted:
                self._release()

    async def _wait_for_post_window(
        self, session_id: str, end: datetime, started_mono: float, post_seconds: float,
    ) -> None:
        # 구간 끝(detectedAt + post) 이후 프레임이 들어올 때까지 기다린다. 최대 post + 1초.
        # 구간 끝이 이미 1초 넘게 지났는데 프레임이 없으면(송출 끊김) 더 기다려도 오지 않는다.
        until_end = (end - datetime.utcnow()).total_seconds()
        wait_deadline = started_mono + min(post_seconds + 1.0, max(0.0, until_end + 1.0))
        while True:
            latest = self._buffer.latest_time(session_id)
            if latest is not None and latest >= end:
                return
            if time.monotonic() >= wait_deadline:
                return
            await asyncio.sleep(_POLL_SEC)

    def _encode_job(self, frames: list[bytes], fps: float, width: int, height: int, deadline: float) -> bytes:
        remaining = deadline - time.monotonic()
        if remaining <= 0:
            raise ClipEncodeError("queue_timeout")
        return self._encoder(
            frames, fps, width, height, self._settings.clip_bitrate, remaining, self._settings.clip_tmp_dir,
        )


clip_service = ClipService(get_settings(), clip_frame_buffer)
