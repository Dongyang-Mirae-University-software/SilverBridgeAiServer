"""H.264(Annex B) -> JPEG 디코더. ffmpeg 서브프로세스 하나가 연결 하나를 맡는다.

stdin 으로 액세스 유닛을 밀어 넣고 stdout 의 MJPEG 스트림을 JPEG 한 장씩 잘라 돌려준다.
프로세스는 close() 에서 반드시 죽인다(누수 방지) - 호출자는 try/finally 로 close() 를 보장해야 한다.
프레임·바이트 원문은 로그에 남기지 않는다.
"""

from __future__ import annotations

import asyncio
import logging
from collections.abc import AsyncIterator

_LOG = logging.getLogger(__name__)

# Access Unit Delimiter(NAL 9) - 한 메시지가 액세스 유닛 하나라는 사실을 파서에 알려 주는 구분자다.
# ffmpeg raw H.264 파서는 "다음 유닛의 시작"을 봐야 앞 프레임을 내보내서, 이것이 없으면 출력이 입력보다 1프레임(15fps에서 약
# 67ms) 늦다. 유닛 뒤에 붙이면 즉시 나온다(실측 2026-10-14: 입력→출력 70ms -> 3ms, 프레임 손실 없음).
_AUD = b"\x00\x00\x00\x01\x09\xf0"

_SOI = b"\xff\xd8"
_EOI = b"\xff\xd9"
_READ_CHUNK = 65536
# 출력 쪽이 이만큼 쌓이도록 소비가 막히면(분석 지연 등) 잘못된 스트림으로 보고 버린다 - 메모리 상한.
_MAX_PENDING_BYTES = 8 * 1024 * 1024


class DecoderUnavailable(RuntimeError):
    """ffmpeg 바이너리를 찾지 못했다."""


def ffmpeg_exe() -> str:
    try:
        import imageio_ffmpeg

        return imageio_ffmpeg.get_ffmpeg_exe()
    except Exception as exc:  # noqa: BLE001
        raise DecoderUnavailable(type(exc).__name__) from None


def split_jpegs(buffer: bytearray) -> list[bytes]:
    """버퍼 앞에서부터 완성된 JPEG(SOI..EOI)를 잘라 돌려주고 버퍼는 남은 부분만 두게 한다.

    ffmpeg mjpeg 출력은 JPEG 사이에 다른 바이트가 없고 EOI(FFD9)는 한 장의 끝에만 나온다
    (엔트로피 부호 안의 0xFF 는 0xFF00 으로 이스케이프된다).
    """
    frames: list[bytes] = []
    while True:
        start = buffer.find(_SOI)
        if start < 0:
            buffer.clear()
            break
        end = buffer.find(_EOI, start + 2)
        if end < 0:
            if start > 0:
                del buffer[:start]
            break
        frames.append(bytes(buffer[start : end + 2]))
        del buffer[: end + 2]
    return frames


class H264ToJpegDecoder:
    def __init__(self, *, framerate: float, quality: int = 4) -> None:
        self._framerate = max(1.0, min(60.0, float(framerate)))
        self._quality = int(quality)
        self._proc: asyncio.subprocess.Process | None = None
        self._closed = False

    @property
    def pid(self) -> int | None:
        return self._proc.pid if self._proc else None

    async def start(self) -> None:
        exe = ffmpeg_exe()
        # 입력은 타임스탬프 없는 raw H.264 라 -framerate 로 가정값을 준다. 출력은 passthrough 로 디코드된 장수 그대로 낸다
        # (프레임 정리·복제 금지). -threads 1 + low_delay 로 프레임 스레딩 지연을 없앤다.
        # ⚠️ -fflags nobuffer 를 넣지 말 것: 실측(2026-10-14)에서 30장 중 0~15장만 나온다(첫 GOP 유실).
        # -probesize 32 -analyzeduration 0 이 없으면 첫 출력이 약 1.5초 늦다. 출력이 입력보다 1장 늦는 파서 대기는 feed()가 유닛 뒤에
        # 붙이는 AUD 로 없앴다.
        self._proc = await asyncio.create_subprocess_exec(
            exe, "-hide_banner", "-loglevel", "error",
            "-flags", "low_delay",
            "-probesize", "32", "-analyzeduration", "0",
            "-framerate", f"{self._framerate:.3f}", "-f", "h264", "-i", "pipe:0",
            "-an", "-threads", "1", "-fps_mode", "passthrough",
            "-c:v", "mjpeg", "-q:v", str(self._quality), "-flush_packets", "1", "-f", "image2pipe", "pipe:1",
            stdin=asyncio.subprocess.PIPE,
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.DEVNULL,
        )  # fmt: skip

    @property
    def alive(self) -> bool:
        return self._proc is not None and self._proc.returncode is None and not self._closed

    async def feed(self, data: bytes, timeout: float = 2.0) -> None:
        """액세스 유닛 하나를 넣는다. 디코더가 못 받으면(파이프 가득·프로세스 종료) 예외."""
        proc = self._proc
        if proc is None or proc.stdin is None or not self.alive:
            raise BrokenPipeError("decoder_not_running")
        try:
            proc.stdin.write(data + _AUD)
            await asyncio.wait_for(proc.stdin.drain(), timeout)
        except (ConnectionResetError, BrokenPipeError):
            raise BrokenPipeError("decoder_pipe_closed") from None

    async def jpegs(self) -> AsyncIterator[bytes]:
        """디코드된 JPEG 를 나오는 대로 돌려준다. 프로세스가 끝나면 종료한다."""
        proc = self._proc
        if proc is None or proc.stdout is None:
            return
        buffer = bytearray()
        while True:
            chunk = await proc.stdout.read(_READ_CHUNK)
            if not chunk:
                return
            buffer.extend(chunk)
            for frame in split_jpegs(buffer):
                yield frame
            if len(buffer) > _MAX_PENDING_BYTES:
                _LOG.warning("[INGEST-WS] 디코더 출력 누적 초과 - 스트림 종료 bytes=%d", len(buffer))
                return

    async def close(self) -> None:
        """프로세스를 죽이고 회수한다. 여러 번 불려도 안전하다."""
        if self._closed:
            return
        self._closed = True
        proc = self._proc
        if proc is None:
            return
        try:
            if proc.stdin is not None and not proc.stdin.is_closing():
                proc.stdin.close()
        except Exception:  # noqa: BLE001
            pass
        if proc.returncode is None:
            try:
                proc.kill()
            except ProcessLookupError:
                pass
        try:
            await asyncio.wait_for(proc.wait(), 5.0)
        except (TimeoutError, asyncio.TimeoutError):
            _LOG.error("[INGEST-WS] ffmpeg 회수 지연 pid=%s", proc.pid)
