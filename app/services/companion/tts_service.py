"""말벗 목소리 (11/13): 말벗 문장을 Supertonic 3 F1 목소리 mp3 로 만들어 메모리에 잠시 둔다.

- 문장이 정해지면 바로 뒤 스레드에서 합성을 시작한다. 응답은 기다리지 않는다.
- 동시 합성 2건, 그 뒤 대기 8건까지. 넘치면 포기한다 (audio 없음 - FE 가 브라우저 읽기로 대신한다, 사용자 결정 11/13).
- 세션 문장은 15분 뒤 지운다 (최대 200개). 디스크·DB 에 쓰지 않는다 (음성은 저장하지 않는다는 정책).
- 매번 같은 문장(AI 고정 문장·백엔드 질문 문장)은 문장 기준으로 따로 보관해 다시 합성하지 않는다. 모델 대답은 이 보관에 넣지 않는다
  (어르신 말을 담을 수 있다 - 세션 보관 15분만).
- 로그에는 문장·세션 원문을 남기지 않는다 (길이·소요 시간·결과만).
"""
from __future__ import annotations

import logging
import subprocess
import threading
import time
from collections import OrderedDict
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable

from app.core.config import get_settings

logger = logging.getLogger("companion.tts")

MAX_AUDIO_BYTES = 1 * 1024 * 1024  # 소리 한 개 상한 (계약: 1MB)
WORKERS = 2  # 동시 합성 수 (gosky 12스레드 가운데 3스레드 x 2)
MAX_PENDING = 8  # 합성 중 2건 뒤에 기다릴 수 있는 수. 넘치면 포기
TTL_SEC = 15 * 60
MAX_ITEMS = 200  # 세션 문장 보관 상한 (mp3 한 개 약 30~60KB)
FIXED_MAX = 300  # 고정 문장 보관 상한
ENCODE_TIMEOUT_SEC = 10.0
MP3_RATE = 24000
MP3_BITRATE = "48k"
SWEEP_INTERVAL_SEC = 60.0  # 요청이 없어도 만료 소리를 지우는 주기 (11/13 점검 M-1)

Synth = Callable[[str], bytes]


class TtsNotFound(Exception):
    pass


class TtsNotReady(Exception):
    pass


class TtsUnavailable(Exception):
    """모델을 못 올린 뒤의 합성 요청. 다시 올리지 않는다 (11/13 점검 L-2)."""


@dataclass
class _Item:
    ward_id: str
    created: float = field(default_factory=time.monotonic)
    done: threading.Event = field(default_factory=threading.Event)
    data: bytes | None = None


def fixed_sentences() -> tuple[str, ...]:
    """기동 때 미리 만들어 둘 AI 고정 문장 (대체·공감·무응답·말 피함·마침·안전 안내·되묻기)."""
    from app.services.companion import reply_service
    from app.services.companion.pipeline import LOW_CONFIDENCE_REPLY

    groups = (reply_service.FALLBACK_FOLLOWUP, reply_service.FALLBACK_EMPATHY, reply_service.FALLBACK_NO_RESPONSE,
              reply_service.FALLBACK_AVOID, reply_service.FALLBACK_LAST,
              (reply_service.SAFETY_REPLY, reply_service.BODY_SAFETY_REPLY, LOW_CONFIDENCE_REPLY))
    out: list[str] = []
    for g in groups:
        out.extend(t for t in g if t not in out)
    return tuple(out)


def encode_mp3(wav, sample_rate: int) -> bytes:
    """float 파형(-1~1)을 24kHz mono mp3 로 바꾼다. 이미지에 있는 imageio-ffmpeg 의 libmp3lame 을 쓴다 (파일을 쓰지 않는다)."""
    import imageio_ffmpeg
    import numpy as np

    pcm = (np.clip(np.asarray(wav, dtype=np.float32).reshape(-1), -1.0, 1.0) * 32767).astype("<i2").tobytes()
    proc = subprocess.run(
        [imageio_ffmpeg.get_ffmpeg_exe(), "-hide_banner", "-loglevel", "error", "-f", "s16le", "-ar", str(sample_rate),
         "-ac", "1", "-i", "pipe:0", "-ar", str(MP3_RATE), "-ac", "1", "-c:a", "libmp3lame", "-b:a", MP3_BITRATE,
         "-f", "mp3", "pipe:1"],
        input=pcm, capture_output=True, timeout=ENCODE_TIMEOUT_SEC, check=True,
    )
    return proc.stdout


def load_supertonic() -> Synth:
    """Supertonic 3 를 올린다. 모델은 MODEL_BASE_PATH/COMPANION_TTS_SUBDIR 에 미리 받아 둔다 (기동 중 내려받지 않는다)."""
    from supertonic import TTS

    s = get_settings()
    model_dir = Path(s.model_base_path) / s.companion_tts_subdir
    # 스레드를 정하지 않으면 onnxruntime 이 모든 코어를 쓴다 - 실시간 분석·받아쓰기와 CPU 를 다툰다
    tts = TTS(model="supertonic-3", model_dir=model_dir, auto_download=False,
              intra_op_num_threads=s.companion_tts_threads, inter_op_num_threads=1)
    style = tts.get_voice_style(s.companion_tts_voice)
    rate = tts.sample_rate

    def synth(text: str) -> bytes:
        wav, _ = tts.synthesize(text, voice_style=style, lang="ko")
        return encode_mp3(wav, rate)

    return synth


class TtsService:
    def __init__(self, loader: Callable[[], Synth] = load_supertonic, workers: int = WORKERS,
                 max_pending: int = MAX_PENDING, ttl_sec: float = TTL_SEC, max_items: int = MAX_ITEMS,
                 fixed_max: int = FIXED_MAX, max_bytes: int = MAX_AUDIO_BYTES,
                 sweep_interval_sec: float = SWEEP_INTERVAL_SEC) -> None:
        self._loader = loader
        self._workers = workers
        self._max_inflight = workers + max_pending
        self._ttl_sec = ttl_sec
        self._max_items = max_items
        self._fixed_max = fixed_max
        self._max_bytes = max_bytes
        self._lock = threading.Lock()
        self._load_lock = threading.Lock()
        self._items: OrderedDict[tuple[str, int], _Item] = OrderedDict()
        self._fixed: OrderedDict[str, bytes] = OrderedDict()
        self._inflight = 0
        self._synth: Synth | None = None
        self._broken = False  # 모델을 못 올렸으면 재시작 전까지 합성하지 않는다 (요청마다 다시 올리며 CPU 를 쓰지 않게)
        # 처음부터 만든다 (11/13 점검 L-1: 처음 쓸 때 만들면 예열과 첫 요청이 겹칠 때 실행기가 2개 생겨 동시 합성이 4건이 된다).
        # 스레드는 첫 제출 때 생기므로 꺼져 있으면 비용이 없다
        self._executor = ThreadPoolExecutor(max_workers=workers, thread_name_prefix="companion-tts")
        self._sweep_interval_sec = sweep_interval_sec
        self._sweeper: threading.Thread | None = None

    # ---- 상태 ----
    def enabled(self) -> bool:
        return bool(get_settings().companion_tts_enabled) and not self._broken

    def _pool(self) -> ThreadPoolExecutor:
        return self._executor

    def sweep(self) -> None:
        """만료·초과 소리를 지운다. 주기 정리 스레드와 요청이 부른다."""
        with self._lock:
            self._purge_locked(time.monotonic())

    def _ensure_sweeper(self) -> None:
        """소리를 처음 보관할 때 주기 정리 스레드를 띄운다 (11/13 점검 M-1).

        요청 때만 정리하면 마지막 대화 뒤 다음 대화까지(예: 21시 -> 다음 날 09시) 모델 대답 소리가 메모리에 남는다.
        """
        if self._sweeper is not None:
            return
        with self._lock:
            if self._sweeper is not None:
                return

            def loop() -> None:
                while True:
                    time.sleep(self._sweep_interval_sec)
                    try:
                        self.sweep()
                    except Exception as exc:  # noqa: BLE001 - 정리 실패로 스레드가 멈추지 않게
                        logger.warning("[COMPANION-TTS] sweep failed (%s)", type(exc).__name__)

            self._sweeper = threading.Thread(target=loop, name="companion-tts-sweep", daemon=True)
            self._sweeper.start()

    def _engine(self) -> Synth:
        if self._synth is None:
            with self._load_lock:
                if self._broken:  # 이미 실패했으면 기다리던 작업마다 다시 올리지 않는다 (11/13 점검 L-2)
                    raise TtsUnavailable()
                if self._synth is None:
                    started = time.monotonic()
                    try:
                        self._synth = self._loader()
                    except Exception as exc:  # noqa: BLE001 - 경로·글자가 섞일 수 있어 클래스명만 남긴다
                        self._broken = True
                        logger.warning("[COMPANION-TTS] load failed (%s) - 소리 없이 동작한다", type(exc).__name__)
                        raise
                    logger.info("[COMPANION-TTS] loaded ms=%d", int((time.monotonic() - started) * 1000))
        return self._synth

    def _purge_locked(self, now: float) -> None:
        stale = [k for k, v in self._items.items() if now - v.created > self._ttl_sec]
        for k in stale:
            del self._items[k]
        while len(self._items) > self._max_items:
            self._items.popitem(last=False)

    # ---- 합성 시작 ----
    def start(self, ward_id: str, session_id: str, index: int, text: str, cache_text: bool = False) -> bool:
        """이 문장의 소리를 만들기 시작한다. 시작했거나 이미 있으면 True, 꺼짐·포화·모델 실패면 False (audio: null).

        cache_text: 매번 같은 문장(백엔드 질문)이면 True - 문장 기준으로도 보관해 다음 대화에서 다시 합성하지 않는다.
        """
        if not self.enabled() or not text or not text.strip():
            return False
        key = (str(session_id), int(index))
        self._ensure_sweeper()
        now = time.monotonic()
        with self._lock:
            self._purge_locked(now)
            cached = self._fixed.get(text)
            if cached is not None:
                self._fixed.move_to_end(text)
                item = _Item(ward_id, data=cached)
                item.done.set()
                self._items[key] = item
                self._items.move_to_end(key)
                self._purge_locked(now)
                logger.info("[COMPANION-TTS] index=%d chars=%d result=cached", key[1], len(text))
                return True
            if self._inflight >= self._max_inflight:
                logger.info("[COMPANION-TTS] index=%d chars=%d result=skip_busy", key[1], len(text))
                return False
            item = _Item(ward_id)
            self._items[key] = item
            self._items.move_to_end(key)
            self._purge_locked(now)
            self._inflight += 1
        try:
            self._pool().submit(self._work, item, key[1], text, cache_text)
        except RuntimeError:  # 종료 중
            with self._lock:
                self._inflight -= 1
            item.done.set()
            return False
        return True

    def _work(self, item: _Item, index: int, text: str, cache_text: bool) -> None:
        started = time.monotonic()
        result = "fail"
        try:
            data = self._engine()(text)
            if not data:
                result = "fail_empty"
            elif len(data) > self._max_bytes:
                result = "fail_too_large"
            else:
                item.data = data
                result = "ok"
                if cache_text:
                    self._remember(text, data)
        except Exception as exc:  # noqa: BLE001 - 원문이 섞일 수 있어 클래스명만
            result = f"fail_{type(exc).__name__}"
        finally:
            with self._lock:
                self._inflight -= 1
            item.done.set()
            logger.info("[COMPANION-TTS] index=%d chars=%d ms=%d bytes=%d result=%s", index, len(text),
                        int((time.monotonic() - started) * 1000), len(item.data or b""), result)

    def _remember(self, text: str, data: bytes) -> None:
        with self._lock:
            self._fixed[text] = data
            self._fixed.move_to_end(text)
            while len(self._fixed) > self._fixed_max:
                self._fixed.popitem(last=False)

    # ---- 조회 ----
    def get(self, session_id: str, index: int, wait_sec: float, ward_id: str | None = None) -> bytes:
        """소리를 돌려준다. 합성 중이면 wait_sec 까지 기다린다. 없음·만료·꺼짐·실패 TtsNotFound, 시간 안에 못 끝나면 TtsNotReady.

        ward_id 를 주면 그 소리를 만든 대화의 주인과 대조해 다르면 없는 것과 똑같이 TtsNotFound 다(BE 점검 L-2, 11/13 -
        소리는 BE 대화 번호로만 찾으므로 BE 두 곳이 같은 AI를 쓰면 같은 번호의 남의 소리가 나갈 수 있다). 기다리기 전에
        대조해 응답 시간으로도 있는지 드러나지 않게 한다. None 이면 대조하지 않는다(헤더를 싣지 않는 옛 BE 호환)."""
        if not get_settings().companion_tts_enabled:
            raise TtsNotFound()
        with self._lock:
            self._purge_locked(time.monotonic())
            item = self._items.get((str(session_id), int(index)))
        if item is None or (ward_id is not None and item.ward_id != ward_id):
            raise TtsNotFound()
        if not item.done.wait(max(0.0, wait_sec)):
            raise TtsNotReady()
        if item.data is None:
            raise TtsNotFound()
        return item.data

    # ---- 정리·예열 ----
    def delete_ward(self, ward_id: str) -> int:
        """피보호자 탈퇴 시 그 피보호자의 세션 소리를 지운다 (고정 문장은 개인 정보가 아니라 남긴다)."""
        with self._lock:
            gone = [k for k, v in self._items.items() if v.ward_id == ward_id]
            for k in gone:
                del self._items[k]
            return len(gone)

    def prewarm(self, texts: tuple[str, ...]) -> int:
        """모델을 올리고 고정 문장을 차례로 만들어 둔다 (기동 뒤 백그라운드 스레드에서 부른다). 만든 개수를 돌려준다."""
        if not self.enabled():
            return 0
        started = time.monotonic()
        try:
            synth = self._engine()
        except Exception:  # noqa: BLE001 - _engine 이 이미 로그를 남겼다
            return 0
        made = 0
        for t in texts:
            if t in self._fixed:
                continue
            with self._lock:
                self._inflight += 1  # 예열도 대기 개수에 센다 (11/13 점검 L-3) - 요청 대기열이 상한보다 길어지지 않게
            try:
                data = self._pool().submit(synth, t).result()  # 실행기를 거쳐 동시 합성 상한(2) 안에서 만든다
            except Exception as exc:  # noqa: BLE001
                logger.warning("[COMPANION-TTS] prewarm fail (%s) chars=%d", type(exc).__name__, len(t))
                continue
            finally:
                with self._lock:
                    self._inflight -= 1
            if data and len(data) <= self._max_bytes:
                self._remember(t, data)
                made += 1
        logger.info("[COMPANION-TTS] prewarm done count=%d/%d ms=%d", made, len(texts),
                    int((time.monotonic() - started) * 1000))
        return made


tts_service = TtsService()
