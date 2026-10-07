from __future__ import annotations

import json
import logging
import os
import re
import shutil
import threading
import time
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime, timedelta
from pathlib import Path
from typing import Any, Callable

from app.core.config import Settings, get_settings

_LOG = logging.getLogger(__name__)

# 같은 (세션, 종류, 이유)는 이 간격 안에서 한 장만 — 연속 프레임이 디스크를 채우지 않게 한다.
MIN_INTERVAL_SEC = 10.0
QUEUE_MAX = 20
_WARN_INTERVAL_SEC = 30.0
_CLEANUP_INTERVAL_SEC = 3600.0
_KST = timedelta(hours=9)
_NON_DETECTION = {"normal", "unknown"}
_UNSAFE = re.compile(r"[^A-Za-z0-9_-]")
_DATE_DIR = re.compile(r"^\d{8}$")
_GB = 1024**3


def _threshold_for(kind: str, settings: Settings) -> float | None:
    return {
        "fire": settings.fire_smoke_danger_threshold,
        "knife": settings.knife_danger_threshold,
        "fall": settings.fall_danger_threshold,
    }.get(kind)


def _dir_size(path: Path) -> int:
    total = 0
    for root, _, files in os.walk(path):
        for name in files:
            try:
                total += os.path.getsize(os.path.join(root, name))
            except OSError:
                pass
    return total


class TrainingSampleCollector:
    """학습용 프레임 수집 — 허용한 시험 세션의 근접·위험 장면을 원본 JPEG 로 디스크에 모은다.

    집 안 영상이다. 기본은 꺼짐이고, 켜도 COLLECT_SESSION_IDS 에 적은 세션만 저장한다(비면 0).
    내려받는 API 는 없다(서버에서 직접 옮긴다). 어떤 예외도 분석 경로로 올리지 않는다.
    저장은 전용 1스레드에서 하고, 로그에는 세션 ID·종류·이유·건수만 남긴다.
    """

    def __init__(
        self,
        *,
        inline: bool = False,
        now_fn: Callable[[], datetime] = datetime.utcnow,
        mono_fn: Callable[[], float] = time.monotonic,
    ) -> None:
        self._inline = inline  # 테스트용: 제출한 작업을 호출 스레드에서 바로 실행
        self._now = now_fn
        self._mono = mono_fn
        self._executor: ThreadPoolExecutor | None = None
        self._lock = threading.Lock()
        self._last_offer: dict[tuple[str, str, str], float] = {}
        self._pending = 0
        self._counts: dict[tuple[str, str], int] = {}  # (세션, 날짜) → 오늘 저장 장수
        self._total_bytes: int | None = None  # None = 아직 스캔 전
        self._last_cleanup: float | None = None
        self._last_warn: dict[str, float] = {}

    # --- 호출부 ---------------------------------------------------------------------------

    def offer(
        self,
        session_id: str,
        frame_bytes: bytes,
        kind_results: list[tuple[str, dict[str, Any]]],
        settings: Settings | None = None,
    ) -> None:
        try:
            self._offer(session_id, frame_bytes, kind_results, settings or get_settings())
        except Exception:  # noqa: BLE001 - 수집이 분석을 깨뜨리면 안 된다
            _LOG.warning("[COLLECT] offer failed")

    def _offer(
        self, session_id: str, frame_bytes: bytes, kind_results: list[tuple[str, dict[str, Any]]], s: Settings,
    ) -> None:
        if not s.collect_enabled or not frame_bytes:
            return
        allowed = {x.strip() for x in s.collect_session_ids.split(",") if x.strip()}
        if session_id not in allowed:
            return
        samples = []
        for kind, result in kind_results:
            classified = self._classify(kind, result, s)
            if classified is None:
                continue
            reason, threshold = classified
            if not self._interval_ok(session_id, kind, reason):
                continue
            samples.append(
                {
                    "kind": kind,
                    "reason": reason,
                    "confidence": float(result.get("confidence") or 0.0),
                    "threshold": threshold,
                    "detections": list(result.get("detections") or []),
                },
            )
        if not samples:
            return
        overview = [
            {
                "kind": k,
                "detectedType": r.get("detectedType"),
                "confidence": float(r.get("confidence") or 0.0),
                "danger": bool(r.get("danger", False)),
            }
            for k, r in kind_results
        ]
        analyzed_at = next((r.get("detectedAt") for _, r in kind_results if r.get("detectedAt")), None)
        job = (session_id, frame_bytes, samples, overview, analyzed_at, s)
        self._submit(job)

    @staticmethod
    def _classify(kind: str, result: dict[str, Any], s: Settings) -> tuple[str, float] | None:
        threshold = _threshold_for(kind, s)
        if threshold is None:
            return None
        confidence = float(result.get("confidence") or 0.0)
        if result.get("danger"):
            return "alert", threshold
        if result.get("detectedType") in _NON_DETECTION:
            return None
        # 위험 직전: 하한 이상이면 near. 낙상은 점수가 기준을 넘어도 유지 조건 전이면 danger=False 라 여기로 온다.
        if confidence >= s.collect_min_confidence:
            return "near", threshold
        return None

    def _interval_ok(self, session_id: str, kind: str, reason: str) -> bool:
        key = (session_id, kind, reason)
        now = self._mono()
        with self._lock:
            last = self._last_offer.get(key)
            if last is not None and now - last < MIN_INTERVAL_SEC:
                return False
            self._last_offer[key] = now
            return True

    def _submit(self, job: tuple[Any, ...]) -> None:
        if self._inline:
            self._run(job)
            return
        with self._lock:
            if self._pending >= QUEUE_MAX:
                self._warn("queue", "[COLLECT] queue full, dropped")
                return
            self._pending += 1
            if self._executor is None:
                self._executor = ThreadPoolExecutor(max_workers=1, thread_name_prefix="collect")
            executor = self._executor
        executor.submit(self._run_async, job)

    def _run_async(self, job: tuple[Any, ...]) -> None:
        try:
            self._run(job)
        finally:
            with self._lock:
                self._pending -= 1

    # --- 저장 스레드 -------------------------------------------------------------------------

    def _run(self, job: tuple[Any, ...]) -> None:
        try:
            self._save(*job)
        except Exception as exc:  # noqa: BLE001
            _LOG.warning("[COLLECT] save failed error=%s", type(exc).__name__)

    def _save(
        self,
        session_id: str,
        frame_bytes: bytes,
        samples: list[dict[str, Any]],
        overview: list[dict[str, Any]],
        analyzed_at: str | None,
        s: Settings,
    ) -> None:
        root = Path(s.collect_dir)
        root.mkdir(parents=True, exist_ok=True)
        if self._total_bytes is None:
            self._total_bytes = _dir_size(root)  # 시작 후 첫 저장 때 한 번만 훑는다
        self._maybe_cleanup(root, s)

        kst_now = self._now() + _KST
        day = kst_now.strftime("%Y%m%d")
        safe_session = _UNSAFE.sub("_", session_id)
        for sample in samples:
            count_key = (session_id, day)
            if self._counts.get(count_key, 0) >= s.collect_daily_max_per_session:
                self._warn(f"daily:{session_id}", f"[COLLECT] daily cap reached sessionId={safe_session}")
                continue
            if (self._total_bytes or 0) >= s.collect_max_total_gb * _GB:
                self._warn("total", "[COLLECT] total size cap reached, stopped")
                return
            if shutil.disk_usage(root).free < s.collect_min_free_gb * _GB:
                self._warn("free", "[COLLECT] free disk too low, stopped")
                return

            folder = root / day / sample["kind"] / sample["reason"]
            folder.mkdir(parents=True, exist_ok=True)
            stem = f"{kst_now.strftime('%H%M%S')}{kst_now.microsecond // 1000:03d}_{safe_session}_{sample['confidence']:.2f}"
            meta = {
                "sessionId": session_id,
                "kind": sample["kind"],
                "reason": sample["reason"],
                "confidence": sample["confidence"],
                "threshold": sample["threshold"],
                "detections": sample["detections"],
                "results": overview,
                "analyzedAt": analyzed_at,
                "bytes": len(frame_bytes),
            }
            meta_bytes = json.dumps(meta, ensure_ascii=False, default=str).encode("utf-8")
            jpg_path, json_path = folder / f"{stem}.jpg", folder / f"{stem}.json"
            self._write_atomic(jpg_path, frame_bytes)
            try:
                self._write_atomic(json_path, meta_bytes)
            except Exception:
                jpg_path.unlink(missing_ok=True)  # 짝이 없는 사진을 남기지 않는다
                raise
            self._total_bytes = (self._total_bytes or 0) + len(frame_bytes) + len(meta_bytes)
            self._counts[count_key] = self._counts.get(count_key, 0) + 1
            _LOG.info(
                "[COLLECT] saved sessionId=%s kind=%s reason=%s count=%d",
                safe_session, sample["kind"], sample["reason"], self._counts[count_key],
            )

    @staticmethod
    def _write_atomic(path: Path, data: bytes) -> None:
        tmp = path.with_name(f".{path.name}.tmp")
        fd = os.open(tmp, os.O_WRONLY | os.O_CREAT | os.O_TRUNC, 0o600)
        try:
            with os.fdopen(fd, "wb") as fh:
                fh.write(data)
            os.replace(tmp, path)
        except Exception:
            tmp.unlink(missing_ok=True)
            raise

    def _maybe_cleanup(self, root: Path, s: Settings) -> None:
        now = self._mono()
        if self._last_cleanup is not None and now - self._last_cleanup < _CLEANUP_INTERVAL_SEC:
            return
        self._last_cleanup = now
        cutoff = ((self._now() + _KST) - timedelta(days=s.collect_retention_days)).strftime("%Y%m%d")
        removed = 0
        for entry in root.iterdir():
            if entry.is_dir() and _DATE_DIR.match(entry.name) and entry.name < cutoff:
                size = _dir_size(entry)
                shutil.rmtree(entry, ignore_errors=True)
                self._total_bytes = max(0, (self._total_bytes or 0) - size)
                removed += 1
        # 지난 날짜의 장수 카운터도 정리
        today = (self._now() + _KST).strftime("%Y%m%d")
        self._counts = {k: v for k, v in self._counts.items() if k[1] == today}
        if removed:
            _LOG.info("[COLLECT] retention cleanup removedDays=%d", removed)

    def _warn(self, key: str, message: str) -> None:
        now = self._mono()
        with self._lock:
            last = self._last_warn.get(key)
            if last is not None and now - last < _WARN_INTERVAL_SEC:
                return
            self._last_warn[key] = now
        _LOG.warning(message)


training_sample_collector = TrainingSampleCollector()
