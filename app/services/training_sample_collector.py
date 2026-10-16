from __future__ import annotations

import json
import logging
import os
import random
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
RANDOM_REASON = "random"
RANDOM_KIND = "random"  # 폴더 이름(<day>/random/). 종류(fire·knife·fall) 폴더 밖이다.
_RANDOM_JITTER = 0.3  # 간격 ±30%


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
        random_fn: Callable[[], float] = random.random,
    ) -> None:
        self._inline = inline  # 테스트용: 제출한 작업을 호출 스레드에서 바로 실행
        self._now = now_fn
        self._mono = mono_fn
        self._random = random_fn  # [0,1) - 무작위 장면 간격 흔들림용(테스트에서 고정)
        self._executor: ThreadPoolExecutor | None = None
        self._lock = threading.Lock()
        self._last_offer: dict[tuple[str, str, str], float] = {}
        self._pending = 0
        self._counts: dict[tuple[str, str], int] = {}  # (세션, 날짜) → 오늘 저장 장수
        self._total_bytes: int | None = None  # None = 아직 스캔 전
        self._last_cleanup: float | None = None
        self._last_warn: dict[str, float] = {}
        self._spec: str | None = None  # 허용 목록 문자열 → 파싱 캐시(설정이 바뀌면 다시 읽는다)
        self._exact: set[str] = set()
        self._prefixes: tuple[str, ...] = ()
        self._allow_all = False
        self._random_spec: str | None = None
        self._random_exact: frozenset[str] = frozenset()
        self._random_next: dict[str, float] = {}  # 세션 → 다음 무작위 장면 저장 시각(monotonic)

    # --- 호출부 ---------------------------------------------------------------------------

    def offer(
        self,
        session_id: str,
        frame_bytes: bytes,
        kind_results: list[tuple[str, dict[str, Any]]],
        settings: Settings | None = None,
        frame_quality: dict[str, Any] | None = None,
    ) -> None:
        try:
            self._offer(session_id, frame_bytes, kind_results, settings or get_settings(), frame_quality)
        except Exception:  # noqa: BLE001 - 수집이 분석을 깨뜨리면 안 된다
            _LOG.warning("[COLLECT] offer failed")

    def _offer(
        self,
        session_id: str,
        frame_bytes: bytes,
        kind_results: list[tuple[str, dict[str, Any]]],
        s: Settings,
        frame_quality: dict[str, Any] | None = None,
    ) -> None:
        if not s.collect_enabled or not frame_bytes:
            return
        samples: list[dict[str, Any]] = []
        if self._random_due(session_id, s):
            samples.append(self._random_sample(kind_results, frame_quality))
        if self._session_allowed(session_id, s.collect_session_ids):
            samples.extend(self._near_alert_samples(session_id, kind_results, s))
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

    def _near_alert_samples(
        self, session_id: str, kind_results: list[tuple[str, dict[str, Any]]], s: Settings,
    ) -> list[dict[str, Any]]:
        samples: list[dict[str, Any]] = []
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
        return samples

    def _random_due(self, session_id: str, s: Settings) -> bool:
        """이번 프레임이 무작위 장면 차례인가. 점수·위험과 무관하고, 판단은 잠금 안의 가벼운 연산뿐이다.

        세션별 '다음 저장 시각'을 간격 x (0.7~1.3)로 잡는다(주기 편향 방지). 처음 본 세션은 첫 장을 바로 저장하지 않고
        한 간격 뒤부터 시작한다.
        """
        interval = s.collect_random_interval_sec
        if interval <= 0 or not self._random_session_allowed(session_id, s.collect_random_session_ids):
            return False
        now = self._mono()
        with self._lock:
            due_at = self._random_next.get(session_id)
            if due_at is None:
                self._random_next[session_id] = now + self._jittered(interval)
                return False
            if now < due_at:
                return False
            self._random_next[session_id] = now + self._jittered(interval)
            return True

    def _jittered(self, interval: float) -> float:
        return interval * (1.0 - _RANDOM_JITTER + 2 * _RANDOM_JITTER * self._random())

    def _random_session_allowed(self, session_id: str, spec: str) -> bool:
        """무작위 수집 대상: 정확한 세션 ID 만. *가 든 항목은 거부하고 WARN(전체 수집으로 번지지 않게)."""
        if spec != self._random_spec:
            exact: set[str] = set()
            rejected = 0
            for raw in spec.split(","):
                item = raw.strip()
                if not item:
                    continue
                if "*" in item:
                    rejected += 1
                else:
                    exact.add(item)
            if rejected:
                _LOG.warning("[COLLECT] COLLECT_RANDOM_SESSION_IDS wildcard entries rejected count=%d", rejected)
            self._random_spec, self._random_exact = spec, frozenset(exact)
        return session_id in self._random_exact

    @staticmethod
    def _random_sample(
        kind_results: list[tuple[str, dict[str, Any]]], frame_quality: dict[str, Any] | None,
    ) -> dict[str, Any]:
        confidences = [float(r.get("confidence") or 0.0) for _, r in kind_results]
        return {
            "kind": RANDOM_KIND,
            "reason": RANDOM_REASON,
            "confidence": max(confidences, default=0.0),  # 파일 이름용 - 종류별 점수는 meta.results 에 있다
            "threshold": None,
            "detections": [],
            "extra": {"frameQuality": frame_quality},
        }

    def _session_allowed(self, session_id: str, spec: str) -> bool:
        """허용 목록: 정확한 ID, 끝에 *를 붙인 접두어(ward_*), 또는 * 하나(전체). 중간 *는 무시한다."""
        if spec != self._spec:
            exact: set[str] = set()
            prefixes: list[str] = []
            allow_all = False
            for raw in spec.split(","):
                item = raw.strip()
                if not item:
                    continue
                if item == "*":
                    allow_all = True
                elif item.endswith("*") and "*" not in item[:-1]:
                    prefixes.append(item[:-1])
                elif "*" not in item:
                    exact.add(item)
            if allow_all:
                _LOG.warning("[COLLECT] COLLECT_SESSION_IDS='*' - all sessions are collected")
            self._spec, self._exact, self._prefixes, self._allow_all = spec, exact, tuple(prefixes), allow_all
        return self._allow_all or session_id in self._exact or session_id.startswith(self._prefixes)

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
            is_random = sample["reason"] == RANDOM_REASON
            # 무작위 장면은 근접·위험 장면과 하루 상한을 따로 센다(서로 몫을 먹지 않게).
            count_key = (f"{RANDOM_REASON}:{session_id}" if is_random else session_id, day)
            cap = s.collect_random_daily_max if is_random else s.collect_daily_max_per_session
            if self._counts.get(count_key, 0) >= cap:
                self._warn(f"daily:{session_id}", f"[COLLECT] daily cap reached sessionId={safe_session}")
                continue
            if (self._total_bytes or 0) >= s.collect_max_total_gb * _GB:
                self._warn("total", "[COLLECT] total size cap reached, stopped")
                return
            if shutil.disk_usage(root).free < s.collect_min_free_gb * _GB:
                self._warn("free", "[COLLECT] free disk too low, stopped")
                return

            folder = root / day / RANDOM_KIND if is_random else root / day / sample["kind"] / sample["reason"]
            folder.mkdir(parents=True, exist_ok=True)
            prefix = "r_" if is_random else ""  # 같은 프레임의 근접·위험 장면과 파일 이름이 겹치지 않게
            stem = f"{prefix}{kst_now.strftime('%H%M%S')}{kst_now.microsecond // 1000:03d}_{safe_session}_{sample['confidence']:.2f}"
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
                **sample.get("extra", {}),
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
