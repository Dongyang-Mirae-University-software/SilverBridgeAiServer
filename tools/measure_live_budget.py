"""라이브 분석 처리 시간 예산 측정 — 합성 화면으로 프레임당 디코드·품질 검사·감지기 3개 시간을 잰다.

실제 가정 영상은 쓰지 않는다(numpy 로 그린 장면만). 운영 서버와 다른 프로세스에서 낮은 우선순위(nice 19)로
돈다. 운영 컨테이너 안에서 돌리지 말고 같은 이미지로 임시 컨테이너를 띄워 쓴다(README 아래 예시).

두 경로를 같은 화면에 번갈아 돌려 비교한다.
  before = 감지기마다 JPEG 디코드(detect_from_jpeg) + 품질 검사 흑백 디코드(assess_jpeg)  ← 2026-10-14 이전
  after  = 프레임당 컬러 디코드 1회(decode_bgr) → 감지기 3개(detect_from_frame) + 품질 검사(assess_bgr)
두 경로의 감지 결과(종류·신뢰도·danger)·품질 판정이 다르거나, 감지기가 공유 배열을 바꾸면 불일치로 센다.

출력은 숫자만이다(프레임·키·경로 원문을 찍지 않는다).

    docker run --rm --gpus all --env-file .env -v "$PWD":/app -w /app <AI 이미지> \\
        python tools/measure_live_budget.py --frames 300
"""
from __future__ import annotations

import argparse
import os
import statistics
import sys
import time
from pathlib import Path
from typing import Any, Callable

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import cv2  # noqa: E402
import numpy as np  # noqa: E402

from app.core.config import get_settings  # noqa: E402
from app.services import frame_quality  # noqa: E402
from app.services.fall_detection_service import FallDetectionService  # noqa: E402
from app.services.fire_smoke_detection_service import FireSmokeDetectionService  # noqa: E402
from app.services.frame_decode import decode_bgr  # noqa: E402
from app.services.knife_detection_service import KnifeDetectionService  # noqa: E402


def synthetic_frames(count: int, width: int, height: int, seed: int) -> list[bytes]:
    """방처럼 보이는 단색 면 + 일부 프레임엔 불꽃 색 덩어리. 장마다 조금씩 달라 캐시 효과를 막는다."""
    rng = np.random.default_rng(seed)
    frames: list[bytes] = []
    for i in range(count):
        img = np.full((height, width, 3), rng.integers(60, 170, 3), np.uint8)
        for _ in range(14):
            x, y = int(rng.integers(0, width - 60)), int(rng.integers(0, height - 60))
            w, h = int(rng.integers(40, width // 3)), int(rng.integers(40, height // 3))
            cv2.rectangle(img, (x, y), (x + w, y + h), tuple(int(c) for c in rng.integers(0, 255, 3)), -1)
        if i % 3 == 0:
            cx, cy = int(rng.integers(width // 4, 3 * width // 4)), int(rng.integers(height // 3, height - 40))
            for r, color in ((90, (0, 80, 255)), (60, (0, 160, 255)), (30, (120, 230, 255))):
                cv2.ellipse(img, (cx, cy), (r, int(r * 1.6)), 0, 180, 360, color, -1)
        noise = rng.normal(0, 6, img.shape)
        img = np.clip(img + noise, 0, 255).astype(np.uint8)
        ok, buf = cv2.imencode(".jpg", img, [cv2.IMWRITE_JPEG_QUALITY, 80])
        if ok:
            frames.append(buf.tobytes())
    return frames


def _timed(fn: Callable[[], Any]) -> tuple[Any, float]:
    started = time.perf_counter()
    out = fn()
    return out, (time.perf_counter() - started) * 1000


def _key(result: dict[str, Any]) -> tuple[str, float, bool]:
    return (str(result.get("detectedType")), float(result.get("confidence") or 0.0), bool(result.get("danger")))


def _p95(values: list[float]) -> float:
    ordered = sorted(values)
    return ordered[min(len(ordered) - 1, int(round(0.95 * (len(ordered) - 1))))]


def _boxes(result: dict[str, Any]) -> list[tuple[str, float, tuple[int, ...]]]:
    return [(d["detectedType"], d["confidence"], tuple(d["bbox"].values())) for d in result.get("detections", [])]


def _compare_boxes(settings: Any, frames: list[bytes], conf: float) -> int:
    """감지 기준을 낮춰(점수 있는 박스가 나오게) 두 입력 경로의 박스·점수를 장마다 비교한다. 시간은 재지 않는다."""
    low = settings.model_copy(update={"fire_smoke_conf_threshold": conf, "knife_conf_threshold": conf,
                                      "fall_conf_threshold": conf})
    detectors = [("fire", FireSmokeDetectionService(low)), ("knife", KnifeDetectionService(low)),
                 ("fall", FallDetectionService(low))]
    for _, det in detectors:
        det.try_load()
    differ = 0
    boxes = {k: 0 for k, _ in detectors}
    for frame_bytes in frames:
        frame = decode_bgr(frame_bytes)
        for kind, det in detectors:
            old, new = det.detect_from_jpeg(frame_bytes), det.detect_from_frame(frame)
            boxes[kind] += len(new.get("detections", []))
            if _key(old) != _key(new) or _boxes(old) != _boxes(new):
                differ += 1
    print(f"[BUDGET] 박스 비교(기준 {conf}) - 불일치 {differ} / 박스 수 " + ", ".join(f"{k} {n}" for k, n in boxes.items()))
    return differ


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--frames", type=int, default=300)
    parser.add_argument("--warmup", type=int, default=20)
    parser.add_argument("--width", type=int, default=1280)
    parser.add_argument("--height", type=int, default=720)
    parser.add_argument("--budget-ms", type=float, default=30.0)
    parser.add_argument("--seed", type=int, default=20261014)
    parser.add_argument("--equality-conf", type=float, default=0.01,
                        help="결과 비교 단계의 감지 기준(낮춰서 점수 있는 박스까지 비교한다, 0=건너뜀)")
    parser.add_argument("--no-nice", action="store_true", help="우선순위를 낮추지 않는다(전용 장비에서만)")
    args = parser.parse_args()
    if not args.no_nice:
        os.nice(19)

    settings = get_settings().model_copy(update={"fire_smoke_enabled": True, "knife_enabled": True, "fall_enabled": True})
    detectors = [("fire", FireSmokeDetectionService(settings)), ("knife", KnifeDetectionService(settings)),
                 ("fall", FallDetectionService(settings))]
    for kind, det in detectors:
        det.try_load()
        if not det.loaded:
            print(f"[BUDGET] {kind} 모델을 불러오지 못했다 - 측정 중단")
            return 2
    thresholds = frame_quality.thresholds_from(settings)

    frames = synthetic_frames(args.frames + args.warmup, args.width, args.height, args.seed)
    print(f"[BUDGET] 합성 화면 {len(frames)}장 {args.width}x{args.height} (예열 {args.warmup}장 제외)")

    stages = ["decode", "quality"] + [k for k, _ in detectors]
    before: dict[str, list[float]] = {s: [] for s in stages}
    after: dict[str, list[float]] = {s: [] for s in stages}
    totals = {"before": [], "after": []}
    mismatches = {"result": 0, "quality": 0, "mutated": 0}
    scored = {k: 0 for k, _ in detectors}  # 점수가 0보다 큰 장(결과 비교가 "둘 다 normal" 만은 아님을 보인다)

    for i, frame_bytes in enumerate(frames):
        measure = i >= args.warmup
        # before: 감지기마다 디코드 + 흑백 디코드 품질 검사 (디코드는 감지기 시간 안에 들어 있다)
        q_old, t = _timed(lambda: frame_quality.assess_jpeg(frame_bytes, thresholds))
        old_times = {"decode": 0.0, "quality": t}
        old_results = {}
        for kind, det in detectors:
            old_results[kind], old_times[kind] = _timed(lambda d=det: d.detect_from_jpeg(frame_bytes))
        # after: 디코드 1회 → 품질 검사 + 감지기 3개
        frame, t_dec = _timed(lambda: decode_bgr(frame_bytes))
        snapshot = frame.copy() if frame is not None else None
        q_new, t_q = _timed(lambda: frame_quality.assess_bgr(frame, thresholds))
        new_times = {"decode": t_dec, "quality": t_q}
        new_results = {}
        for kind, det in detectors:
            new_results[kind], new_times[kind] = _timed(lambda d=det: d.detect_from_frame(frame))
        if snapshot is not None and not np.array_equal(snapshot, frame):
            mismatches["mutated"] += 1
        if q_old != q_new:
            mismatches["quality"] += 1
        for k, _ in detectors:
            scored[k] += _key(new_results[k])[1] > 0
        if any(_key(old_results[k]) != _key(new_results[k]) for k, _ in detectors):
            mismatches["result"] += 1
        if measure:
            for s in stages:
                before[s].append(old_times[s])
                after[s].append(new_times[s])
            totals["before"].append(sum(old_times.values()))
            totals["after"].append(sum(new_times.values()))

    print(f"{'구간':<10}{'before 평균':>12}{'before p95':>12}{'after 평균':>12}{'after p95':>12}")
    for s in stages:
        print(f"{s:<10}{statistics.mean(before[s]):>12.2f}{_p95(before[s]):>12.2f}"
              f"{statistics.mean(after[s]):>12.2f}{_p95(after[s]):>12.2f}")
    b_mean, a_mean = statistics.mean(totals["before"]), statistics.mean(totals["after"])
    b_p95, a_p95 = _p95(totals["before"]), _p95(totals["after"])
    print(f"{'합계':<10}{b_mean:>12.2f}{b_p95:>12.2f}{a_mean:>12.2f}{a_p95:>12.2f}")
    print(f"[BUDGET] 프레임당 절약 평균 {b_mean - a_mean:.2f}ms ({(b_mean - a_mean) / b_mean * 100:.0f}%)")
    print(f"[BUDGET] 불일치 - 감지 결과 {mismatches['result']} / 품질 판정 {mismatches['quality']} "
          f"/ 공유 배열 변형 {mismatches['mutated']} (전체 {len(frames)}장)")
    print("[BUDGET] 점수가 나온 장 - " + ", ".join(f"{k} {n}" for k, n in scored.items()))
    if args.equality_conf > 0:
        mismatches["boxes"] = _compare_boxes(settings, frames, args.equality_conf)
    verdict = "PASS" if a_p95 <= args.budget_ms else "FAIL"
    print(f"[BUDGET] 예산 {args.budget_ms:.0f}ms 대비 after p95 {a_p95:.2f}ms → {verdict}")
    return 0 if verdict == "PASS" and not any(mismatches.values()) else 1


if __name__ == "__main__":
    raise SystemExit(main())
