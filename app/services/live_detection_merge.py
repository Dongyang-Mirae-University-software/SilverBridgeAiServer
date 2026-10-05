from __future__ import annotations

from typing import Any

# danger 가 둘 이상일 때 대표로 올릴 순서. 백엔드는 대표 하나(detectedType/confidence/danger)만 본다.
KIND_PRIORITY: tuple[str, ...] = ("fire", "knife", "fall")

_NON_DETECTION_TYPES = {"normal", "unknown"}


def merge_kind_results(kind_results: list[tuple[str, dict[str, Any]]]) -> dict[str, Any]:
    """종류별 감지 결과를 백엔드 계약의 대표 결과 하나로 합친다.

    kind_results: [(종류, 감지기 결과)] — 감지기 결과는 detect_from_jpeg 형태
    (낙상은 유지 판정이 반영된 danger/confidence).

    대표 선택: danger 인 것 중 KIND_PRIORITY 순(화재 > 흉기 > 낙상) → danger 가 없으면 점수가 가장 높은
    감지(동점이면 우선순위 순) → 감지가 하나도 없으면 normal(모든 감지기가 판정 불가면 unknown).
    켜진 감지기가 화재 하나면 대표는 화재 감지기 결과 그대로다(기존 응답과 같다).
    """
    ordered = sorted(kind_results, key=lambda item: _priority(item[0]))

    primary: dict[str, Any] | None = next((r for _, r in ordered if r.get("danger")), None)
    if primary is None:
        detected = [r for _, r in ordered if _is_detection(r)]
        if detected:
            # sorted 는 안정 정렬이라 동점이면 우선순위가 앞선 쪽이 남는다.
            primary = sorted(detected, key=lambda r: -float(r.get("confidence") or 0.0))[0]

    if primary is not None:
        detected_type = primary.get("detectedType", "normal")
        confidence = float(primary.get("confidence") or 0.0)
        danger = bool(primary.get("danger", False))
    else:
        all_unknown = bool(ordered) and all(r.get("detectedType") == "unknown" for _, r in ordered)
        detected_type = "unknown" if all_unknown else "normal"
        confidence = 0.0
        danger = False

    detections: list[dict[str, Any]] = []
    for _, r in ordered:
        detections.extend(r.get("detections") or [])

    return {
        "detectedType": detected_type,
        "confidence": confidence,
        "danger": danger,
        "detections": detections,
        "analyzedAt": ordered[0][1].get("detectedAt") if ordered else None,
        "results": [
            {
                "detectedType": r.get("detectedType") if _is_detection(r) else kind,
                "confidence": float(r.get("confidence") or 0.0),
                "danger": bool(r.get("danger", False)),
                "available": r.get("detectedType") != "unknown",
            }
            for kind, r in ordered
        ],
    }


def _priority(kind: str) -> int:
    return KIND_PRIORITY.index(kind) if kind in KIND_PRIORITY else len(KIND_PRIORITY)


def _is_detection(result: dict[str, Any]) -> bool:
    return result.get("detectedType") not in _NON_DETECTION_TYPES
