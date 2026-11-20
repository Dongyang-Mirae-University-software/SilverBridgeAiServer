"""회피 → 기분 보완 (2026-11-19 정서분석 고도화, 2026-11-20 점검 L-4로 턴 판정까지).

기분 어댑터(e1600soup)는 노년의 에두른 거부·회피("말하기 싫어", "됐어, 그만 물어봐", "묻지 마")를
평온(CALM)으로 읽는다. 회피는 평온이 아니라 괴로움에 가깝다. 회피 턴이고 글 기분이 CALM이면 WORRY로 본다.

근거(평가 세트 142턴, e1600soup, main 판본 avoidance): 3갈래 F1 69.0 → 70.4, 부정 놓침 19 → 17/55,
'괜찮은데 부정으로' 오탐 2/81 그대로 (docs/eval-results/2026-11-20-companion-quality-merge.md).

적용 위치 (11/20 점검 L-4, 사용자 결정 "전부 반영"): 처음에는 응답의 textMood.label만 바꿔서 백엔드가 저장하는
턴 기분(turnResult)에는 들어가지 않았다. 지금은 pipeline.analyze_turn_result 가 표정과 합치기 전에 적용한다 -
글 라벨·턴 라벨을 WORRY 쪽으로 두고, 확률 결합(prob_add)에서는 CALM·WORRY 확률을 맞바꿔 "모델이 WORRY로 본 것처럼" 합친다.
textMood.scores 는 모델 원래 확률 그대로 둔다(측정용). 대답 선택은 말 피함(AVOID) 문장이 먼저라 바뀌지 않는다.
COMPANION_AVOID_MOOD=false 면 전과 같다(보정 없음).
"""
from __future__ import annotations

from app.services.companion.avoidance import is_avoidance

ADJUSTED_AVOID = "avoid"  # textMood.adjusted 값 (더한 필드 - 백엔드는 몰라도 된다)


def applies(transcript: str, label: str | None, usable: bool) -> bool:
    """회피 턴이고 쓸 수 있는 글 기분이 CALM인가 (그때만 WORRY로 본다)."""
    return bool(usable and label == "CALM" and is_avoidance(transcript))


def swap_calm_worry(probs: dict[str, float]) -> dict[str, float]:
    """4코드 확률에서 CALM·WORRY 를 맞바꾼다 - 모델이 CALM 에 준 확신을 WORRY 쪽으로 옮긴다 (합은 그대로)."""
    out = dict(probs)
    out["CALM"], out["WORRY"] = probs["WORRY"], probs["CALM"]
    return out
