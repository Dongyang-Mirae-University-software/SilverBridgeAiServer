"""회피 → 기분 보완 (2026-11-19, 정서분석 고도화).

기분 어댑터(e1600soup)는 노년의 에두른 거부·회피("그만해", "말하기 싫어", "그런 건 왜 물어", "됐어, 그만 물어봐")를
평온(CALM)으로 읽는다. 회피는 평온이 아니라 괴로움에 가깝다. 회피 턴이고 보고 기분이 CALM이면 WORRY로 바로잡는다.

근거(평가 세트 142턴 실측, 2026-11-19): 이 보정만 적용하면
  정확도 0.652 → 0.667, 부정 놓침 34.6% → 30.9%, '괜찮은데 부정으로' 오탐 0개 추가(2.5% 유지).
작게 하지만 깨끗한 이득(회복 2턴, 새 오탐 없음). 라벨만으로 직전 부정을 잇는 규칙(C)은 오탐이 크게 늘어 쓰지 않는다.

보고되는 글 기분(textMood.label, 백엔드 여러 날 기분 집계용)만 바꾼다. 대답 선택·회피 처리(turnLabel)는 그대로 둔다.
"""
from __future__ import annotations

from app.services.companion.avoidance import is_avoidance


def adjust_avoid_mood(transcript: str, text_mood: dict | None) -> None:
    """회피이고 보고 기분이 쓸 수 있는 CALM이면 WORRY로 바꾼다 (text_mood 를 제자리에서 수정).

    label(보고용)만 바꾸고 turnLabel(대답 선택용)은 두어, 대답 흐름은 건드리지 않는다.
    """
    if (text_mood and text_mood.get("usable")
            and text_mood.get("label") == "CALM" and is_avoidance(transcript)):
        text_mood["label"] = "WORRY"
        text_mood["adjusted"] = "avoid"  # 더한 표시 (백엔드는 몰라도 된다). 보정된 턴임을 드러낸다
