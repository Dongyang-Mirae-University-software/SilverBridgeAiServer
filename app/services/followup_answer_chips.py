"""되묻기 답 버튼(응답 suggestedQuestions + suggestedKind="answer", 2026-11-02 → 2026-11-09 단계 문답).

되묻기 턴에 어르신이 타이핑하지 않고 눌러서 답할 수 있게, 지금 질문 하나에 맞는 버튼 2~3개를 준다("네"/"아니요",
"오늘부터요"/"며칠 됐어요"/"오래됐어요"). 문장은 followup_steps 표의 것만 나간다 - 모델이 만들지 않는다.
- 버튼을 눌러 보낸 문장도 일반 문장과 똑같이 판정된다(응급·자해·위기는 단계 처리보다 먼저).
- 보호자 상담(단계 상태 없는 옛 한 번 되묻기, 2026-11-11)은 아래 #160 묶음 표 4개를 준다.
- 피보호자 단계 되묻기는 단계 상태(meta.followupStep)로 지금 질문의 버튼을 준다. 응급 답(high·경고 박스)에는 주지 않는다.
  위험 신호 낱말이 있는 첫 턴(redFlag119, 119 한 줄이 함께 나간다)에도 위험 신호 질문의 "네"/"아니요"를 준다 -
  "네"는 규칙으로 바로 119 답이라 라우터에 기대지 않는다(2026-11-02 의 "칩 없음"은 그 칩이 라우터 판정에 기댔기 때문이다).
"""

from __future__ import annotations

import re
from typing import Any

from app.services import followup_steps

KIND = "answer"


# ---- 보호자 상담 옛 한 번 되묻기의 답 칩(#160 표 그대로, 2026-11-11 되살림) ----------------------------------------
# 단계 상태가 없는 되묻기(보호자 상담 - 질문 3개를 한 번에)에 기간 2·정도 1·동반 1 문장 4개를 준다.
# 위험 신호 되묻기(redFlag119 - 마무리가 119 한 줄)에는 주지 않는다(옛 기준 그대로 - 그 칩은 라우터 판정에 기댄다).
# (묶음 이름, 공백 뺀 문장에 맞출 정규식, 본인용 4개, 대리 상담용 4개). 위에서부터 처음 맞는 묶음을 쓴다.
# 저림·어지럼을 머리보다 먼저("머리가 어지러워요"는 어지럼), 관절을 기침·열보다 먼저("손목"의 목). 맨 끝 기본 묶음은 모두 맞는다.
_LEGACY_GROUPS: tuple[tuple[str, re.Pattern[str], tuple[str, ...], tuple[str, ...]], ...] = (
    (
        "numb",
        re.compile(r"저리|저려|저린|저림"),
        ("오늘부터 그래요", "몇 달째 그래요", "밤에 더 저려요", "손발이 다 저려요"),
        ("오늘부터 그러시대요", "몇 달째 그러시대요", "밤에 더 저리시대요", "손발이 다 저리시대요"),
    ),
    (
        "dizzy",
        re.compile(r"어지"),
        ("오늘 처음 그래요", "며칠째 계속 그래요", "일어설 때 심해요", "속도 메스꺼워요"),
        ("오늘 처음 그러시대요", "며칠째 계속 그러시대요", "일어서실 때 심하시대요", "속도 메스꺼우시대요"),
    ),
    (
        "head",
        re.compile(r"머리|두통"),
        ("오늘부터 아파요", "며칠째 계속 아파요", "점점 심해져요", "열도 나요"),
        ("오늘부터 아프시대요", "며칠째 계속 아프시대요", "점점 심해지신대요", "열도 나신대요"),
    ),
    (
        "belly",
        re.compile(r"배|속|구토|토할|설사|변비|소화"),
        ("오늘부터 아파요", "며칠째 계속 아파요", "먹고 나면 더 아파요", "설사도 해요"),
        ("오늘부터 아프시대요", "며칠째 계속 아프시대요", "드시고 나면 더 아프시대요", "설사도 하신대요"),
    ),
    (
        "joint",
        # 손·발은 "손이 아파요" 같은 통증 말만(손 떨림·발 부음은 기본 묶음), 목은 뻐근·결림만 - "목이 아파요"는 기침·열 묶음(2026-11-03 점검 L-1).
        re.compile(r"무릎|허리|어깨|관절|뼈|손목|발목|다리|팔|손가락|발가락|손이아|발이아|손이쑤|발이쑤|목이뻐근|목뻐근|목이결|목결"),
        ("며칠 전부터 아파요", "몇 달째 아파요", "걸을 때 더 아파요", "붓고 열감이 있어요"),
        ("며칠 전부터 아프시대요", "몇 달째 아프시대요", "걸으실 때 더 아프시대요", "붓고 열감이 있으시대요"),
    ),
    (
        "cough",
        re.compile(r"기침|가래|감기|열|목|코|콧물"),
        ("오늘부터 그래요", "며칠째 계속돼요", "밤에 더 심해요", "가래도 나와요"),
        ("오늘부터 그러시대요", "며칠째 계속되신대요", "밤에 더 심하시대요", "가래도 나오신대요"),
    ),
    (
        "default",
        re.compile(r""),
        ("오늘부터 그래요", "며칠 전부터 그래요", "점점 심해져요", "참을 만해요"),
        ("오늘부터 그러시대요", "며칠 전부터 그러시대요", "점점 심해지신대요", "참으실 만하시대요"),
    ),
)


def legacy_all_chips() -> list[str]:
    """표의 모든 문장(테스트용)."""
    return [chip for _, _, self_chips, proxy_chips in _LEGACY_GROUPS for chip in (*self_chips, *proxy_chips)]


def legacy_group_name(message: str) -> str:
    compact = re.sub(r"\s+", "", message or "")
    return next(name for name, pattern, _, _ in _LEGACY_GROUPS if pattern.search(compact))


def legacy_chips_for(message: str, *, proxy: bool) -> list[str]:
    compact = re.sub(r"\s+", "", message or "")
    for _, pattern, self_chips, proxy_chips in _LEGACY_GROUPS:
        if pattern.search(compact):
            return list(proxy_chips if proxy else self_chips)
    return []


def answer_chips(result: dict[str, Any], message: str) -> list[str]:
    """이 답이 답 버튼을 줄 단계 되묻기 턴이면 지금 질문의 버튼, 아니면 빈 목록."""
    meta = result.get("meta") if isinstance(result.get("meta"), dict) else {}
    if not meta.get("followupAsked"):
        return []
    if result.get("riskLevel") == "high" or result.get("emergencyWarning"):
        return []
    reply = result.get("reply")
    if not isinstance(reply, str) or not reply.strip() or not message:
        return []
    state = followup_steps.valid_state(meta.get("followupStep"))
    if state:
        return followup_steps.chips(state)
    if meta.get("followupStep") is not None or meta.get("redFlag119") or "119" in reply:
        return []  # 손상된 단계 상태·위험 신호 되묻기에는 주지 않는다
    return legacy_chips_for(message, proxy=bool(meta.get("followupProxy")))
