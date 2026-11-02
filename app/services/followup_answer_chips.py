"""되묻기 답 칩(응답 suggestedQuestions + suggestedKind="answer", 2026-11-02).

짧은 증상 한마디("머리가 아파요")에 되묻기(언제부터·얼마나·같이 나타나는 증상)를 하는 턴에,
어르신이 타이핑하지 않고 눌러서 답할 수 있게 증상 묶음별 고정 문장 4개를 준다(기간 2·정도 1·동반 1).
- 모델이 만들지 않는다 - 이 표의 문장만 나간다. 판정(응급·자해·위기·약 + 술·예약·진료과)은 바꾸지 않는다.
- 칩을 눌러 보낸 문장도 일반 문장과 똑같이 판정되고, 되묻기 다음 턴의 "앞 질문 / 이번 답" 이어 붙이기를 탄다.
- 위험 신호 되묻기(redFlag119, 마무리가 119 한 줄)·답에 119가 든 답·응급 답에는 주지 않는다(이어서 물어볼 질문과 같은 기준).
- 문장 조건(tests/test_followup_answer_chips.py 가 고정): 22자 이하, 응급·자해·위험 신호·예약 낱말·약 + 술·
  되묻기 다음 턴 종료어(chat_service._FOLLOWUP_CLOSE_WORDS - "됐어요" 등) 없음. "며칠 됐어요"는 종료어라 쓰지 않는다.
- 위험 신호 칩("갑자기 아주 심해요")은 두지 않는다 - 키워드 응급 규칙에 걸리지 않아 119 여부가 라우터에 따라 흔들린다.
"""

from __future__ import annotations

import re
from typing import Any

KIND = "answer"

# (묶음 이름, 공백 뺀 문장에 맞출 정규식, 본인용 4개, 대리 상담용 4개). 위에서부터 처음 맞는 묶음을 쓴다.
# 저림·어지럼을 머리보다 먼저("머리가 어지러워요"는 어지럼), 관절을 기침·열보다 먼저("손목"의 목). 맨 끝 기본 묶음은 모두 맞는다.
_GROUPS: tuple[tuple[str, re.Pattern[str], tuple[str, ...], tuple[str, ...]], ...] = (
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
        re.compile(r"무릎|허리|어깨|관절|뼈|손목|발목|다리|팔"),
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


def all_chips() -> list[str]:
    """표의 모든 문장(테스트용)."""
    return [chip for _, _, self_chips, proxy_chips in _GROUPS for chip in (*self_chips, *proxy_chips)]


def group_name(message: str) -> str:
    compact = re.sub(r"\s+", "", message or "")
    return next(name for name, pattern, _, _ in _GROUPS if pattern.search(compact))


def chips_for(message: str, *, proxy: bool) -> list[str]:
    compact = re.sub(r"\s+", "", message or "")
    for _, pattern, self_chips, proxy_chips in _GROUPS:
        if pattern.search(compact):
            return list(proxy_chips if proxy else self_chips)
    return []


def answer_chips(result: dict[str, Any], message: str) -> list[str]:
    """이 답이 답 칩을 줄 되묻기 턴이면 칩 4개, 아니면 빈 목록. 말투는 되묻기 답이 정한 followupProxy 를 따른다."""
    meta = result.get("meta") if isinstance(result.get("meta"), dict) else {}
    if not meta.get("followupAsked") or meta.get("redFlag119"):
        return []
    if result.get("riskLevel") == "high" or result.get("emergencyWarning"):
        return []
    reply = result.get("reply")
    if not isinstance(reply, str) or not reply.strip() or "119" in reply or not message:
        return []
    return chips_for(message, proxy=bool(meta.get("followupProxy")))
