"""보호자용 '예시 질문 자동 생성' (v2, 2026-10-25): 주제(카테고리)별로 예시 질문을 만든다.

설계
- 어르신께 나가는 질문은 보호자가 직접 쓴다. 이 예시는 생각이 안 날 때 보는 참고용이다.
- 자동 발송되지 않는다.
- v2 변경: 어르신의 사실을 가정하는 표현(지난 여행, 어제 일, 특정 장소·사람 이름 등)을 금지한다.
  이런 표현이 들어간 문장은 자동 검사에서 탈락시킨다.
- 나머지 검사(존댓말, 물음표, 길이, 의료 표현 금지, 영어·이모지 금지)는 generate_questions.py 와 같다.

결과: examples.json (이전 버전은 examples_v1.json 으로 보관)
실행 (저장소 루트에서):
    venv\Scripts\python.exe companion\generate_examples.py
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import generate_questions as gq  # noqa: E402
from generate_questions import ask, check, clean  # noqa: E402

OUT = Path(__file__).resolve().parent / "examples.json"
PER_CATEGORY = 6

# 사실을 가정하는 표현 (v2 금지 목록)
ASSUMPTION_WORDS = ("지난", "어제", "예전", "옛날", "그제", "작년", "지난번", "여행", "손자분", "손녀", "아드님", "따님")

SYSTEM_V2 = gq.SYSTEM + """
7. 어르신의 과거 일, 여행, 특정 가족이나 장소를 가정하지 않습니다. 지금 이 순간의 안부나 일상만 묻습니다.
8. 얼마나, 어떠신가요 같은 평가형 표현을 자연스럽게 쓰고, 어색한 수식어를 붙이지 않습니다."""

CATEGORIES = [
    {"name": "날씨·계절", "topic": "오늘 날씨나 계절에 대해 가볍게 안부 묻기"},
    {"name": "식사·입맛", "topic": "오늘 식사를 하셨는지, 식사는 편안한지 묻기 (약이나 치료 이야기는 하지 않음)"},
    {"name": "가족·손주", "topic": "가족과 연락이 잘 되는지 가볍게 묻기 (구체적인 이름이나 과거 일은 말하지 않음)"},
    {"name": "취미·여가", "topic": "요즘 하시는 일상 활동이나 산책에 대해 묻기"},
    {"name": "이웃·친구", "topic": "오늘 이웃이나 친구와 가볍게 인사를 나누셨는지 묻기"},
    {"name": "잠·휴식", "topic": "오늘 잠은 편안했는지, 쉬는 시간은 어땠는지 묻기"},
    {"name": "추억·이야기", "topic": "좋아하는 음식이나 노래처럼 가볍게 떠올릴 수 있는 이야기 묻기 (과거 사건은 묻지 않음)"},
]


def assumption_free(q: str) -> bool:
    return not any(w in q for w in ASSUMPTION_WORDS)


def main() -> None:
    gq.SYSTEM = SYSTEM_V2  # ask() 가 읽는 시스템 지시문을 v2 로 바꾼다
    result = []
    for cat in CATEGORIES:
        seen: set[str] = set()
        passed: list[str] = []
        tries = 0
        while len(passed) < PER_CATEGORY and tries < PER_CATEGORY * 4:
            tries += 1
            temp = 0.7 + 0.05 * (tries % 6)
            q = clean(ask(cat["topic"], temp))
            if not q or q in seen:
                continue
            seen.add(q)
            if not check(q) and assumption_free(q):
                passed.append(q)
        result.append({"category": cat["name"], "examples": passed})
        print(f"{cat['name']}: {len(passed)}개", flush=True)
    OUT.write_text(json.dumps({"version": "v2", "note": "보호자 참고용 예시. 자동 발송되지 않음. 사실 가정 표현 금지.",
                               "categories": result}, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"저장: {OUT}", flush=True)


if __name__ == "__main__":
    main()
