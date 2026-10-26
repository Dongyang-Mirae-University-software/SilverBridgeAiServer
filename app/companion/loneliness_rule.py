"""외로움·단절 신호 규칙 (2026-10-25, 사용자 제안 A안).

모델이 외로움을 기쁨이나 당황으로 잘못 읽는 문제를 보완한다.
단어만 보면 틀리므로 세 조건을 모두 만족할 때만 "확인 필요"로 본다.

1. 외로움·단절 표현이 있다        (혼자, 쓸쓸, 외롭, 적막 등)
2. 그 상황이 없음·부재를 말한다  (아무도 ... 않/못/없, 연락이 없, 찾아오지 않 등)
3. 연락·방문이 있었다는 말이 없다 (왔어, 전화, 만났, 들렀 등)  <- 있으면 조건 2를 취소한다

예) "아무도 안 올 줄 알았는데 오늘 누가 왔어" -> 3번 '왔어'가 있어서 신호 없음
    "오늘도 아무도 찾아오지 않았어요"            -> 확인 필요
    "영감이 떠난 뒤로 집이 너무 조용해요"        -> 1번 '조용' 은 빼고 '떠난'/'혼자' 계열만 본다 (아래 참고)

주의: 이 규칙은 진단이 아니다. 결과는 '확인 필요'(보호자 확인 권장)로만 쓴다.
"""

from __future__ import annotations

import re

# 1. 외로움·단절 표현 (단어가 그대로 들어간 경우)
LONELY_WORDS = ("혼자", "외롭", "외로", "쓸쓸", "적막", "고독", "아무도", "찾아오는 사람", "연락이 없", "연락 없")

# 2. 없음·부재를 말하는 표현 (근처에 있으면 조건 2 만족)
ABSENCE = ("않", "못", "없", "안 ", "안왔", "안 왔", "안 와", "안 오", "안올", "안 올")

# 3. 연락·방문이 있었다는 표현 (있으면 신호를 취소)
CONTACT = ("왔어", "왔다", "왔는데", "왔어요", "와서", "전화 왔", "전화해", "통화", "만났", "들렀", "찾아왔", "방문")

WINDOW = 12  # 조건 1과 2 사이를 이 글자 수 안에서 찾는다
LOSS_WINDOW = 20  # 상실 표현과 부가 조건 사이 거리


# 4. 상실 표현 (떠나다 계열). 이것만으로는 신호가 아니고, 아래 부가 조건이 함께 있어야 한다.
LOSS_WORDS = ("떠난", "떠나", "떠났", "먼저 간", "세상을 떠", "돌아가신", "돌아가시", "사별", "보내고", "보낸 뒤")
# 부가 조건: 고요함·빈집·그리움·부재
EXTRA_WORDS = ("조용", "적막", "텅", "썰렁", "허전", "서운", "그립", "그리", "보고 싶", "보고싶", "없", "빈 집", "빈집")
# 제외: 외출·여행처럼 상실이 아닌 말
EXCLUDE_WORDS = ("여행", "출근", "소풍", "나가서", "나갔", "놀러")


def _near_absence(text: str, word_pos: int) -> bool:
    start = max(0, word_pos - WINDOW)
    end = min(len(text), word_pos + WINDOW)
    window = text[start:end]
    return any(a in window for a in ABSENCE)


NEGATION_BEFORE = ("안 ", "안", "못 ", "못", "않")


def _has_positive_contact(text: str) -> bool:
    """연락 표현이 긍정으로 쓰였는지 본다. '안 와서', '못 왔다'처럼 부정 앞에 붙은 것은 제외한다."""
    for c in CONTACT:
        pos = text.find(c)
        while pos != -1:
            before = text[max(0, pos - 3):pos]
            if not any(n in before for n in NEGATION_BEFORE):
                return True
            pos = text.find(c, pos + 1)
    return False


def loneliness_signal(text: str) -> bool:
    """외로움·단절 신호가 있는지 본다. True 면 '확인 필요' 후보다."""
    t = re.sub(r"\s+", " ", text.strip())
    if _has_positive_contact(t):
        return False  # 조건 3: 연락·방문이 있었다고 말하면 신호를 취소
    for w in LONELY_WORDS:
        pos = t.find(w)
        if pos == -1:
            continue
        if w in ("혼자", "쓸쓸", "외롭", "외로", "적막", "고독"):
            # 이런 단어 자체가 외로움을 뜻하므로 조건 2 없이도 신호로 본다
            return True
        if _near_absence(t, pos):
            return True
    if any(x in t for x in EXCLUDE_WORDS):
        return False
    for w in LOSS_WORDS:
        pos = t.find(w)
        if pos == -1:
            continue
        start, end = max(0, pos - LOSS_WINDOW), min(len(t), pos + LOSS_WINDOW)
        if any(x in t[start:end] for x in EXTRA_WORDS):
            return True
    return False


def _self_test() -> None:
    cases = [
        ("아무도 안 올 줄 알았는데 오늘 누가 왔어", False),
        ("오늘도 아무도 찾아오지 않았어요", True),
        ("요즘 밤에 혼자 있으니 자꾸 눈물이 나요", True),
        ("세상에 나 혼자 남은 것 같아 쓸쓸해요", True),
        ("오늘 손주가 와서 같이 밥 먹어서 너무 좋았어요", False),
        ("아무도 전화가 안 와서 서운했어요", True),
        ("오늘 친구가 전화해서 웃다가 왔어요", False),
        ("연락이 없어서 마음이 졸여요", True),
        ("날씨가 좋아서 산책 다녀왔더니 기분이 좋아요", False),
        ("혼자 점심 먹었어요", True),
        ("영감이 떠난 뒤로 집이 너무 조용해요", True),
        ("손주가 떠나서 허전해요", True),
        ("친구가 먼저 떠나서 그리워요", True),
        ("아들이 여행 떠나서 집이 조용해요", False),
        ("버스가 떠났어요 그냥 기다렸어요", False),
    ]
    bad = [(t, got, want) for t, want in cases for got in [loneliness_signal(t)] if got != want]
    if bad:
        raise SystemExit(f"실패: {bad}")
    print(f"외로움 규칙 검사 {len(cases)}건 모두 통과")


if __name__ == "__main__":
    _self_test()
