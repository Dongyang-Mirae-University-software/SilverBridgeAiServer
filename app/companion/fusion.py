"""Phase 6: 텍스트 감정과 표정 감정 결과를 하나의 정서 라벨로 합친다. (2026-10-25 규칙 갱신)

규칙
- 긍정·중립 신호는 확신도 0.6 이상일 때만 인정한다 (UNKNOWN_BELOW).
- 부정 신호(분노·슬픔·불안·상처·당황)는 텍스트에서 확신도 0.4 이상이면 인정한다 (NEG_MIN, 사용자 결정 A).
  놓치면 안 되는 신호라서 조금 더 넓게 본다. 진단이 아니라 '확인 필요'로만 표시한다.
- 애매한 대답(글쎄, 모르겠어, 그냥 등)과 신호가 약한 경우는 '판단 보류'로 둔다 (사용자 결정).
  판단 보류가 나오면 다음 질문을 드리고, 여러 대답을 합쳐 하루 결과를 확정한다 (api_mock.py).
- '판단 불가'는 대답 자체가 없을 때만 쓴다.

출력 라벨: 기쁨 / 평온 / 확인 필요 / 판단 보류 / 판단 불가
"""

from __future__ import annotations

from dataclasses import dataclass

UNKNOWN_BELOW = 0.6  # 긍정·중립·표정 신호의 최소 확신도
NEG_MIN = 0.4        # 텍스트 부정 신호의 최소 확신도 (사용자 결정 A, 2026-10-25)

# 텍스트 감정 (GGARA02 6개 라벨) -> 극성
TEXT_POLARITY = {
    "기쁨": "pos",
    "분노": "neg",
    "슬픔": "neg",
    "불안": "neg",
    "상처": "neg",
    "당황": "neg",
}

# 표정 감정 (HSEmotion 8개 라벨, trpakov 7개 라벨 모두 포함) -> 극성. surprise 는 방향이 없어서 중립.
FACE_POLARITY = {
    "happiness": "pos",
    "happy": "pos",
    "neutral": "neu",
    "surprise": "neu",
    "anger": "neg",
    "angry": "neg",
    "contempt": "neg",
    "disgust": "neg",
    "fear": "neg",
    "sadness": "neg",
    "sad": "neg",
}

# 애매한 대답 표현 (답은 했지만 감정이 드러나지 않는 말)
AMBIGUOUS_WORDS = ("글쎄", "모르겠", "그냥", "별일 없", "그럭저럭", "뭐 그렇", "그저 그래", "그럴 뿐")


@dataclass(frozen=True)
class Signal:
    label: str | None  # 모델이 낸 라벨. None 이면 신호 없음
    score: float  # 모델 확신도 (0~1)


def is_ambiguous(text: str) -> bool:
    return any(w in text for w in AMBIGUOUS_WORDS)


def polarity(sig: Signal | None, table: dict[str, str], neg_min: float = UNKNOWN_BELOW) -> str | None:
    """극성(pos/neu/neg)을 돌려준다. 확신이 기준보다 낮으면 None."""
    if sig is None or sig.label is None:
        return None
    pol = table.get(str(sig.label)) or table.get(str(sig.label).lower())
    if pol is None:
        return None
    if pol == "neg":
        return "neg" if sig.score >= neg_min else None
    return pol if sig.score >= UNKNOWN_BELOW else None


def fuse(text: Signal | None, face: Signal | None, ambiguous: bool = False) -> dict:
    """두 신호를 합쳐 정서 라벨과 이유를 돌려준다. text·face 가 모두 None 이면 답이 없는 것이다."""
    if ambiguous:
        return _result("판단 보류", "애매한 대답 (여러 질문으로 확인)", None, None)
    if text is None and face is None:
        return _result("판단 불가", "쓸 수 있는 신호가 없음 (답변 없음, 얼굴 없음 등)", None, None)

    p_text = polarity(text, TEXT_POLARITY, NEG_MIN)
    p_face = polarity(face, FACE_POLARITY)
    valid = [p for p in (p_text, p_face) if p is not None]

    if not valid:
        return _result("판단 보류", "신호가 약함 (확신 낮음, 다음 질문으로 확인)", p_text, p_face)

    if len(valid) == 1:
        p = valid[0]
        only = "텍스트" if p_text is not None else "표정"
        if p == "pos":
            label, reason = "기쁨", f"{only} 신호만 있고 긍정"
        elif p == "neu":
            label, reason = "평온", f"{only} 신호만 있고 중립"
        elif p_face is None:
            label, reason = "확인 필요", "텍스트만으로 부정 신호 (진단 아님, 보호자 확인 권장)"
        else:
            label, reason = "판단 보류", "표정 신호 하나만 부정 (확인 전에는 단정하지 않음)"
        return _result(label, reason, p_text, p_face)

    if p_text == p_face:
        mapping = {"pos": ("기쁨", "두 신호 모두 긍정"), "neu": ("평온", "두 신호 모두 중립"),
                   "neg": ("확인 필요", "두 신호 모두 부정 (진단 아님, 보호자 확인 권장)")}
        label, reason = mapping[p_text]
    elif {p_text, p_face} == {"pos", "neg"}:
        label, reason = "판단 보류", "말과 표정이 반대 방향 (긍정·부정 엇갈림)"
    elif "pos" in (p_text, p_face):
        label, reason = "평온", "중립 신호와 긍정 신호가 섞임"
    else:
        label, reason = "판단 보류", "부정 신호와 중립 신호가 섞임"
    return _result(label, reason, p_text, p_face)


def _result(label: str, reason: str, p_text, p_face) -> dict:
    return {"label": label, "reason": reason, "text_polarity": p_text, "face_polarity": p_face}


def _self_test() -> None:
    T = lambda l, s=0.9: Signal(l, s)  # noqa: E731
    cases = [
        ("둘 다 기쁨", T("기쁨"), T("Happiness"), False, "기쁨"),
        ("표정만 중립", None, T("Neutral"), False, "평온"),
        ("둘 다 부정", T("슬픔"), T("Sadness"), False, "확인 필요"),
        ("말은 긍정, 표정 부정", T("기쁨"), T("Sadness"), False, "판단 보류"),
        ("말은 부정, 표정 긍정", T("분노"), T("Happiness"), False, "판단 보류"),
        ("텍스트만 부정 (확신 높음)", T("불안"), None, False, "확인 필요"),
        ("텍스트만 부정 (확신 0.45) -> 확인 필요 (A안)", T("상처", 0.45), None, False, "확인 필요"),
        ("텍스트만 부정 (확신 0.3) -> 판단 보류", T("분노", 0.3), None, False, "판단 보류"),
        ("텍스트만 긍정 (확신 높음)", T("기쁨", 0.9), None, False, "기쁨"),
        ("텍스트만 긍정 (확신 0.5) -> 판단 보류", T("기쁨", 0.5), None, False, "판단 보류"),
        ("애매한 대답 -> 판단 보류", T("불안", 0.7), None, True, "판단 보류"),
        ("대답 없음 -> 판단 불가", None, None, False, "판단 불가"),
        ("표정 확신 낮음 무시", T("기쁨"), T("Sadness", 0.3), False, "기쁨"),
        ("놀람은 중립", T("기쁨"), T("Surprise"), False, "평온"),
    ]
    failed = []
    for name, t, f, amb, want in cases:
        got = fuse(t, f, ambiguous=amb)["label"]
        if got != want:
            failed.append((name, got, want))
    if failed:
        raise SystemExit(f"실패: {failed}")
    print(f"규칙 검사 {len(cases)}건 모두 통과")


if __name__ == "__main__":
    _self_test()


def decide_turn(text_pol: str | None, face_pol: str | None) -> dict:
    """대답 한 턴의 결과 (2026-10-25 사용자 규칙).

    - 말이 기쁨(pos)·평온(neu)이면 그대로 통과. 표정은 보지 않는다.
    - 말이 부정(neg)이면 표정을 본다: 표정도 긍정이면 판단 보류(엇갈림), 아니면 확인 필요.
    - 말이 애매(None)하면: 표정이 부정이면 확인 필요, 표정이 긍정·중립이면 판단 보류,
      표정도 애매하면 확인 필요.
    """
    if text_pol == "pos":
        return {"label": "기쁨", "reason": "말이 긍정이라 표정은 보지 않음"}
    if text_pol == "neu":
        return {"label": "평온", "reason": "말이 중립이라 표정은 보지 않음"}
    if text_pol == "neg":
        if face_pol == "pos":
            return {"label": "판단 보류", "reason": "말은 부정, 표정은 긍정 (엇갈림, 확인 전 단정하지 않음)"}
        return {"label": "확인 필요", "reason": "말이 부정 (진단 아님, 보호자 확인 권장)"}
    if face_pol == "neg":
        return {"label": "확인 필요", "reason": "말은 애매하고 표정은 부정 (진단 아님, 보호자 확인 권장)"}
    if face_pol in ("pos", "neu"):
        return {"label": "판단 보류", "reason": "말은 애매, 표정은 긍정·중립 (다음 질문으로 확인)"}
    return {"label": "확인 필요", "reason": "말도 표정도 애매함 (보호자 확인 권장)"}
