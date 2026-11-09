from __future__ import annotations

from dataclasses import dataclass

# 사용자 결정 (2026-11-02): 턴 라벨 5개 + 판단 보류. 화면·집계에 쓰는 값이다.
TURN_LABELS = ("기쁨", "평온", "슬픔", "화남", "우울")
UNDECIDED = "판단 보류"  # 말과 표정이 서로 다르거나 둘 다 쓸 수 없을 때

# 사용자 결정 (2026-11-02, 제안 매핑 채택): 텍스트 감정 모델 GGARA02 6개 라벨 → 턴 라벨
GGARA_TO_TURN = {
    "기쁨": "기쁨",
    "슬픔": "슬픔",
    "분노": "화남",
    "상처": "우울",
    "불안": "우울",
    "당황": "평온",
}

# 사용자 결정 (2026-11-02, 제안 매핑 채택): 표정 모델 HSEmotion 8개 라벨 → 턴 라벨
# 키는 emotiefflib 가 내는 이름 그대로 (첫 글자 대문자)
HSEMOTION_TO_TURN = {
    "Happiness": "기쁨",
    "Neutral": "평온",
    "Sadness": "슬픔",
    "Anger": "화남",
    "Disgust": "화남",
    "Contempt": "화남",
    "Fear": "우울",
    "Surprise": UNDECIDED,
}

# 11/10 글 기분 e900 의 4코드 → 턴 라벨 (턴 라벨 5개는 그대로). 걱정은 '슬픔'으로 남겨 화면 기분에서 다시 걱정(WORRY)이 된다
MOOD_CODES = ("JOY", "CALM", "WORRY", "DEPRESSED")
MOOD_TO_TURN = {"JOY": "기쁨", "CALM": "평온", "WORRY": "슬픔", "DEPRESSED": "우울"}
# 글 감정 라벨(재희 GGARA02 6감정 또는 e900 4코드) → 턴 라벨. 이름이 겹치지 않아 한 표로 쓴다
TEXT_TO_TURN = {**GGARA_TO_TURN, **MOOD_TO_TURN}
# 표정 → 합칠 때 더해 줄 코드. 웃음 JOY · 평온 CALM · 부정 WORRY (DGX fuse_cmp 와 같다)
_FACE_TURN_TO_CODE = {"기쁨": "JOY", "평온": "CALM", "슬픔": "WORRY", "화남": "WORRY", "우울": "WORRY"}


def face_mood_code(face_label: str | None) -> str | None:
    """표정 라벨(HSEmotion 이름)을 기분 코드로. 놀람·모르는 이름은 None (더하지 않는다)."""
    if not face_label:
        return None
    return _FACE_TURN_TO_CODE.get(HSEMOTION_TO_TURN.get(face_label, ""))


# 말과 표정의 방향 비교용 (CONFLICT 판정에만 쓴다). 평온·판단 보류는 중립으로 본다
_POSITIVE = {"기쁨"}
_NEGATIVE = {"슬픔", "화남", "우울"}


@dataclass
class TurnFusion:
    source: str  # TEXT | FACE | CONFLICT | NO_RESPONSE (아키텍처 08 결합 표)
    label: str | None  # 턴 라벨. 무응답이면 None


def _polarity(label: str) -> int:
    if label in _POSITIVE:
        return 1
    if label in _NEGATIVE:
        return -1
    return 0


def fuse_turn(
    no_response: bool,
    text_label: str | None,
    text_usable: bool,
    face_label: str | None,
    face_usable: bool,
) -> TurnFusion:
    """아키텍처 08 결합 표를 그대로 옮긴 함수. 입력 라벨은 이미 턴 라벨 5개로 바꾼 값이다.

    - 발화 없음 → NO_RESPONSE (라벨 없음, 집계 제외)
    - 말과 표정이 둘 다 쓸 수 있는데 방향이 반대 → CONFLICT, 판단 보류
    - 둘 다 쓸 수 있고 방향이 같거나 중립 → TEXT (말을 채택, 표정은 참고)
    - 한쪽만 쓸 수 있으면 그쪽 결과
    - 둘 다 못 쓰면 판단 보류 (근거 부족을 그대로 드러낸다)
    """
    if no_response:
        return TurnFusion("NO_RESPONSE", None)
    if text_usable and face_usable and text_label and face_label:
        if _polarity(text_label) * _polarity(face_label) < 0:
            return TurnFusion("CONFLICT", UNDECIDED)
        return TurnFusion("TEXT", text_label)
    if text_usable and text_label:
        return TurnFusion("TEXT", text_label)
    if face_usable and face_label:
        return TurnFusion("FACE", face_label)
    return TurnFusion("TEXT", UNDECIDED)


def fuse_prob_add(
    no_response: bool,
    text_probs: dict[str, float] | None,
    face_code: str | None,
    weight: float,
) -> TurnFusion:
    """11/10 합치기 (COMPANION_FUSION_RULE=prob_add): 글 기분 4코드 확률에 표정 코드 쪽으로 weight 를 더해 가장 큰 코드를 고른다.

    재희 fuse_turn 은 말과 표정의 방향이 반대면 보류한다. 새 표정 층은 무표정을 '평온'으로 제대로 잡아서,
    무표정인 어르신의 걱정·우울 말이 거의 다 보류됐다 (시험에서 걱정·우울 찾음 84% → 0.3%). 그래서 보류(CONFLICT)를 만들지 않는다.
    - 무응답 → NO_RESPONSE
    - 글 확률이 있으면: 표정이 판정을 바꿨으면 FACE, 아니면 TEXT
    - 글 확률이 없고 표정만 있으면 표정 코드 (FACE), 둘 다 없으면 판단 보류
    같은 값이면 JOY·CALM·WORRY·DEPRESSED 순서로 앞의 것 (DGX 시험과 같다).
    """
    if no_response:
        return TurnFusion("NO_RESPONSE", None)
    if text_probs:
        q = {c: float(text_probs.get(c, 0.0)) for c in MOOD_CODES}
        text_best = max(MOOD_CODES, key=lambda c: q[c])
        if face_code in q:
            q[face_code] += weight
        best = max(MOOD_CODES, key=lambda c: q[c])
        return TurnFusion("TEXT" if best == text_best else "FACE", MOOD_TO_TURN[best])
    if face_code in MOOD_TO_TURN:
        return TurnFusion("FACE", MOOD_TO_TURN[face_code])
    return TurnFusion("TEXT", UNDECIDED)
