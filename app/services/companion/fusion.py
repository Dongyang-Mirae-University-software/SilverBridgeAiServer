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
