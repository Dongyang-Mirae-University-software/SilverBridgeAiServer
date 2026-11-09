from __future__ import annotations

from app.services.companion.fusion import (
    GGARA_TO_TURN,
    HSEMOTION_TO_TURN,
    TURN_LABELS,
    UNDECIDED,
    fuse_turn,
)


def test_no_response_has_no_label():
    r = fuse_turn(no_response=True, text_label=None, text_usable=False, face_label=None, face_usable=False)
    assert (r.source, r.label) == ("NO_RESPONSE", None)


def test_text_only_when_face_is_unusable():
    r = fuse_turn(False, "슬픔", True, "기쁨", False)
    assert (r.source, r.label) == ("TEXT", "슬픔")


def test_face_only_when_text_is_unusable():
    r = fuse_turn(False, None, False, "기쁨", True)
    assert (r.source, r.label) == ("FACE", "기쁨")


def test_same_direction_takes_text():
    r = fuse_turn(False, "슬픔", True, "우울", True)
    assert (r.source, r.label) == ("TEXT", "슬픔")


def test_opposite_direction_is_conflict_and_undecided():
    r = fuse_turn(False, "기쁨", True, "슬픔", True)
    assert (r.source, r.label) == ("CONFLICT", UNDECIDED)


def test_neutral_face_with_negative_text_is_not_conflict():
    r = fuse_turn(False, "슬픔", True, "평온", True)
    assert (r.source, r.label) == ("TEXT", "슬픔")


def test_nothing_usable_is_undecided():
    r = fuse_turn(False, None, False, None, False)
    assert (r.source, r.label) == ("TEXT", UNDECIDED)


def test_mapping_tables_only_use_turn_labels_or_undecided():
    allowed = set(TURN_LABELS) | {UNDECIDED}
    assert set(GGARA_TO_TURN.values()) <= allowed
    assert set(HSEMOTION_TO_TURN.values()) <= allowed


def test_ggara_keys_are_the_six_model_labels():
    assert set(GGARA_TO_TURN) == {"기쁨", "슬픔", "분노", "상처", "불안", "당황"}


def test_hsemotion_keys_match_the_8_class_model():
    from emotiefflib import facial_analysis as fa  # noqa: F401  (설치 확인)
    names = {"Anger", "Contempt", "Disgust", "Fear", "Happiness", "Neutral", "Sadness", "Surprise"}
    assert set(HSEMOTION_TO_TURN) == names


def _clf(label: str, conf: float = 0.9):
    from app.services.companion.text_emotion_service import TextEmotionResult

    return lambda text: TextEmotionResult(label, conf, conf >= 0.6, {})


def test_pipeline_long_text_uses_text_label():
    from app.services.companion import pipeline

    r = pipeline.analyze_turn_result("오늘은 손주가 와서 정말 기분이 좋아요", {"label": None, "usable": False}, _clf("기쁨"))
    assert r == {"label": "기쁨", "source": "TEXT"}


def test_pipeline_short_text_falls_back_to_face(monkeypatch):
    from app.core.config import get_settings
    from app.services.companion import pipeline

    monkeypatch.setattr(get_settings(), "companion_text_mood_engine", "ggara02")  # 재희: 10자 미만은 말을 쓰지 않는다

    face = {"label": "Sadness", "usable": True}
    r = pipeline.analyze_turn_result("네 좋아요", face, _clf("기쁨"))
    assert r == {"label": "슬픔", "source": "FACE"}


def test_pipeline_text_model_missing_uses_face_only():
    from app.services.companion import pipeline
    from app.services.companion.text_emotion_service import TextEmotionUnavailable

    def broken(_text):
        raise TextEmotionUnavailable("없음")

    face = {"label": "Happiness", "usable": True}
    r = pipeline.analyze_turn_result("오늘은 손주가 와서 정말 기분이 좋아요", face, broken)
    assert r == {"label": "기쁨", "source": "FACE"}


def test_every_fusion_outcome_uses_only_the_allowed_vocabulary():
    # S7-6: 턴 라벨은 기쁨·평온·슬픔·화남·우울 5개와 판단 보류만 나온다
    import itertools

    allowed = set(TURN_LABELS) | {UNDECIDED}
    labels = [None, *TURN_LABELS, UNDECIDED]
    for no_resp, tl, tu, fl, fu in itertools.product([False, True], labels, [False, True], labels, [False, True]):
        r = fuse_turn(no_resp, tl, tu, fl, fu)
        if r.label is not None:
            assert r.label in allowed, (no_resp, tl, tu, fl, fu, r)
        assert r.source in {"TEXT", "FACE", "CONFLICT", "NO_RESPONSE"}
