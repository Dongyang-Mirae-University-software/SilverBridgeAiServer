"""글 기분 4클래스 확률(textMood.scores) 노출 시험 (2026-11-19, 정서분석 고도화).

B(부정 민감 보정)·E(섞인 감정)·품질 측정에 쓰려면 승자 라벨만이 아니라 4클래스 확률이 필요하다.
토큰 확률을 얻은 턴만 채우고, 못 얻으면(파싱 경로) null.
"""
from __future__ import annotations

from app.services.companion.mood_service import MOOD_CODES
from app.services.companion.pipeline import analyze_turn_result
from app.services.companion.text_emotion_service import TextEmotionResult


def _face():
    return {"label": None, "usable": False, "confidence": 0.0, "frameCount": 0, "totalFrames": 0, "facesFound": 0}


def test_scores_exposed_when_token_probs_available():
    scores = {"JOY": 0.1, "CALM": 0.5, "WORRY": 0.3, "DEPRESSED": 0.1}
    out: dict = {}
    analyze_turn_result("오늘은 기분이 좀 가라앉네요", _face(),
                        lambda t: TextEmotionResult("CALM", 0.5, True, scores), text_out=out)
    assert out["scores"] == {c: round(scores[c], 4) for c in MOOD_CODES}
    assert out["label"] == "CALM"


def test_scores_null_when_only_label_parsed():
    out: dict = {}
    analyze_turn_result("오늘은 기분이 좀 가라앉네요", _face(),
                        lambda t: TextEmotionResult("CALM", 1.0, True, {}), text_out=out)
    assert out["scores"] is None


def test_scores_null_when_classifier_unavailable():
    out: dict = {}
    analyze_turn_result("오늘은 기분이 좀 가라앉네요", _face(), lambda t: None, text_out=out)
    assert out["scores"] is None
