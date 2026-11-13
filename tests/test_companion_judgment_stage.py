"""말벗 판단 단계 교체 (11/10): 기분 e900 · 위험 v9 + kw4 · 대답 cc5 · 표정 head3 · 합치기 prob_add · 요약 머리말.

모델은 모두 가짜다 (실제 가중치 측정은 DGX). 스위치를 재희 값으로 돌리면 재희 동작인지도 본다.
"""

from __future__ import annotations

import io
import threading
import time

import numpy as np
import pytest
from PIL import Image

from app.core.config import get_settings
from app.services.companion import pipeline
from app.services.companion.answer_summary import summarize_answer
from app.services.companion.face_service import (
    HEAD3_LABELS,
    FaceService,
    expand_box,
    head3_decide,
    head3_probs,
)
from app.services.companion.fusion import UNDECIDED, face_mood_code, fuse_prob_add
from app.services.companion.llm_client import (
    MOOD_ADAPTER,
    REPLY_ADAPTER,
    AdapterGate,
    ExaoneLlm,
    LlmUnavailable,
)
from app.services.companion.mood_service import PREFIX as MOOD_PREFIX
from app.services.companion.mood_service import MoodService, parse_mood
from app.services.companion.reply_service import build_reply_prompt, generate_reply
from app.services.companion.risk import (
    keyword_risk,
    risk_probability,
    risk_threshold,
)
from app.services.companion.text_emotion_service import TextEmotionResult, TextEmotionUnavailable


@pytest.fixture
def settings(monkeypatch):
    s = get_settings()

    def set_(**kw):
        for k, v in kw.items():
            monkeypatch.setattr(s, k, v)
    return set_


JAEHEE = dict(companion_text_mood_engine="ggara02", companion_fusion_rule="plan41", companion_face_head="hsemotion8",
              companion_risk_prompt="v4", companion_risk_threshold=None, companion_risk_keyword_rule="jaehee",
              companion_reply_lora=False, companion_summary_strip_prefix=False, companion_min_text_chars=None)


def test_new_switch_defaults():
    s = get_settings()
    assert (s.companion_text_mood_engine, s.companion_fusion_rule, s.companion_face_head) == ("e900", "prob_add", "head3")
    assert (s.companion_risk_prompt, s.companion_risk_keyword_rule) == ("v9", "kw4")
    assert s.companion_face_weight == 0.2 and s.companion_face_neg_min == 0.8
    assert s.companion_reply_lora is True and s.companion_summary_strip_prefix is True
    assert risk_threshold() == 0.99 and pipeline.min_text_chars() == 2


@pytest.mark.parametrize("env_value, expected", [
    (None, "companion/llm/lora-reply-cc7"),  # 11/13 기본값 cc7 (환경 변수 없이 띄우면 cc7)
    ("companion/llm/lora-reply-cc5", "companion/llm/lora-reply-cc5"),  # 되돌림: 환경 변수가 이긴다
])
def test_reply_lora_subdir_default_is_cc7(env_value, expected):
    # 설정 기본값은 import 때 정해진다. 여기서 config를 다시 import하면 다른 테스트가 쓰는 설정까지 바뀌므로
    # 별도 프로세스에서 읽는다 (.env는 읽지 않게 막는다)
    import os
    import subprocess
    import sys

    env = {k: v for k, v in os.environ.items() if k != "COMPANION_REPLY_LORA_SUBDIR"}
    if env_value is not None:
        env["COMPANION_REPLY_LORA_SUBDIR"] = env_value
    code = ("import dotenv; dotenv.load_dotenv = lambda *a, **k: False\n"
            "from app.core.config import Settings; print(Settings().companion_reply_lora_subdir)")
    out = subprocess.run([sys.executable, "-c", code], env=env, capture_output=True, text=True, timeout=60, check=True)
    assert out.stdout.strip().splitlines()[-1] == expected


def test_jaehee_values_restore_old_thresholds(settings):
    settings(**JAEHEE)
    assert risk_threshold() == 0.5 and pipeline.min_text_chars() == 10


def test_explicit_threshold_and_min_chars_win(settings):
    settings(companion_risk_threshold=0.7, companion_min_text_chars=5)
    assert risk_threshold() == 0.7 and pipeline.min_text_chars() == 5


# --- 어댑터 전환 ---------------------------------------------------------------------------

def test_gate_switches_only_when_adapter_changes():
    gate, calls = AdapterGate(), []
    for name in (None, "mood", "mood", None, "reply"):
        with gate.use(name, calls.append):
            pass
    assert calls == ["mood", None, "reply"]  # 처음 None 은 이미 기본 상태라 바꾸지 않는다


def test_gate_runs_same_adapter_together_and_waits_for_others():
    gate, log = AdapterGate(), []
    inside = threading.Barrier(2, timeout=2)

    def same():
        with gate.use(None, log.append):
            inside.wait()  # 둘 다 동시에 안에 있어야 통과한다
            time.sleep(0.05)
            log.append("same-done")

    def other(started):
        started.wait()
        with gate.use("reply", log.append):
            log.append(("reply-running", gate.current))

    started = threading.Event()
    t1, t2 = threading.Thread(target=same), threading.Thread(target=same)
    t3 = threading.Thread(target=other, args=(started,))
    t1.start(); t2.start(); time.sleep(0.01); started.set(); t3.start()  # noqa: E702
    for t in (t1, t2, t3):
        t.join(3)
    # 다른 어댑터는 진행 중인 두 호출이 끝난 뒤에야 바꾼다
    assert log == ["same-done", "same-done", "reply", ("reply-running", "reply")]


class _FakeBase:
    def __init__(self):
        self.calls = []

    def disable_adapter_layers(self):
        self.calls.append("off")

    def enable_adapter_layers(self):
        self.calls.append("on")


class _FakePeft:
    def __init__(self):
        self.base_model = _FakeBase()

    def set_adapter(self, name):
        self.base_model.calls.append(name)


def test_switch_turns_adapters_on_and_off():
    llm = ExaoneLlm()
    llm._model, llm._adapters = _FakePeft(), {MOOD_ADAPTER, REPLY_ADAPTER}
    llm._switch(MOOD_ADAPTER)
    llm._switch(None)
    llm._switch(REPLY_ADAPTER)
    assert llm._model.base_model.calls == ["on", "mood", "off", "on", "reply"]


def test_missing_adapter_is_unavailable_and_reported(monkeypatch):
    llm = ExaoneLlm()
    llm._adapters = {MOOD_ADAPTER}
    monkeypatch.setattr(llm, "_load", lambda: (None, None))
    assert llm.has_adapter(MOOD_ADAPTER) and not llm.has_adapter(REPLY_ADAPTER)
    with pytest.raises(LlmUnavailable):
        llm._need_adapter(REPLY_ADAPTER)
    llm._need_adapter(None)  # 기본 모델은 언제나 쓸 수 있다


def test_no_adapter_folders_keeps_the_plain_model(settings, tmp_path):
    settings(model_base_path=str(tmp_path))
    llm, model = ExaoneLlm(), object()
    assert llm._attach_adapters(model) is model and llm._adapters == set()


# --- 글 기분 e900 ---------------------------------------------------------------------------

@pytest.mark.parametrize("raw,code", [
    ('{"mood": "WORRY"}', "WORRY"), ("{mood: JOY}", "JOY"), ('{"mood": "calm"}', "CALM"),
    ('{"mood": "UNKNOWN"}', "UNKNOWN"), ("모르겠어요", "UNKNOWN"), ("", "UNKNOWN"), ('{"mood": "HAPPY"}', "UNKNOWN"),
])
def test_parse_mood_is_lenient(raw, code):
    assert parse_mood(raw) == code


class MoodLlm:
    def __init__(self, probs=None, raw=None):
        self.probs, self.raw, self.calls = probs, raw, []

    def message_token_probs(self, system, user, prefix, candidates, *, adapter=None, timeout):
        self.calls.append(("probs", system, user, prefix, candidates, adapter))
        return self.probs

    def generate_adapter(self, user, *, adapter, system=None, temperature, num_predict, timeout):
        self.calls.append(("gen", system, user, adapter))
        return self.raw


def test_mood_prob_add_reads_four_codes_with_mood_adapter():
    llm = MoodLlm(probs={"JOY": 0.1, "CALM": 0.2, "WORRY": 0.6, "DEPRESSED": 0.1})
    r = MoodService(llm).classify("잠을 못 자서 걱정이야")
    kind, system, user, prefix, cands, adapter = llm.calls[0]
    assert kind == "probs" and adapter == MOOD_ADAPTER and prefix == MOOD_PREFIX == '{"mood": "'
    assert cands == ("JOY", "CALM", "WORRY", "DEPRESSED") and user == "대답:\n잠을 못 자서 걱정이야"
    assert "WORRY: 걱정" in system and 'UNKNOWN' in system
    assert (r.label, r.usable, r.scores["WORRY"]) == ("WORRY", True, 0.6)


def test_mood_plan41_generates_and_unknown_is_unusable(settings):
    settings(companion_fusion_rule="plan41")
    r = MoodService(MoodLlm(raw="{mood: DEPRESSED}")).classify("기운이 없어")
    assert (r.label, r.usable, r.scores) == ("DEPRESSED", True, {})
    r = MoodService(MoodLlm(raw='{"mood": "UNKNOWN"}')).classify("음")
    assert (r.label, r.usable) == ("UNKNOWN", False)


def test_unknown_mood_falls_back_to_face(settings):
    settings(companion_fusion_rule="plan41")
    clf = MoodService(MoodLlm(raw="???")).classify
    r = pipeline.analyze_turn_result("그냥 그렇지 뭐", {"label": "Happiness", "usable": True}, clf)
    assert r == {"label": "기쁨", "source": "FACE"}
    r = pipeline.analyze_turn_result("그냥 그렇지 뭐", {}, clf)
    assert r == {"label": UNDECIDED, "source": "TEXT"}


def test_mood_without_llm_features_is_unavailable():
    class Plain:
        def generate(self, *_a, **_k):
            return ""
    with pytest.raises(TextEmotionUnavailable):
        MoodService(Plain()).classify("좋아요")


def _e900(probs):
    return lambda _t: TextEmotionResult(max(probs, key=probs.get), max(probs.values()), True, dict(probs))


def test_two_char_answer_is_judged_with_e900():
    r = pipeline.analyze_turn_result("좋아", {}, _e900({"JOY": 0.8, "CALM": 0.1, "WORRY": 0.05, "DEPRESSED": 0.05}))
    assert r == {"label": "기쁨", "source": "TEXT"}


def test_two_char_answer_is_skipped_with_jaehee_engine(settings):
    settings(**JAEHEE)
    r = pipeline.analyze_turn_result("좋아", {}, lambda _t: TextEmotionResult("기쁨", 0.9, True, {}))
    assert r == {"label": UNDECIDED, "source": "TEXT"}


def test_router_picks_text_classifier_by_engine(settings):
    from app.routers.companion_router import get_text_classifier
    from app.services.companion.mood_service import mood_service
    from app.services.companion.text_emotion_service import text_emotion_service

    assert get_text_classifier() == mood_service.classify
    settings(companion_text_mood_engine="ggara02")
    assert get_text_classifier() == text_emotion_service.classify


# --- 말·표정 합치기 -------------------------------------------------------------------------

WORRY_TEXT = {"JOY": 0.05, "CALM": 0.25, "WORRY": 0.6, "DEPRESSED": 0.1}


def test_worry_words_with_calm_face_stay_worry_not_undecided():
    r = pipeline.analyze_turn_result("요즘 잠을 못 자서 걱정이야", {"label": "Neutral", "usable": True}, _e900(WORRY_TEXT))
    assert r == {"label": "슬픔", "source": "TEXT"}  # 슬픔 → 화면 기분 걱정(WORRY)


def test_calm_words_with_low_confidence_and_smile_become_joy():
    probs = {"JOY": 0.35, "CALM": 0.45, "WORRY": 0.1, "DEPRESSED": 0.1}
    r = pipeline.analyze_turn_result("그냥 그래요", {"label": "Happiness", "usable": True}, _e900(probs))
    assert r == {"label": "기쁨", "source": "FACE"}


def test_no_face_uses_text_only():
    for face in ({}, {"label": "Happiness", "usable": False}):
        r = pipeline.analyze_turn_result("요즘 잠을 못 자서 걱정이야", face, _e900(WORRY_TEXT))
        assert r == {"label": "슬픔", "source": "TEXT"}


def test_prob_add_never_makes_conflict_but_plan41_does(settings):
    face = {"label": "Happiness", "usable": True}
    assert pipeline.analyze_turn_result("요즘 잠을 못 자서 걱정이야", face, _e900(WORRY_TEXT))["source"] != "CONFLICT"
    settings(companion_fusion_rule="plan41")
    r = pipeline.analyze_turn_result("요즘 잠을 못 자서 걱정이야", face, _e900(WORRY_TEXT))
    assert r == {"label": UNDECIDED, "source": "CONFLICT"}  # 재희 동작: 방향이 반대면 보류


def test_face_weight_switch(settings):
    probs = {"JOY": 0.35, "CALM": 0.45, "WORRY": 0.1, "DEPRESSED": 0.1}
    settings(companion_face_weight=0.05)
    r = pipeline.analyze_turn_result("그냥 그래요", {"label": "Happiness", "usable": True}, _e900(probs))
    assert r == {"label": "평온", "source": "TEXT"}


def test_fuse_prob_add_rules():
    assert fuse_prob_add(True, None, "JOY", 0.2).source == "NO_RESPONSE"
    assert (fuse_prob_add(False, None, "WORRY", 0.2).label, fuse_prob_add(False, None, "WORRY", 0.2).source) == ("슬픔", "FACE")
    assert fuse_prob_add(False, None, None, 0.2).label == UNDECIDED
    tie = {"JOY": 0.25, "CALM": 0.25, "WORRY": 0.25, "DEPRESSED": 0.25}
    assert fuse_prob_add(False, tie, None, 0.2).label == "기쁨"  # 같은 값이면 JOY·CALM·WORRY·DEPRESSED 순서


@pytest.mark.parametrize("label,code", [("Happiness", "JOY"), ("Neutral", "CALM"), ("Sadness", "WORRY"),
                                        ("Anger", "WORRY"), ("Fear", "WORRY"), ("Surprise", None), (None, None)])
def test_face_codes(label, code):
    assert face_mood_code(label) == code


def test_ggara_result_under_prob_add_uses_jaehee_table():
    r = pipeline.analyze_turn_result("오늘은 손주가 와서 정말 좋았어요", {"label": "Sadness", "usable": True},
                                     lambda _t: TextEmotionResult("기쁨", 0.9, True, {"기쁨": 0.9}))
    assert r == {"label": UNDECIDED, "source": "CONFLICT"}  # 4코드 확률이 없으면 재희 표


# --- 위험 v9 + 낱말 규칙 kw4 --------------------------------------------------------------

class RiskLlm:
    def __init__(self, p):
        self.p, self.calls, self.v4_calls = p, [], 0

    def generate(self, *_a, **_k):
        raise LlmUnavailable("off")

    def message_token_probs(self, system, user, prefix, candidates, *, adapter=None, timeout):
        self.calls.append((system, user, prefix, candidates, adapter))
        return {"있": self.p, "없": 1 - self.p}

    def next_token_probs(self, prompt, prefix, candidates, timeout):
        self.v4_calls += 1
        return {"있": self.p, "없": 1 - self.p}


def test_v9_uses_system_prompt_and_base_model():
    llm = RiskLlm(0.995)
    assert risk_probability(llm, "애들한테 짐만 되니 사라지는 게 낫지") == 0.995
    system, user, prefix, cands, adapter = llm.calls[0]
    assert "노인의 돌려 말한 자살 신호" in system and user == "대답:\n애들한테 짐만 되니 사라지는 게 낫지"
    assert prefix == '{"risk": "' and cands == ("있", "없") and adapter is None and llm.v4_calls == 0


def _risk_turn(llm, text="요즘 그냥 다 귀찮아요"):
    clf = _e900({"JOY": 0.1, "CALM": 0.2, "WORRY": 0.2, "DEPRESSED": 0.5})
    return pipeline.run_turn(transcript=text, face={}, classify_text=clf, llm=llm, question="오늘 어떠세요?",
                             prev_signals=None, is_last_turn=False, turn_no=1)


@pytest.mark.parametrize("p,flag", [(0.99, True), (0.9899, False), (0.5, False)])
def test_v9_threshold_is_point_99(p, flag):
    assert _risk_turn(RiskLlm(p))["riskFlag"] is flag


def test_v4_switch_uses_jaehee_prompt_and_half(settings):
    settings(companion_risk_prompt="v4")
    llm = RiskLlm(0.6)
    assert _risk_turn(llm)["riskFlag"] is True and llm.v4_calls == 1 and llm.calls == []


def test_v9_without_probability_feature_gives_none():
    class Old:
        def next_token_probs(self, *_a, **_k):
            return {"있": 1.0, "없": 0.0}
    assert risk_probability(Old(), "죽고 싶어") is None


@pytest.mark.parametrize("text", ["혼자 살다보니까 적적해", "혼자살다 보니 밥을 대충 먹어", "허리가 아파 죽겠어", "무릎이 쑤셔서 죽겠네",
                                  "손주가 걱정돼 죽겠어", "더워 죽겠다"])
def test_kw4_drops_known_false_alarms(text, settings):
    assert keyword_risk(text) is False
    settings(companion_risk_keyword_rule="jaehee")
    assert keyword_risk(text) is True  # 재희 규칙은 잡았다 (오탐)


@pytest.mark.parametrize("text", ["자살하고 싶어", "자 살 생각이 나", "그냥 죽고 싶다", "아이고 죽겠네", "약을 모아 뒀어",
                                  "영감 따라가고 싶어", "숨이 안 쉬어져", "119 불러 줘", "자해를 했어"])
def test_kw4_keeps_real_signals(text):
    assert keyword_risk(text) is True


def test_keyword_rule_still_alerts_when_model_is_off():
    r = _risk_turn(RiskLlm(0.0), "그냥 죽고 싶어")
    assert r["riskFlag"] is True and r["replySource"] == "SAFETY"
    r = _risk_turn(RiskLlm(0.0), "허리가 아파 죽겠어")
    assert r["riskFlag"] is False


# --- 공감 대답 (기본 cc7) ---------------------------------------------------------------------------

class ReplyLlm:
    def __init__(self, text, has=True):
        self.text, self.has, self.calls = text, has, []

    def has_adapter(self, name):
        return self.has

    def generate_adapter(self, user, *, adapter, system=None, temperature, num_predict, timeout):
        self.calls.append(("adapter", adapter, user, system))
        return self.text

    def generate(self, prompt, *, temperature, num_predict, timeout):
        self.calls.append(("base", None, prompt, None))
        return self.text


Q, A = "오늘 점심은 드셨어요?", "응 국수 먹었어"


def test_followup_reply_uses_reply_adapter_with_same_prompt():
    llm = ReplyLlm("국수 드셨군요. 어떤 국수를 드셨어요?")
    r = generate_reply(llm, Q, A, None, False, 1)
    assert llm.calls == [("adapter", REPLY_ADAPTER, build_reply_prompt(Q, A, None, False), None)]
    assert r.source == "LLM" and r.text.endswith("?")


def test_reply_adapter_output_still_goes_through_rules():
    llm = ReplyLlm("오늘 날씨가 참 좋네요.")  # 물음으로 끝나지 않음 + 대답과 무관
    r = generate_reply(llm, Q, A, None, False, 1)
    assert r.source == "FALLBACK" and "물음으로 끝나지 않음" in r.failures


def test_last_turn_switch_off_and_missing_adapter_use_base_model(settings):
    llm = ReplyLlm("오늘도 이야기 나눠 주셔서 고맙습니다.")
    generate_reply(llm, Q, A, None, True, 8)
    assert llm.calls[0][0] == "base"
    llm = ReplyLlm("국수 드셨군요. 어떤 국수를 드셨어요?", has=False)
    generate_reply(llm, Q, A, None, False, 1)
    assert llm.calls[0][0] == "base"
    settings(companion_reply_lora=False)
    llm = ReplyLlm("국수 드셨군요. 어떤 국수를 드셨어요?")
    generate_reply(llm, Q, A, None, False, 1)
    assert llm.calls[0][0] == "base"


# --- 한 줄 요약 머리말 -----------------------------------------------------------------------

class SummaryLlm:
    def __init__(self, raw):
        self.raw = raw

    def generate(self, *_a, **_k):
        return self.raw


@pytest.mark.parametrize("raw,out", [("요약: 아침에 산책을 다녀옴", "아침에 산책을 다녀옴"), ("기록 ： 점심에 국수", "점심에 국수"),
                                     ('"요약:국수를 먹음"', "국수를 먹음"), ("아침에 요약: 산책", "아침에 요약: 산책"),
                                     ("요약된 내용 없음", "요약된 내용 없음")])
def test_summary_header_is_removed(raw, out):
    assert summarize_answer(SummaryLlm(raw), "대답") == out


def test_summary_header_kept_when_switch_off(settings):
    settings(companion_summary_strip_prefix=False)
    assert summarize_answer(SummaryLlm("요약: 국수"), "대답") == "요약: 국수"


# --- 표정 head3 -----------------------------------------------------------------------------

@pytest.mark.parametrize("probs,j", [([0.1, 0.1, 0.8], 2), ([0.15, 0.06, 0.79], 0), ([0.05, 0.16, 0.79], 1), ([0.0, 0.0, 1.0], 2)])
def test_head3_negative_needs_point_8(probs, j):
    assert head3_decide(np.array(probs), 0.8) == j


def test_head3_probs_and_box():
    head = {"weight": np.eye(3, 4, dtype=np.float32), "bias": np.zeros(3, np.float32),
            "mu": np.zeros(4, np.float32), "sd": np.ones(4, np.float32)}
    p = head3_probs(np.array([0, 0, 5, 0], np.float32), head)
    assert abs(p.sum() - 1) < 1e-6 and int(np.argmax(p)) == 2
    assert expand_box((100, 100, 100, 100), 640, 480) == (85, 60, 215, 220)
    assert expand_box((0, 0, 100, 100), 90, 90) == (0, 0, 90, 90)


def _jpeg(w=320, h=240) -> bytes:
    buf = io.BytesIO()
    Image.new("RGB", (w, h), (120, 110, 100)).save(buf, "JPEG")
    return buf.getvalue()


class _Det:
    def __init__(self, faces):
        self.faces = faces

    def setInputSize(self, _size):
        pass

    def detect(self, _img):
        return 1, self.faces


class _Feat:
    def __init__(self):
        self.shapes = []

    def extract_features(self, img):
        self.shapes.append(img.shape)
        return np.array([[0.0, 0.0, 9.0]], np.float32)


def _head3_service(faces, settings, neg_logit=9.0):
    settings(companion_face_head="head3")
    svc, feat = FaceService(), _Feat()
    head = {"weight": np.eye(3, dtype=np.float32), "bias": np.zeros(3, np.float32),
            "mu": np.zeros(3, np.float32), "sd": np.ones(3, np.float32)}
    svc._h3 = (_Det(faces), feat, head)
    return svc, feat


def test_head3_analyze_picks_largest_face_and_reports_contract_label(settings):
    faces = np.array([[10, 10, 20, 20, 0.9], [50, 40, 120, 120, 0.8]], np.float32)  # 20px 는 짧은 변/8(30) 미만이라 버린다
    svc, feat = _head3_service(faces, settings)
    r = svc.analyze([_jpeg(), _jpeg(), b"not an image"])
    assert (r.label, r.usable, r.facesFound, r.totalFrames) == ("Sadness", True, 2, 3)
    assert feat.shapes == [(224, 224, 3), (224, 224, 3)] and r.label in HEAD3_LABELS


def test_head3_no_face_is_unusable(settings):
    svc, _ = _head3_service(None, settings)
    r = svc.analyze([_jpeg()])
    assert (r.label, r.usable) == (None, False)


def test_head3_files_missing_falls_back_to_hsemotion8(settings, tmp_path):
    settings(model_base_path=str(tmp_path), companion_face_head="head3")
    svc = FaceService()
    assert svc._use_head3() is False
    settings(companion_face_head="hsemotion8")
    svc._h3 = ("set", "but", "off")
    assert svc._use_head3() is False  # 재희 값이면 파일이 있어도 쓰지 않는다


# --- 스위치를 모두 재희 값으로 ---------------------------------------------------------------

class JaeheeLlm:
    def __init__(self):
        self.base_calls, self.adapter_calls, self.v4_calls = 0, 0, 0

    def generate(self, prompt, **_k):
        self.base_calls += 1
        if "요약" in prompt or "한 줄" in prompt:
            return "요약: 국수를 먹음"
        return "국수 드셨군요. 어떤 국수를 드셨어요?"

    def generate_adapter(self, *_a, **_k):
        self.adapter_calls += 1
        return "어댑터"

    def message_token_probs(self, *_a, **_k):
        self.adapter_calls += 1
        return {"있": 1.0, "없": 0.0}

    def next_token_probs(self, *_a, **_k):
        self.v4_calls += 1
        return {"있": 0.6, "없": 0.4}


def test_all_switches_off_is_jaehee(settings):
    settings(**JAEHEE)
    llm = JaeheeLlm()
    r = pipeline.run_turn(transcript="혼자 살다 보니 국수를 먹었어", face={"label": "Happiness", "usable": True},
                          classify_text=lambda _t: TextEmotionResult("슬픔", 0.9, True, {}), llm=llm,
                          question=Q, prev_signals=None, is_last_turn=False, turn_no=1)
    assert llm.adapter_calls == 0 and llm.v4_calls == 1
    assert r["riskFlag"] is True  # 재희: v4 0.6 ≥ 0.5, '혼자 살다'도 낱말 규칙에 걸린다
    assert r["turnResult"] == {"label": UNDECIDED, "source": "CONFLICT"}  # 재희 fuse_turn
