"""고도화 평가 세트(tests/companion_eval/companion_quality_v1.json - tests/eval 은 챗봇 형식 *_v*.json 을 다른 시험이 모두 읽으므로 따로 둔다)가 품질 기준대로 만들어졌는지 확인한다 (2026-11-15, 6-2).

기준: docs/companion-quality-criteria.md. 세트가 틀리면 그 뒤 모든 측정(6-4·6-8)이 틀리므로 모양·라벨·겹침을 고정한다.
"""
from __future__ import annotations

import json
import re
from collections import Counter
from pathlib import Path

import pytest

from app.routers.companion_router import ReplyText
from app.services.companion import mood_service

ROOT = Path(__file__).resolve().parents[1]
SET_PATH = ROOT / "tests" / "companion_eval" / "companion_quality_v1.json"
CRITERIA = ROOT / "docs" / "companion-quality-criteria.md"
MOOD_PROMPT = ROOT / "app" / "services" / "companion" / "prompts" / "mood-e900.txt"

TYPES = {"longing", "resignation", "hurt", "worry", "pain", "risk", "avoid", "short", "joy", "calm", "others", "future"}
POSITIVE, NEGATIVE = {"JOY", "CALM"}, {"WORRY", "DEPRESSED"}
# 백엔드 질문 목록 (SilverBridgeBe src/main/resources/db/migration/V65__add_companion.sql, 11/15 확인)
BACKEND_QUESTIONS = {
    "09:00": {"지난밤에는 잠을 어떻게 주무셨어요?", "아침에 일어나셨을 때 몸은 좀 어떠셨어요?",
              "아침에 편안하게 일어나셨나요? 어떠셨는지 들려주세요.", "오늘 하루는 어떻게 시작하셨어요?"},
    "12:00": {"점심은 무엇을 드셨어요?", "요즘 입맛은 어떠세요?", "점심은 어디서, 누구와 드셨어요?",
              "점심은 맛있게 드셨나요? 어떤 음식이었어요?"},
    "15:00": {"오늘 오후에는 무엇을 하며 지내셨어요?", "요즘 즐겨 하시는 일이 있으세요? 말씀해 주세요.",
              "오늘 바깥 바람은 쐬셨어요? 어디에 다녀오셨는지 들려주세요.", "오늘 기분은 어떠세요? 그런 이유가 있을까요?"},
    "18:00": {"저녁은 어떻게 챙겨 드셨어요?", "오늘 누구와 이야기를 나누셨어요?", "요즘 가족이나 이웃 소식은 어떠세요?"},
    "21:00": {"오늘 하루는 어떠셨어요? 기억에 남는 일이 있으세요?", "내일은 어떤 일을 하실 생각이세요?",
              "요즘 마음은 어떠신지 편하게 말씀해 주세요.", "오늘 하루 중 가장 좋았던 때는 언제였어요?"},
}


@pytest.fixture(scope="module")
def data():
    return json.loads(SET_PATH.read_text(encoding="utf-8"))


def _turns(data):
    return [(s, t) for s in data["scenarios"] for t in s["turns"]]


def test_header_and_ids(data):
    assert data["version"] == "companion-quality-v1"
    ids = [s["id"] for s in data["scenarios"]]
    assert len(ids) == len(set(ids))


def test_every_type_has_at_least_three_scenarios(data):
    counts = Counter(s["type"] for s in data["scenarios"])
    assert set(counts) == TYPES
    assert min(counts.values()) >= 3


def test_opening_is_a_real_backend_question_for_that_slot(data):
    for s in data["scenarios"]:
        assert s["slot"] in BACKEND_QUESTIONS, s["id"]
        assert s["opening"] in BACKEND_QUESTIONS[s["slot"]], s["id"]
    assert {s["slot"] for s in data["scenarios"]} == set(BACKEND_QUESTIONS)  # 시간대 5개 모두 (결정 1: 시간대 공감)


def test_turn_counts_and_final_greeting_scenarios(data, monkeypatch):
    for s in data["scenarios"]:
        assert 3 <= len(s["turns"]) <= 8, s["id"]
    eight = [s["id"] for s in data["scenarios"] if len(s["turns"]) == 8]
    assert len(eight) >= 4  # maxTurns 8 - 마지막 인사를 재는 시나리오
    from app.core.config import get_settings
    assert get_settings().companion_max_turns == 8


def test_turn_fields_and_mood_labels(data):
    codes = set(mood_service.MOOD_CODES) | {mood_service.UNKNOWN}
    for s, t in _turns(data):
        where = f"{s['id']}: {t['elder']}"
        assert set(t) == {"elder", "mood", "risk", "follow", "review"}, where
        ReplyText(answer=t["elder"])  # 실제 글 대답 API 검사를 통과하는 말
        assert 1 <= len(t["mood"]) <= 2 and set(t["mood"]) <= codes, where
        if len(t["mood"]) == 2:  # 섞인 감정 = 긍정 하나 + 부정 하나 (결정 3)
            assert len(set(t["mood"]) & POSITIVE) == 1 and len(set(t["mood"]) & NEGATIVE) == 1, where
        assert isinstance(t["risk"], bool) and isinstance(t["review"], bool), where
        assert all(isinstance(f, str) and f for f in t["follow"]), where


def test_mood_codes_match_the_running_mood_prompt():
    prompt = MOOD_PROMPT.read_text(encoding="utf-8")
    assert set(re.findall(r"^- ([A-Z]+):", prompt, re.M)) == set(mood_service.MOOD_CODES)


def test_set_covers_mixed_moods_risks_and_risk_idioms(data):
    turns = [t for _, t in _turns(data)]
    assert sum(len(t["mood"]) == 2 for t in turns) >= 5
    for s in data["scenarios"]:
        if s["type"] == "risk":
            assert s["turns"][0]["risk"] is True, s["id"]
    risky = [t["elder"] for t in turns if t["risk"]]
    assert len(risky) >= 4
    # 관용 표현·헷갈리는 말은 위험이 아니다 (헛경보 측정)
    for phrase in ("배불러 죽겠어", "웃겨 죽는 줄", "자다가 깨지만 않으면"):
        hits = [t for t in turns if phrase in t["elder"]]
        assert hits and all(t["risk"] is False for t in hits), phrase


def test_no_duplicate_elder_lines(data):
    lines = [t["elder"] for _, t in _turns(data)]
    assert len(lines) == len(set(lines))


def test_no_overlap_with_other_eval_sets(data):
    # 다른 평가·시험 세트 문장과 겹치면 학습·기준 고르기에 쓰였을 수 있다 (평가 점수 부풀림 방지)
    others = set()
    for p in (ROOT / "tests" / "eval").glob("*.json"):
        if p.name == SET_PATH.name:
            continue

        def walk(x):
            if isinstance(x, str):
                others.add(x.strip())
            elif isinstance(x, dict):
                for v in x.values():
                    walk(v)
            elif isinstance(x, list):
                for v in x:
                    walk(v)
        walk(json.loads(p.read_text(encoding="utf-8")))
    overlap = [t["elder"] for _, t in _turns(data) if t["elder"] in others]
    assert overlap == []


def test_criteria_document_has_every_rule_the_set_uses():
    doc = CRITERIA.read_text(encoding="utf-8")
    for needle in ("R1", "R10", "결정 1", "결정 2", "결정 3", "결정 4", "몸이 불편하다는 사실만", "감정 없이 말을 피함"):
        assert needle in doc, needle

def test_relief_is_joy_like_e1300(data):
    # 11/16 사용자 결정: 기분 어댑터 e1300(노년 '안도' 문장을 기쁨으로 학습)에 맞춰 안도 = JOY (기준 v1.3)
    relief = {t["elder"]: t["mood"] for _, t in _turns(data)
              if t["elder"] in ("딸이 같이 가 준다니 마음은 놓이는데 그래도 떨려", "조금 쉬니까 괜찮아진 것 같기도 하고")}
    assert relief == {"딸이 같이 가 준다니 마음은 놓이는데 그래도 떨려": ["JOY", "WORRY"], "조금 쉬니까 괜찮아진 것 같기도 하고": ["JOY"]}
    assert "안도" in CRITERIA.read_text(encoding="utf-8")
