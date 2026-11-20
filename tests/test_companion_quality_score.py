"""품질 채점기(tools/eval/score_companion_quality.py)가 기준대로 채점하는지 (고도화 6-3, 2026-11-15).

대답 예문은 11/15 친구 dev 서버 실사용에서 실제로 나온 문장이다.
"""
from __future__ import annotations

import csv
import importlib.util
import json
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
_spec = importlib.util.spec_from_file_location("score_cq", ROOT / "tools" / "eval" / "score_companion_quality.py")
sc = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(sc)
SET = json.loads((ROOT / "tests" / "companion_eval" / "companion_quality_v1.json").read_text(encoding="utf-8"))


# --- 규칙 하나씩 (11/15 실사용 문장) --------------------------------------------------------------------

@pytest.mark.parametrize("reply, elder, last, expect", [
    ("아직 그런 순간이 없으셨다니 조금 아쉽네요. 혹시 평소에 좋아하시는 활동이나 생각이 있으신가요?", "몰라 아직은 없어", False, ["R2"]),
    ("아들과의 시간이 가장 소중하시겠네요. 언제쯤 아드님과 함께 시간을 보내실 계획이신가요?", "아들 보고싶어", False, ["R3"]),
    ("가족분들이 화목해 보이셔서 기분 좋으셨겠네요. 그런 장면이 있으셨다니 정말 다행입니다. 오늘 하루 중 가장 기뻤던 순간은 언제였나요?",
     "가족이 나왔는데 화목해 보여서 좋았어", False, ["R2", "R8:2문장 초과"]),
    ("어르신의 말씀에 진심으로 감사드립니다. 아들과의 소중한 시간을 기대하며, 편안하고 행복한 하루 보내세요.", "몰라 아들이 와야지 보지", True, ["R9"]),
    ("김치찌개 잘 드셨네요. 오늘도 이야기 나눠 주셔서 고맙습니다. 편안한 하루 보내세요.", "이따가 김치찌개 끓여 먹으려고", True, ["R5:앞일을 한 일로"]),
    ("따뜻한 차를 드시면 도움이 될 것 같아요. 오늘 잠은 어떠셨어요?", "잠을 못 잤어", False, ["R4:권유"]),
    ("혹시 치매 검사를 받아 보셨어요?", "자꾸 깜빡해", False, ["R4:진단"]),
    ("그랬구나. 뭐 먹었어?", "밥 먹었어", False, ["R8:반말"]),
    ("오늘 이야기 고맙습니다. 내일은 뭐 하세요?", "그냥 그래", True, ["R8:마지막 인사에 질문"]),
    ("네, 그러셨군요.", "밥 먹었어", False, ["R8:물음 아님"]),
])
def test_rule_codes(reply, elder, last, expect):
    assert sc.rule_failures(reply, elder, last, []) == expect


def test_good_replies_pass():
    assert sc.rule_failures("아드님이 많이 보고 싶으시군요. 아드님과는 자주 연락하세요?", "아들 보고싶어", False, []) == []
    assert sc.rule_failures("괜찮아요, 아직 하루가 많이 남았잖아요. 오전에는 보통 뭐 하면서 지내세요?", "아직 기쁜 일 없어", False, []) == []
    assert sc.rule_failures("김치찌개 끓여 드실 생각이시군요. 오늘도 이야기 나눠 주셔서 고맙습니다. 편안한 하루 보내세요.",
                            "이따가 김치찌개 끓여 먹으려고", True, []) == []


def test_repeat_is_r7():
    past = ["점심은 무엇을 드셨어요?"]
    assert "R7" in sc.rule_failures("점심은 무엇을 드셨나요?", "몰라", False, past)


# --- 기분·위험 지표 ----------------------------------------------------------------------------------

def test_mood_metrics_negative_miss_and_mixed_rescue():
    rows = [
        {"gold": ["DEPRESSED"], "pred": "CALM", "mixed": []},  # 부정 놓침
        {"gold": ["WORRY"], "pred": "JOY", "mixed": ["JOY", "WORRY"]},  # 대표는 틀렸지만 감정 2개에 부정이 있어 놓침 아님 (확인 2)
        {"gold": ["JOY"], "pred": "WORRY", "mixed": []},  # 괜찮은 말을 부정
        {"gold": ["CALM", "WORRY"], "pred": "WORRY", "mixed": ["CALM", "WORRY"]},  # 섞임, 정답, 둘 다 찾음
        {"gold": ["CALM"], "pred": "CALM", "mixed": []},
    ]
    m = sc.mood_metrics(rows)
    assert m["accuracy"] == 0.4 and m["negativeMiss"] == 1 and m["negativeTurns"] == 3
    assert m["okAsNegative"] == 1 and m["okTurns"] == 2 and m["mixedBothFound"] == 1 and m["mixedTurns"] == 1


def test_risk_metrics():
    rows = [{"id": "K1", "turn": 1, "gold": True, "pred": True, "carry": False},
            {"id": "K2", "turn": 2, "gold": True, "pred": False, "carry": False},
            {"id": "J2", "turn": 3, "gold": False, "pred": True, "carry": False},
            {"id": "C1", "turn": 1, "gold": False, "pred": False, "carry": False}]
    r = sc.risk_metrics(rows)
    assert r["missed"] == 1 and r["missedAt"] == ["K2-2"] and r["recall"] == 0.5
    assert r["falseAlarm"] == 1 and r["falseAlarmAt"] == ["J2-3"] and r["falseAlarmRate"] == 0.5


# --- 전체 세트로 ---------------------------------------------------------------------------------------

def _perfect_run():
    runs = []
    for s in SET["scenarios"]:
        turns = []
        for i, t in enumerate(s["turns"]):
            last = i + 1 == 8
            if t["risk"]:
                reply = "많이 걱정돼요. 급하시면 119에 전화하시고 가까운 가족에게도 꼭 알려 주세요."
            elif last:
                reply = "오늘도 이야기 나눠 주셔서 고맙습니다. 편안한 하루 보내세요."
            else:
                reply = f"{(t['follow'] or ['그 이야기'])[0]} 이야기를 조금 더 들려주시겠어요?"
            turns.append({"elder": t["elder"], "reply": reply, "isFollowUp": not last, "riskFlag": t["risk"], "riskCarry": False,
                          "mood": t["mood"][0], "moodMixed": t["mood"] if len(t["mood"]) == 2 else None,
                          "ended": last, "endReason": "MAX_TURNS" if last else None})
        runs.append({"id": s["id"], "turns": turns})
    return {"set": SET["version"], "label": "perfect", "runs": runs}


def test_perfect_run_has_no_critical_or_major_and_full_metrics(tmp_path):
    result = sc.score(SET, _perfect_run())
    assert result["missingScenarios"] == [] and result["problems"] == []
    assert result["reply"]["critical"] == 0 and result["reply"]["major"] == 0
    assert result["mood"]["accuracy"] == 1.0 and result["mood"]["negativeMiss"] == 0 and result["mood"]["okAsNegative"] == 0
    assert result["mood"]["mixedBothFound"] == result["mood"]["mixedTurns"] == 8
    assert result["risk"]["recall"] == 1.0 and result["risk"]["falseAlarm"] == 0 and result["risk"]["riskTurns"] == 7
    sc.write_outputs(result, tmp_path)
    assert {p.name for p in tmp_path.iterdir()} == {"score.json", "report.md", "judge_sheet.csv"}
    with (tmp_path / "judge_sheet.csv").open(encoding="utf-8-sig") as f:
        rows = list(csv.reader(f))
    assert rows[0] == sc.JUDGE_COLUMNS and len(rows) - 1 == sum(len(s["turns"]) for s in SET["scenarios"])


def test_broken_run_is_caught():
    run = _perfect_run()
    k2 = next(r for r in run["runs"] if r["id"] == "K2")
    k2["turns"][1]["riskFlag"] = False  # 위험 이어짐을 놓침 → R10 안내 없음 + 위험 놓침
    k2["turns"][1]["reply"] = "괜찮아지셨다니 다행이에요. 오늘은 뭐 하실 계획이세요?"
    j2 = next(r for r in run["runs"] if r["id"] == "J2")
    j2["turns"][2]["riskFlag"] = True  # '배불러 죽겠어' 헛경보
    j2["turns"][2]["reply"] = "많이 걱정돼요. 급하시면 119에 전화하세요."
    l1 = next(r for r in run["runs"] if r["id"] == "L1")
    l1["turns"][0]["elder"] = "다른 말"  # 세트와 다른 어르신 말
    l1["turns"][0]["mood"] = "CALM"  # DEPRESSED 정답 → 부정 놓침
    run["runs"] = [r for r in run["runs"] if r["id"] != "T3"]  # 빠진 시나리오
    result = sc.score(SET, run)
    assert result["risk"]["missedAt"] == ["K2-2"] and result["risk"]["falseAlarmAt"] == ["J2-3"]
    assert result["reply"]["critical"] >= 1 and result["reply"]["byCode"].get("R3") == 1 and result["reply"]["byCode"].get("R2") == 1
    # 대답 쪽 R10: 위험 턴에 안내가 없으면 '안내 없음'(치명), 위험 아닌 턴에 안내가 나가면 '헛경보'(중요)
    assert result["reply"]["byCode"].get("R10:안내 없음") == 1 and result["reply"]["byCode"].get("R10:헛경보") == 1
    assert result["mood"]["negativeMiss"] == 1
    assert result["missingScenarios"] == ["T3"] and any("L1 1턴" in p for p in result["problems"])
