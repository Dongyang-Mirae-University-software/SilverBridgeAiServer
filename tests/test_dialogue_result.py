"""대화 결과 1건의 형식 (노션 요청사항 1). 서버를 띄우지 않고 결과 함수만 시험한다."""

from app.services.companion.dialogue_store import Dialogue, TurnRecord
from app.services.companion.turn_record import build_dialogue_result


def _dialogue(labels: list[str | None], risk: bool = False) -> Dialogue:
    d = Dialogue("dlg-1", "ward-1", "sess-1", "09:00", "아침 식사는 하셨어요?", 8, questionId="q-7")
    for i, label in enumerate(labels, start=1):
        d.turns.append(TurnRecord(turnNo=i, label=label, source="TEXT" if label else "NO_RESPONSE"))
    d.ended = True
    d.endReason = "최대 턴 수"
    d.endedAt = "2026-11-02T09:12:00+00:00"
    d.riskSeen = risk
    return d


def test_result_has_request_fields_and_no_transcript():
    result = build_dialogue_result(_dialogue(["기쁨", "평온"]))
    assert result["dialogueId"] == "dlg-1"
    assert result["wardId"] == "ward-1"
    assert result["questionId"] == "q-7"
    assert result["slot"] == "09:00"
    assert result["endReason"] == "최대 턴 수"
    assert result["turnCount"] == 2
    assert result["summary"] is None
    assert "transcript" not in result


def test_answered_count_excludes_no_response_turns():
    result = build_dialogue_result(_dialogue(["기쁨", None, "판단 보류", None]))
    assert result["turnCount"] == 4
    assert result["noResponseCount"] == 2
    assert result["answeredCount"] == 2  # 판단 보류는 답한 것으로 센다


def test_mood_groups_sad_and_angry_as_worry():
    assert build_dialogue_result(_dialogue(["슬픔", "슬픔", "화남"]))["mood"] == "WORRY"


def test_mood_is_unknown_when_nothing_decided():
    result = build_dialogue_result(_dialogue(["판단 보류", None]))
    assert result["mood"] == "UNKNOWN"  # 백엔드 결정 D: null 이 아니라 UNKNOWN
    assert result["answeredCount"] == 1


def test_alert_is_urgent_only_when_risk_seen():
    assert build_dialogue_result(_dialogue(["우울"], risk=True))["alertLevel"] == "URGENT"
    assert build_dialogue_result(_dialogue(["우울"], risk=False))["alertLevel"] == "INFO"
