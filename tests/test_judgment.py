from __future__ import annotations

from app.services.companion.judgment import DayRecord, baseline_ready, judge, negative_ratio


def _days(n: int, labels: list[str | None], risk: int = 0) -> list[DayRecord]:
    return [DayRecord(f"2026-10-{i + 1:02d}", list(labels), risk) for i in range(n)]


def test_baseline_needs_20_days_of_data():
    # S10-2 (사용자 결정): 30일 중 20일 이상 기록이 있어야 판정을 시작한다
    assert baseline_ready(_days(19, ["기쁨"])) is False
    assert baseline_ready(_days(20, ["기쁨"])) is True
    assert baseline_ready(_days(25, [None])) is False  # 무응답만 있는 날은 기록 있는 날로 세지 않는다


def test_insufficient_baseline_is_not_judged():
    result = judge(_days(5, ["기쁨"]), _days(7, ["슬픔"]))
    assert result["state"] == "BASELINE_INSUFFICIENT"
    assert result["alertLevel"] is None


def test_negative_share_doubling_gives_notice():
    # S10-3 (사용자 결정): 부정 신호 비율이 기준선의 2배 이상이면 변화
    baseline = _days(20, ["기쁨", "기쁨", "기쁨", "슬픔"])  # 부정 25%
    recent = _days(7, ["슬픔", "슬픔", "기쁨", "화남"])  # 부정 75%
    result = judge(baseline, recent)
    assert result["state"] == "NOTICE"
    assert result["alertLevel"] == "NOTICE"
    assert result["risk"] is False


def test_same_share_is_normal_and_info_only():
    baseline = _days(20, ["기쁨", "슬픔"])
    recent = _days(7, ["기쁨", "슬픔"])
    result = judge(baseline, recent)
    assert result["state"] == "NORMAL"
    assert result["alertLevel"] == "INFO"  # 알림은 보내지 않고 기록만 남긴다


def test_small_gap_from_zero_baseline_is_not_a_change():
    # 기준선이 0%여도 10%(0.1)는 최소 차이 0.2에 못 미치므로 변화가 아니다
    baseline = _days(20, ["기쁨"])
    recent = _days(7, ["기쁨"] * 9 + ["슬픔"])
    assert judge(baseline, recent)["state"] == "NORMAL"


def test_risk_flag_is_urgent_regardless_of_baseline():
    recent = _days(7, ["기쁨"])
    recent[3].riskCount = 1
    result = judge(_days(20, ["기쁨"]), recent)
    assert result["state"] == "URGENT"
    assert result["alertLevel"] == "URGENT"
    assert result["risk"] is True


def test_negative_ratio_ignores_no_response_turns():
    assert negative_ratio([DayRecord("d", [None, None])]) is None
    assert negative_ratio([DayRecord("d", [None, "슬픔", "기쁨", None])]) == 0.5
