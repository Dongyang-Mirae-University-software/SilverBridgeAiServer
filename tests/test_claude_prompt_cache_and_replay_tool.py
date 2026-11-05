"""라우터 프롬프트 캐싱(2026-11-05)과 오프라인 재생 도구(tools/eval/replay_offline.py).

캐싱: 고정 지시문(system)은 캐시 표시가 붙은 블록 하나, 요청마다 바뀌는 값은 user 쪽에만. 사용량 로그는 숫자만.
도구: API 를 적게 쓰게 - stub·cache 는 키 없이, live 는 키와 호출 상한이 있어야만. 본문·캐시 파일은 저장소 밖에만.
"""
from __future__ import annotations

import importlib.util
import json
import logging
import pathlib
import sys
from types import SimpleNamespace

import pytest

from app.core.config import Settings
from app.services import chat_router_service as crs
from app.services import follow_up_questions as fq
from app.services.claude_structured_client import ClaudeStructuredClient

REPO = pathlib.Path(__file__).resolve().parents[1]
_spec = importlib.util.spec_from_file_location("replay_offline", REPO / "tools/eval/replay_offline.py")
ro = importlib.util.module_from_spec(_spec)
sys.modules["replay_offline"] = ro
_spec.loader.exec_module(ro)


def _client(cache: bool = True) -> ClaudeStructuredClient:
    client = ClaudeStructuredClient.__new__(ClaudeStructuredClient)
    client._settings = SimpleNamespace(claude_max_tokens=Settings().claude_max_tokens, claude_prompt_cache=cache)
    client._state = SimpleNamespace(model_name="claude-opus-5-5")
    return client


# ---- 프롬프트 캐싱 -------------------------------------------------------------------------------------------

def test_system_is_one_cached_block_with_stable_bytes() -> None:
    c = _client()
    first = c._request_kwargs(crs.SYSTEM_PROMPT, "사용자 문장 1", crs.ROUTE_SCHEMA)
    second = c._request_kwargs(crs.SYSTEM_PROMPT, "다른 문장과 이력", crs.ROUTE_SCHEMA)
    assert isinstance(first["system"], list) and len(first["system"]) == 1
    assert first["system"][0]["cache_control"] == {"type": "ephemeral"}
    assert json.dumps(first["system"], ensure_ascii=False) == json.dumps(second["system"], ensure_ascii=False)
    assert first["output_config"] == second["output_config"]  # 스키마·effort 도 요청마다 같다
    assert "사용자 문장 1" not in json.dumps(first["system"], ensure_ascii=False)  # 바뀌는 값은 user 쪽


def test_cache_switch_off_sends_plain_string() -> None:
    assert _client(cache=False)._request_kwargs("지시문", "u", None)["system"] == "지시문"


def test_follow_up_prompt_also_goes_through_cached_block() -> None:
    kwargs = _client()._request_kwargs(fq._SYSTEM_PROMPT, "u", None)
    assert kwargs["system"][0]["cache_control"]["type"] == "ephemeral"


def test_router_system_prompt_is_a_module_constant() -> None:
    # 요청마다 지시문을 새로 만들면 캐시가 조용히 깨진다 - 날짜·시각 같은 값이 섞이지 않았는지
    import importlib
    again = importlib.reload(crs).SYSTEM_PROMPT
    assert again == crs.SYSTEM_PROMPT


def test_usage_log_has_numbers_only(caplog) -> None:
    usage = SimpleNamespace(input_tokens=120, output_tokens=40, cache_read_input_tokens=2000, cache_creation_input_tokens=0)
    with caplog.at_level(logging.INFO):
        ClaudeStructuredClient._log_usage("router", SimpleNamespace(usage=usage))
    line = [r.getMessage() for r in caplog.records if "[CLAUDE-USAGE]" in r.getMessage()][0]
    assert line == "[CLAUDE-USAGE] stage=router in=120 out=40 cache_read=2000 cache_write=0"


def test_usage_log_tolerates_missing_cache_fields(caplog) -> None:
    with caplog.at_level(logging.INFO):
        ClaudeStructuredClient._log_usage("follow_up", SimpleNamespace(usage=SimpleNamespace(input_tokens=1, output_tokens=2)))
    assert any("cache_read=0 cache_write=0" in r.getMessage() for r in caplog.records)


def test_setting_defaults_to_on() -> None:
    assert Settings().claude_prompt_cache is True


# ---- 재생 도구 ---------------------------------------------------------------------------------------------

def test_outputs_must_be_outside_repo(tmp_path) -> None:
    assert ro.outside_repo(tmp_path / "x.json") == (tmp_path / "x.json").resolve()
    with pytest.raises(SystemExit):
        ro.outside_repo(REPO / "docs" / "bodies.json")


def test_cache_key_is_deterministic_and_follows_prompt_changes() -> None:
    k1 = ro.router_cache_key("m", "sys", {"a": 1, "b": 2}, "문장", [], None)
    assert k1 == ro.router_cache_key("m", "sys", {"b": 2, "a": 1}, "문장", [], None)  # 스키마 키 순서 무관
    assert k1 != ro.router_cache_key("m", "sys 2", {"a": 1, "b": 2}, "문장", [], None)  # 지시문이 바뀌면 다른 키
    assert k1 != ro.router_cache_key("m", "sys", {"a": 1, "b": 2}, "문장", [{"role": "user", "content": "x"}], None)


def test_call_budget_stops_at_limit() -> None:
    b = ro.CallBudget(2)
    b.take()
    b.take()
    with pytest.raises(ro.BudgetExceeded):
        b.take()


def test_stub_route_maps_expectations() -> None:
    r = ro.stub_route({"expect": {"route": "symptom", "cpr_aed": True}}, "문장")
    assert r["emergency"] is True and r["cardiac_arrest"] is True and r["engine"] == "stub"
    r2 = ro.stub_route({"expect": {"crisis": True}}, "문장")
    assert r2["self_harm"] is True


def test_planned_calls_count_replayed_turns() -> None:
    items = [{"id": "a", "q": "x", "history_user": ["1", "2"]}, {"id": "b", "q": "y", "history_user": []}]
    assert ro.planned_router_calls(items, 2) == (3 + 1) * 2


def test_live_without_key_is_refused(monkeypatch, tmp_path) -> None:
    monkeypatch.delenv("ANTHROPIC_API_KEY", raising=False)
    s = tmp_path / "set.json"
    s.write_text(json.dumps({"items": [{"id": "a", "q": "x", "expect": {}}]}), encoding="utf-8")
    with pytest.raises(SystemExit):
        ro.main(["run", "--set", str(s), "--router", "live", "--cache-file", str(tmp_path / "c.json"), "--out", str(tmp_path / "o.json")])


def test_live_over_budget_does_not_start(monkeypatch, tmp_path) -> None:
    monkeypatch.setenv("ANTHROPIC_API_KEY", "not-a-real-key")
    s = tmp_path / "set.json"
    s.write_text(json.dumps({"items": [{"id": f"i{n}", "q": "x", "expect": {}} for n in range(5)]}), encoding="utf-8")
    code = ro.main(["run", "--set", str(s), "--router", "live", "--max-router-calls", "3",
                    "--cache-file", str(tmp_path / "c.json"), "--out", str(tmp_path / "o.json")])
    assert code == 2 and not (tmp_path / "o.json").exists()


def test_router_variance_over_budget_does_not_start(monkeypatch, tmp_path) -> None:
    monkeypatch.setenv("ANTHROPIC_API_KEY", "not-a-real-key")
    s = tmp_path / "set.json"
    s.write_text(json.dumps({"items": [{"id": "a", "q": "x"}, {"id": "b", "q": "y"}]}), encoding="utf-8")
    assert ro.main(["router-variance", "--set", str(s), "--ids", "a,b", "--times", "5", "--max-router-calls", "9"]) == 2


class _FakeReal:
    def __init__(self) -> None:
        self._loader = SimpleNamespace(state=SimpleNamespace(model_name="claude-opus-5-5"))
        self.calls = 0

    def route(self, message, history, ctx, prev_reservation=None):
        self.calls += 1
        return {"route": "symptom", "emergency": False, "engine": "claude"}


def test_cached_router_hit_miss_and_live_fill() -> None:
    real, stats, cache = _FakeReal(), {"hit": 0, "miss": 0, "live": 0}, {}
    live = ro._CachedRouter(real, cache, "live", ro.CallBudget(5), stats)
    live.route("문장", [], None)  # 실제 호출 1회 + 기록
    live.route("문장", [], None)  # 기록 재사용
    assert real.calls == 1 and stats == {"hit": 1, "miss": 0, "live": 1}
    cached = ro._CachedRouter(real, cache, "cache", ro.CallBudget(0), stats)
    assert cached.route("문장", [], None)["engine"] == "claude"
    with pytest.raises(ro.RouterCacheMiss):
        cached.route("기록에 없는 문장", [], None)
    assert real.calls == 1  # cache 모드는 절대 부르지 않는다


# ---- 2026-11-05 점검 H-1·M-1·L-1 ----------------------------------------------------------------------------

@pytest.fixture()
def restore_claude_method():
    from app.services import claude_structured_client as csc

    original = csc.ClaudeStructuredClient.generate_structured_text
    yield csc
    csc.ClaudeStructuredClient.generate_structured_text = original


def test_stub_and_cache_block_every_claude_call(restore_claude_method) -> None:
    csc = restore_claude_method
    stats = {"blocked": 0, "claude": 0}
    ro.guard_claude_calls("stub", ro.CallBudget(100), stats)
    client = csc.ClaudeStructuredClient.__new__(csc.ClaudeStructuredClient)
    for stage in ("router", "reservation_extract", "follow_up"):
        with pytest.raises(csc.StructuredCallError):
            client.generate_structured_text("s", "u", None, stage=stage)
    assert stats == {"blocked": 3, "claude": 0}


def test_live_budget_counts_every_claude_stage(restore_claude_method, monkeypatch) -> None:
    csc = restore_claude_method
    monkeypatch.setattr(csc.ClaudeStructuredClient, "generate_structured_text", lambda self, *a, **k: "{}")
    stats = {"blocked": 0, "claude": 0}
    budget = ro.CallBudget(2)
    ro.guard_claude_calls("live", budget, stats)
    client = csc.ClaudeStructuredClient.__new__(csc.ClaudeStructuredClient)
    client.generate_structured_text("s", "u", None, stage="router")
    client.generate_structured_text("s", "u", None, stage="reservation_extract")  # 예약 추출도 상한에 든다
    with pytest.raises(ro.BudgetExceeded):
        client.generate_structured_text("s", "u", None, stage="follow_up")
    assert stats["claude"] == 2 and budget.used == 2


def test_planned_claude_calls_include_reservation_and_follow_up() -> None:
    items = [{"id": "a", "q": "x", "expect": {"route": "reservation"}, "history_user": ["1"]},
             {"id": "b", "q": "y", "expect": {"route": "symptom"}, "history_user": []}]
    assert ro.planned_claude_calls(items, 1, follow_up=False) == 2 * 2 + 1  # 예약 문항은 턴마다 라우터 + 추출
    assert ro.planned_claude_calls(items, 1, follow_up=True) == 2 * 3 + 2


def test_summary_must_be_outside_repo(tmp_path) -> None:
    s = tmp_path / "set.json"
    s.write_text(json.dumps({"items": [{"id": "a", "q": "x", "expect": {}}]}), encoding="utf-8")
    with pytest.raises(SystemExit):
        ro.main(["run", "--set", str(s), "--out", str(tmp_path / "o.json"), "--summary", str(REPO / "docs" / "s.json")])


def test_dotenv_cannot_refill_key_in_stub_mode(tmp_path, monkeypatch) -> None:
    """.env 가 있어도 stub·cache 모드는 키가 빈 값으로 남는다(load_dotenv 는 이미 있는 값을 덮지 않는다)."""
    from dotenv import load_dotenv

    env = tmp_path / ".env"
    env.write_text("ANTHROPIC_API_KEY=fake-for-test\nCLAUDE_API_KEY=fake-too\n", encoding="utf-8")
    import argparse
    import os

    # _prepare_env 가 바꾸는 값을 먼저 등록해 두면 테스트가 끝날 때 원래대로 돌아간다(다른 테스트로 새지 않게).
    for var in ("ANTHROPIC_API_KEY", "CLAUDE_API_KEY", "DATABASE_URL", "FOLLOW_UP_ENABLED", "STREAM_STATE_BACKEND", "HF_HUB_OFFLINE"):
        monkeypatch.setenv(var, os.environ.get(var, "x"))
    monkeypatch.delenv("ANTHROPIC_API_KEY")
    monkeypatch.delenv("CLAUDE_API_KEY")
    ro._prepare_env(argparse.Namespace(follow_up=False, router="stub"))
    load_dotenv(env)  # config.py 가 하는 일과 같다
    assert os.environ["ANTHROPIC_API_KEY"] == "" and os.environ["CLAUDE_API_KEY"] == ""
