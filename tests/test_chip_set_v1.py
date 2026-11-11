"""칩 문장 시험지(tests/eval/chip_set_v1.json, 2026-11-11) - 칩을 그 칩이 나오는 앞 턴 다음에 누른 것으로 stub 재생.

피보호자 대화는 대부분 칩(되묻기 답 버튼·FE 빠른 질문·이어서 물어볼 질문)으로 진행되는데 시험지는 사람이 쓴 자유 문장뿐이었다
(11/11 "신경과는 어떤 검사…"·"설사할 때 무엇을 먹으면…"이 모두 칩 문장). 여기서는 경로·생성에 넘긴 문장·응급·자해 오판만 본다 -
답 내용(must)은 실제 MedGemma 재생(DGX replay_offline)에서 본다. CHIP_OUT 을 주면 문항별 판정 칸만 JSON 줄로 남긴다(본문 없음).
"""
from __future__ import annotations

import json
import os
import pathlib
from typing import Any

import pytest

from tests.test_followup_steps import _send, env, flow  # noqa: F401

_SET = json.loads((pathlib.Path(__file__).parent / "eval" / "chip_set_v1.json").read_text(encoding="utf-8"))
ITEMS = _SET["items"]


def _path(out: dict[str, Any], called: bool, dept_info: bool) -> str:
    """응답 칸만으로 경로를 가른다(API 응답에는 meta 가 실리지 않는다)."""
    reply = out.get("reply") or ""
    if out.get("riskLevel") == "high":
        return "119"
    if dept_info:
        return "dept_info"
    if out.get("intent") == "hospital_reservation" or out.get("reservationRequired") and not called and out.get("ui"):
        return "reservation"
    if out.get("suggestedKind") == "answer" or "여쭤볼게요" in reply:
        return "question"
    if called:
        return "generate"
    if "오늘 바로" in reply:
        return "visit"
    return "rule"


def _play(env, item: dict[str, Any], monkeypatch) -> tuple[dict[str, Any], str, str]:
    from app.services.medical_llm_service import MedicalLlmService as M
    marks: list[str] = []
    original = M.department_info_response

    def tagged(kind, department, proxy):
        marks.append(kind)
        return original(kind, department, proxy)

    monkeypatch.setattr(M, "department_info_response", staticmethod(tagged))
    env.router.route_value = item["expect"].get("route") or "symptom"
    ward = item.get("wardId")
    for text in item["history_user"]:
        _send(env, text, session=item["id"], ward=ward, role=item["role"])
    marks.clear()
    before = len(env.llm.calls)
    out = _send(env, item["q"], session=item["id"], ward=ward, role=item["role"])
    called = len(env.llm.calls) > before
    return out, _path(out, called, bool(marks)), (env.llm.calls[-1]["message"] if called else "")


def test_chip_set_shape() -> None:
    ids = [it["id"] for it in ITEMS]
    assert len(ids) == len(set(ids)) and len(ITEMS) >= 150
    assert {it["path"] for it in ITEMS} <= {"question", "119", "visit", "generate", "dept_info", "reservation"}


@pytest.mark.parametrize("item", ITEMS, ids=[it["id"] for it in ITEMS])
def test_chip_is_handled_on_expected_path(flow, item: dict[str, Any], monkeypatch) -> None:
    out, path, message = _play(flow, item, monkeypatch)
    if os.environ.get("CHIP_OUT"):
        from app.services.chat_keywords import symptom_department
        with open(os.environ["CHIP_OUT"], "a", encoding="utf-8") as f:
            f.write(json.dumps({"id": item["id"], "cat": item["cat"], "role": item["role"], "expect": item["path"], "path": path,
                                "risk": out.get("riskLevel"), "has109": "109" in (out.get("reply") or ""),
                                "genHasOk": all(w in message for w in item.get("gen_has") or []),
                                "dept": symptom_department(message) if message else None}, ensure_ascii=False) + "\n")
    assert path == item["path"]
    assert "109" not in (out.get("reply") or "")  # 칩 문장은 자해·위기 답으로 가지 않는다
    if item["path"] != "119":
        assert out.get("riskLevel") != "high"
    for word in item.get("gen_has") or []:
        assert word in message
