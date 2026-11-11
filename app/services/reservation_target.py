"""보호자 챗봇 예약 대상 고르기(2026-11-11, 사용자 결정).

보호자 챗봇은 평소 보호자 본인 상담으로 시작한다(피보호자 선택 칸 없음). 예약은 피보호자만 대상이라(보호자 웹 예약 화면과 같다)
wardId 없이 예약 흐름이 시작되면 예약을 진행하지 않고 먼저 "누구의 병원 예약인가요?"를 묻는다.

계약(BE·AI·FE 공통):
- BE 는 보호자 경로 context 에 guardianWards = [{wardId, name, relation}](ACTIVE 연결만)를 넣는다. 칸이 없으면(옛 BE) 묻지 않는다.
- 선택지 값은 각 wardId(보호자 본인 "나"는 없다). FE 는 고른 값을 uiSelection 과 함께 그 턴부터 context.wardId 에 싣는다(BE 가 검증).
- 이 턴의 wardId 가 고른 값과 같을 때만 그 피보호자로 예약을 시작한다 - AI 는 wardId 를 스스로 정하지 않는다.

이름·관계는 선택지 표시 글자로만 쓰고 로그에는 개수만 남긴다.
"""

from __future__ import annotations

import json
import time
from typing import Any

from app.schemas.chat_schema import ChatRequest, GuardianWard, UiOption, UiPayload

# BE(#472 ChatRelayService.PATIENT_CHOICE_FIELD)가 이 칸의 선택 값을 ACTIVE 연결로 검증한다 - 이름을 바꾸면 BE·FE 도 함께.
TARGET_FIELD = "reservation_patient"
# 대상 질문은 확인 카드와 같은 30분 안에 골라야 한다(오래된 질문으로 예약이 시작되지 않게).
TARGET_TTL_SECONDS = 1800
MAX_OPTIONS = 20
_NAME_MAX = 20
_RELATION_MAX = 10
QUESTION = "누구의 병원 예약인가요?"
_PICK_HINT = "아래에서 예약할 분을 골라 주세요."
NO_WARD_REPLY = "챗봇 병원 예약은 연결된 피보호자의 예약만 할 수 있어요. 피보호자와 연결한 뒤 다시 말씀해 주세요."
STALE_REPLY = "예약할 분을 고르는 시간이 지났어요. 예약하실 내용을 다시 말씀해 주세요."


def guardian_wards(payload: ChatRequest) -> list[GuardianWard] | None:
    """보호자 경로에서 BE 가 넣은 연결 피보호자 목록. 보호자 경로가 아니거나 칸이 없으면 None(대상을 묻지 않는다).
    role 은 BE 가 경로로 덮어쓴 값이라 피보호자 경로에 끼워 넣은 목록은 쓰지 않는다. 같은 wardId 는 한 번만, 최대 20명."""
    ctx = payload.context
    if ctx is None or ctx.guardianWards is None or str(ctx.role or "").strip().upper() != "GUARDIAN":
        return None
    seen: set[str] = set()
    wards: list[GuardianWard] = []
    for ward in ctx.guardianWards:
        if ward.wardId in seen:
            continue
        seen.add(ward.wardId)
        wards.append(ward)
        if len(wards) >= MAX_OPTIONS:
            break
    return wards


def option_label(ward: GuardianWard) -> str:
    """선택지 표시 글자 - "이름(관계)", 관계가 없으면 이름만, 이름이 없으면 "피보호자"."""
    name = " ".join((ward.name or "").split())[:_NAME_MAX] or "피보호자"
    relation = " ".join((ward.relation or "").split())[:_RELATION_MAX]
    return f"{name}({relation})" if relation else name


def _base(reply: str, model_name: str | None, meta: dict[str, Any], intent: str = "hospital_reservation") -> dict[str, Any]:
    return {
        "reply": reply,
        "riskLevel": "medium",
        "recommendedAction": "guardian_contact",
        "reservationRequired": intent == "hospital_reservation",
        "intent": intent,
        "engine": "rule",
        "modelName": model_name,
        "type": "message",
        "ui": None,
        "tool": None,
        "data": None,
        "summary": None,
        "possibleCauses": [],
        "homeCare": [],
        "visitHospitalIf": [],
        "emergencyWarning": [],
        "meta": meta,
        "decisionTrace": list(meta.get("decisionTrace") or []),
    }


def ask_reply(
    wards: list[GuardianWard], *, model_name: str | None, start_ui: dict[str, str] | None, retry: bool = False,
    carry: dict[str, str] | None = None,
) -> dict[str, Any]:
    """대상 질문 답. 기록 meta 에는 허용한 wardId 목록(이름 없음)·시작 버튼·시각만 남긴다 - 선택 턴이 이것과 맞춰 본다.
    시작 문장은 이 기록의 message 칸, 규격화 문장은 router.normalized_message 에 이미 있다. 선택 턴에서 다시 묻는 기록(carry)은
    그 칸들이 비어 있어 앞 질문의 시작 문장·규격화 문장을 옮겨 둔다(같은 사용자의 같은 챗 기록 표 - 앞 행에 이미 있는 값)."""
    meta: dict[str, Any] = {
        "reservationFlowHint": True,
        "reservationTargetAsk": {
            "wardIds": [w.wardId for w in wards],
            "startUi": start_ui,
            "createdAt": int(time.time()),
            **({"carry": carry} if carry else {}),
        },
        "decisionTrace": [{"event": "reservation_target_asked", "options": len(wards), "retry": retry}],
    }
    out = _base(f"{_PICK_HINT}\n{QUESTION}" if retry else QUESTION, model_name, meta)
    out["type"] = "ui"
    out["ui"] = UiPayload(
        kind="select", field=TARGET_FIELD, label=QUESTION,
        options=[UiOption(value=w.wardId, label=option_label(w)) for w in wards],
    )
    return out


def no_ward_reply(model_name: str | None) -> dict[str, Any]:
    """연결된 피보호자가 없는 보호자 - 보호자 본인 예약은 하지 않는다."""
    return _base(NO_WARD_REPLY, model_name, {"reservationTargetNotice": True, "decisionTrace": [{"event": "reservation_target_no_ward"}]}, intent="general_chat")


def stale_reply(model_name: str | None) -> dict[str, Any]:
    """대상 질문이 없거나 지났거나 목록에 없는 값 - 예약을 시작하지 않는다. 예약 흐름이 끝난 답이라 intent 는 일반이다
    (FE 는 이 답 뒤로 글 입력에 wardId 를 싣지 않는다)."""
    return _base(STALE_REPLY, model_name, {"reservationTargetNotice": True, "decisionTrace": [{"event": "reservation_target_stale"}]}, intent="general_chat")


def pending_ask(latest: Any, session_id: str) -> dict[str, Any] | None:
    """직전 기록(같은 세션)이 아직 유효한 대상 질문이면 그 내용 - wardIds·startUi·시작 문장·규격화 문장. 서버 기록에서만 읽는다."""
    if latest is None:
        return None
    try:
        ctx = json.loads(latest.context_json or "{}")
    except (json.JSONDecodeError, TypeError):
        return None
    if not isinstance(ctx, dict) or ctx.get("sessionId") != session_id:
        return None
    meta = ctx.get("upstreamMeta") if isinstance(ctx.get("upstreamMeta"), dict) else {}
    ask = meta.get("reservationTargetAsk")
    if not isinstance(ask, dict) or not isinstance(ask.get("createdAt"), int):
        return None
    if not 0 <= int(time.time()) - ask["createdAt"] <= TARGET_TTL_SECONDS:
        return None
    ward_ids = [w for w in ask.get("wardIds") or [] if isinstance(w, str) and w]
    if not ward_ids:
        return None
    start_ui = ask.get("startUi")
    if not (isinstance(start_ui, dict) and isinstance(start_ui.get("field"), str) and isinstance(start_ui.get("value"), str)):
        start_ui = None
    router = meta.get("router") if isinstance(meta.get("router"), dict) else {}
    normalized = router.get("normalized_message") if router.get("route") == "reservation" else None
    carry = ask.get("carry")
    if isinstance(carry, dict) and isinstance(carry.get("message"), str) and isinstance(carry.get("normalized"), str):
        return {"wardIds": ward_ids, "startUi": start_ui, "message": "" if start_ui else carry["message"],
                "normalized": carry["normalized"] or "병원 예약:"}
    return {
        "wardIds": ward_ids,
        "startUi": start_ui,
        # 버튼으로 시작했으면 그 버튼을 다시 넘기고 글 문장은 없다("[선택] …" 기록 글자를 예약 문장으로 쓰지 않는다).
        "message": "" if start_ui else str(latest.message or "").strip(),
        "normalized": str(normalized or "").strip() or "병원 예약:",
    }


def selection_outcome(ask: dict[str, Any] | None, value: str, ward_id: str) -> str:
    """대상 선택 턴의 결과 - "go"(그 피보호자로 예약 시작) / "reask"(이 턴 wardId 가 고른 값과 다름 - FE 가 안 실음) / "stale"."""
    if ask is None or value not in ask["wardIds"]:
        return "stale"
    return "go" if ward_id == value else "reask"
