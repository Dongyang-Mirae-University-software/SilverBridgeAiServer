"""예약 최종 확인·취소·변경 흐름.

- 예약은 확인 카드에서 "예약하기"를 눌렀을 때만 만든다. 취소도 확인을 거친 뒤에만 부른다.
- 확인 카드가 가리키는 초안은 서버가 직전 chat_log(meta.pendingAction)에 저장해 두고, 버튼 값에는 토큰만 싣는다.
  (버튼 값에 초안을 실으면 클라이언트가 바꿔 보낼 수 있다.) 토큰이 직전 턴의 것과 같을 때만 실행한다.
- 고치기·변경 중에는 초안을 매 턴 이력에서 다시 만들지 않고 meta.flowState 의 초안을 이어 쓴다.
"""
from __future__ import annotations

import logging
import time
import uuid
from datetime import timedelta
from typing import Any

from app.schemas.chat_schema import ChatRequest, UiOption, UiPayload
from datetime import datetime

from sqlalchemy import delete
from sqlalchemy.exc import IntegrityError, SQLAlchemyError

from app.database.session import SessionLocal
from app.models.reservation_token_claim import ReservationTokenClaim
from app.services.reservation_common import DEMO_NOTICE, _SEOUL_TZ, _date_label, _digits, _phone_last4, _trace, with_demo_notice
from app.services.reservation_intake import ReservationDraft, build_reservation_draft

CONFIRM_FIELDS = frozenset(
    {
        "reservation_confirm",
        "reservation_edit",
        "reservation_cancel_pick",
        "reservation_cancel_confirm",
        "reservation_change_pick",
    }
)
_CANCELLABLE_STATUSES = {"BOOKED", "CONFIRMED"}  # 예약 서비스가 취소를 받아 주는 상태
_STOP = "__stop__"
_EDIT_ITEMS = (("hospital", "병원"), ("department", "진료과"), ("date", "날짜"), ("time", "시간"))
_DRAFT_KEYS = (
    "location", "hospital_name", "hospital_id", "department", "reservation_date", "reservation_time",
    "patient_name", "phone", "birth_date", "symptom_summary", "memo",
)
_PENDING_TTL_SECONDS = 1800  # 확인 카드는 30분 안에 눌러야 한다(오래된 카드가 며칠 뒤에 예약되지 않게)
_STALE_REPLY = "확인 시간이 지났어요. 예약하실 병원과 진료과, 날짜를 다시 알려 주세요."


class _TokenClaims:
    """확인 토큰을 한 번만 쓰게 하는 선점 저장소(DB).

    토큰 검증은 직전 chat_log 를 읽고 소비(새 로그 기록)는 응답 뒤에 일어나므로, 같은 버튼을 거의 동시에 두 번 누르면
    둘 다 검증을 통과한다. 예약·취소를 부르기 전에 (사용자, 토큰)을 먼저 INSERT 해 첫 요청만 호출하게 한다.
    기본키 충돌이 곧 "이미 선점됨"이라 서버를 여러 대로 늘려도 유효하다.
    채팅 세션과 따로 연 짧은 세션으로 바로 커밋한다(채팅 트랜잭션이 롤백돼도 선점은 남아야 한다).
    실패한 뒤에도 선점은 풀지 않는다 - 실패 응답 뒤 카드는 어차피 만료돼 처음부터 다시 안내한다.
    DB 오류는 삼키지 않는다: 선점하지 못한 채 예약하는 것보다 요청이 실패하는 쪽이 안전하다."""

    TTL_SECONDS = 600

    def claim(self, user_id: str, token: str) -> bool:
        with SessionLocal() as db:
            try:
                db.add(ReservationTokenClaim(user_id=user_id, token=token, created_at=datetime.utcnow()))
                db.commit()
            except IntegrityError:
                db.rollback()
                return False
        self._purge_expired()
        return True

    def _purge_expired(self) -> None:
        """오래된 행 정리. 실패해도 선점 결과에는 영향이 없다."""
        try:
            with SessionLocal() as db:
                db.execute(delete(ReservationTokenClaim).where(ReservationTokenClaim.created_at < datetime.utcnow() - timedelta(seconds=self.TTL_SECONDS)))
                db.commit()
        except SQLAlchemyError as exc:
            _LOG.warning("[예약토큰선점정리실패] %s", type(exc).__name__)


_LOG = logging.getLogger(__name__)
_CLAIMS = _TokenClaims()
_BUSY_REPLY = "이미 처리 중이에요. 잠시 뒤 예약 목록에서 확인해 주세요."


def draft_to_dict(draft: ReservationDraft) -> dict[str, Any]:
    return {k: getattr(draft, k) for k in _DRAFT_KEYS if getattr(draft, k) not in (None, "")}


def draft_from_dict(raw: dict[str, Any]) -> ReservationDraft:
    draft = ReservationDraft(message="")
    for key in _DRAFT_KEYS:
        value = raw.get(key)
        if value not in (None, ""):
            setattr(draft, key, value)
    return draft


def _mask_phone(phone: str | None) -> str:
    digits = _digits(phone or "")
    return f"{digits[:3]}-****-{digits[-4:]}" if len(digits) >= 8 else "등록된 번호"


def _status(row: dict[str, Any]) -> str:
    return str(row.get("status") or "").strip().upper()


def _error_text(data: Any, default: str) -> tuple[str, str]:
    if isinstance(data, dict):
        return str(data.get("message") or default), str(data.get("code") or "REQUEST_FAILED")
    return default, "REQUEST_FAILED"


class ReservationFlowMixin:
    """ReservationOrchestrator 에 섞이는 확인·취소·변경 흐름. self 의 _ui_reply·_advance 등을 쓴다."""

    CANCEL_CUES = ("취소",)
    CHANGE_CUES = ("변경", "바꿔", "바꾸", "바꿀", "옮겨", "미뤄")
    # 요청이 아니라 거절·부정이다: "취소 안 할게요", "취소하지 않을래요"
    NEGATION_CUES = ("안 할", "안할", "안 해", "안해", "하지 않", "하지않", "하지 마", "하지마", "않을", "않아", "싫")
    # 방법을 묻는 말: 목록을 띄우기보다 말하는 법을 안내한다
    QUESTION_CUES = ("방법", "어떻게", "알려", "궁금", "하는 법", "할 수 있")
    HOW_TO_REPLY = "예약 취소나 변경은 여기서 바로 도와드려요. \"예약 취소할래\" 또는 \"예약 바꿔줘\"라고 말씀해 주세요."

    @classmethod
    def _intent_text(cls, message: str) -> str:
        """"취소 말고 변경할래"처럼 '말고' 앞은 거절된 요청이라 뒤쪽만 본다."""
        return message.split("말고", 1)[1] if "말고" in message else message

    @classmethod
    def _has_cue(cls, message: str, cues: tuple[str, ...]) -> bool:
        if not message or "예약" not in message:
            return False
        text = cls._intent_text(message)
        return any(cue in text for cue in cues) and not any(neg in text for neg in cls.NEGATION_CUES)

    @classmethod
    def _is_question(cls, message: str) -> bool:
        return any(cue in message for cue in cls.QUESTION_CUES)

    @classmethod
    def _is_cancel_request(cls, message: str) -> bool:
        # "취소된 예약 내역 보여줘"처럼 조회 문장은 취소 요청이 아니다.
        lookup = getattr(cls, "LOOKUP_KEYWORDS", ())
        return cls._has_cue(message, cls.CANCEL_CUES) and not cls._is_question(message) and not any(k in message for k in lookup)

    @classmethod
    def _is_change_request(cls, message: str) -> bool:
        return cls._has_cue(message, cls.CHANGE_CUES) and not cls._is_question(message)

    @classmethod
    def _is_cancel_change_question(cls, message: str) -> bool:
        """"예약 취소 방법 알려줘" - 취소·변경하는 법을 묻는 말."""
        return (cls._has_cue(message, cls.CANCEL_CUES) or cls._has_cue(message, cls.CHANGE_CUES)) and cls._is_question(message)

    # ---- 공통 ----------------------------------------------------------------

    @staticmethod
    def _ward_id(req: ChatRequest) -> str:
        return (req.context.wardId or "").strip() if req.context and req.context.wardId else ""

    @classmethod
    def _pin_ward_patient(cls, req: ChatRequest, draft: ReservationDraft) -> None:
        """보호자 대리 예약의 환자는 BE가 넘긴 피보호자다. 채팅 문장(LLM 추출)·입력값이 이름·전화를 덮지 못한다.
        이름은 wardName(없으면 chat_service 가 옮겨 둔 name), 전화는 wardPhone 만 본다 - BE는 피보호자 전화가 있으면 wardPhone 을,
        없으면 phone 까지 지워 보낸다. 피보호자 전화가 없으면 입력받은 연락처를 그대로 쓴다(막으면 대리 예약 자체가 불가능하다)."""
        ctx = req.context
        if not cls._ward_id(req) or ctx is None:
            return
        name = (ctx.wardName or ctx.name or "").strip()
        phone = (ctx.wardPhone or "").strip()
        if name:
            draft.patient_name = name
        if phone:
            draft.phone = phone

    @staticmethod
    def _followup_ui() -> UiPayload:
        return UiPayload(
            kind="select",
            field="reservation_followup",
            label="예약 목록을 보여드릴까요?",
            options=[UiOption(value="show_list", label="예약 목록 보기"), UiOption(value="stay", label="계속 상담하기")],
        )

    def _flow_message(self, meta: dict[str, Any], reply: str, ui: UiPayload | None = None) -> dict[str, Any]:
        out = self._ui_reply(meta, reply, ui)  # type: ignore[attr-defined]
        if ui is None:
            out["type"] = "message"
        return out

    def _flow_error(self, meta: dict[str, Any], tool: str, message: str, code: str, data: Any = None) -> dict[str, Any]:
        out = self._flow_message(meta, message)
        out.update({"type": "tool_error", "tool": tool, "data": data, "error": {"code": code, "message": message}})
        return out

    def _stale(self, meta: dict[str, Any], kind: str) -> dict[str, Any]:
        _trace(meta, "reservation_flow_stale", kind=kind)
        return self._flow_message(meta, _STALE_REPLY)

    @staticmethod
    def _valid_pending(stored: dict[str, Any], kind: str, token: str, ward_id: str) -> dict[str, Any] | None:
        pending = stored.get("pendingAction")
        if (
            isinstance(pending, dict)
            and pending.get("kind") == kind
            and token
            and pending.get("token") == token
            and (pending.get("wardId") or "") == ward_id
            and isinstance(pending.get("createdAt"), int)
            and 0 <= int(time.time()) - pending["createdAt"] <= _PENDING_TTL_SECONDS
        ):
            return pending
        return None

    @staticmethod
    def _brief(row: dict[str, Any]) -> str:
        hospital = str(row.get("hospitalName") or row.get("hospital") or "병원").strip()
        day = str(row.get("reservationDate") or row.get("date") or "").strip()[:10]
        tm = str(row.get("reservationTime") or row.get("time") or "").strip()
        return " ".join(part for part in (hospital, _date_label(day) if day else "", tm) if part)

    def _remember_flow(
        self, out: dict[str, Any], meta: dict[str, Any], draft: ReservationDraft, replaces: dict[str, Any] | None, ward_id: str
    ) -> dict[str, Any]:
        """고치기·변경 중이면 다음 턴이 이 초안을 이어 쓰도록 남긴다(확인 카드가 나온 턴은 pendingAction 이 대신한다).
        초안은 시작한 피보호자(wardId)에게 묶인다 - 다른 피보호자 상담에서 이어 쓰면 환자 정보가 섞인다."""
        target = out.get("meta") if isinstance(out.get("meta"), dict) else meta
        if "pendingAction" not in target:
            target["flowState"] = {"draft": draft_to_dict(draft), "replaces": replaces, "wardId": ward_id}
        out["meta"] = target
        return out

    # ---- 확인 카드 -------------------------------------------------------------

    def _confirm_reply(
        self, req: ChatRequest, meta: dict[str, Any], draft: ReservationDraft, replaces: dict[str, Any] | None = None
    ) -> dict[str, Any]:
        token = uuid.uuid4().hex[:12]
        ward_id = self._ward_id(req)
        meta["pendingAction"] = {
            "kind": "create", "token": token, "draft": draft_to_dict(draft), "wardId": ward_id, "replaces": replaces,
            "createdAt": int(time.time()),
        }
        name = (draft.patient_name or "").strip()
        lines: list[str] = []
        if ward_id:
            # 계정 주인은 BE가 넘긴 피보호자다(초안 이름은 _pin_ward_patient 로 이미 같은 값).
            who = (req.context.wardName if req.context and req.context.wardName else "") or name or "피보호자"
            lines.append(f"{who}님의 예약을 보호자님이 대신 잡고 있어요.")
        lines.append("아래 내용으로 예약할까요?")
        lines += [
            f"· 환자: {name}님" if name else "· 환자: -",
            f"· 병원: {draft.hospital_name or '-'}",
            f"· 진료과: {draft.department or '-'}",
            f"· 날짜: {_date_label(draft.reservation_date or '')}",
            f"· 시간: {draft.reservation_time or '-'}",
            f"· 연락처: {_mask_phone(draft.phone)}",
        ]
        if replaces:
            lines.append(f"※ 기존 예약({replaces.get('label')})은 새 예약이 완료되면 취소돼요.")
        lines.append(DEMO_NOTICE)
        _trace(
            meta, "reservation_confirm_shown", hospital_id=draft.hospital_id, department=draft.department,
            reservation_date=draft.reservation_date, reservation_time=draft.reservation_time,
            phone_last4=_phone_last4(draft.phone), on_behalf=bool(ward_id), replaces=bool(replaces),
        )
        return self._flow_message(
            meta,
            "\n".join(lines),
            UiPayload(
                kind="select",
                field="reservation_confirm",
                label="예약 내용을 확인해 주세요",
                options=[UiOption(value=f"confirm:{token}", label="예약하기"), UiOption(value=f"edit:{token}", label="고치기")],
            ),
        )

    # ---- 예약 생성 (확인 뒤에만) -------------------------------------------------

    def _create_from_draft(
        self,
        req: ChatRequest,
        meta: dict[str, Any],
        draft: ReservationDraft,
        key: str | None,
        replaces: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        self._pin_ward_patient(req, draft)  # 저장된 초안에서 바로 예약하는 경로도 피보호자 값으로 한 번 더 맞춘다
        _trace(
            meta, "reservation_submit", hospital_id=draft.hospital_id, hospital_name=draft.hospital_name,
            department=draft.department, reservation_date=draft.reservation_date, reservation_time=draft.reservation_time,
            patient_name_len=len(draft.patient_name or ""), phone_last4=_phone_last4(draft.phone),
        )
        # 같은 날짜·시간에 이미 예약이 있으면 만들지 않고 시간을 다시 고르게 한다(원자적이진 않지만 중복 창을 줄인다).
        # 조회가 실패하면 통과시키고 예약 서비스의 중복 거절에 맡긴다.
        booked = self._booked_times(key)  # type: ignore[attr-defined]
        if draft.reservation_time and draft.reservation_time in booked.get(draft.reservation_date or "", set()):
            _trace(meta, "reservation_precheck_duplicate")
            reselect = self._reselect_time(meta, draft, key, replaces, self._ward_id(req))
            if reselect is not None:
                return reselect
        args = draft.to_make_appointment_args()
        if draft.hospital_id is not None:
            args["hospital_id"] = draft.hospital_id
        cr = self._reservation.create_reservation(  # type: ignore[attr-defined]
            hospital_id=int(args["hospital_id"]),
            department=str(args["department"]),
            reservation_date=str(args["date"]),
            reservation_time=str(args["time"]),
            patient_name=str(args["patient_name"]),
            phone=str(args["phone"]),
            api_key_override=key,
            birth_date=str(args["birth_date"]).strip() if args.get("birth_date") else None,
            symptom_summary=str(args["symptom_summary"]).strip() if args.get("symptom_summary") else None,
            memo=str(args["memo"]).strip() if args.get("memo") else None,
        )
        st = int(cr.get("httpStatus") or 0)
        data = cr.get("data")
        if 200 <= st < 300 and isinstance(data, dict):
            shaped = self._shape_make_appointment(data, str(args["patient_name"]), str(args["phone"]))  # type: ignore[attr-defined]
            _trace(meta, "reservation_submit_ok", reservation_id=shaped.get("reservation_id"), status=shaped.get("status"))
            return self._after_created(meta, shaped, key, replaces, req)

        msg, code = _error_text(data, "예약 처리에 실패했습니다.")
        _trace(meta, "reservation_submit_failed", http_status=st, code=code)  # 예약 서비스 문구는 입력값을 되돌려 줄 수 있어 남기지 않는다
        if code == "DUPLICATE_RESERVATION":
            reselect = self._reselect_time(meta, draft, key, replaces, self._ward_id(req))
            if reselect is not None:
                return reselect
        return self._flow_error(meta, "make_appointment", msg, code, data)

    def _reselect_time(
        self, meta: dict[str, Any], draft: ReservationDraft, key: str | None, replaces: dict[str, Any] | None, ward_id: str
    ) -> dict[str, Any] | None:
        """같은 시간대 중복은 막다른 길이 아니라 시간을 다시 고르게 한다. 다시 고를 시간이 없으면 None."""
        if draft.hospital_id is None or not draft.reservation_date:
            return None
        draft.reservation_time = None
        slot_date, slots = self._find_slots(draft.hospital_id, draft.reservation_date, key, meta, draft.department)  # type: ignore[attr-defined]
        if slot_date is None or not slots:
            return None
        out = self._time_select_reply(  # type: ignore[attr-defined]
            meta, draft, draft.reservation_date, slot_date, slots,
            prefix="같은 시간대에 이미 예약이 있어요. 다른 시간을 골라 주세요.",
        )
        return self._remember_flow(out, meta, draft, replaces, ward_id)

    def _after_created(
        self, meta: dict[str, Any], shaped: dict[str, Any], key: str | None, replaces: dict[str, Any] | None, req: ChatRequest
    ) -> dict[str, Any]:
        reply = "예약이 완료되었습니다."
        ui: UiPayload | None = self._followup_ui()
        if replaces:
            # 새 예약이 확정된 뒤에만 기존 예약을 취소한다. 새 예약이 실패했다면 여기까지 오지 않는다.
            cancelled = self._reservation.cancel_reservation(int(replaces["id"]), api_key_override=key)  # type: ignore[attr-defined]
            if 200 <= int(cancelled.get("httpStatus") or 0) < 300:
                _trace(meta, "reservation_change_old_cancelled", old_id=replaces.get("id"))
                reply = f"예약이 완료되었습니다. 기존 예약({replaces.get('label')})은 취소했어요."
            else:
                _trace(meta, "reservation_change_old_cancel_failed", old_id=replaces.get("id"), http_status=cancelled.get("httpStatus"))
                token = uuid.uuid4().hex[:12]
                meta["pendingAction"] = {
                    "kind": "cancel", "token": token, "reservationId": int(replaces["id"]),
                    "wardId": self._ward_id(req), "label": replaces.get("label"), "createdAt": int(time.time()),
                }
                # 질문이 마지막 줄에 오도록 시연용 안내를 앞에 둔다.
                reply = f"{DEMO_NOTICE}\n새 예약은 완료되었지만 기존 예약({replaces.get('label')}) 취소에 실패했어요. 기존 예약을 취소할까요?"
                ui = self._cancel_confirm_ui(token)
        out = self._flow_message(meta, with_demo_notice(reply), ui)
        out.update({"type": "tool_result", "tool": "make_appointment", "data": shaped})
        return out

    # ---- 취소·변경 목록 ----------------------------------------------------------

    MAX_PICK = 10

    def _cancellable_rows(self, key: str | None) -> tuple[list[dict[str, Any]], int] | None:
        """취소할 수 있는 예약 - 오늘 이후만, 날짜·시간순으로 MAX_PICK건. (보여줄 목록, 조건에 맞는 전체 수). 조회 실패는 None."""
        res = self._reservation.list_my_appointments(api_key_override=key)  # type: ignore[attr-defined]
        data = res.get("data")
        if not (200 <= int(res.get("httpStatus") or 0) < 300) or not isinstance(data, list):
            return None
        today = datetime.now(_SEOUL_TZ).date().isoformat()

        def day(row: dict[str, Any]) -> str:
            return str(row.get("reservationDate") or row.get("date") or "")[:10]

        rows = [
            row for row in data
            if isinstance(row, dict) and _status(row) in _CANCELLABLE_STATUSES and row.get("reservationId") is not None
            and (not day(row) or day(row) >= today)  # 날짜를 읽을 수 없으면 숨기지 않는다
        ]
        rows.sort(key=lambda r: (day(r), str(r.get("reservationTime") or r.get("time") or "")))
        return rows[: self.MAX_PICK], len(rows)

    def _start_pick(self, meta: dict[str, Any], key: str | None, *, change: bool) -> dict[str, Any]:
        found = self._cancellable_rows(key)
        if found is None:
            return self._flow_error(meta, "list_my_appointments", "예약 목록을 불러오지 못했습니다.", "LOOKUP_FAILED")
        rows, total = found
        if not rows:
            reply = "바꿀 수 있는 예약이 없어요. 새로 예약할까요?" if change else "취소할 수 있는 예약이 없어요."
            return self._flow_message(meta, reply)
        options = [UiOption(value=str(row["reservationId"]), label=self._brief(row)) for row in rows]
        options.append(UiOption(value=_STOP, label="그만두기"))
        field = "reservation_change_pick" if change else "reservation_cancel_pick"
        label = "어떤 예약을 바꿀까요?" if change else "어떤 예약을 취소할까요?"
        if total > len(rows):
            label += f" (가까운 {len(rows)}건만 보여드려요)"
        _trace(meta, "reservation_pick_shown", change=change, count=len(rows), total=total)
        return self._flow_message(meta, label, UiPayload(kind="select", field=field, label=label, options=options))

    @staticmethod
    def _cancel_confirm_ui(token: str) -> UiPayload:
        return UiPayload(
            kind="select",
            field="reservation_cancel_confirm",
            label="예약을 취소할까요?",
            options=[UiOption(value=f"yes:{token}", label="취소하기"), UiOption(value=f"no:{token}", label="그대로 두기")],
        )

    def _load_cancellable(self, meta: dict[str, Any], value: str, key: str | None) -> tuple[dict[str, Any] | None, dict[str, Any] | None]:
        """선택한 예약을 예약 서비스에서 다시 읽는다(본인 소유가 아니면 서비스가 404로 거절한다). (행, 오류 응답)."""
        try:
            rid = int(value.strip())
        except ValueError:
            rid = 0
        if rid <= 0:
            return None, self._flow_message(meta, "선택한 예약을 찾지 못했어요. 다시 말씀해 주세요.")
        res = self._reservation.get_reservation(rid, api_key_override=key)  # type: ignore[attr-defined]
        st, data = int(res.get("httpStatus") or 0), res.get("data")
        if not (200 <= st < 300) or not isinstance(data, dict):
            msg, code = _error_text(data, "예약 정보를 불러오지 못했습니다.")
            _trace(meta, "reservation_pick_failed", http_status=st, code=code)
            return None, self._flow_error(meta, "get_reservation", msg, code, data)
        if _status(data) not in _CANCELLABLE_STATUSES:
            return None, self._flow_message(meta, "이미 취소되었거나 변경할 수 없는 예약이에요.")
        data.setdefault("reservationId", rid)
        return data, None

    # ---- 선택 턴 처리 ------------------------------------------------------------

    def _handle_flow_selection(
        self, req: ChatRequest, meta: dict[str, Any], field: str, value: str, key: str | None
    ) -> dict[str, Any]:
        stored = meta.get("storedFlow") if isinstance(meta.get("storedFlow"), dict) else {}
        ward_id = self._ward_id(req)
        value = (value or "").strip()

        if field == "reservation_cancel_pick":
            if value == _STOP:
                return self._flow_message(meta, "알겠습니다. 예약은 그대로 둘게요.")
            row, err = self._load_cancellable(meta, value, key)
            if err:
                return err
            token = uuid.uuid4().hex[:12]
            label = self._brief(row)
            meta["pendingAction"] = {
                "kind": "cancel", "token": token, "reservationId": int(row["reservationId"]), "wardId": ward_id, "label": label,
                "createdAt": int(time.time()),
            }
            return self._flow_message(meta, f"{label} 예약을 취소할까요?", self._cancel_confirm_ui(token))

        if field == "reservation_cancel_confirm":
            action, _, token = value.partition(":")
            pending = self._valid_pending(stored, "cancel", token, ward_id)
            if pending is None or action not in {"yes", "no"}:
                return self._stale(meta, "cancel")
            if action == "no":
                return self._flow_message(meta, "알겠습니다. 예약은 그대로 둘게요.")
            if not _CLAIMS.claim(req.userId, token):
                _trace(meta, "reservation_token_busy", kind="cancel")
                return self._flow_message(meta, _BUSY_REPLY)
            res = self._reservation.cancel_reservation(int(pending["reservationId"]), api_key_override=key)  # type: ignore[attr-defined]
            st, data = int(res.get("httpStatus") or 0), res.get("data")
            if 200 <= st < 300:
                _trace(meta, "reservation_cancel_ok", reservation_id=pending["reservationId"])
                out = self._flow_message(meta, f"{pending.get('label')} 예약을 취소했어요.", self._followup_ui())
                out.update({"type": "tool_result", "tool": "cancel_reservation", "data": {"reservationId": pending["reservationId"], "status": "CANCELED"}})
                return out
            msg, code = _error_text(data, "예약 취소에 실패했습니다.")
            _trace(meta, "reservation_cancel_failed", http_status=st, code=code)
            return self._flow_error(meta, "cancel_reservation", msg, code, data)

        if field == "reservation_change_pick":
            if value == _STOP:
                return self._flow_message(meta, "알겠습니다. 예약은 그대로 둘게요.")
            row, err = self._load_cancellable(meta, value, key)
            if err:
                return err
            _history, loc = self._conversation_context(req)  # type: ignore[attr-defined]
            normalized = self._normalize_location_hint(loc)  # type: ignore[attr-defined]
            draft = build_reservation_draft(
                "", location=normalized or loc, user_context=req.context.model_dump(exclude_none=True) if req.context else None,
            )
            draft.hospital_name = str(row.get("hospitalName") or "").strip() or None
            draft.department = str(row.get("department") or "").strip() or None
            replaces = {"id": int(row["reservationId"]), "label": self._brief(row)}
            _trace(meta, "reservation_change_started", old_id=replaces["id"])
            out = self._advance(req, meta, draft, key, replaces)  # type: ignore[attr-defined]
            if isinstance(out.get("reply"), str):
                out["reply"] = f"{replaces['label']} 예약을 바꿀게요. {out['reply']}"
            return self._remember_flow(out, meta, draft, replaces, ward_id)

        if field == "reservation_confirm":
            action, _, token = value.partition(":")
            pending = self._valid_pending(stored, "create", token, ward_id)
            if pending is None or action not in {"confirm", "edit"}:
                return self._stale(meta, "create")
            draft = draft_from_dict(pending.get("draft") or {})
            replaces = pending.get("replaces") if isinstance(pending.get("replaces"), dict) else None
            if action == "edit":
                meta["flowState"] = {"draft": draft_to_dict(draft), "replaces": replaces, "editToken": token, "wardId": ward_id}
                return self._flow_message(
                    meta,
                    "무엇을 고칠까요?",
                    UiPayload(
                        kind="select",
                        field="reservation_edit",
                        label="무엇을 고칠까요?",
                        options=[UiOption(value=f"{item}:{token}", label=label) for item, label in _EDIT_ITEMS],
                    ),
                )
            if draft.required_gap() is not None:
                return self._stale(meta, "create")
            if not _CLAIMS.claim(req.userId, token):
                _trace(meta, "reservation_token_busy", kind="create")
                return self._flow_message(meta, _BUSY_REPLY)
            return self._create_from_draft(req, meta, draft, key, replaces)

        if field == "reservation_edit":
            item, _, token = value.partition(":")
            flow = stored.get("flowState") if isinstance(stored.get("flowState"), dict) else {}
            if (
                not token
                or flow.get("editToken") != token
                or "wardId" not in flow  # 배포 전에 저장된 옛 상태
                or (flow.get("wardId") or "") != ward_id
                or item not in {i for i, _ in _EDIT_ITEMS}
            ):
                return self._stale(meta, "edit")
            draft = draft_from_dict(flow.get("draft") or {})
            replaces = flow.get("replaces") if isinstance(flow.get("replaces"), dict) else None
            draft.reservation_time = None  # 무엇을 고치든 시간은 다시 고른다(병원·날짜가 바뀌면 그 시간이 비어 있다는 보장이 없다)
            if item == "hospital":
                draft.hospital_name = draft.hospital_id = None
            elif item == "department":
                draft.department = draft.hospital_name = draft.hospital_id = None
            elif item == "date":
                draft.reservation_date = None
            _trace(meta, "reservation_edit", item=item)
            out = self._advance(req, meta, draft, key, replaces)  # type: ignore[attr-defined]
            return self._remember_flow(out, meta, draft, replaces, ward_id)

        return self._stale(meta, "unknown")
