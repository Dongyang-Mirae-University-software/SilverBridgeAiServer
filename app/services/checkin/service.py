"""안부 말벗 흐름.

- 세션·턴 상태와 대화 원문(받아쓴 글자·AI 문장)은 이 서버 DB 에 둔다. 백엔드는 세션 ID·결과만 갖는다.
- 매 턴은 받아쓰기 + 공감·꼬리질문만 한다(어르신이 기다리는 시간을 줄이려고). 정서·생활 신호 판정은 finish 에서 대화 전체로 한 번.
- 평소 대비 판정은 백엔드가 한다 - 여기서는 그날 대화에서 나온 값만 낸다.
- 음성·사진은 저장하지 않는다. 로그에는 세션 ID·턴 번호·결과 코드·소요 시간만 남긴다(대화 내용 금지).
"""

from __future__ import annotations

import json
import logging
import re
import time
from datetime import datetime, timedelta

from sqlalchemy import delete, select
from sqlalchemy.exc import IntegrityError
from sqlalchemy.orm import Session

from app.models.checkin import CheckinSession, CheckinTurn
from app.services import chat_keywords
from app.services.checkin import audio as audio_mod
from app.services.checkin.config import get_checkin_settings
from app.services.checkin.models import get_models

log = logging.getLogger(__name__)


class CheckinError(Exception):
    def __init__(self, status: int, code: str, message: str):
        super().__init__(code)
        self.status, self.code, self.message = status, code, message


END_WORDS = ("그만할게", "그만 할게", "그만하자", "그만 하자", "끊을게", "끊자", "이만할게", "이만 할게", "됐어 그만", "나중에 얘기")
FAREWELL = "말씀 나눠 주셔서 고마워요. 편안한 시간 보내세요."
EMERGENCY_REPLY = "많이 걱정돼요. 위급하시면 지금 바로 119에 연락하시고, 보호자분께도 꼭 알려 주세요."
SELF_HARM_REPLY = "많이 힘드셨겠어요. 혼자 견디지 마시고 109(자살예방 상담전화)나 보호자분께 꼭 이야기해 주세요."
FALLBACK_REPLIES = ("그러셨군요. 말씀해 주셔서 고마워요. 오늘 하루는 어떻게 지내셨어요?",
                    "그렇군요. 요즘 몸은 좀 어떠세요?",
                    "말씀 잘 들었어요. 식사는 잘 챙겨 드시고 계세요?")
LONELY_WORDS = ("외롭", "외로", "적적", "허전", "쓸쓸")
EVIDENCE_WORDS = {
    "sleep": ("잠", "잤", "깼", "뒤척", "설쳤"),
    "meal": ("밥", "식사", "먹", "끼니", "입맛"),
    "pain": ("아파", "아프", "쑤", "결려", "결리", "저려", "저리", "쓰려", "지끈", "시큰", "통증"),
    "lonely": LONELY_WORDS + ("혼자", "아무도", "찾아오", "말할 사람"),
}


def _risk(text: str) -> dict:
    c = chat_keywords.compact(text)
    return {"emergency": any(k in c for k in chat_keywords.EMERGENCY_KEYWORDS),
            "selfHarm": any(k in c for k in chat_keywords.SELF_HARM_KEYWORDS)}


def _clean_reply(raw: str) -> str:
    """두 문장까지, 줄바꿈·이모지·괄호 제거. 비면 None."""
    text = re.sub(r"[\(\[\{（].*?[\)\]\}）]", "", raw or "")
    text = re.sub(r"[^\w\s가-힣.,?!~·'\"-]", "", text)
    text = re.sub(r"\s+", " ", text).strip()
    parts = re.findall(r"[^.?!]+[.?!]?", text)
    out = " ".join(p.strip() for p in parts[:2] if p.strip())
    return out[:160] if out else ""


def _post_rules(sig: dict, text: str) -> dict:
    """서버 규칙 R1~R3 (2026-10-20 DGX 시험과 같다)."""
    sig = dict(sig)
    if "깼" in text and "안 깼" not in text:
        sig["sleep"] = "나쁨"
    if any(w in text for w in LONELY_WORDS):
        sig["lonely"] = "있음"
    if sig.get("lonely") == "있음":
        sig["mood"] = "나쁨"
    return sig


def _summary_text(sig: dict, risk: dict) -> str:
    """보호자 화면 한 줄. 사실을 단정하지 않고 '말씀하셨어요'로 쓴다. 원문은 옮기지 않는다."""
    lines = []
    if risk["selfHarm"]:
        lines.append("마음이 많이 힘들다는 말씀을 하셨어요.")
    if risk["emergency"]:
        lines.append("위급할 수 있는 말씀을 하셨어요.")
    if sig.get("pain") == "있음":
        lines.append("아픈 곳이 있다고 하셨어요.")
    if sig.get("sleep") == "나쁨":
        lines.append("잠을 잘 못 주무셨다고 하셨어요.")
    if sig.get("meal") == "거름":
        lines.append("식사를 거르셨다고 하셨어요.")
    if sig.get("lonely") == "있음":
        lines.append("외롭다는 말씀을 하셨어요.")
    elif sig.get("mood") == "나쁨":
        lines.append("기분이 좋지 않다고 하셨어요.")
    if not lines:
        lines.append("기분이 좋다고 하셨어요." if sig.get("mood") == "좋음" else "특별히 불편한 점은 말씀하지 않으셨어요.")
    return " ".join(lines[:2])


def _session_or_404(db: Session, session_id: str, user_id: str) -> CheckinSession:
    s = db.get(CheckinSession, session_id)
    if s is None or s.user_id != user_id:  # 남의 세션도 없는 것과 같게 답한다
        raise CheckinError(404, "SESSION_NOT_FOUND", "대화를 찾을 수 없습니다.")
    return s


def _turn_view(t: CheckinTurn) -> dict:
    return {"turnNo": t.turn_no, "transcript": t.transcript, "reply": t.reply, "done": bool(t.done),
            "voice": json.loads(t.voice_json or "{}"), "face": json.loads(t.face_json) if t.face_json else None,
            "risk": _risk(t.transcript)}


def start(db: Session, session_id: str, user_id: str, slot: str, question: str) -> dict:
    s = db.get(CheckinSession, session_id)
    if s is not None:
        if s.user_id != user_id or s.slot != slot:
            raise CheckinError(409, "SESSION_CONFLICT", "이미 다른 대화에 쓰인 세션 ID 입니다.")
        return {"sessionId": s.session_id, "startedAt": s.started_at.isoformat(timespec="seconds") + "Z"}
    s = CheckinSession(session_id=session_id, user_id=user_id, slot=slot, question=question.strip())
    db.add(s)
    try:
        db.commit()
    except IntegrityError:
        db.rollback()
        return start(db, session_id, user_id, slot, question)
    log.info("[CHECKIN] start session=%s slot=%s", session_id, slot)
    return {"sessionId": s.session_id, "startedAt": s.started_at.isoformat(timespec="seconds") + "Z"}


def turn(db: Session, session_id: str, user_id: str, turn_no: int, audio_bytes: bytes, frames: list[bytes]) -> dict:
    cfg = get_checkin_settings()
    s = _session_or_404(db, session_id, user_id)
    existing = db.scalar(select(CheckinTurn).where(CheckinTurn.session_id == session_id, CheckinTurn.turn_no == turn_no))
    if existing is not None:  # 같은 턴을 다시 보내면 저장된 결과를 그대로 돌려준다
        return _turn_view(existing)
    if s.finished_at is not None:
        raise CheckinError(409, "SESSION_FINISHED", "이미 끝난 대화입니다.")
    turns = db.scalars(select(CheckinTurn).where(CheckinTurn.session_id == session_id).order_by(CheckinTurn.turn_no)).all()
    if turns and turns[-1].done:
        raise CheckinError(409, "SESSION_FINISHED", "이미 끝난 대화입니다.")
    if turn_no > cfg.max_turns:
        raise CheckinError(409, "TURN_LIMIT", "대화 턴 수를 넘었습니다.")
    if turn_no != len(turns) + 1:
        raise CheckinError(409, "TURN_OUT_OF_ORDER", "턴 번호가 순서에 맞지 않습니다.")
    t0 = time.monotonic()
    try:
        arr = audio_mod.decode(audio_bytes)
    except audio_mod.AudioError as e:
        raise CheckinError(422, e.code, "음성 파일을 읽을 수 없습니다.") from None
    if len(arr) / audio_mod.SR > cfg.max_audio_sec + 1:
        raise CheckinError(413, "AUDIO_TOO_LONG", "대답 음성이 너무 깁니다.")
    models = get_models()
    transcript = models.transcribe(arr)
    if not transcript:
        raise CheckinError(422, "SPEECH_NOT_RECOGNIZED", "말씀을 알아듣지 못했어요.")
    voice = audio_mod.voice_metrics(arr, transcript)
    question = s.question if not turns else turns[-1].reply
    risk = _risk(transcript)
    last = turn_no >= cfg.max_turns or any(w in transcript for w in END_WORDS)
    if risk["selfHarm"]:
        reply = SELF_HARM_REPLY
    elif risk["emergency"]:
        reply = EMERGENCY_REPLY
    elif last and turn_no < cfg.max_turns:
        reply = FAREWELL
    else:
        reply = _clean_reply(models.reply(question, transcript, last)) or (FAREWELL if last else FALLBACK_REPLIES[turn_no % 3])
    row = CheckinTurn(session_id=session_id, user_id=user_id, turn_no=turn_no, question=question, transcript=transcript,
                      reply=reply, done=int(last), voice_json=json.dumps(voice), face_json=None)
    db.add(row)
    try:
        db.commit()
    except IntegrityError:  # 같은 턴이 동시에 들어오면 먼저 저장된 것을 쓴다
        db.rollback()
        row = db.scalar(select(CheckinTurn).where(CheckinTurn.session_id == session_id, CheckinTurn.turn_no == turn_no))
        if row is None:
            raise
    log.info("[CHECKIN] turn session=%s no=%s done=%s ms=%d frames=%d", session_id, turn_no, last,
             int((time.monotonic() - t0) * 1000), len(frames))
    return _turn_view(row)


def finish(db: Session, session_id: str, user_id: str, self_report: str | None, reason: str) -> dict:
    s = _session_or_404(db, session_id, user_id)
    if s.result_json:
        return json.loads(s.result_json)
    turns = db.scalars(select(CheckinTurn).where(CheckinTurn.session_id == session_id).order_by(CheckinTurn.turn_no)).all()
    t0 = time.monotonic()
    utts = [t.transcript for t in turns]
    risk = {"emergency": False, "selfHarm": False}
    for t in turns:
        r = _risk(t.transcript)
        risk = {k: risk[k] or r[k] for k in risk}
    if utts:
        models = get_models()
        sig = _post_rules(models.summarize(utts), " ".join(utts))
        emotion = models.emotion(utts) or "판단 보류"
    else:
        sig, emotion = {k: None for k in ("sleep", "meal", "pain", "mood", "lonely")}, "판단 보류"
    neg = {"sleep": "나쁨", "meal": "거름", "pain": "있음", "lonely": "있음"}
    evidence = []
    for field, bad in neg.items():
        if sig.get(field) == bad:
            hit = next((t.turn_no for t in turns if any(w in t.transcript for w in EVIDENCE_WORDS[field])), None)
            evidence.append({"signal": field, "turnNo": hit})
    voices = [json.loads(t.voice_json or "{}") for t in turns]
    avg = lambda key: (round(sum(v[key] for v in voices if v.get(key) is not None) / n, 2)  # noqa: E731
                       if (n := sum(1 for v in voices if v.get(key) is not None)) else None)
    result = {
        "sessionId": session_id,
        "turns": len(turns),
        "endReason": reason,
        "emotion": emotion,
        "selfReport": self_report,
        "signals": sig,
        "risk": risk,
        "summary": _summary_text(sig, risk) if utts else "대답이 없었어요.",
        "evidence": evidence,
        "voice": {"avgAnswerSec": avg("answerSec"), "avgSpeechSec": avg("speechSec"), "avgPauseRatio": avg("pauseRatio"),
                  "avgCharsPerSec": avg("charsPerSec"), "avgLoudnessDb": avg("loudnessDb")},
        "face": None,  # 표정 모델은 한국인 얼굴 시험 뒤에 연결한다(그 전까지 null)
    }
    s.result_json = json.dumps(result, ensure_ascii=False)
    s.finished_at = datetime.utcnow()
    db.commit()
    log.info("[CHECKIN] finish session=%s turns=%d emotion_set=%s ms=%d", session_id, len(turns), emotion != "판단 보류",
             int((time.monotonic() - t0) * 1000))
    return result


def purge_user(db: Session, user_id: str) -> dict:
    turns = db.execute(delete(CheckinTurn).where(CheckinTurn.user_id == user_id)).rowcount or 0
    sessions = db.execute(delete(CheckinSession).where(CheckinSession.user_id == user_id)).rowcount or 0
    db.commit()
    log.info("[CHECKIN] purge user=%s sessions=%d turns=%d", user_id, sessions, turns)
    return {"sessions": sessions, "turns": turns}


def purge_expired(db: Session, now: datetime | None = None) -> dict:
    cutoff = (now or datetime.utcnow()) - timedelta(days=get_checkin_settings().retention_days)
    ids = db.scalars(select(CheckinSession.session_id).where(CheckinSession.started_at < cutoff)).all()
    if not ids:
        return {"sessions": 0, "turns": 0}
    turns = db.execute(delete(CheckinTurn).where(CheckinTurn.session_id.in_(ids))).rowcount or 0
    sessions = db.execute(delete(CheckinSession).where(CheckinSession.session_id.in_(ids))).rowcount or 0
    db.commit()
    return {"sessions": sessions, "turns": turns}
