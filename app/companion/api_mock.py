"""말벗 API 목업 (계획서 Phase 2·7, 백엔드와 합의 전 로컬 확인용). 버전 0.3 (백엔드 응답 규칙 반영).

응답 규칙 (백엔드와 같은 봉투)
- 성공: {"success": true, "message": "...", "data": {...}, "code": null}
- 실패: {"success": false, "message": "...", "data": null, "code": "ERROR_CODE"}

엔드포인트 (wardId 는 백엔드가 ACTIVE 연결을 확인한 뒤 넘긴다고 가정)
- GET  /companion/question?wardId=&slot=09:00   지금 시각의 보호자 질문 (없으면 404 QUESTION_NOT_SET)
- PUT  /companion/questions/{slot}?wardId=      보호자 질문 저장, 본문 {"question": "..."}
- GET  /companion/examples                      예시 질문 (참고용)
- POST /companion/answers                       글자 대답 제출, 본문 {"wardId", "sessionId", "slot", "text"}
- POST /companion/answers/audio                 음성 대답 제출, 폼 필드 wardId, sessionId, slot, 파일 audio (wav)
- GET  /companion/emotions?wardId=&date=        그날 시각별 정서 결과

인증: 헤더 X-API-Key 가 .env 의 COMPANION_API_KEY 와 같아야 한다. 키는 .env 에만 둔다.

저장 구조 (답변 1건): wardId, sessionId, slot, date, question, text, audioSeconds, label, reason, createdAt
주의: 목업 저장소는 data/api_mock/ (대답 원문 포함, 실제 서비스에서는 쓰지 않음). 얼굴 결과는 아직 연결하지 않음.
"""

from __future__ import annotations

import concurrent.futures
from contextlib import asynccontextmanager
import hmac
import io
import json
import os
import sys
import uuid
from datetime import datetime
from pathlib import Path
from typing import Optional

from dotenv import load_dotenv
from fastapi import APIRouter, Depends, FastAPI, File, Form, Header, HTTPException, Query, Request, UploadFile
from fastapi.responses import JSONResponse
from fastapi.staticfiles import StaticFiles

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[2]  # 저장소 루트 (data/, .env 위치)
sys.path.insert(0, str(HERE))  # 같은 폴더의 모듈을 짧은 이름으로 부른다
load_dotenv(ROOT / ".env")

import dialogue as dlg  # noqa: E402
import review_log  # noqa: E402
from dialogue import MAX_TURNS, TIME_LIMIT_SEC, Dialogue  # noqa: E402
from pydantic import BaseModel, Field

from fusion import Signal, fuse, is_ambiguous  # noqa: E402

# 판단 보류일 때 물어볼 다음 질문 (돌아가며 쓴다, 사용자 결정 2026-10-25)
FOLLOW_UPS = ["조금 더 자세히 말씀해 주실 수 있을까요?", "오늘 기분을 한 단어로 말씀해 주실래요?", "요즘 마음에 가장 남는 일은 무엇인가요?"]

DATA = ROOT / "data" / "api_mock"
DATA.mkdir(parents=True, exist_ok=True)
QUESTIONS_FILE = DATA / "guardian_questions.json"
ANSWERS_FILE = DATA / "answers.json"
EXAMPLES_FILE = HERE / "examples.json"
SLOTS = {"09:00", "12:00", "15:00", "18:00", "21:00"}
MAX_AUDIO_BYTES = 10 * 1024 * 1024
ANALYSIS_TIMEOUT_SEC = 30  # 감정 분석 한 번의 상한 (계획서 7.4: 시간 초과는 분석 실패로 기록)
_pool = concurrent.futures.ThreadPoolExecutor(max_workers=1)

_models: dict = {"emo_tok": None, "emo": None, "stt": None}


@asynccontextmanager
async def lifespan(_app: FastAPI):
    # 서버를 켤 때 텍스트 모델을 미리 읽는다. 첫 대답이 모델 읽기 시간 때문에 시간 초과(분석 실패)로 남지 않도록 한다.
    _pool.submit(_emotion_model).result(timeout=300)
    _stt_model()  # 음성 인식 모델도 미리 읽는다 (첫 음성 대답이 2분 대화 시간을 쓰지 않도록)
    yield


# 말벗 API 라우터. AI 서버 앱(app/main.py)에 붙여서 Swagger 에 나타난다.
router = APIRouter(tags=["말벗 (Companion)"])


async def http_error(request: Request, exc: HTTPException):
    return JSONResponse(status_code=exc.status_code,
                        content={"success": False, "message": exc.detail, "data": None, "code": exc.detail})


def ok(data, message: str = "요청 처리 완료") -> dict:
    return {"success": True, "message": message, "data": data, "code": None}


def _load(path: Path, default):
    return json.loads(path.read_text(encoding="utf-8")) if path.exists() else default


def _save(path: Path, data) -> None:
    path.write_text(json.dumps(data, ensure_ascii=False, indent=2), encoding="utf-8")


def require_key(x_api_key: Optional[str] = Header(default=None)) -> None:
    expected = os.getenv("COMPANION_API_KEY", "")
    if not expected or not x_api_key or not hmac.compare_digest(x_api_key, expected):
        raise HTTPException(status_code=401, detail="AUTH_INVALID_KEY")


def _check_slot(slot: str) -> None:
    if slot not in SLOTS:
        raise HTTPException(status_code=422, detail="SLOT_INVALID")


def _emotion_model():
    from transformers import AutoModelForSequenceClassification, AutoTokenizer

    if _models["emo"] is None:
        mid = "GGARA02/kcelectra-korean-emotion"
        cache = str(ROOT / "data" / "models_cache")
        _models["emo_tok"] = AutoTokenizer.from_pretrained(mid, cache_dir=cache)
        _models["emo"] = AutoModelForSequenceClassification.from_pretrained(mid, cache_dir=cache).eval()
    return _models["emo_tok"], _models["emo"]


def _stt_model():
    from faster_whisper import WhisperModel

    if _models["stt"] is None:
        _models["stt"] = WhisperModel("large-v3", device="cpu", compute_type="int8",
                                      download_root=str(ROOT / "data" / "models_cache"))
    return _models["stt"]


def text_label(text: str, face: Optional[dict] = None) -> dict:
    """대답 한 턴의 결과. 말을 먼저 보고, 말이 부정이거나 애매할 때만 표정(face)을 함께 본다 (decide_turn 규칙)."""
    import torch
    from fusion import FACE_POLARITY, NEG_MIN, TEXT_POLARITY, decide_turn, polarity
    from loneliness_rule import loneliness_signal

    tok, model = _emotion_model()
    with torch.no_grad():
        batch = tok([text], truncation=True, max_length=128, return_tensors="pt")
        p = torch.softmax(model(**batch).logits, dim=-1)[0]
    i = int(torch.argmax(p))
    sig = Signal(model.config.id2label[i], float(p[i]))
    amb = is_ambiguous(text)
    tp = None if amb else polarity(sig, TEXT_POLARITY, NEG_MIN)
    # 외로움·단절 표현은 부정으로 본다 (2026-10-25 규칙, 긍정 판단을 뒤집음)
    if not amb and tp != "neg" and loneliness_signal(text):
        tp = "neg"
    fp = polarity(Signal(str(face["label"]), float(face["score"])), FACE_POLARITY) if face else None
    res = decide_turn(tp, fp)
    return {"textEmotion": sig.label, "textScore": round(sig.score, 3), "textPolarity": tp,
            "facePolarity": fp, "label": res["label"], "reason": res["reason"]}


def _store_answer(ward_id: str, session_id: str, slot: str, text: str, audio_seconds: Optional[float],
                  face: Optional[dict] = None) -> dict:
    now = datetime.now()
    question = _load(QUESTIONS_FILE, {}).get(f"{ward_id}|{slot}")
    try:
        result = _pool.submit(text_label, text).result(timeout=ANALYSIS_TIMEOUT_SEC)
    except concurrent.futures.TimeoutError:
        result = {"textEmotion": None, "label": "분석 실패", "reason": "분석 시간 초과 (다음 시각에는 영향 없음)"}
    except Exception:
        result = {"textEmotion": None, "label": "분석 실패", "reason": "분석 중 오류 (다음 시각에는 영향 없음)"}
    rows = _load(ANSWERS_FILE, [])
    rows.append({"wardId": ward_id, "sessionId": session_id, "slot": slot, "date": now.date().isoformat(),
                 "question": question, "text": text, "audioSeconds": audio_seconds,
                 "label": result["label"], "reason": result["reason"], "face": face,
                 "createdAt": now.isoformat(timespec="seconds")})
    _save(ANSWERS_FILE, rows)
    body = {"slot": slot, "transcript": text, "audioSeconds": audio_seconds, "face": face, **result}
    if result["label"] == "판단 보류":
        same = [r for r in rows if r["wardId"] == ward_id and r["slot"] == slot and r["date"] == now.date().isoformat()]
        body["followUp"] = {"question": FOLLOW_UPS[len(same) % len(FOLLOW_UPS)], "note": "판단 보류: 다음 질문으로 확인합니다"}
    return ok(body, "대답이 저장되었습니다.")


class QuestionBody(BaseModel):
    question: str


class AnswerBody(BaseModel):
    wardId: str
    sessionId: str
    slot: str
    text: str
    # 대답 후 얼굴이 보이는지 확인할 시간(초). 0이면 얼굴 확인을 하지 않는다 (AI 서버 스트림 필요)
    faceCheckSec: int = Field(default=0, ge=0, le=10)


@router.get("/companion/question", dependencies=[Depends(require_key)])
def get_question(wardId: str = Query(...), slot: str = Query(...)) -> dict:
    _check_slot(slot)
    saved = _load(QUESTIONS_FILE, {})
    key = f"{wardId}|{slot}"
    if key not in saved:
        raise HTTPException(status_code=404, detail="QUESTION_NOT_SET")
    return ok({"slot": slot, "question": saved[key]})


@router.put("/companion/questions/{slot}", dependencies=[Depends(require_key)])
def put_question(slot: str, body: QuestionBody, wardId: str = Query(...)) -> dict:
    _check_slot(slot)
    q = body.question.strip()
    if not q or len(q) > 200:
        raise HTTPException(status_code=422, detail="QUESTION_INVALID")
    saved = _load(QUESTIONS_FILE, {})
    saved[f"{wardId}|{slot}"] = q
    _save(QUESTIONS_FILE, saved)
    return ok({"slot": slot, "question": q}, "질문이 저장되었습니다.")


@router.get("/companion/examples", dependencies=[Depends(require_key)])
def get_examples() -> dict:
    return ok(json.loads(EXAMPLES_FILE.read_text(encoding="utf-8")))


@router.post("/companion/expression-check", dependencies=[Depends(require_key)])
async def post_expression_check(frame: UploadFile = File(...)) -> dict:
    """테스트 화면용: 사진 한 장의 얼굴·표정을 읽는다. 사진은 저장하지 않는다."""
    if frame.content_type not in {"image/jpeg", "image/jpg"}:
        raise HTTPException(status_code=415, detail="JPEG 파일만 보낼 수 있습니다")
    data = await frame.read(2 * 1024 * 1024 + 1)
    if len(data) > 2 * 1024 * 1024:
        raise HTTPException(status_code=413, detail="사진이 너무 큽니다 (2MB 이하)")
    import expression_check

    return ok(expression_check.classify_jpeg(data), "표정 확인 완료")


DIALOGUES: dict = {}  # 진행 중인 대화 (목업이라 메모리에만 둔다)


class DialogueStart(BaseModel):
    wardId: str
    sessionId: str
    slot: str


class FaceIn(BaseModel):
    label: str = Field(min_length=1, max_length=40)
    score: float = Field(ge=0, le=1)


class DialogueReply(BaseModel):
    text: str = Field(min_length=1, max_length=500)
    face: Optional[FaceIn] = None  # 같은 순간의 표정 결과 (없으면 말만으로 판단)
    consentPhoto: bool = False  # 어르신 본인이 사진 전달에 동의했을 때만 true (기본 꺼짐)
    photo: Optional[str] = Field(default=None, max_length=90000)  # 확인 필요 턴의 얼굴 사진 한 장 (JPEG, base64)


PHOTO_MAX_BYTES = 60 * 1024


def _check_photo(photo_b64: str) -> str:
    """사진이 JPEG 인지, 크기가 맞는지 확인한다. 저장하지 않는다."""
    import base64
    try:
        raw = base64.b64decode(photo_b64, validate=True)
    except Exception:
        raise HTTPException(status_code=422, detail="PHOTO_INVALID")
    if raw[:2] != b"\xff\xd8":
        raise HTTPException(status_code=422, detail="PHOTO_NOT_JPEG")
    if len(raw) > PHOTO_MAX_BYTES:
        raise HTTPException(status_code=413, detail="PHOTO_TOO_LARGE")
    return photo_b64


def _analyze_turn(text: str, face: Optional[dict] = None) -> dict:
    try:
        return _pool.submit(text_label, text, face).result(timeout=ANALYSIS_TIMEOUT_SEC)
    except concurrent.futures.TimeoutError:
        return {"textEmotion": None, "label": "분석 실패", "reason": "분석 시간 초과 (다음 턴에는 영향 없음)"}
    except Exception:
        return {"textEmotion": None, "label": "분석 실패", "reason": "분석 중 오류 (다음 턴에는 영향 없음)"}


def _persist_turn(ward_id: str, session_id: str, slot: str, dialogue_id: str, row: dict) -> None:
    now = datetime.now()
    rows = _load(ANSWERS_FILE, [])
    rows.append({"wardId": ward_id, "sessionId": session_id, "slot": slot, "dialogueId": dialogue_id,
                 "turn": row["turn"], "date": now.date().isoformat(), "question": row["question"],
                 "text": row["text"], "audioSeconds": None, "label": row["label"], "reason": row["reason"],
                 "face": None, "createdAt": now.isoformat(timespec="seconds")})
    _save(ANSWERS_FILE, rows)


REVIEW_FILE = DATA / "review_replies.json"


def _generate_and_log(question: str, answer: str, turn_index: int, slot: str) -> dict:
    """AI 문장을 만들고, 검수 목록에 쌓는다 (어르신 대답 원문은 저장하지 않음)."""
    out = dlg.generate_reply(question, answer, turn_index)
    problems = dlg.check_reply(out["reply"]) if out["source"] == "model" else []
    item = review_log.make_item(question, out["reply"], out["source"], problems, slot)
    rows = _load(REVIEW_FILE, [])
    rows.append(item)
    _save(REVIEW_FILE, rows)
    out["reviewId"] = item["id"]
    return out


class ReviewDecision(BaseModel):
    decision: str  # approved 또는 rejected


@router.get("/companion/review/replies", dependencies=[Depends(require_key)])
def list_review(status: Optional[str] = Query(default=None)) -> dict:
    if status is not None and status not in review_log.STATUSES:
        raise HTTPException(status_code=422, detail="STATUS_INVALID")
    return ok(review_log.list_items(_load(REVIEW_FILE, []), status))


@router.post("/companion/review/replies/{item_id}", dependencies=[Depends(require_key)])
def decide_review(item_id: str, body: ReviewDecision) -> dict:
    rows = _load(REVIEW_FILE, [])
    try:
        item = review_log.decide(rows, item_id, body.decision)
    except ValueError:
        raise HTTPException(status_code=422, detail="DECISION_INVALID")
    if item is None:
        raise HTTPException(status_code=404, detail="REVIEW_ITEM_NOT_FOUND")
    _save(REVIEW_FILE, rows)
    return ok(item, "검수 결과를 저장했습니다.")


@router.post("/companion/dialogues", dependencies=[Depends(require_key)])
def start_dialogue(body: DialogueStart) -> dict:
    _check_slot(body.slot)
    saved = _load(QUESTIONS_FILE, {})
    key = f"{body.wardId}|{body.slot}"
    if key not in saved:
        raise HTTPException(status_code=404, detail="QUESTION_NOT_SET")
    did = uuid.uuid4().hex
    DIALOGUES[did] = Dialogue(
        dialogue_id=did, ward_id=body.wardId, session_id=body.sessionId, slot=body.slot, question=saved[key],
        analyze=_analyze_turn,
        persist=lambda row: _persist_turn(body.wardId, body.sessionId, body.slot, did, row),
        generate=lambda q, a, i: _generate_and_log(q, a, i, body.slot),
    )
    return ok({"dialogueId": did, "question": saved[key], "timeLimitSec": TIME_LIMIT_SEC,
               "maxTurns": MAX_TURNS, "secondsLeft": TIME_LIMIT_SEC}, "대화를 시작했습니다.")


@router.post("/companion/dialogues/{did}/reply", dependencies=[Depends(require_key)])
def reply_dialogue(did: str, body: DialogueReply) -> dict:
    d = DIALOGUES.get(did)
    if d is None:
        raise HTTPException(status_code=404, detail="DIALOGUE_NOT_FOUND")
    face = body.face.model_dump() if body.face else None
    photo = _check_photo(body.photo) if body.photo else None
    try:
        data = d.reply(body.text.strip(), face)
    except ValueError:
        raise HTTPException(status_code=409, detail="DIALOGUE_ENDED")
    data["photoForBackend"] = _photo_for_backend(data, body.consentPhoto, photo)
    return ok(data, "대답이 저장되었습니다.")


def _photo_for_backend(data: dict, consent: bool, photo: Optional[str]) -> Optional[str]:
    """동의가 있고 확인 필요 턴일 때만 사진을 돌려준다. AI 서버는 사진을 저장하지 않는다 (백엔드가 받아 저장)."""
    if consent and photo and data.get("label") == "확인 필요":
        return photo
    return None


@router.post("/companion/dialogues/{did}/reply/audio", dependencies=[Depends(require_key)])
async def reply_dialogue_audio(did: str, audio: UploadFile = File(...), faceLabel: Optional[str] = Form(None),
                               faceScore: Optional[float] = Form(None), consentPhoto: bool = Form(False),
                               photo: Optional[str] = Form(None)) -> dict:
    """대화 중 음성 대답: WAV 를 글자로 바꾼 뒤 텍스트 대답과 같은 흐름으로 처리한다. 음성은 저장하지 않는다."""
    d = DIALOGUES.get(did)
    if d is None:
        raise HTTPException(status_code=404, detail="DIALOGUE_NOT_FOUND")
    if d.ended:
        raise HTTPException(status_code=409, detail="DIALOGUE_ENDED")
    if not (audio.filename or "").lower().endswith(AUDIO_EXTENSIONS):
        raise HTTPException(status_code=422, detail="AUDIO_FORMAT_UNSUPPORTED")
    raw = await audio.read()
    if not raw:
        raise HTTPException(status_code=422, detail="AUDIO_EMPTY")
    if len(raw) > MAX_AUDIO_BYTES:
        raise HTTPException(status_code=413, detail="AUDIO_TOO_LARGE")
    text, seconds = _transcribe(raw)
    try:
        face = {"label": faceLabel, "score": faceScore} if faceLabel and faceScore is not None else None
        data = d.reply(text, face)
    except ValueError:
        raise HTTPException(status_code=409, detail="DIALOGUE_ENDED")
    data.update({"transcript": text, "audioSeconds": seconds})
    data["photoForBackend"] = _photo_for_backend(data, consentPhoto, _check_photo(photo) if photo else None)
    return ok(data, "대답이 저장되었습니다.")


@router.post("/companion/dialogues/{did}/finish", dependencies=[Depends(require_key)])
def finish_dialogue(did: str) -> dict:
    d = DIALOGUES.get(did)
    if d is None:
        raise HTTPException(status_code=404, detail="DIALOGUE_NOT_FOUND")
    return ok(d.finish(), "대화를 마쳤습니다.")


@router.post("/companion/answers", dependencies=[Depends(require_key)])
def post_answer(body: AnswerBody) -> dict:
    _check_slot(body.slot)
    text = body.text.strip()
    if not text:
        raise HTTPException(status_code=422, detail="ANSWER_EMPTY")
    face = None
    if body.faceCheckSec > 0:
        import face_window

        face = face_window.collect(body.sessionId, body.faceCheckSec)
    return _store_answer(body.wardId, body.sessionId, body.slot, text, None, face=face)


@router.post("/companion/answers/audio", dependencies=[Depends(require_key)])
async def post_answer_audio(wardId: str = Form(...), sessionId: str = Form(...), slot: str = Form(...),
                            audio: UploadFile = File(...)) -> dict:
    _check_slot(slot)
    if not (audio.filename or "").lower().endswith(AUDIO_EXTENSIONS):
        raise HTTPException(status_code=422, detail="AUDIO_FORMAT_UNSUPPORTED")
    raw = await audio.read()
    if not raw:
        raise HTTPException(status_code=422, detail="AUDIO_EMPTY")
    if len(raw) > MAX_AUDIO_BYTES:
        raise HTTPException(status_code=413, detail="AUDIO_TOO_LARGE")
    text, seconds = _transcribe(raw)
    return _store_answer(wardId, sessionId, slot, text, seconds)


AUDIO_EXTENSIONS = (".wav", ".webm", ".ogg", ".mp3", ".m4a", ".mp4")


def _decode_audio(raw: bytes):
    """소리 바이트를 16kHz 모노 배열로 바꾼다. WAV 는 soundfile, 그 밖의 형식(webm·ogg·mp3·m4a)은 PyAV 로 읽는다."""
    import numpy as np

    if raw[:4] == b"RIFF":
        import soundfile as sf

        try:
            arr, sr = sf.read(io.BytesIO(raw), dtype="float32")
        except Exception:
            raise HTTPException(status_code=422, detail="AUDIO_UNREADABLE")
        if arr.ndim > 1:
            arr = arr.mean(axis=1)
        if sr != 16000:
            n = int(len(arr) * 16000 / sr)
            arr = np.interp(np.linspace(0, len(arr) - 1, n), np.arange(len(arr)), arr).astype("float32")
        return arr
    import av

    try:
        container = av.open(io.BytesIO(raw))
        if not container.streams.audio:
            raise ValueError("no audio stream")
        resampler = av.audio.resampler.AudioResampler(format="flt", layout="mono", rate=16000)
        chunks = []
        for frame in container.decode(audio=0):
            for out in resampler.resample(frame):
                chunks.append(out.to_ndarray().reshape(-1))
        for out in resampler.resample(None):
            chunks.append(out.to_ndarray().reshape(-1))
        container.close()
    except Exception:
        raise HTTPException(status_code=422, detail="AUDIO_UNREADABLE")
    if not chunks:
        raise HTTPException(status_code=422, detail="AUDIO_UNREADABLE")
    return np.concatenate(chunks).astype("float32")


def _transcribe(raw: bytes) -> tuple[str, float]:
    """소리 파일을 글자로 바꾼다 (16kHz 모노, faster-whisper). 음성 파일은 저장하지 않는다."""
    arr = _decode_audio(raw)
    seconds = round(len(arr) / 16000, 2)
    segments, _ = _stt_model().transcribe(arr, language="ko", beam_size=5)
    text = " ".join(seg.text.strip() for seg in segments).strip()
    if not text:
        raise HTTPException(status_code=422, detail="SPEECH_NOT_RECOGNIZED")
    return text, seconds


@router.get("/companion/emotions", dependencies=[Depends(require_key)])
def get_emotions(wardId: str = Query(...), date: str = Query(...)) -> dict:
    rows = _load(ANSWERS_FILE, [])
    items = [{"slot": r["slot"], "label": r["label"], "reason": r["reason"]}
             for r in rows if r.get("wardId") == wardId and r["date"] == date]
    labels = [i["label"] for i in items]
    day = dlg.combine_label(labels)  # 가장 많이 나온 결과, 동점이면 확인 필요 (세션 결과와 같은 규칙)
    dist = dlg.distribution(labels)
    summary = dlg.summarize(labels)
    return ok({"date": date, "dayResult": day, "summary": summary, "distribution": dist, "items": items})


# 독립 실행용 목업 서버 (개발·시험). AI 서버에서는 위의 router 만 쓴다.
app = FastAPI(title="companion API mock", version="0.3", lifespan=lifespan)
app.include_router(router)
app.add_exception_handler(HTTPException, http_error)
# 테스트용 화면 (로컬). 같은 주소에서 열리므로 API 호출 설정이 필요 없다.
app.mount("/ui", StaticFiles(directory=HERE / "ui", html=True), name="ui")
