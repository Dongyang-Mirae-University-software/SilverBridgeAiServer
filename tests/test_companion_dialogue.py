from __future__ import annotations

import json
import re
import threading
import time
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

import pytest
from fastapi.testclient import TestClient

from app.core.config import get_settings
from app.services.companion.backend_client import BackendUnavailable
from app.routers.companion_router import get_question_fetcher, get_text_classifier, get_transcriber
from app.services.companion.dialogue_store import store
from app.services.companion.review_store import review_store
from app.services.companion.stt_service import SttResult, SttUnavailable, confidence_from_logprobs
from app.services.companion.text_emotion_service import TextEmotionResult
from app.services.companion.timeouts import call_with_timeout
from app.services.companion.turn_record import send_dialogue_result
from scripts.companion.dev_app import app

KEY = "test-companion-key"
HDR = {"X-API-Key": KEY}


def _wav_bytes(size: int = 64) -> bytes:
    return b"RIFF" + (size).to_bytes(4, "little") + b"WAVE" + b"\x00" * size


def _multipart(name: str, filename: str, content_type: str, data: bytes) -> tuple[bytes, str]:
    boundary = "----testboundary"
    body = (
        f"--{boundary}\r\n"
        f'Content-Disposition: form-data; name="{name}"; filename="{filename}"\r\n'
        f"Content-Type: {content_type}\r\n\r\n"
    ).encode() + data + f"\r\n--{boundary}--\r\n".encode()
    return body, f"multipart/form-data; boundary={boundary}"


def _fetcher_returning(question):
    calls = []

    def fetch(settings, ward_id, slot):
        calls.append((ward_id, slot))
        if isinstance(question, Exception):
            raise question
        return None if question is None else (question, "q-test")  # 백엔드는 (질문, 질문 ID)를 돌려준다

    fetch.calls = calls
    return fetch


def _fake_transcriber(audio: bytes) -> SttResult:
    # 시험용 가짜 STT: 소리가 있으면 고정 문장 (실제 모델은 test_stt_integration.py 에서 본다)
    return SttResult(text="테스트 대답입니다" if audio else "", confidence=0.9, elapsed_ms=1)


@pytest.fixture
def make_client():
    def _make(fetcher, transcriber=_fake_transcriber):
        settings = get_settings().model_copy(update={
            "companion_api_key": KEY,
            "companion_backend_base_url": "",  # 턴 기록 전송은 끈다 (실제 전송 시험은 아래에서 따로 한다)
        })
        app.dependency_overrides[get_settings] = lambda: settings
        app.dependency_overrides[get_question_fetcher] = lambda: fetcher
        app.dependency_overrides[get_transcriber] = lambda: transcriber
        app.dependency_overrides[get_text_classifier] = lambda: (lambda t: TextEmotionResult("기쁨", 0.9, True, {}))
        return TestClient(app)

    yield _make
    app.dependency_overrides.clear()


def _open(client, slot="09:00"):
    return client.post("/companion/dialogues", json={"wardId": "u1", "sessionId": "s1", "slot": slot}, headers=HDR)


def test_open_requires_key(make_client):
    client = make_client(_fetcher_returning("잘 주무셨어요?"))
    r = client.post("/companion/dialogues", json={"wardId": "u1", "sessionId": "s1", "slot": "09:00"})
    assert r.status_code == 401
    assert r.json()["code"] == "AUTH_INVALID_KEY"


def test_open_rejects_unknown_slot(make_client):
    client = make_client(_fetcher_returning("잘 주무셨어요?"))
    r = _open(client, slot="10:00")
    assert r.status_code == 422
    assert r.json()["errorCode"] == "SLOT_INVALID"


def test_open_asks_backend_for_question(make_client):
    fetch = _fetcher_returning("잘 주무셨어요?")
    client = make_client(fetch)
    r = _open(client, slot="09:00")
    assert r.status_code == 200
    data = r.json()["data"]
    assert data["question"] == "잘 주무셨어요?"
    assert data["maxTurns"] == 8
    assert fetch.calls == [("u1", "09:00")]


def test_open_question_not_set_is_404(make_client):
    r = _open(make_client(_fetcher_returning(None)))
    assert r.status_code == 404
    assert r.json()["code"] == "QUESTION_NOT_SET"


def test_open_backend_down_is_503(make_client):
    r = _open(make_client(_fetcher_returning(BackendUnavailable("down"))))
    assert r.status_code == 503
    assert r.json()["code"] == "BACKEND_UNAVAILABLE"


def _dialogue_id(client) -> str:
    return _open(client).json()["data"]["dialogueId"]


def test_text_reply_returns_turn_without_echoing_text(make_client):
    client = make_client(_fetcher_returning("잘 주무셨어요?"))
    did = _dialogue_id(client)
    r = client.post(f"/companion/dialogues/{did}/reply", json={"text": "잘 잤어요"}, headers=HDR)
    assert r.status_code == 200
    data = r.json()["data"]
    assert data["turnNo"] == 1
    assert data["ended"] is False
    assert data["reply"]["isFollowUp"] is True
    assert "transcript" not in data
    assert "잘 잤어요" not in r.text  # 답변 원문은 응답에도 저장에도 남기지 않는다 (백엔드에만)


def test_empty_text_is_422(make_client):
    client = make_client(_fetcher_returning("잘 주무셨어요?"))
    did = _dialogue_id(client)
    r = client.post(f"/companion/dialogues/{did}/reply", json={"text": "   "}, headers=HDR)
    assert r.status_code == 422
    assert r.json()["code"] == "ANSWER_EMPTY"


def test_eighth_answer_ends_dialogue_and_ninth_is_rejected(make_client):
    client = make_client(_fetcher_returning("잘 주무셨어요?"))
    did = _dialogue_id(client)
    last = None
    for _ in range(8):
        last = client.post(f"/companion/dialogues/{did}/reply", json={"text": "네"}, headers=HDR)
        assert last.status_code == 200
    data = last.json()["data"]
    assert data["turnNo"] == 8
    assert data["ended"] is True
    assert data["endReason"] == "최대 턴 수"
    assert data["reply"]["isFollowUp"] is False
    r = client.post(f"/companion/dialogues/{did}/reply", json={"text": "네"}, headers=HDR)
    assert r.status_code == 409
    assert r.json()["code"] == "DIALOGUE_ENDED"


def test_finish_sets_user_stop(make_client):
    client = make_client(_fetcher_returning("잘 주무셨어요?"))
    did = _dialogue_id(client)
    client.post(f"/companion/dialogues/{did}/reply", json={"text": "네"}, headers=HDR)
    r = client.post(f"/companion/dialogues/{did}/finish", headers=HDR)
    assert r.status_code == 200
    data = r.json()["data"]
    assert data["endReason"] == "USER_STOP"  # D16: 그때까지 턴 수와 결과를 그대로 기록
    assert data["turnCount"] == 1


def test_unknown_dialogue_is_404(make_client):
    client = make_client(_fetcher_returning("잘 주무셨어요?"))
    r = client.post("/companion/dialogues/nope/reply", json={"text": "네"}, headers=HDR)
    assert r.status_code == 404
    assert r.json()["code"] == "DIALOGUE_NOT_FOUND"


def test_audio_wav_is_accepted_and_transcribed(make_client):
    client = make_client(_fetcher_returning("잘 주무셨어요?"))
    did = _dialogue_id(client)
    body, ctype = _multipart("audio", "answer.wav", "audio/wav", _wav_bytes())
    r = client.post(f"/companion/dialogues/{did}/reply/audio", content=body,
                    headers={**HDR, "content-type": ctype})
    assert r.status_code == 200
    data = r.json()["data"]
    assert data["turnNo"] == 1
    assert data["transcript"]  # 가짜 STT 결과


def test_audio_stt_missing_model_is_503(make_client):
    def broken(_audio):
        raise SttUnavailable("음성 인식 모델 폴더가 없습니다")

    client = make_client(_fetcher_returning("잘 주무셨어요?"), transcriber=broken)
    did = _dialogue_id(client)
    body, ctype = _multipart("audio", "answer.wav", "audio/wav", _wav_bytes())
    r = client.post(f"/companion/dialogues/{did}/reply/audio", content=body,
                    headers={**HDR, "content-type": ctype})
    assert r.status_code == 503
    assert r.json()["code"] == "STT_UNAVAILABLE"


def test_audio_silence_is_no_response_turn(make_client):
    # 아키텍처 3·4절: 글자가 없으면 무응답 턴 (200, NO_RESPONSE). 422 로 막지 않는다.
    def silent(_audio):
        return SttResult(text="", confidence=0.0, elapsed_ms=1)

    client = make_client(_fetcher_returning("잘 주무셨어요?"), transcriber=silent)
    did = _dialogue_id(client)
    body, ctype = _multipart("audio", "answer.wav", "audio/wav", _wav_bytes())
    r = client.post(f"/companion/dialogues/{did}/reply/audio", content=body,
                    headers={**HDR, "content-type": ctype})
    assert r.status_code == 200
    data = r.json()["data"]
    assert data["turnResult"] == {"label": None, "source": "NO_RESPONSE"}
    assert data["turnNo"] == 1
    assert data["signals"] is None
    assert data["transcript"] == ""


def test_confidence_is_exp_of_mean_logprob():
    # D13: 평균 로그확률을 지수 변환해 0~1 (로그확률 0 이면 1, 낮을수록 0 에 가까움)
    assert confidence_from_logprobs([]) == 0.0
    assert confidence_from_logprobs([0.0]) == pytest.approx(1.0)
    assert confidence_from_logprobs([-0.5, -0.5]) == pytest.approx(0.6065, abs=1e-3)
    assert 0.0 <= confidence_from_logprobs([-30.0]) < 1e-6


def test_audio_unsupported_format_is_422(make_client):
    client = make_client(_fetcher_returning("잘 주무셨어요?"))
    did = _dialogue_id(client)
    body, ctype = _multipart("audio", "answer.ogg", "audio/ogg", b"OggS" + b"\x00" * 64)
    r = client.post(f"/companion/dialogues/{did}/reply/audio", content=body,
                    headers={**HDR, "content-type": ctype})
    assert r.status_code == 422
    assert r.json()["code"] == "AUDIO_FORMAT_UNSUPPORTED"


def test_audio_missing_part_is_422(make_client):
    client = make_client(_fetcher_returning("잘 주무셨어요?"))
    did = _dialogue_id(client)
    body, ctype = _multipart("other", "x.wav", "audio/wav", _wav_bytes())
    r = client.post(f"/companion/dialogues/{did}/reply/audio", content=body,
                    headers={**HDR, "content-type": ctype})
    assert r.status_code == 422
    assert r.json()["code"] == "AUDIO_MISSING"


def test_audio_over_10mb_is_413(make_client):
    client = make_client(_fetcher_returning("잘 주무셨어요?"))
    did = _dialogue_id(client)
    big = _wav_bytes(10 * 1024 * 1024 + 10)
    body, ctype = _multipart("audio", "big.wav", "audio/wav", big)
    r = client.post(f"/companion/dialogues/{did}/reply/audio", content=body,
                    headers={**HDR, "content-type": ctype})
    assert r.status_code == 413
    assert r.json()["code"] == "AUDIO_TOO_LARGE"


# ---- 아래는 S7·S8·S9·S10·S12 항목 시험 (2026-11-02 결정 반영) ----

def _silent_audio_post(client, did):
    body, ctype = _multipart("audio", "answer.wav", "audio/wav", _wav_bytes())
    return client.post(f"/companion/dialogues/{did}/reply/audio", content=body,
                       headers={**HDR, "content-type": ctype})


def test_three_consecutive_no_response_turns_end_dialogue(make_client):
    # D15 (사용자 결정): 3번 연속 무응답이면 대화 종료
    client = make_client(_fetcher_returning("잘 주무셨어요?"),
                         transcriber=lambda _a: SttResult(text="", confidence=0.0, elapsed_ms=1))
    did = _dialogue_id(client)
    last = [_silent_audio_post(client, did).json()["data"] for _ in range(3)]
    assert last[1]["ended"] is False
    assert last[2]["ended"] is True
    assert last[2]["endReason"] == "연속 무응답"
    assert _silent_audio_post(client, did).status_code == 409


def test_an_answer_resets_the_no_response_streak(make_client):
    texts = iter(["", "", "네 잘 잤어요", "", ""])
    client = make_client(_fetcher_returning("잘 주무셨어요?"),
                         transcriber=lambda _a: SttResult(text=next(texts), confidence=0.9, elapsed_ms=1))
    did = _dialogue_id(client)
    results = [_silent_audio_post(client, did).json()["data"] for _ in range(5)]
    assert [r["ended"] for r in results] == [False, False, False, False, False]


def test_turn_order_is_stt_then_text_emotion(make_client):
    # S7-7: 음성 글자 변환이 끝난 뒤에 글자 감정을 본다
    events = []
    client = make_client(_fetcher_returning("잘 주무셨어요?"),
                         transcriber=lambda a: (events.append("stt"), _fake_transcriber(a))[1])
    app.dependency_overrides[get_text_classifier] = lambda: (
        lambda t: (events.append("text"), TextEmotionResult("기쁨", 0.9, True, {}))[1])
    did = _dialogue_id(client)
    body, ctype = _multipart("audio", "answer.wav", "audio/wav", _wav_bytes())
    r = client.post(f"/companion/dialogues/{did}/reply/audio", content=body,
                    headers={**HDR, "content-type": ctype})
    assert r.status_code == 200
    assert events == ["stt", "text"]


def test_transcript_is_returned_but_not_kept_on_ai_side(make_client):
    # S8-2 (D10): 원문은 응답과 백엔드 전송에만 실린다. AI 저장소에는 남지 않는다.
    client = make_client(_fetcher_returning("잘 주무셨어요?"))
    did = _dialogue_id(client)
    body, ctype = _multipart("audio", "answer.wav", "audio/wav", _wav_bytes())
    r = client.post(f"/companion/dialogues/{did}/reply/audio", content=body,
                    headers={**HDR, "content-type": ctype})
    assert r.json()["data"]["transcript"] == "테스트 대답입니다"
    turn = store.get(did).turns[0]
    assert set(vars(turn)) == {"turnNo", "label", "source"}
    assert all("테스트 대답입니다" not in item["text"] for item in review_store.list_items())


def test_key_is_never_echoed_in_responses(make_client):
    # S9-4: 키 값은 응답 본문에 나오지 않는다 (잘못된 키 응답 포함)
    client = make_client(_fetcher_returning("잘 주무셨어요?"))
    bad = client.post("/companion/dialogues", json={"wardId": "u1", "sessionId": "s1", "slot": "09:00"},
                      headers={"X-API-Key": "wrong-key"})
    ok = _open(client)
    status = client.get("/companion/status", headers=HDR)
    for r in (bad, ok, status):
        assert KEY not in r.text


def test_dialogue_result_is_posted_once_at_the_end_without_transcript(make_client):
    # S8-5 (대화 종료 시 1건), S8-6 (prompt_version), 원문 제외 (10/26 합의). 로컬 임시 백엔드에 실제로 보내 본다.
    received = []

    class Handler(BaseHTTPRequestHandler):
        def do_POST(self):
            size = int(self.headers.get("content-length", 0))
            received.append((self.path, self.headers.get("X-API-Key"), json.loads(self.rfile.read(size))))
            self.send_response(201)
            self.end_headers()

        def log_message(self, *args):
            pass

    server = ThreadingHTTPServer(("127.0.0.1", 0), Handler)
    threading.Thread(target=server.serve_forever, daemon=True).start()
    try:
        client = make_client(_fetcher_returning("잘 주무셨어요?"))
        base = f"http://127.0.0.1:{server.server_address[1]}"
        app.dependency_overrides[get_settings] = lambda: get_settings().model_copy(update={
            "companion_api_key": KEY, "companion_backend_base_url": base,
            "companion_backend_api_key": "backend-test-key", "companion_record_timeout_sec": 5.0})
        did = _dialogue_id(client)
        r = client.post(f"/companion/dialogues/{did}/reply", json={"text": "오늘 아침에 공원을 한 바퀴 걷고 왔어요"}, headers=HDR)
        assert r.status_code == 200
        assert received == []  # 턴마다 보내지 않는다
        client.post(f"/companion/dialogues/{did}/finish", headers=HDR)
        client.post(f"/companion/dialogues/{did}/finish", headers=HDR)  # 두 번 끝내도 한 번만 보낸다
    finally:
        server.shutdown()
    assert len(received) == 1
    path, key, record = received[0]
    assert path == "/companion/dialogue-results"
    assert key == "backend-test-key"
    assert record["mood"] == "기쁨"
    assert record["turnCount"] == 1
    assert record["noResponseCount"] == 0
    assert record["endReason"] == "USER_STOP"
    assert record["alertLevel"] == "INFO"
    assert record["endedAt"]
    assert "transcript" not in record
    assert "오늘 아침에" not in json.dumps(record, ensure_ascii=False)
    assert record["promptVersion"]["signals"] == "signals-v1"
    assert record["promptVersion"]["reply"] in ("reply-v1", "fallback-v1")


def test_undecided_dialogue_gives_null_mood(make_client):
    # 답이 짧아 말로 판단하지 못한 대화는 mood 를 null 로 보낸다 (화면에서는 '확인할 수 없어요')
    client = make_client(_fetcher_returning("잘 주무셨어요?"))
    did = _dialogue_id(client)
    client.post(f"/companion/dialogues/{did}/reply", json={"text": "네"}, headers=HDR)
    r = client.post(f"/companion/dialogues/{did}/finish", headers=HDR)
    data = r.json()["data"]
    assert data["mood"] is None
    assert data["result"] == "판단 보류"  # 이전 응답 형식 호환용
    assert data["endedAt"]


def test_send_dialogue_result_without_address_does_nothing():
    settings = get_settings().model_copy(update={"companion_backend_base_url": ""})
    assert send_dialogue_result(settings, {"dialogueId": "x"}) is False


def test_call_with_timeout_gives_up_after_limit():
    # S9-1: 제한 시간을 넘긴 작업은 TimeoutError 로 끊는다
    with pytest.raises(TimeoutError):
        call_with_timeout(time.sleep, 0.05, 0.5)
    assert call_with_timeout(lambda x: x + 1, 1.0, 1) == 2


def test_error_codes_in_contract_exist_in_code():
    # S9-2: 계약 문서의 오류 코드 표가 실제 코드와 맞는지 본다
    root = Path(__file__).resolve().parents[1]
    doc = (root / "docs" / "companion-contract.md").read_text(encoding="utf-8")
    codes = re.findall(r"^\| \d{3} \| ([A-Z_]+) \|", doc, flags=re.M)
    source = "".join(p.read_text(encoding="utf-8") for p in (root / "app").rglob("*.py")
                     if "companion" in p.name or "companion" in str(p.parent))
    assert codes
    assert [c for c in codes if c not in source] == []


def _companion_sources() -> list[Path]:
    root = Path(__file__).resolve().parents[1]
    return [root / "app" / "routers" / "companion_router.py",
            *(root / "app" / "services" / "companion").glob("*.py")]


def test_companion_code_does_not_print_or_log_user_text():
    # S9-3: 말벗 코드는 print 를 쓰지 않고, 로그에 원문·음성·키를 넣지 않는다
    logged_secret = re.compile(r"(logger|logging)\.\w+\(.*(transcript|\btext\b|audio|api_key|KEY)")
    offenders = []
    for path in _companion_sources():
        for no, line in enumerate(path.read_text(encoding="utf-8").splitlines(), 1):
            if "print(" in line or logged_secret.search(line):
                offenders.append(f"{path.name}:{no}")
    assert offenders == []


def test_companion_code_does_not_read_service_database():
    # S10-6: AI 서버는 백엔드(서비스) DB를 읽지 않는다. 말벗 코드는 자기 저장소만 쓴다.
    pattern = re.compile(r"^\s*(from|import) app\.(database|models)\.(\w+)")
    allowed = {"companion_base"}
    offenders = []
    for path in _companion_sources():
        for line in path.read_text(encoding="utf-8").splitlines():
            m = pattern.match(line)
            if m and m.group(3) not in allowed:
                offenders.append(f"{path.name}: {line.strip()}")
    assert offenders == []


def test_risk_flag_turn_is_marked_in_review_list():
    # S7-5: 위험 플래그가 켜진 턴은 검수 항목에 risk 표시가 붙는다
    from app.routers.companion_router import Dialogue, _queue_for_review

    import uuid

    dialogue_id = uuid.uuid4().hex  # 실행할 때마다 새 ID (개발 DB에 예전 기록이 남아 있어도 섞이지 않게)
    d = Dialogue(dialogue_id, "u1", "s1", "09:00", "질문", 8)
    result = {"reply": {"text": "괜찮으세요?"}, "replySource": "FALLBACK",
              "signals": {"riskFlag": True}}
    _queue_for_review(d, result)
    _queue_for_review(d, {**result, "signals": {"riskFlag": False}})
    items = [i for i in review_store.list_items() if i["dialogueId"] == dialogue_id]
    assert sorted(i["risk"] for i in items) == [False, True]


def test_daily_summary_sentence_and_request_schema():
    # S10-5 (규칙 기반 요약 문장), S10-1 (요청 양식)
    from pydantic import ValidationError

    from app.schemas.companion_daily import DailySummaryRequest, SlotSummary
    from app.services.companion.daily_summary import summary_sentence

    assert summary_sentence([]) == "오늘은 말벗 대화가 없었습니다."
    slots = [SlotSummary(slot="09:00", label="기쁨"), SlotSummary(slot="12:00", label="기쁨"),
             SlotSummary(slot="18:00", noResponse=True)]
    assert summary_sentence(slots) == ("오늘 3번 대화 중 가장 많이 나온 기분은 '기쁨'입니다."
                                       " 답을 주시지 않은 경우는 1번입니다.")
    DailySummaryRequest(wardId="u1", date="2026-11-02", slots=slots)
    try:
        DailySummaryRequest(wardId="u1", date="2026-11-02", slots=[SlotSummary(slot="10:00")])
        raise AssertionError("잘못된 시각이 통과했습니다")
    except ValidationError:
        pass


def test_no_time_limit_and_no_seconds_left_field(make_client):
    # S7-2 (D1): 시간 제한 없음. 응답에 secondsLeft 를 두지 않는다
    assert get_settings().companion_session_limit_sec == 0
    client = make_client(_fetcher_returning("잘 주무셨어요?"))
    did = _dialogue_id(client)
    r = client.post(f"/companion/dialogues/{did}/reply", json={"text": "네 잘 잤어요"}, headers=HDR)
    assert "secondsLeft" not in r.text


def test_eighth_answer_is_last_turn_with_closing_reply(make_client):
    # S7-3: 8번째 답은 마지막 턴이라 마무리 인사(isFollowUp false)이고, 대화는 최대 턴 수로 끝난다
    client = make_client(_fetcher_returning("잘 주무셨어요?"))
    did = _dialogue_id(client)
    last = None
    for _ in range(get_settings().companion_max_turns):
        last = client.post(f"/companion/dialogues/{did}/reply", json={"text": "네 잘 잤어요"}, headers=HDR).json()["data"]
    assert last["reply"]["isFollowUp"] is False
    assert last["ended"] is True
    assert last["endReason"] == "최대 턴 수"


def test_dialogue_result_carries_question_id_and_screen_mood():
    # 노션 요구: 질문 ID를 되돌려 주기, 기분 값은 화면 4개(기쁨·평온·걱정·우울)로 묶기
    from app.services.companion.dialogue_store import Dialogue, TurnRecord
    from app.services.companion.turn_record import build_dialogue_result

    d = Dialogue("dlg-test-1", "ward-x", "s1", "09:00", "질문", 8, questionId="q-0900")
    d.turns = [TurnRecord(1, "슬픔", "TEXT"), TurnRecord(2, "화남", "TEXT"), TurnRecord(3, "기쁨", "TEXT"),
               TurnRecord(4, None, "NO_RESPONSE")]
    result = build_dialogue_result(d)
    assert result["questionId"] == "q-0900"
    assert result["mood"] == "걱정"  # 슬픔·화남 두 번 → 걱정이 최빈
    assert result["noResponseCount"] == 1
    assert result["turnCount"] == 4


def test_undecided_only_dialogue_has_null_mood_and_screen_moods_for_others():
    from app.services.companion.dialogue_store import Dialogue, TurnRecord
    from app.services.companion.turn_record import build_dialogue_result

    only_undecided = Dialogue("dlg-test-2", "ward-x", "s1", "09:00", "질문", 8)
    only_undecided.turns = [TurnRecord(1, "판단 보류", "CONFLICT")]
    assert build_dialogue_result(only_undecided)["mood"] is None
    sad = Dialogue("dlg-test-3", "ward-x", "s1", "09:00", "질문", 8)
    sad.turns = [TurnRecord(1, "우울", "TEXT")]
    assert build_dialogue_result(sad)["mood"] == "우울"


def test_ward_data_deletion_removes_only_that_ward(make_client):
    # 탈퇴 시 말벗 데이터 삭제: 그 피보호자의 대화와 검수 항목만 지운다
    from app.services.companion.review_store import review_store

    client = make_client(_fetcher_returning("잘 주무셨어요?"))
    keep = _dialogue_id(client)
    client.post(f"/companion/dialogues/{keep}/reply", json={"text": "오늘 아침에 공원을 한 바퀴 걷고 왔어요"}, headers=HDR)
    review_store.add("other-dialogue", 1, "다른 피보호자 응답", "reply-v1", ward_id="ward-other")
    r = client.delete("/companion/wards/ward-demo-test-delete/data", headers=HDR)
    assert r.status_code == 200
    assert r.json()["data"] == {"wardId": "ward-demo-test-delete", "dialogues": 0, "reviewItems": 0}
    r2 = client.delete("/companion/wards/ward-demo-test-delete/data", headers=HDR)
    assert r2.status_code == 200
    assert r2.json()["data"]["dialogues"] == 0
    assert any(i["dialogueId"] == "other-dialogue" for i in review_store.list_items())
