from __future__ import annotations

from collections import Counter

from app.core.config import Settings
from app.services.companion.dialogue_store import Dialogue
from app.services.companion.face_service import face_label_ko
from app.services.companion.reply_service import reply_prompt_version
from app.services.companion.signals import PROMPT_VERSION as SIGNALS_PROMPT_VERSION



# 화면에 쓰는 기분 값 (노션 요구: 기쁨·평온·걱정·우울). 턴 라벨을 이 값으로 묶는다.
# 제안 매핑: 슬픔·화남 → 걱정. 팀이 정하면 이 표만 고치면 된다.
SCREEN_MOOD = {"기쁨": "기쁨", "평온": "평온", "우울": "우울", "슬픔": "걱정", "화남": "걱정"}

# 백엔드 결정 (2026-11-02): 기분을 코드로 보낸다. 판단 못 하면 UNKNOWN (백엔드가 0으로 세지 않도록)
MOOD_CODE = {"기쁨": "JOY", "평온": "CALM", "걱정": "WORRY", "우울": "DEPRESSED"}


def build_dialogue_result(d: Dialogue) -> dict:
    """대화 1회 결과. 원문(대답 글)은 넣지 않는다 (10/26 합의: 원문은 AI 쪽에만 둔다).

    mood: 답한 턴 중 가장 많이 나온 기분. 판단 보류뿐이거나 답이 없으면 None (화면에서는 '확인할 수 없어요').
    alertLevel: 위험 플래그가 있었으면 URGENT, 없으면 INFO. 부정 비율 기반 NOTICE는 기준선이 필요해서
    백엔드가 계산한다 (AI는 대화 기록을 쌓지 않는다).
    """
    answered = [SCREEN_MOOD[t.label] for t in d.turns if t.label in SCREEN_MOOD]
    mood = Counter(answered).most_common(1)[0][0] if answered else None
    no_response = sum(1 for t in d.turns if t.label is None)
    # 대화 전체에서 가장 많이 인식된 감정과 표정 (턴마다 나온 횟수). 판단 보류는 '가장 많이'에서 뺀다
    emotion_counts = Counter(t.label for t in d.turns if t.label is not None)
    decided_emotions = Counter({k: v for k, v in emotion_counts.items() if k != "판단 보류"})
    face_counts = Counter(t.face for t in d.turns if t.face)
    turn_summaries = [t.summary for t in d.turns if t.summary]
    return {
        "emotionCounts": dict(emotion_counts),  # 예: {"기쁨": 2, "평온": 1, "판단 보류": 1}
        "dominantEmotion": decided_emotions.most_common(1)[0][0] if decided_emotions else None,
        "faceCounts": dict(face_counts),  # 표정 모델 라벨별 횟수 (예: {"Happiness": 2, "Neutral": 1})
        "dominantFace": face_counts.most_common(1)[0][0] if face_counts else None,
        # 가장 많이 나온 표정의 한국어 이름 (11/13). 없거나 모르는 값이면 None. faceCounts 키는 영어 그대로
        "dominantFaceKo": face_label_ko(face_counts.most_common(1)[0][0]) if face_counts else None,
        # 어르신이 답한 턴 수 (무응답 제외, 판단 보류는 답한 것으로 센다). 응답률 = answeredCount ÷ 질문 수 (노션 요청 1-3)
        "answeredCount": d.turnNo - no_response,
        "dialogueId": d.dialogueId,
        "wardId": d.wardId,
        "questionId": d.questionId,  # 백엔드가 준 질문 ID를 그대로 돌려준다 (응답률 계산용)
        "slot": d.slot,
        "endedAt": d.endedAt,
        "endReason": d.endReason,
        "turnCount": d.turnNo,
        "noResponseCount": no_response,
        # 백엔드 결정 D (2026-11-03): 기분은 영문 코드로 보낸다. 판단 못 하면 UNKNOWN (null 아님)
        "mood": MOOD_CODE.get(mood, "UNKNOWN"),
        "moodCode": MOOD_CODE.get(mood, "UNKNOWN"),  # 같은 값 (이전 응답과의 호환)
        "moodKo": mood,  # 화면 표시용 한글 (판단 못 하면 null)
        "turnSummaries": turn_summaries,  # 턴마다 한 줄 요약 (대답 원문 없음)
        "needsCheckCount": sum(1 for t in d.turns if t.needsCheck),  # 음성 인식이 불확실해 다시 확인한 턴 수
        "summary": " · ".join(turn_summaries)[:300] if turn_summaries else None,  # 대화 전체 요약 (보호자 화면용)
        "riskFlag": d.riskSeen,
        "alertLevel": "URGENT" if d.riskSeen else "INFO",
        "promptVersion": {"reply": reply_prompt_version(), "signals": SIGNALS_PROMPT_VERSION},
    }
