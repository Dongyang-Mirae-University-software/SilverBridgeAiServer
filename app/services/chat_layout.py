"""답 모양 - 핵심 한 줄·짧은 본문·칸별 카드·줄 머리 이모지(2026-11-01 가독성).

모든 판정(위험도·119·응급·자해·위기·약 + 술·되묻기·진료과·빼기 규칙·이어서 물어볼 질문)이 끝난 답을 받아
**모양만** 바꾼다. 문장을 새로 만들지 않고, 상투 문장 "말씀해 주신 내용을 정리했어요." 말고는 어떤 문장도 버리지 않는다.

- layout: {"headline": str|None, "body": [...], "questions": [...], "homeCare": [...], "visitHospitalIf": [...]}
  FE 는 layout 이 있으면 layout 으로, 없으면 reply 로 그린다(옛 AI·옛 기록 호환).
- reply: layout 을 한 줄에 한 문장으로 펼친 글(줄마다 이모지). 기록·옛 FE·이력은 이 글을 쓴다.
- 응급·자해·위기 답: 맨 위 "🚨 119(위기는 109)" 한 줄, 다른 줄에는 이모지를 붙이지 않는다.
- 예약 흐름(버튼·도구 결과) 답은 건드리지 않는다(layout 없음).
"""
from __future__ import annotations

import re
from typing import Any

from app.services import chat_keywords as kw
from app.services import medical_llm_service as _mls

_SENTENCE_SPLIT = re.compile(r"(?<=[.!?。])\s+")
_FILLER = re.compile(r"^말씀해 주신 내용을 정리했어요[.!]?$")
_SUMMARY_LINE = re.compile(r"^·\s*(증상|특징)\s*:\s*(.+)$")
_QUESTION_LINE = re.compile(r"^·\s*(.+)$")
_CALL_119 = re.compile(r"119")
_CALL_109 = re.compile(r"109")
_HEADLINE_CALL = re.compile(r"전화|연락")
# 따옴표 안 낱말은 안내할 말의 예시다("병원 찾아줘"라고 말씀해 주세요) - 병원 문장·핵심 줄 판단에서 뺀다(2026-11-01 점검 M-2).
_QUOTED = re.compile(r"\"[^\"]*\"|“[^”]*”|'[^']*'|‘[^’]*’")
BODY_MAX_LINES = 4
# 피보호자(WARD) 본인 답(2026-11-04): 모델이 쓴 문장은 핵심 줄 포함 3문장·공백 포함 120자까지.
# 119·109 문장과 서버 고정 문장(약·약 + 술·진료 권유 보장·위험 신호)은 세지 않고 늘 남긴다.
WARD_MODEL_MAX_SENTENCES = 3
WARD_MODEL_MAX_CHARS = 120


def _sentences(line: str) -> list[str]:
    return [s.strip() for s in _SENTENCE_SPLIT.split(line or "") if s.strip()]


def _same(a: str, b: str) -> bool:
    return re.sub(r"\s+", "", a) == re.sub(r"\s+", "", b)


def _in(sentence: str, pool: list[str]) -> bool:
    return any(_same(sentence, p) for p in pool)


def _kind(result: dict[str, Any]) -> str | None:
    if result.get("intent") == "hospital_reservation":
        return None
    if result.get("type") not in (None, "message") or result.get("ui") or result.get("tool"):
        return None
    if not isinstance(result.get("reply"), str) or not result["reply"].strip():
        return None
    meta = result.get("meta") if isinstance(result.get("meta"), dict) else {}
    if meta.get("crisisFollowup"):
        return "crisis"
    if meta.get("selfHarm") or result.get("emergencyWarning"):
        return "emergency"
    if meta.get("followupAsked"):
        return "followup"
    return "general"


def _hospital(sentence: str) -> bool:
    """병원·의사·119 이야기인가(따옴표 안 낱말 제외)."""
    return bool(kw.LAYOUT_HOSPITAL.search(_QUOTED.sub("", sentence)))


def _without_emoji(text: str) -> str:
    """모델이 문장 안에 쓴 이모지를 지운 글(줄 구분은 그대로). 줄 머리 이모지는 이 뒤에 허용 목록에서 하나만 붙인다
    - 한 줄 하나·허용 목록 19개·응급 답 🚨 하나 규칙이 모델 출력과 상관없이 지켜지게(2026-11-01 점검 M-1)."""
    def clean(line: str) -> str:
        plain = kw.strip_layout_emoji(line)
        return line if plain == line else re.sub(r"[ \t]{2,}", " ", plain).strip()  # 이모지 없는 줄은 원문 그대로
    return "\n".join(clean(line) for line in (text or "").split("\n"))


def _emoji(sentence: str) -> str:
    return f"{kw.layout_emoji(sentence)} {sentence}"


def _urgent_layout(
    reply: str, marker: re.Pattern[str], parts: dict[str, Any] | None = None, concise: bool = False
) -> dict[str, Any]:
    """응급·자해·위기: marker(119/109) 전화 문장 하나를 🚨 핵심 줄로, 나머지는 순서대로 이모지 없이 본문."""
    lines: list[str] = []
    for line in reply.split("\n"):
        m = _SUMMARY_LINE.match(line.strip())
        if m:
            lines.append(f"{m.group(1)}: {m.group(2).strip()}")
            continue
        lines.extend(s for s in _sentences(line) if not _FILLER.match(s))
    # "전화" 문장 > "연락" 문장 > marker 가 든 첫 문장 순으로 고른다(119 "전화하세요"가 할 일이다).
    headline_idx = next(
        (i for i, s in enumerate(lines) if marker.search(s) and "전화" in s),
        next((i for i, s in enumerate(lines) if marker.search(s) and _HEADLINE_CALL.search(s)),
             next((i for i, s in enumerate(lines) if marker.search(s)), None)),
    )
    headline = None
    if headline_idx is not None:
        headline = f"{kw.LAYOUT_EMERGENCY_EMOJI} {lines.pop(headline_idx)}"
    # 행동·119 안내(서버 문장)를 공감·증상 줄보다 먼저 - 순서만 바꾼다(공감은 생성 경로의 공감 칸 문장).
    empathy_pool = _sentences(str((parts or {}).get("empathy") or ""))
    summary = [s for s in lines if re.match(r"^(증상|특징): ", s)]
    empathy = [s for s in lines if s not in summary and _in(s, empathy_pool)]
    rest = [s for s in lines if s not in summary and s not in empathy]
    # 피보호자 본인 응급 답은 공감·증상 줄만 뺀다 - 119 줄·행동 문장·경고 박스는 그대로(concise).
    return {"headline": headline, "body": rest if concise else rest + empathy + summary, "questions": [], "homeCare": [], "visitHospitalIf": []}


def _followup_layout(reply: str) -> dict[str, Any]:
    body, questions, hospital = [], [], []
    headline = None
    for line in reply.split("\n"):
        text = line.strip()
        if not text:
            continue
        q = _QUESTION_LINE.match(text)
        if q:
            questions.append(f"{kw.LAYOUT_QUESTION_EMOJI} {q.group(1).strip()}")  # 질문 줄은 "?" 뒤 괄호까지 한 줄
            continue
        for s in _sentences(text):
            if headline is None and _hospital(s) and not kw.LAYOUT_CONDITIONAL.search(s):
                headline = _emoji(s)
            elif _hospital(s):
                hospital.append(_emoji(s))
            else:
                body.append(f"{kw.LAYOUT_EMPATHY_EMOJI} {s}")
    return {"headline": headline, "body": body, "questions": questions, "homeCare": [], "visitHospitalIf": hospital}


def _fixed_patterns() -> list[re.Pattern[str]]:
    """여러 문장으로 된 고정 문장(약 + 술 제외) - 한 카드에 묶는다(설명과 할 일이 갈라지지 않게)."""
    out = []
    for f in _mls.FIXED_SENTENCES:
        if f.name.startswith("_DRUG_ALCOHOL_RULES."):
            continue
        for text in filter(None, (str(f), f.proxy)):
            if len(_sentences(text.replace("{department}", "과"))) < 2:
                continue
            out.append(re.compile("^" + re.escape(text).replace(re.escape("{department}"), "[가-힣]+") + "$"))
    return out


_MULTI_FIXED = _fixed_patterns()


def _single_fixed_patterns() -> list[re.Pattern[str]]:
    """서버 고정 문장(약 + 술 포함, 본인·대리 말투 모두)을 한 문장씩 - 피보호자 답 줄이기에서 지우지 않을 문장."""
    out = []
    for f in _mls.FIXED_SENTENCES:
        for text in filter(None, (str(f), f.proxy)):
            for sentence in _sentences(text):
                out.append(re.compile("^" + re.escape(sentence).replace(re.escape("{department}"), "[가-힣]+") + "$"))
    return out


_SINGLE_FIXED = _single_fixed_patterns()


def _protected(sentence: str, drug_alcohol: list[str]) -> bool:
    """피보호자 답에서도 남길 문장: 119·109 안내, 서버 고정 문장, 약 + 술 문장."""
    return bool(
        _CALL_119.search(sentence) or _CALL_109.search(sentence)
        or any(p.match(sentence) for p in _SINGLE_FIXED) or _in(sentence, drug_alcohol)
    )


def _ward_trim(placed: list[tuple[str, str]], headline: str | None, drug_alcohol: list[str]) -> tuple[list[tuple[str, str]], int]:
    """피보호자 본인 답: 공감·증상·특징 줄을 빼고, 모델 문장은 핵심 줄을 포함해 3문장·120자까지(넘치면 뒤에서부터 뺀다).
    지키는 문장(_protected)은 세지 않고 자리 그대로 둔다. 핵심 줄(과 안내)은 길이와 상관없이 남긴다."""
    count = 1 if headline else 0
    chars = len(headline) if headline else 0
    kept: list[tuple[str, str]] = []
    dropped = 0
    for where, s in placed:
        if where in ("empathy", "summary"):
            dropped += 1
            continue
        if _protected(s, drug_alcohol):
            kept.append((where, s))
            continue
        if count < WARD_MODEL_MAX_SENTENCES and (chars + len(s) <= WARD_MODEL_MAX_CHARS or count == 0):
            kept.append((where, s))
            count += 1
            chars += len(s)
        else:
            dropped += 1
    return kept, dropped


def _general_layout(reply: str, parts: dict[str, Any], drug_alcohol: list[str], concise: bool = False) -> dict[str, Any]:
    empathy = _sentences(str(parts.get("empathy") or ""))
    advice = _sentences(str(parts.get("advice") or ""))
    body: list[str] = []
    home: list[str] = []
    hospital: list[str] = []
    headline_candidates: list[tuple[int, str]] = []  # (우선순위, 문장)
    placed: list[tuple[str, str]] = []  # (자리, 문장)

    for line in reply.split("\n"):
        text = line.strip()
        if not text:
            continue
        m = _SUMMARY_LINE.match(text)
        if m:
            placed.append(("summary", f"{m.group(1)}: {m.group(2).strip()}"))
            continue
        if any(p.match(text) for p in _MULTI_FIXED):
            group = _sentences(text)
            where = "hospital" if any(_hospital(x) for x in group) else "home"
            placed.extend((where, x) for x in group)
            continue
        for s in _sentences(text):
            if _FILLER.match(s):
                continue
            if _in(s, empathy):
                placed.append(("empathy", s))
            elif _in(s, drug_alcohol):
                if _hospital(s) and not _CALL_119.search(s):
                    headline_candidates.append((0, s))
                placed.append(("body", s))
            elif _hospital(s):
                if not _CALL_119.search(s) and not kw.LAYOUT_CONDITIONAL.search(s):
                    headline_candidates.append((1 if _in(s, advice) else 2, s))
                placed.append(("hospital", s))
            elif _in(s, advice) or parts.get("structured"):
                placed.append(("home", s))  # 상담 답에서 병원 이야기가 아닌 권고·고정 안내
            else:
                placed.append(("body", s))  # 일반 대화·대체 응답의 문장

    headline = None
    if headline_candidates:
        headline = min(headline_candidates, key=lambda x: x[0])[1]
        idx = next(i for i, (_, s) in enumerate(placed) if s == headline)
        placed.pop(idx)
    ward_dropped = 0
    if concise:
        placed, ward_dropped = _ward_trim(placed, headline, drug_alcohol)

    tagged: list[tuple[str, str]] = []  # 본문 (자리표, 줄)
    for where, s in placed:
        if where == "empathy":
            tagged.append(("empathy", f"{kw.LAYOUT_EMPATHY_EMOJI} {s}"))
        elif where == "summary":
            tagged.append(("summary", f"{kw.LAYOUT_SUMMARY_EMOJI} {s}"))
        elif where == "body":
            tagged.append(("body", _emoji(s)))
        elif where == "home":
            home.append(_emoji(s))
        else:
            hospital.append(_emoji(s))
    # 본문은 4줄까지 - 첫 줄과 증상·특징 줄은 남기고, 넘치는 줄은 버리지 않고 뒤에서부터 "집에서 이렇게" 칸 맨 앞으로 옮긴다.
    moved: list[str] = []
    while len(tagged) > BODY_MAX_LINES:
        idx = next((i for i in range(len(tagged) - 1, 0, -1) if tagged[i][0] != "summary"), None)
        if idx is None:
            break
        moved.insert(0, tagged.pop(idx)[1])
    body = [line for _, line in tagged]
    home = moved + home
    return {
        "headline": _emoji(headline) if headline else None,
        "body": body,
        "questions": [],
        "homeCare": home,
        "visitHospitalIf": hospital,
        **({"_wardDropped": ward_dropped} if concise else {}),
    }


def flatten(layout: dict[str, Any]) -> str:
    """layout 을 한 줄에 한 문장으로 펼친 reply."""
    lines = [layout.get("headline")] if layout.get("headline") else []
    for key in ("body", "questions", "homeCare", "visitHospitalIf"):
        lines.extend(layout.get(key) or [])
    return "\n".join(lines)


def split_warning(items: list[str] | None) -> list[str]:
    """응급 경고 박스는 내용 그대로, 한 문장씩 나눈 목록으로(문장 3개가 한 항목에 들어 있었다)."""
    out: list[str] = []
    for item in items or []:
        out.extend(_sentences(str(item)) or [str(item)])
    return out


def apply(result: dict[str, Any], drug_alcohol_sentences: list[str] | None = None, concise: bool = False) -> dict[str, Any]:
    """result 의 reply 를 새 모양으로 바꾸고 layout 을 붙인다(제자리 수정 후 반환). 판정 값은 건드리지 않는다.
    result["layoutParts"](생성 경로의 공감·권고 칸)은 여기서 꺼내 버린다 - 기록·응답에 남기지 않는다.
    concise: 피보호자 본인(WARD) 답 - 일반 답은 모델 문장을 줄이고, 응급 답은 공감·증상 줄만 뺀다.
    위기(109)·되묻기·예약 답은 concise 여도 그대로다(질문·전화 안내가 답의 전부라 줄일 것이 없다)."""
    parts = result.pop("layoutParts", None) or {}
    kind = _kind(result)
    if kind is None:
        result["layout"] = None
        return result
    reply = _without_emoji(result["reply"])
    parts = {**parts, **{k: _without_emoji(str(parts.get(k) or "")) for k in ("empathy", "advice")}}
    if kind == "emergency":
        layout = _urgent_layout(reply, _CALL_119, parts, concise=concise)
        result["emergencyWarning"] = split_warning(result.get("emergencyWarning"))
    elif kind == "crisis":
        layout = _urgent_layout(reply, _CALL_109)
        result["emergencyWarning"] = split_warning(result.get("emergencyWarning"))
    elif kind == "followup":
        layout = _followup_layout(reply)
    else:
        pool = [x for sentence in (drug_alcohol_sentences or []) for x in _sentences(str(sentence))]
        layout = _general_layout(reply, parts, pool, concise=concise)
    ward_dropped = layout.pop("_wardDropped", None)
    if concise:
        before = len([s for line in reply.split("\n") for s in _sentences(line) if not _FILLER.match(s)])
        after = len(_sentences(flatten(layout).replace("\n", " ")))
        result.setdefault("meta", {})
        if isinstance(result["meta"], dict):
            # 기록용(문장 수만, 본문 없음) - 측정·점검에서 줄어든 정도를 본다.
            result["meta"]["wardConcise"] = {"kind": kind, "dropped": ward_dropped if ward_dropped is not None else max(0, before - after)}
    result["layout"] = layout
    result["reply"] = flatten(layout)
    return result
