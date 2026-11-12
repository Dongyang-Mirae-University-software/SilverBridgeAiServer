"""답 모양 - 핵심 한 줄·짧은 본문·칸별 카드·줄 머리 이모지(2026-11-01 가독성).

모든 판정(위험도·119·응급·자해·위기·약 + 술·되묻기·진료과·빼기 규칙·이어서 물어볼 질문)이 끝난 답을 받아
**모양만** 바꾼다. 문장을 새로 만들지 않고, 상투 문장 "말씀해 주신 내용을 정리했어요." 말고는 어떤 문장도 버리지 않는다.

- layout: {"headline": str|None, "body": [...], "questions": [...], "homeCare": [...], "visitHospitalIf": [...]}
  FE 는 layout 이 있으면 layout 으로, 없으면 reply 로 그린다(옛 AI·옛 기록 호환).
- reply: layout 을 한 줄에 한 문장으로 펼친 글(줄마다 이모지). 기록·옛 FE·이력은 이 글을 쓴다.
- 응급·자해·위기 답: 맨 위 "🚨 119(위기는 109)" 한 줄, 다른 줄에는 이모지를 붙이지 않는다.
- 2026-11-09: 챗 서비스는 emoji=False(CHAT_LAYOUT_EMOJI 기본 끔)로 부른다 - 모든 줄 머리 이모지를 빼고, 응급 답 핵심 줄은
  119 전화 문장(없으면 CALL_119_HEADLINE). 단계 되묻기 질문 턴은 _step_layout(질문 하나 + 본문, 카드 없음).
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
# 이모지를 끈 응급 답의 핵심 줄에 전화 안내가 없을 때 쓰는 문장(2026-11-09).
CALL_119_HEADLINE = "119에 바로 전화하세요."
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


def _step_layout(reply: str) -> dict[str, Any]:
    """단계 되묻기 질문 턴(2026-11-09): 질문 하나는 questions 칸, 나머지(첫 턴 앞말·위험 신호 119 줄·다시 묻기 앞말)는 본문. 카드는 비운다."""
    body, questions = [], []
    for line in reply.split("\n"):
        text = line.strip()
        if not text:
            continue
        q = _QUESTION_LINE.match(text)
        if q:
            questions.append(f"{kw.LAYOUT_QUESTION_EMOJI} {q.group(1).strip()}")
        else:
            body.extend(_emoji(s) if _hospital(s) else f"{kw.LAYOUT_EMPATHY_EMOJI} {s}" for s in _sentences(text))
    return {"headline": None, "body": body, "questions": questions, "homeCare": [], "visitHospitalIf": []}


def _strip_emoji(layout: dict[str, Any]) -> dict[str, Any]:
    """줄 머리 이모지를 뺀 layout(CHAT_LAYOUT_EMOJI=false, 2026-11-09 기본). 문장·칸·순서는 그대로다."""
    def plain(s: str) -> str:
        return kw.strip_layout_emoji(s).strip()
    out = dict(layout)
    out["headline"] = plain(layout["headline"]) if layout.get("headline") else layout.get("headline")
    for key in ("body", "questions", "homeCare", "visitHospitalIf"):
        out[key] = [plain(s) for s in layout.get(key) or [] if plain(s)]
    return out


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


def append_notice(result: dict[str, Any], notice: str, emoji: bool = True) -> bool:
    """모양을 다 정한 답 끝에 안내 문장을 붙인다(2026-11-11 라우터 실패 안내). 붙였으면 True.
    apply 뒤에 부른다 - 피보호자 답 줄이기·본문 4줄 규칙이 이 문장을 자르거나 카드로 옮기지 않게.
    layout 이 있으면 본문(body) 맨 끝에 한 줄 한 문장으로 넣고 reply 를 다시 펼친다(화면과 글이 같은 자리).
    layout 이 없는 답(예약·선택지)은 reply 끝에만 붙인다. 이미 든 문장은 다시 붙이지 않는다.
    긴급 표시(urgent) layout 에는 붙이지 않는다 - 응급·위기 답은 부르는 쪽이 먼저 거르고, 여기서 한 번 더 막는다."""
    reply = result.get("reply")
    if not isinstance(reply, str) or not reply.strip():
        return False
    layout = result.get("layout")
    if isinstance(layout, dict) and layout.get("urgent"):
        return False
    present = [kw.strip_layout_emoji(s).strip() for line in reply.split("\n") for s in _sentences(line)]
    missing = [s for s in _sentences(notice) if not _in(s, present)]
    if not missing:
        return False
    if isinstance(layout, dict):
        layout["body"] = [*(layout.get("body") or []), *((_emoji(s) if emoji else s) for s in missing)]
        result["reply"] = flatten(layout)
    else:
        result["reply"] = "\n".join([reply.rstrip("\n"), *missing])
    return True


def strip_notices(text: str, notices: Any) -> str:
    """다음 턴 이력용: 답에 붙였던 안내 문장 줄을 뺀 글. 모델이 앞 답의 안내 문장을 따라 쓰지 않게 한다."""
    pool = [s for notice in notices for s in _sentences(str(notice))]
    return "\n".join(line for line in (text or "").split("\n") if not _in(kw.strip_layout_emoji(line).strip(), pool))


def split_warning(items: list[str] | None) -> list[str]:
    """응급 경고 박스는 내용 그대로, 한 문장씩 나눈 목록으로(문장 3개가 한 항목에 들어 있었다)."""
    out: list[str] = []
    for item in items or []:
        out.extend(_sentences(str(item)) or [str(item)])
    return out


def _crisis_followup_side_layout(result: dict[str, Any]) -> dict[str, Any] | None:
    """예약·선택지·도구 답(모양을 바꾸지 않는 답)에 위기 뒤 해제 109 줄이 붙었을 때만 작은 layout(2026-11-09, FE #229 점검 L-5).
    핵심 줄 = 맨 앞 109 줄, 본문 = 나머지 reply 줄 그대로(문장·순서 불변), urgent = true - FE 가 109 줄을 "긴급" 상자에 넣는다.
    reply·ui·tool·type·intent·판정 칸은 건드리지 않는다. 그 밖의 예약·선택지 답은 지금처럼 layout 없음(None)."""
    meta = result.get("meta") if isinstance(result.get("meta"), dict) else {}
    reply = result.get("reply")
    if not meta.get("crisisFollowup") or not isinstance(reply, str):
        return None
    lines = [line.strip() for line in reply.split("\n") if line.strip()]
    if not lines or not _CALL_109.search(lines[0]):
        return None
    return {"headline": lines[0], "body": lines[1:], "questions": [], "homeCare": [], "visitHospitalIf": [], "urgent": True}


def apply(
    result: dict[str, Any], drug_alcohol_sentences: list[str] | None = None, concise: bool = False, emoji: bool = True
) -> dict[str, Any]:
    """result 의 reply 를 새 모양으로 바꾸고 layout 을 붙인다(제자리 수정 후 반환). 판정 값은 건드리지 않는다.
    result["layoutParts"](생성 경로의 공감·권고 칸)은 여기서 꺼내 버린다 - 기록·응답에 남기지 않는다.
    concise: 피보호자 본인(WARD) 답 - 일반 답은 모델 문장을 줄이고, 응급 답은 공감·증상 줄만 뺀다.
    위기(109)·되묻기·예약 답은 concise 여도 그대로다(질문·전화 안내가 답의 전부라 줄일 것이 없다).
    emoji=False(챗 서비스 기본, CHAT_LAYOUT_EMOJI): 모양을 다 정한 뒤 줄 머리 이모지만 뺀다 - 응급 핵심 줄의 🚨도."""
    parts = result.pop("layoutParts", None) or {}
    kind = _kind(result)
    if kind is None:
        result["layout"] = _crisis_followup_side_layout(result)
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
        meta = result.get("meta") if isinstance(result.get("meta"), dict) else {}
        layout = _step_layout(reply) if meta.get("followupStep") else _followup_layout(reply)
    else:
        pool = [x for sentence in (drug_alcohol_sentences or []) for x in _sentences(str(sentence))]
        layout = _general_layout(reply, parts, pool, concise=concise)
    ward_dropped = layout.pop("_wardDropped", None)
    if not emoji:
        layout = _strip_emoji(layout)
        if kind == "emergency" and "전화" not in (layout.get("headline") or ""):
            # 🚨 표시 대신 할 일을 문장으로(2026-11-09): 핵심 줄에 전화 안내가 없으면(폴백 "119 또는 가까운 응급실 방문을 권장합니다")
            # "119에 바로 전화하세요."를 핵심 줄로 두고 원래 핵심 줄은 본문 맨 위로 - 문장은 버리지 않는다.
            if layout.get("headline"):
                layout["body"] = [layout["headline"], *layout["body"]]
            layout["headline"] = CALL_119_HEADLINE
    if kind in ("emergency", "crisis"):
        # 응급·위기(119·109) 답 표시(2026-11-09, FE #227 점검 M-1): FE 가 핵심 줄을 "긴급" 경고 상자에 넣는 근거.
        # 위기 뒤 해제 턴(crisisFollowup)은 경고 박스·위험도 high 가 없어 FE 가 알아볼 칸이 따로 필요했다.
        # 표시 전용 - riskLevel·emergencyWarning·되묻기·답 칩 판정은 이 칸을 보지 않는다. 그 밖의 답에는 칸을 두지 않는다.
        layout["urgent"] = True
    if concise:
        before = len([s for line in reply.split("\n") for s in _sentences(line) if not _FILLER.match(s)])
        after = len(_sentences(flatten(layout).replace("\n", " ")))
        result.setdefault("meta", {})
        if isinstance(result["meta"], dict):
            # 기록용(문장 수만, 본문 없음) - 측정·점검에서 줄어든 정도를 본다.
            result["meta"]["wardConcise"] = {"kind": kind, "dropped": ward_dropped if ward_dropped is not None else max(0, before - after)}
    result["layout"] = layout
    result["reply"] = flatten(layout)
    if kind in ("emergency", "crisis") and result.get("emergencyWarning"):
        # FE 는 경고 박스(emergencyWarning)를 목록으로 그린 뒤 본문(layout.body)을 그린다 - 경고 박스와 같은 문장만 본문에서 뺀다
        # (2026-11-12 화면 확인: 단계 되묻기 119 규칙 답·생성 실패 대체 답은 본문이 경고 박스 문장 그 자체라 두 번 보였다).
        # 표시 전용 - reply 를 layout 에서 만든 뒤에 빼므로 reply(기록·이력·판정)·emergencyWarning·핵심 줄(긴급 표시 근거)은 그대로다.
        shown = {s.strip() for s in result["emergencyWarning"] if isinstance(s, str)}
        layout["body"] = [line for line in layout.get("body") or []
                          if not (_sentences(_without_emoji(line))
                                  and all(s.strip() in shown for s in _sentences(_without_emoji(line))))]
    return result
