"""말벗 안부 질문 후보를 로컬 EXAONE(Ollama)으로 만들고 규칙으로 걸러낸다.

Phase 1 산출물
- 시간대(09·12·15·18·21시)마다 후보 질문을 여러 개 만든다.
- 규칙(존댓말, 질문 형태, 길이, 의료 조언 금지)을 통과한 것만 남긴다.
- 결과는 questions_candidates.json 에 저장한다. 사람이 검토해서 최종 세트를 고른다.

실행 (companion-emotion 폴더에서):
    venv\\Scripts\\python.exe companion\\generate_questions.py
"""

from __future__ import annotations

import json
import re
import urllib.request
from pathlib import Path

OLLAMA_URL = "http://localhost:11434/api/generate"
MODEL = "exaone3.5:2.4b"
OUT = Path(__file__).resolve().parent / "questions_candidates.json"
PER_SLOT = 6
MAX_CHARS = 45

SLOTS = [
    {"time": "09:00", "topic": "아침 안부: 밤에 잘 주무셨는지"},
    {"time": "12:00", "topic": "점심: 식사를 했는지"},
    {"time": "15:00", "topic": "오후: 오늘 기분이 어떤지"},
    {"time": "18:00", "topic": "저녁: 저녁 식사를 했는지"},
    {"time": "21:00", "topic": "하루 마무리: 오늘 하루가 어땠는지"},
]

SYSTEM = """당신은 혼자 사는 어르신께 안부를 묻는 따뜻한 말벗입니다.
규칙:
1. 존댓말로 질문 한 문장만 만듭니다. 끝은 반드시 물음표(?)입니다.
2. "예/아니오"로만 끝나지 않게, 어르신이 이야기를 꺼낼 수 있도록 묻습니다.
3. 40자 이내로 짧게 씁니다.
4. 병, 약 복용 지시, 진단, 치료에 대한 말은 하지 않습니다.
5. 인사말, 설명, 따옴표 없이 질문 문장만 출력합니다.
6. 이모지, 괄호, 영어를 쓰지 않습니다. 물음표는 한 번만 씁니다."""

BANNED = ("진단", "처방", "치료", "병원", "수술", "약을 ")


def ask(topic: str, temperature: float) -> str:
    prompt = f"주제: {topic}\n위 주제로 어르신께 드릴 안부 질문 한 개를 만드세요."
    body = json.dumps(
        {
            "model": MODEL,
            "system": SYSTEM,
            "prompt": prompt,
            "stream": False,
            "options": {"temperature": temperature, "num_predict": 60},
        }
    ).encode("utf-8")
    req = urllib.request.Request(OLLAMA_URL, data=body, headers={"Content-Type": "application/json"})
    with urllib.request.urlopen(req, timeout=300) as res:
        return json.loads(res.read().decode("utf-8"))["response"]


EMOJI = re.compile("[\U0001F000-\U0001FAFF☀-➿️]")


def clean(raw: str) -> str:
    """모델 출력에서 질문 한 문장만 남긴다. 이모지, (?), 영어 꼬리, 중복 물음표를 정리한다."""
    line = next((ln.strip() for ln in raw.splitlines() if ln.strip()), "")
    line = EMOJI.sub("", line)
    line = re.sub(r"^[\s\"'“”*\-\d\.\)]+", "", line).strip().strip("\"'“”")
    line = re.sub(r"\(\?\)|\?\s*\?+", "?", line)  # (?) 와 ?? 를 ? 하나로
    line = re.sub(r"[!！]+", "", line)
    # 물음표가 있으면 첫 질문 문장까지만 쓴다
    if "?" in line:
        line = line[: line.index("?") + 1].strip()
    line = re.sub(r"\s+", " ", line).strip()
    return line


def has_english(q: str) -> bool:
    return bool(re.search(r"[A-Za-z]{3,}", q))


def check(q: str) -> list[str]:
    problems = []
    if not q.endswith("?"):
        problems.append("물음표 없음")
    if len(q) > MAX_CHARS:
        problems.append(f"{MAX_CHARS}자 초과")
    if has_english(q):
        problems.append("영어 포함")
    if not re.search(r"(요|나요|까요|세요|시나요|셨어요|셨나요|신가요|습니까)\?$", q):
        problems.append("존댓말 질문 어미 아님")
    if any(b in q for b in BANNED):
        problems.append("의료 관련 표현")
    return problems


def main() -> None:
    results = []
    for slot in SLOTS:
        seen: set[str] = set()
        cands = []
        for i in range(PER_SLOT * 2):  # 탈락을 감안해 두 배로 생성
            if len([c for c in cands if not c["problems"]]) >= PER_SLOT:
                break
            temp = 0.7 + 0.05 * (i % 6)
            q = clean(ask(slot["topic"], temp))
            if not q or q in seen:
                continue
            seen.add(q)
            cands.append({"text": q, "problems": check(q)})
        passed = [c["text"] for c in cands if not c["problems"]]
        results.append({"time": slot["time"], "topic": slot["topic"], "passed": passed, "all": cands})
        print(f"{slot['time']} 통과 {len(passed)}/{len(cands)}")
    OUT.write_text(json.dumps(results, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"저장: {OUT}")


if __name__ == "__main__":
    main()
