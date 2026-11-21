#!/usr/bin/env python3
"""cc8b 겹침 평가 (eval_cc8.py 의 topic 부분을 지금 main 코드로). 짧은 대답이 이어지는 대화에서 말벗 질문이 앞서 한 질문과
같은 화제를 다시 묻는 비율. DGX, 실제 EXAONE 2.4B + 어댑터, API 0.
사용: eval_cc8b.py <어댑터 cc7|cc8b> <off|summary> <last true|false>  →  cc8b/eval-<이름>.json
"""
import os, sys, json, logging, pathlib, random, re, collections, time, warnings
warnings.filterwarnings("ignore")
which, hist, last = sys.argv[1], sys.argv[2], sys.argv[3]
name = f"{which}-{hist}" + ("" if hist == "off" else f"-last{last}")
H = pathlib.Path.home(); W = H / "llmbench/cc8b"; R = H / "llmbench/results"
os.environ.update(MODEL_BASE_PATH=str(W / "models"), COMPANION_REPLY_LORA_SUBDIR=f"companion/llm/lora-reply-{which}",
                  COMPANION_REPLY_HISTORY=hist, COMPANION_REPLY_HISTORY_LAST=last, COMPANION_LLM_TIMEOUT_SEC="300",
                  COMPANION_TURN_CAP_SEC="900", COMPANION_TEXT_TIMEOUT_SEC="120", HF_HUB_OFFLINE="1", COMPANION_LLM_WARMUP="false",
                  COMPANION_REVIEW_DATABASE_URL=f"sqlite:///{W}/review-{name}.db")
sys.path.insert(0, str(W / "srv")); sys.path.insert(1, str(H / "llmbench/cc8"))
import cc8_filters as F
logging.basicConfig(level=logging.WARNING)
from app.services.companion.llm_client import llm_client
from app.services.companion.mood_service import mood_service
from app.services.companion.dialogue_store import store
from app.services.companion.tts_service import tts_service
from app.routers import companion_router
tts_service.start = lambda *a, **k: False
NOFACE = {"label": None, "confidence": 0.0, "frameCount": 0, "usable": False, "totalFrames": 0, "facesFound": 0}
S203 = ["아침에 일어났을 때 너무 피곤했어", "알 거 없잖아", "아니 없어", "없다고", "계획 없어", "오늘은 하루 보낼 거 계획 없다니까"]
FLAT = ["그냥 그래", "별일 없어", "몰라", "없어", "그냥 있지 뭐", "할 거 없어", "글쎄", "별로", "그냥 집에 있어", "아니 괜찮아",
        "뭐 그렇지", "모르겠어", "그냥 지내", "없다니까", "그저 그래", "됐어", "그냥 쉬어", "별거 없어"]
D = [("203", "좋은 아침이에요. 아침에 일어나셨을 때 몸은 좀 어떠셨어요?", S203),
     ("212", "아침에 편안하게 일어나셨나요? 어떠셨는지 들려주세요.", S203),
     ("flat", "오늘 오후는 어떻게 보내고 계세요?", ["그냥 그래", "별일 없어", "그냥 집에 있지", "몰라", "할 거 없어", "그냥 있다니까"]),
     ("lunch", "점심은 드셨어요? 무엇을 드셨는지 궁금해요.", ["점심 먹었어", "김치찌개 먹었지", "맛있었어", "혼자 먹었어", "그냥 그래", "별로 할 말 없어"]),
     ("plan", "저녁 시간이에요. 내일은 무엇을 하실 계획이세요?", ["내일은 쉬려고", "알빠야", "계획 없어", "그냥 쉬어", "몰라", "쉰다니까"])]
test = [d["guid"] for d in json.loads((R / "acc_days.json").read_text())]
C = {x["guid"]: x for x in json.loads((H / "llmbench/data/carecall-corpus/data/carecall_filtered_10k.json").read_text())}
rnd = random.Random(1113)
for g in test:
    x = C.get(g)
    if not x: continue
    q0 = next((" ".join(u["text"].split()) for u in x["data"] if u["role"] == "system" and u["text"].strip().endswith("?")), None)
    if q0: D.append((g[:8], q0, rnd.sample(FLAT, 6)))
    if len(D) == 25: break
llm_client._load(); t0 = time.time(); rows = []
for dn, q0, answers in D:
    d = store.create("ward-eval", "s-" + dn, "09:00", q0, 8, None); asked = [q0]
    for i, a in enumerate(answers):
        r = companion_router._run_turn(store.get(d.dialogueId), a, NOFACE, mood_service.classify, llm_client)
        companion_router._turn_response(store.get(d.dialogueId), r, face=NOFACE)
        y = r["reply"]["text"]; src = r.get("replySource")
        rep = sorted(F.repeats(y, asked, a)) if src == "LLM" else []
        rows.append(dict(d=dn, t=i + 1, a=a, y=y, src=src, rep=rep, act=bool(re.search(r"활동|취미", y))))
        if src == "LLM": asked.append(y)
llm = [x for x in rows if x["src"] == "LLM"]
res = dict(name=name, turns=len(rows), llm=len(llm), repeat=sum(bool(x["rep"]) for x in llm),
           repeatRate=round(sum(bool(x["rep"]) for x in llm) / max(1, len(llm)), 3), act=sum(x["act"] for x in rows),
           src=dict(collections.Counter(x["src"] for x in rows)), seconds=round(time.time() - t0))
(W / f"eval-{name}.json").write_text(json.dumps(dict(res, rows=rows), ensure_ascii=False, indent=1))
print("RESULT", json.dumps(res, ensure_ascii=False))
