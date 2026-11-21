#!/usr/bin/env python3
"""11/21 대답 LoRA cc8b = cc7 학습 자료(흐름 없음 = 첫 대답) + 자기 대화 교사 정답(teach_cc8b.py, 흐름 있음).
입력은 운영 코드의 build_reply_prompt 그대로 (COMPANION_REPLY_HISTORY=summary → reply-v4 꼬리질문 · reply-v2-last 마지막 인사,
흐름 최근 3쌍). 흐름의 어르신 줄은 운영 summarize_answer 로 만든 한 줄 요약. 설정은 cc7 과 같다 (nf4·r16·2 epoch).
사용: ft_reply8b.py data (요약·자료만, 개수·예시 출력) / ft_reply8b.py train
"""
import json, math, os, pathlib, random, re, sys, time, collections
H = pathlib.Path.home(); R = H / "llmbench/results"; W = H / "llmbench/cc8b"
os.environ.update(COMPANION_REPLY_HISTORY="summary", COMPANION_REPLY_HISTORY_TURNS="3", COMPANION_REPLY_HISTORY_LAST="true",
                  MODEL_BASE_PATH=str(W / "models"), COMPANION_LLM_WARMUP="false", COMPANION_LLM_TIMEOUT_SEC="300", HF_HUB_OFFLINE="1")
OUT = H / "llmbench/ft/EXAONE-3.5-2.4B-reply-lora-cc8b"
sys.path.insert(0, str(W / "srv")); sys.path.insert(1, str(H / "llmbench/cc8")); sys.path.insert(2, str(H / "llmbench/medcmp")); sys.path.insert(2, str(H / "llmbench"))
import cc8_filters as F
from app.services.companion import reply_service as rs
mode = sys.argv[1]
DATA = R / "reply_lora_data8b.json"
SUMM = R / "cc8b_summaries.json"


def prompt_of(e):
    return rs.build_reply_prompt(e["q"], e["a"], None, bool(e.get("last")), tuple(tuple(x) for x in e.get("hist", ())))


if mode == "data":
    dlg = json.loads((R / "reply_teacher78_cc8b.json").read_text())
    summ = json.loads(SUMM.read_text()) if SUMM.exists() else {}
    todo = sorted({x["a"] for d in dlg for x in d["turns"]} - set(summ))
    if todo:
        from app.services.companion.llm_client import llm_client
        from app.services.companion.answer_summary import summarize_answer
        llm_client._load(); t0 = time.time()
        for i, a in enumerate(todo, 1):
            summ[a] = summarize_answer(llm_client, a)
            if i % 200 == 0:
                print("summary", i, "/", len(todo), f"{time.time() - t0:.0f}s", flush=True)
                SUMM.write_text(json.dumps(summ, ensure_ascii=False))
        SUMM.write_text(json.dumps(summ, ensure_ascii=False))
    print("요약 없음(None)", sum(1 for v in summ.values() if not v), "/", len(summ), flush=True)
    new = []; seen = collections.Counter(); tails = collections.Counter()
    for d in dlg:
        hist = []
        for x in d["turns"]:
            if x.get("train"):
                k = F.qsent(x["y"]) if not x["last"] else x["y"]
                if seen[x["y"]] < 2 and tails[k] < 3:
                    seen[x["y"]] += 1; tails[k] += 1
                    new.append(dict(q=x["q"], a=x["a"], y=x["y"], last=x["last"], hist=[list(h) for h in hist], dlg=d["id"], kind="cc8b"))
            hist.append((x["q"], summ.get(x["a"])))
    print("교사 정답", sum(1 for d in dlg for x in d["turns"] if x.get("train")), "통과", len(new),
          "마지막 인사", sum(1 for e in new if e["last"]), "흐름 길이", dict(collections.Counter(min(len(e["hist"]), 3) for e in new)), flush=True)
    print("꼬리질문 상위", collections.Counter(F.qsent(e["y"]) for e in new if not e["last"]).most_common(5), flush=True)
    base = json.loads((R / "reply_lora_data7.json").read_text())  # cc7 자료 (흐름 없음 = 첫 대답 형식)
    rnd = random.Random(1121)
    ids = sorted({e["dlg"] for e in new}); rnd.shuffle(ids); vid = set(ids[:max(20, len(ids) // 10)])  # 대화 단위로 검증 분리
    val = base["val"] + [e for e in new if e["dlg"] in vid]
    train = base["train"] + [e for e in new if e["dlg"] not in vid]
    rnd.shuffle(train)
    DATA.write_text(json.dumps({"train": train, "val": val}, ensure_ascii=False))
    print("학습", len(train), "검증", len(val), flush=True)
    for e in rnd.sample([e for e in new if e["hist"]], 2) + rnd.sample([e for e in new if e["last"]], 1):
        print("-----\n" + prompt_of(e)[:900] + "\n=> " + e["y"], flush=True)
    sys.exit()

import torch
from peft import LoraConfig, get_peft_model
from llm_bench import LM
d = json.loads(DATA.read_text()); train, val = d["train"], d["val"]
t0 = time.time()
lm = LM("exaone35", "nf4"); tok = lm.tok
TM, CL = type(lm.m.transformer), type(lm.m)
TM.get_input_embeddings = lambda self: self.wte; TM.set_input_embeddings = lambda self, v: setattr(self, "wte", v)
CL.get_input_embeddings = lambda self: self.transformer.wte; CL.get_output_embeddings = lambda self: self.lm_head
model = get_peft_model(lm.m, LoraConfig(r=16, lora_alpha=32, lora_dropout=0.05, target_modules="all-linear", task_type="CAUSAL_LM"))


def enc(e):
    p = tok.apply_chat_template([{"role": "user", "content": prompt_of(e)}], add_generation_prompt=True, tokenize=False)
    pi = tok(p, add_special_tokens=False)["input_ids"]; yi = tok(e["y"] + tok.eos_token, add_special_tokens=False)["input_ids"]
    return torch.tensor(pi + yi), torch.tensor([-100] * len(pi) + yi)


tr = [enc(e) for e in train]; va = [enc(e) for e in val]
opt = torch.optim.AdamW([p for p in model.parameters() if p.requires_grad], lr=2e-4); EP, ACC = 2, 8
total = EP * math.ceil(len(tr) / ACC)
sched = torch.optim.lr_scheduler.LambdaLR(opt, lambda s: min(1.0, (s + 1) / 10) * max(0.0, 1 - s / total))


def loss_of(ids, labels):
    lg = model(input_ids=ids[None].to(model.device)).logits[0, :-1].float()
    return torch.nn.functional.cross_entropy(lg, labels[1:].to(model.device), ignore_index=-100)


for ep in range(EP):
    model.train(); runl = 0
    for i, (ids, labels) in enumerate(tr, 1):
        loss = loss_of(ids, labels) / ACC; loss.backward(); runl += loss.item() * ACC
        if i % ACC == 0 or i == len(tr): opt.step(); sched.step(); opt.zero_grad()
        if i % 500 == 0: print(f"ep{ep + 1} {i}/{len(tr)} loss {runl / i:.4f} {time.time() - t0:.0f}s", flush=True)
    model.eval()
    with torch.no_grad(): vl = sum(loss_of(i, l).item() for i, l in va) / len(va)
    print(f"epoch {ep + 1} train {runl / len(tr):.4f} val {vl:.4f} {time.time() - t0:.0f}s", flush=True)
model.save_pretrained(OUT)
print("SAVED", OUT, f"{time.time() - t0:.0f}s")
