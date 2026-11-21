#!/usr/bin/env python3
"""11/21 cc8b 자료: EXAONE 3.5 7.8B(교사)가 말벗·어르신을 번갈아 맡아 대화를 이어 간다 (자기 대화, API 0).

cc8 실패 원인(흐름 자리에 상담원 말)을 피하려고 흐름의 말벗 줄이 말벗 자신의 문장이 되게 만든다.
말벗 정답은 후보 3개 중 거르기(cc8_filters + 운영 check_reply + 품질 채점기 rule_failures)를 통과한 첫 것만 쓴다.
품질 세트(companion_quality_v1) 어르신 문장과 같거나 비슷한(0.8) 말은 쓰지 않는다 (시험 전용).
사용: teach_cc8b.py probe N | gen N  →  results/reply_teacher78_cc8b[_probe].json
"""
import json, pathlib, random, re, sys, time, collections
from difflib import SequenceMatcher
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

H = pathlib.Path.home(); R = H / "llmbench/results"; W = H / "llmbench/cc8b"; M = H / "llmbench/exaone78_patched"
sys.path.insert(0, str(H / "llmbench/cc8")); sys.path.insert(0, str(W / "srv"))
import cc8_filters as F
from app.services.companion import reply_service as rs
from app.services.companion.risk import keyword_risk
import importlib.util as _u
_s = _u.spec_from_file_location("score", W / "srv/tools/eval/score_companion_quality.py"); SC = _u.module_from_spec(_s); _s.loader.exec_module(SC)

mode, N = sys.argv[1], int(sys.argv[2])
TURNS = 6
OPENINGS = [x.strip() for x in (W / "openings.txt").read_text(encoding="utf-8").splitlines() if x.strip()]
SET = json.loads((W / "srv/tests/companion_eval/companion_quality_v1.json").read_text())
SET_ELDER = [t["elder"] for s in SET["scenarios"] for t in s["turns"]]

def near_set(a):
    return any(a == x or SequenceMatcher(None, a, x).ratio() >= 0.8 for x in SET_ELDER)

# 어르신 말투 예시 - CareCall 어르신 대답 (상담원 말은 쓰지 않는다)
C = json.loads((H / "llmbench/data/carecall-corpus/data/carecall_filtered_10k.json").read_text())
FLATRE = re.compile(r"^(그냥|몰라|없어|없다|별로|아니|글쎄|뭐|괜찮|그래|됐어|할 거 없|계획 없|별일 없|모르겠)")
# 짧은 대답 어르신 역은 생성이 길어지기 쉬워, 60%는 CareCall 실제 어르신의 짧은 대답을 쓴다 (짧은 대답 반복이 cc8 의 원래 문제)
FLAT_REAL = sorted({" ".join(u["text"].split()) for x in C for u in x["data"]
                    if u["role"] == "user" and len(u["text"].strip()) <= 12 and FLATRE.search(u["text"].strip())})
ELDER_STYLE = [" ".join(u["text"].split()) for x in C for u in x["data"] if u["role"] == "user" and 4 <= len(u["text"].strip()) <= 40]

SITU = ["무릎이 시큰거려 걷기가 불편하다", "텃밭에 상추와 고추를 키운다", "손주가 주말에 다녀갔다", "딸이 멀리 살아 자주 못 본다",
        "경로당에 가끔 나간다", "입맛이 없어 밥을 대충 먹는다", "밤에 자주 깬다", "아침마다 동네를 걷는다", "혼자 텔레비전을 오래 본다",
        "이웃 할머니와 가끔 차를 마신다", "병원에 정기적으로 다닌다", "강아지를 키운다", "아들이 바빠 연락이 뜸하다", "장 보러 시장에 간다",
        "허리가 아파 오래 서 있기 힘들다", "교회에 다닌다", "복지관 노래 교실에 다닌다", "요즘 별일 없이 집에만 있다", "약을 꼬박꼬박 챙겨 먹는다",
        "날이 추워져 바깥에 잘 안 나간다", "옛날 친구가 요즘 많이 아프다", "반찬을 만들어 아들네 보내려 한다", "화분 가꾸는 걸 좋아한다",
        "귀가 잘 안 들려 전화가 불편하다", "곧 김장을 하려고 한다"]
MOODS = [("평온한", 40), ("기분 좋은", 20), ("조금 걱정이 있는", 25), ("기운이 없고 쓸쓸한", 15)]
LENGTH = [("아주 짧게, 2~6글자로만 (예: 그냥 그래, 몰라, 없어, 먹었어)", 30), ("한 문장으로 10~30글자", 50), ("두 문장까지 40글자 안으로", 20)]

def wpick(rnd, xs):
    return rnd.choices([x for x, _ in xs], weights=[w for _, w in xs])[0]

BOT_SYS = ("당신은 혼자 사시는 어르신께 안부를 묻는 말벗입니다. 지금까지의 대화는 흐름을 이해하는 데만 쓰고, 어르신이 방금 하신 대답에 답합니다.\n"
           "정확히 두 문장으로 씁니다. 1) 방금 하신 말을 받아 주는 짧은 공감 한 문장. 2) 방금 하신 말 속 내용을 하나 골라 더 묻는 짧은 질문 한 문장(물음표로 끝).\n"
           "지난 대화에서 이미 물었거나 나온 화제는 다시 묻지 않고, 지난 대화 내용을 대답에 이어 붙이지 않습니다. 어르신 대답이 짧고 별 내용이 없으면(그냥, 몰라, 없어) "
           "캐묻지 말고 받아 준 뒤 아직 묻지 않은 가까운 일상(식사, 잠, 몸, 가족, 이웃, 집안일 등) 하나를 가볍게 묻습니다.\n"
           "지키기: 권유·조언(드세요, 하세요, 어떨까요) 금지. 날씨·막연한 활동·취미 질문 금지. 계획·이유·언제를 캐묻지 않습니다. 말벗 자신의 이야기 금지. 병 이름 금지. "
           "앞으로 하실 일을 이미 하신 일처럼 말하지 않습니다. 60자 안팎, 존댓말, 답 문장만 씁니다.")
LAST_SYS = ("당신은 혼자 사시는 어르신께 안부를 묻는 말벗입니다. 오늘 대화를 마무리하는 짧은 인사를 한두 문장으로 씁니다.\n"
            "질문하지 않습니다. 어르신이 방금 하신 말을 짧게 받아 주고 편안한 하루를 빕니다. 지난 대화는 흐름을 이해하는 데만 쓰고, "
            "어르신이 하지 않은 일을 했다고 말하지 않습니다(앞으로 하신다는 일은 앞으로의 일로). 느낌표·'감사합니다'·말벗 자신의 이야기 금지. 50자 안팎, 존댓말, 답 문장만 씁니다.")
BOT_SHOTS = [([], "점심은 드셨어요?", "응 국수 삶아 먹었어", "국수 맛있게 드셨군요. 어떤 국수 해 드셨어요?"),
             ([("요즘 잠은 잘 주무세요?", "새벽에 자꾸 깨")], "새벽에 자꾸 깨시면 피곤하시겠어요. 깨시면 다시 잠드시기 어려우세요?", "응 한참 뒤척여",
              "한참 뒤척이시면 힘드시겠어요. 그럴 때는 누워서 뭐 하세요?"),
             ([("오늘 뭐 하셨어요?", "그냥 집에 있었지")], "집에서 쉬셨군요. 집에 계실 때는 주로 뭐 하면서 지내세요?", "몰라 그냥",
              "그냥 쉬시는군요. 오늘 식사는 챙겨 드셨어요?"),
             ([("몸은 좀 어떠세요?", "무릎이 좀 아파")], "무릎이 아프시면 불편하시겠어요. 언제부터 그러셨어요?", "아들이 파스 사다 줬어",
              "아드님이 챙겨 주셨군요. 아드님은 자주 들르세요?")]
LAST_SHOTS = [([("점심은 드셨어요?", "응 국수 먹었어")], "국수 맛있게 드셨군요. 저녁은 뭐 드실 거예요?", "이따가 된장찌개 끓이려고",
               "된장찌개 끓여 드신다니 든든하시겠어요. 오늘 저녁도 편안하게 보내세요."),
              ([("오늘 기분은 어떠세요?", "그냥 그래")], "그냥 그러시군요. 오늘 누구와 이야기 나누셨어요?", "아무하고도 안 했어",
               "오늘은 조용히 보내셨군요. 저와 이야기 나눠 주셔서 반가웠어요. 편히 쉬세요.")]

def ctx(hist, q, a):
    h = "\n".join(f"- 말벗: {b}\n- 어르신: {e}" for b, e in hist) if hist else "(아직 없음)"
    return f"지금까지의 대화:\n{h}\n말벗이 방금 한 말: {q}\n어르신 대답: {a}"

def bot_msgs(hist, q, a, last):
    m = [{"role": "system", "content": LAST_SYS if last else BOT_SYS}]
    for sh, sq, sa, sy in (LAST_SHOTS if last else BOT_SHOTS):
        m += [{"role": "user", "content": ctx(sh, sq, sa)}, {"role": "assistant", "content": sy}]
    return m + [{"role": "user", "content": ctx(hist, q, a)}]

def elder_msgs(d, q):
    sys_ = (f"당신은 혼자 사는 {d['age']}세 {d['gender']} 어르신입니다. 요즘 형편: {d['situ']}. 오늘은 {d['mood']} 기분입니다.\n"
            f"말벗(돌봄 앱)이 묻는 말에 실제 어르신처럼 편한 말투(반말 섞어도 됨)로 대답합니다. 길이: {d['length']}.\n"
            "대답 문장만 씁니다. 질문을 되묻지 않습니다. 죽음·자해·응급 이야기는 하지 않습니다.\n"
            "말투 예시: " + " / ".join(d["style"]))
    h = "\n".join(f"말벗: {b}\n나: {e}" for b, e in d["hist"])
    return [{"role": "system", "content": sys_}, {"role": "user", "content": (h + "\n" if h else "") + f"말벗: {q}\n나:"}]

def clean_elder(t):
    t = t.strip().splitlines()[0] if t.strip() else ""
    t = re.sub(r"^(나|어르신)\s*[:：]\s*", "", t).strip().strip("\"'“”")
    return t if 1 <= len(t) <= 80 else ""

HABIT = re.compile(r"순간|조용|책|음악|기억에 남")  # cc8 에서 생긴 버릇 (메모 companion-reply-cc8-shelved)

def ok_follow(y, a, q, past):
    if not F.ok_ctx(y, a, q, past[-3:]): return False
    if HABIT.search(y) and not HABIT.search(a): return False
    if rs.check_reply(y, False, a) not in ([], ["대답과 무관"]): return False
    return [c for c in SC.rule_failures(y, a, False, past) if c != "R2"] == []  # R2 는 채점기에서 '개선'(실패 아님)

def ok_last(y, a, past):
    if "?" in y or "!" in y or "감사합니다" in y or not 10 <= len(y) <= 90: return False
    if re.search(r"저는|저도|제가|저희", y): return False
    return [c for c in SC.rule_failures(y, a, True, past) if c != "R2"] == []

rnd = random.Random(1121)
dlg = []
for i in range(N):
    dlg.append(dict(id=i, opening=OPENINGS[i % len(OPENINGS)], age=rnd.randint(68, 88), gender=rnd.choice(["할머니", "할아버지"]),
                    situ=", ".join(rnd.sample(SITU, 2)), mood=wpick(rnd, MOODS), length=wpick(rnd, LENGTH),
                    style=rnd.sample(ELDER_STYLE, 6), hist=[], turns=[], q=OPENINGS[i % len(OPENINGS)], done=False, stop=None))

tok = AutoTokenizer.from_pretrained(M, trust_remote_code=True); tok.padding_side = "left"
if tok.pad_token is None: tok.pad_token = tok.eos_token
m = AutoModelForCausalLM.from_pretrained(M, trust_remote_code=True, dtype=torch.bfloat16, device_map="cuda"); m.eval()
torch.manual_seed(1121); t0 = time.time()

def gen(msg_list, n, temp, max_new, bs):
    out = []
    for i in range(0, len(msg_list), bs):
        ps = [tok.apply_chat_template(x, add_generation_prompt=True, tokenize=False) for x in msg_list[i:i + bs]]
        enc = tok(ps, return_tensors="pt", padding=True, add_special_tokens=False).to(m.device)
        with torch.no_grad():
            o = m.generate(**enc, max_new_tokens=max_new, do_sample=True, temperature=temp, top_p=0.9, num_return_sequences=n,
                           pad_token_id=tok.pad_token_id)
        txt = [re.sub(r"[ \t]+", " ", tok.decode(x[enc["input_ids"].shape[1]:], skip_special_tokens=True)).strip() for x in o]
        out += [txt[k * n:(k + 1) * n] for k in range(len(ps))]
    return out

stops = collections.Counter()
for t in range(1, TURNS + 1):
    act = [d for d in dlg if not d["done"]]
    if not act: break
    el = gen([elder_msgs(d, d["q"]) for d in act], 1, 0.9, 60, 32)
    el = [[rnd.choice(FLAT_REAL)] if d["length"].startswith("아주 짧게") and rnd.random() < 0.6 else x for d, x in zip(act, el)]
    keep = []
    for d, (a,) in zip(act, el):
        a = clean_elder(a)
        why = "empty" if not a else "risk" if keyword_risk(a) else "set" if near_set(a) else None
        if why: d["done"], d["stop"] = True, why; stops[why] += 1; continue
        d["a"] = a; keep.append(d)
    last = t == TURNS
    cands = gen([bot_msgs(d["hist"], d["q"], d["a"], last) for d in keep], 4, 0.8, 90, 16)
    for d, cs in zip(keep, cands):
        past = [d["opening"]] + [x["y"] for x in d["turns"]]
        good = [c for c in cs if (ok_last(c, d["a"], past) if last else ok_follow(c, d["a"], d["q"], past))]
        good.sort(key=lambda c: bool(SC.R2_PATTERN.search(c)))  # R2(다행이·아쉽…) 없는 후보 먼저
        y = good[0] if good else None
        if y is None:
            # 운영과 같이 대체 문장으로 대화를 이어 가고, 이 턴은 학습에서 뺀다 (train=False)
            stops["nofit"] += 1
            fb = rnd.choice(rs.FALLBACK_LAST if last else rs.FALLBACK_FOLLOWUP)
            d["turns"].append(dict(t=t, q=d["q"], a=d["a"], y=fb, last=last, train=False))
            d["hist"].append((d["q"], d["a"])); d["q"] = fb
            if last: d["done"] = True
            continue
        d["turns"].append(dict(t=t, q=d["q"], a=d["a"], y=y, last=last, train=True))
        d["hist"].append((d["q"], d["a"])); d["q"] = y
        if last: d["done"] = True
    print(f"turn {t} active {len(act)} kept {len(keep)} train {sum(1 for d in keep if d['turns'] and d['turns'][-1]['t'] == t and d['turns'][-1]['train'])} "
          f"{time.time() - t0:.0f}s stops {dict(stops)}", flush=True)

out = R / ("reply_teacher78_cc8b_probe.json" if mode == "probe" else "reply_teacher78_cc8b.json")
for d in dlg: d.pop("style", None)
out.write_text(json.dumps(dlg, ensure_ascii=False, indent=1))
n_ex = sum(1 for d in dlg for x in d["turns"] if x["train"])
print("flat pool", len(FLAT_REAL))
print("DONE dialogues", len(dlg), "examples", n_ex, "last", sum(1 for d in dlg for x in d["turns"] if x["last"] and x["train"]),
      "turns/dlg", dict(collections.Counter(len(d["turns"]) for d in dlg)), f"{time.time() - t0:.0f}s", flush=True)
if mode == "probe":
    for d in dlg[:3]:
        print("==", d["opening"], "|", d["mood"], "|", d["length"][:10])
        for x in d["turns"]: print("  어르신:", x["a"], "\n  말벗" + ("" if x["train"] else "(대체)") + ":", x["y"])
