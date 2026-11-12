"""말벗 판단 단계(11/10) DGX 측정: 운영 코드(app/) 그대로, int8, 실제 가중치. 부분: mood risk reply fuse face.
DGX 전용 경로(~/llmbench/results 자료, ~/llmbench/cmp 모듈)를 쓴다. 결과: docs/eval-results/2026-11-10-companion-judgment-stage.md"""
import os, sys, json, time, math, random, collections, pathlib, hashlib
H = pathlib.Path.home(); T = H / "llmbench/tmp_judgment_b"; R = H / "llmbench/results"
os.environ["MODEL_BASE_PATH"] = str(T / "models"); os.environ["COMPANION_LLM_TIMEOUT_SEC"] = "120"; os.environ["COMPANION_TEXT_TIMEOUT_SEC"] = "120"
for p in ("llmbench/medcmp", "llmbench", "llmbench/cmp"): sys.path.append(str(H / p))
import torch
from app.core.config import get_settings
from app.services.companion.llm_client import llm_client, MOOD_ADAPTER
from app.services.companion.mood_service import MoodService, MOOD_CODES, _PROMPT as MOOD_PROMPT
from app.services.companion import risk as rk, reply_service as rs, fusion as fu, pipeline as pl
S = get_settings(); PARTS = sys.argv[1:] or ["mood", "risk", "reply", "fuse", "face"]
def out(*a): print(*a, flush=True)
NEG = {"WORRY", "DEPRESSED"}
def metrics(rows):  # fuse_cmp·every_neg 와 같은 정의 (gold, out)
    g3 = lambda c: "NEG" if c in NEG else c; f1 = []
    for c in ("JOY", "CALM", "NEG"):
        tp = sum(g3(o) == c and g3(g) == c for g, o in rows); fp = sum(g3(o) == c and g3(g) != c for g, o in rows); fn = sum(g3(o) != c and g3(g) == c for g, o in rows)
        pr = tp / (tp + fp) if tp + fp else 0; rc = tp / (tp + fn) if tp + fn else 0; f1.append(2 * pr * rc / (pr + rc) if pr + rc else 0)
    pos = [(g, o) for g, o in rows if g not in NEG]; neg = [(g, o) for g, o in rows if g in NEG]
    return dict(n=len(rows), f1_3=round(sum(f1) / 3, 3), pos_as_neg=round(sum(o in NEG for _, o in pos) / max(1, len(pos)), 3),
                neg_found=round(sum(o in NEG for _, o in neg) / max(1, len(neg)), 3), undecided=round(sum(o is None for _, o in rows) / len(rows), 3))
torch.cuda.reset_peak_memory_stats(); t = time.time(); tok, model = llm_client._load(); torch.cuda.synchronize()
out("LOAD", round(time.time() - t, 1), "s adapters", sorted(llm_client._adapters), "gb", round(torch.cuda.memory_allocated() / 2**30, 2), "quant", S.companion_llm_quant)
def mood_prob(text):
    S.companion_fusion_rule = "prob_add"; return MoodService(llm_client).classify(text)
def mood_gen(text):
    S.companion_fusion_rule = "plan41"; r = MoodService(llm_client).classify(text); S.companion_fusion_rule = "prob_add"; return r
def mood_4sent(text):  # fuse_cmp 방식: 4코드 정답 문장 로그확률 softmax (같은 어댑터·같은 입력)
    p = llm_client._chat_ids(tok, MOOD_PROMPT.read_text(encoding="utf-8"), "대답:\n" + text); lps = []
    with llm_client._gate.use(MOOD_ADAPTER, llm_client._switch):
        for c in MOOD_CODES:
            tt = torch.tensor(tok(json.dumps({"mood": c}), add_special_tokens=False)["input_ids"]); x = torch.cat([p, tt])[None].to(model.device)
            with torch.inference_mode(): lg = model(input_ids=x).logits[0, len(p) - 1:-1].float().log_softmax(-1)
            lps.append(float(lg.gather(1, tt.to(model.device)[:, None]).sum()))
    mx = max(lps); e = [math.exp(v - mx) for v in lps]; s = sum(e); return dict(zip(MOOD_CODES, [v / s for v in e]))
def timed(fn, *a):
    torch.cuda.synchronize(); s = time.time(); r = fn(*a); torch.cuda.synchronize(); return r, time.time() - s
def pct(xs, q): xs = sorted(xs); return round(xs[min(len(xs) - 1, int(q * len(xs)))], 3)

if "mood" in PARTS:
    cc = [x for x in json.loads((R / "every_neg_cand2_labels.json").read_text()) if x["split"] == "test"]
    rows_g, rows_p, rows_4, lat_p, lat_4, agree, unk5 = [], [], [], [], [], 0, 0
    for x in cc:
        g = mood_gen(x["text"]); rows_g.append((x["code"], g.label if g.usable else None))
        p, tp = timed(mood_prob, x["text"]); q, tq = timed(mood_4sent, x["text"]); lat_p.append(tp); lat_4.append(tq)
        a = max(MOOD_CODES, key=lambda c: q[c]); rows_p.append((x["code"], p.label)); rows_4.append((x["code"], a)); agree += p.label == a
    out("RESULT mood CareCall200 생성(plan41 경로)", json.dumps(metrics(rows_g)))
    out("RESULT mood CareCall200 첫토큰(prob_add 글만)", json.dumps(metrics(rows_p)))
    out("RESULT mood CareCall200 4문장", json.dumps(metrics(rows_4)), "판정일치", f"{agree}/{len(cc)}")
    out("RESULT mood 지연 첫토큰 중간", pct(lat_p, .5), "느린5%", pct(lat_p, .95), "| 4문장 중간", pct(lat_4, .5), "느린5%", pct(lat_4, .95))

if "risk" in PARTS:
    sets = json.loads((R / "risk_jh1109.json").read_text())["sets"]; res = {}
    for name in ("ho100", "tricky", "everyday", "carecall"):
        rows = sets[name]; lat = []
        for r in rows:
            r["p_int8"], dt = timed(rk.risk_probability, llm_client, r["text"]); lat.append(dt); r["kw4"] = rk.keyword_risk(r["text"])
        groups = collections.defaultdict(list)
        for r in rows: groups[r.get("cat") or ("위험" if r["gold"] else "평범")].append(r)
        for gname, g in groups.items():
            f = lambda fn: round(sum(map(fn, g)) / len(g), 3)
            res[f"{name}/{gname}"] = dict(n=len(g), v9_int8=f(lambda r: r["p_int8"] is not None and r["p_int8"] >= .99), v9_bf16_기록=f(lambda r: r["p9"] >= .99),
                                         kw4=f(lambda r: r["kw4"]), 운영_v9_or_kw4=f(lambda r: r["kw4"] or (r["p_int8"] or 0) >= .99), 재희낱말=f(lambda r: bool(r.get("kw"))))
            out("RESULT risk", f"{name}/{gname}", json.dumps(res[f"{name}/{gname}"], ensure_ascii=False))
        out("RESULT risk 지연", name, "중간", pct(lat, .5), "느린5%", pct(lat, .95))
        same = sum((r["p_int8"] >= .99) == (r["p9"] >= .99) for r in rows); out("RESULT risk int8·bf16 판정일치", name, f"{same}/{len(rows)}")

if "reply" in PARTS:
    rows = json.loads((R / "reply_eval2.json").read_text()); res = {}
    for key, lora in (("대답어댑터", True), ("기본모델(재희)", False)):
        S.companion_reply_lora = lora; fb = 0; lat = []; random.seed(1110); torch.manual_seed(1110)
        for r in rows:
            tm = {}; pl.analyze_turn_result(r["text"], {"label": None, "usable": False}, mood_prob, text_out=tm)  # 운영처럼 글 기분을 넘긴다 (11/12 L-1)
            rr, dt = timed(rs.generate_reply, llm_client, r["q"], r["text"], None, False, 1, text_mood=tm if S.companion_risk_first else None); lat.append(dt); fb += rr.source == "FALLBACK"; r[key] = rr.text
        out("RESULT reply", key, "대체비율", round(fb / len(rows), 3), f"({fb}/{len(rows)})", "지연 중간", pct(lat, .5), "느린5%", pct(lat, .95))
    S.companion_reply_lora = True
    for r in rows[:5]: out("SAMPLE reply", r["text"][:30], "→", r["대답어댑터"][:60])

if "fuse" in PARTS:
    import actual_base as ab
    items = [x for x in json.loads((R / "text_kcelectra_aihub263.json").read_text())["items"] if x["gold"] != "surprise"]
    turbo = json.loads((R / "e2e_stt_hf_turbo.json").read_text()); agree = n = 0
    for x in items:
        t = turbo.get(x["id"], "").strip(); x["t"] = t
        if t:
            x["p"] = mood_prob(t).scores; q = mood_4sent(t); n += 1
            agree += max(MOOD_CODES, key=lambda c: x["p"][c]) == max(MOOD_CODES, key=lambda c: q[c]); x["q"] = q
        else: x["p"] = x["q"] = None
    out("RESULT fuse 글 판정 첫토큰·4문장 일치", f"{agree}/{n}")
    face = json.loads((R / "cmp_face_rows.json").read_text()); pool = collections.defaultdict(list)
    for r in face: pool[r["label"]].append(r)
    FC = {0: "JOY", 1: "CALM", 2: "WORRY"}
    def face_code(lbl, rng):
        v = [r["head3_yunet"] for r in rng.sample(pool[lbl], 3) if r["head3_yunet"] >= 0]
        if len(v) < 2: return None
        k, c = collections.Counter(v).most_common(1)[0]; return FC[k] if c >= 2 else None
    def scen(s, g, rng):
        if s == "S1": return ab.FACE_OF[g]
        if s == "S2": return "중립"
        return ab.FACE_OF[g] if rng.random() < 0.5 else "중립"
    TURN_CODE = {"기쁨": "JOY", "평온": "CALM", "슬픔": "WORRY", "우울": "DEPRESSED"}
    FACE_LABEL = {"JOY": "Happiness", "CALM": "Neutral", "WORRY": "Sadness"}
    for key in ("p", "q"):
        for sc in ("S1", "S2", "S3"):
            for rule in ("prob_add", "plan41"):
                for part in ("test", "all"):
                    rng = random.Random(1109); rows = []
                    for i, x in enumerate(items):
                        g = ab.G7[x["gold"]]; fc = face_code(scen(sc, g, rng), rng)
                        if part == "test" and not i % 2: continue
                        if rule == "prob_add":
                            fz = fu.fuse_prob_add(not x["t"], x[key], fc, 0.2)
                        else:
                            tl = fu.MOOD_TO_TURN[max(MOOD_CODES, key=lambda c: x[key][c])] if x[key] else None
                            fz = fu.fuse_turn(not x["t"], tl, tl is not None, fu.HSEMOTION_TO_TURN[FACE_LABEL[fc]] if fc else None, fc is not None)
                        rows.append((g, TURN_CODE.get(fz.label)))
                    out("RESULT fuse", "글확률=" + ("첫토큰" if key == "p" else "4문장"), sc, rule, part, json.dumps(metrics(rows)))

if "face" in PARTS:
    sys.argv = ["x", "0"]; import cmp_face as cf
    from app.services.companion.face_service import FaceService
    from PIL import Image, ImageOps
    import numpy as np
    va, Wv, bv = cf.body(f"{cf.F}/enet_b0_8_best_vgaf.pt")
    h = torch.load(f"{cf.F}/head3_yunet.pt", map_location="cuda"); head = torch.nn.Linear(1280, 3).cuda(); head.load_state_dict(h["state"]); head.eval(); mu, sd = h["mu"].cuda(), h["sd"].cuda()
    def pt_code(path):  # turn_bench.face_code 그대로
        im = ImageOps.exif_transpose(Image.open(path)).convert("RGB"); im.thumbnail((cf.MAX_SIDE, cf.MAX_SIDE))
        x = cf.detect("yunet", im, np.ascontiguousarray(np.asarray(im)[:, :, ::-1]))
        if x is None: return None
        with torch.no_grad(): p = torch.softmax(head((va(x[None].cuda()) - mu) / sd), -1)[0]
        return 2 if float(p[2]) >= 0.8 else int(p[:2].argmax())
    svc = FaceService(); S.companion_face_head = "head3"; assert svc._use_head3()
    faces = [r for r in cf.fe.collect() if int(hashlib.md5(r["person"].encode()).hexdigest(), 16) % 10 < 3]
    random.Random(1110).shuffle(faces); faces = faces[:400]; same = both_none = 0; lat = []; conf = collections.Counter()
    NAME = {"Happiness": 0, "Neutral": 1, "Sadness": 2}
    for r in faces:
        a = pt_code(r["path"]); raw = open(r["path"], "rb").read(); s = time.time(); b = svc._head3_frame(raw); lat.append(time.time() - s)
        b = NAME[b[0]] if b else None; same += a == b; conf[(a, b)] += 1
    out("RESULT face ONNX(운영 코드)·pt(시험 코드) 판정일치", f"{same}/{len(faces)}", "엇갈림", {f"{k}": v for k, v in conf.items() if k[0] != k[1]}, "프레임 지연 중간", pct(lat, .5), "느린5%", pct(lat, .95))
torch.cuda.synchronize(); out("PEAK_GB", round(torch.cuda.max_memory_allocated() / 2**30, 2))
