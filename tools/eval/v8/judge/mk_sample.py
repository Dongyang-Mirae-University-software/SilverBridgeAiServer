#!/usr/bin/env python3
"""학생 이중 채점 표본 HTML(DGX 안에서만 실행). 본문은 출력하지 않는다."""
import json, os, random, html, collections, hashlib
H = os.path.expanduser
O = H('~/chat-eval/out'); J = H('~/chat-eval/judge')
items = [json.loads(l) for l in open(f'{O}/ha_judge_in.jsonl')]
summ = json.load(open(f'{O}/ha_judge_summary.json'))
split = set(summ['split_ids'])                      # "id#run#cid"
key = lambda it: f"{it['id']}#{it.get('run', 1)}"
has_split = [it for it in items if any(f"{key(it)}#{c['cid']}" in split for c in it['criteria'])]
rest = [it for it in items if it not in has_split]
rng = random.Random(20261122)
pri = [it for it in rest if it['area'] in ('F', 'H', 'G')]; oth = [it for it in rest if it not in pri]
rng.shuffle(pri); rng.shuffle(oth)
N = 40
pick = has_split + pri[:max(0, N - len(has_split))]
pick += oth[:max(0, N - len(pick))]
rng.shuffle(pick)
crit_split = sum(1 for it in pick for c in it['criteria'] if f"{key(it)}#{c['cid']}" in split)
E = lambda s: html.escape(str(s or ''))
rows, mapping, order = [], [], []
for n, it in enumerate(pick, 1):
    sid = f"S{n:02d}"
    turns = it.get('user_turns') or []
    q = ''.join(f'<div class="turn"><b>({i}) {"어르신" if it.get("speaker","본인")=="본인" else E(it.get("speaker"))}</b> {E(t)}</div>' for i, t in enumerate(turns, 1))
    refs = ''.join(f'<li>{E(r)}</li>' for r in (it.get('reference_points') or []))
    cs = ''
    for c in it['criteria']:
        k = f"{sid}.{c['cid']}"; order.append(k)
        mapping.append({'sample': k, 'id': key(it), 'cid': c['cid'], 'area': it['area'], 'split': f"{key(it)}#{c['cid']}" in split})
        cs += f'''<div class="crit"><div class="ct"><b>{E(c["cid"])}</b> {E(c["text"])}</div>
<div class="ex"><span class="ok">충족 예시</span> {E(c.get("met_example"))}<br><span class="no">미충족 예시</span> {E(c.get("unmet_example"))}</div>
<div class="pick"><label><input type="radio" name="{k}" value="1"> 충족</label><label><input type="radio" name="{k}" value="0"> 미충족</label></div></div>'''
    rows.append(f'''<section><h2>{sid} <small>말하는 사람: {E(it.get("speaker","본인"))}</small></h2>
<h3>사용자 질문</h3>{q}<h3>챗봇 답</h3><pre>{E(it.get("answer"))}</pre>
<details><summary>이 문항의 참조 요점(기준 판단에 참고)</summary><ul>{refs}</ul></details>{cs}</section>''')
ver = hashlib.sha256(json.dumps(order).encode()).hexdigest()[:8]
page = f'''<!doctype html><html lang="ko"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">
<title>챗봇 답 채점표</title><style>
:root{{--bg:#fff;--fg:#1d1d1f;--mut:#666;--line:#ddd;--card:#f7f7f8;--ok:#0a7a3e;--no:#b42318}}
@media (prefers-color-scheme:dark){{:root{{--bg:#151516;--fg:#eee;--mut:#aaa;--line:#333;--card:#1f1f21;--ok:#4cc38a;--no:#ff7a6e}}}}
body{{background:var(--bg);color:var(--fg);font:16px/1.6 system-ui,"Apple SD Gothic Neo","Malgun Gothic",sans-serif;max-width:860px;margin:0 auto;padding:16px}}
section{{border:1px solid var(--line);border-radius:10px;padding:14px 16px;margin:18px 0;background:var(--card)}}
h2{{margin:0 0 6px;font-size:20px}} h2 small{{font-weight:400;color:var(--mut);font-size:14px;margin-left:8px}} h3{{font-size:15px;margin:12px 0 4px;color:var(--mut)}}
pre{{white-space:pre-wrap;word-break:break-word;font:inherit;background:var(--bg);border:1px solid var(--line);border-radius:8px;padding:10px;margin:0}}
.turn{{margin:2px 0}} .crit{{border-top:1px dashed var(--line);margin-top:12px;padding-top:10px}} .ex{{font-size:14px;color:var(--mut);margin:4px 0}}
.ok{{color:var(--ok);font-weight:600}} .no{{color:var(--no);font-weight:600}} .pick label{{margin-right:18px;cursor:pointer}} .pick input{{transform:scale(1.25);margin-right:6px}}
.guide{{border-left:4px solid var(--ok);padding:8px 12px;background:var(--card)}}
#out{{position:sticky;bottom:0;background:var(--bg);border-top:1px solid var(--line);padding:10px 0}}
textarea{{width:100%;box-sizing:border-box;font:13px monospace;background:var(--card);color:var(--fg);border:1px solid var(--line);border-radius:6px}}
button{{font:inherit;padding:6px 14px;border-radius:6px;border:1px solid var(--line);background:var(--card);color:var(--fg);cursor:pointer}}
</style></head><body>
<h1>챗봇 답 채점표 <small style="font-size:14px;color:var(--mut)">표본 {len(pick)}문항 · 기준 {len(order)}개 · 판 {ver}</small></h1>
<div class="guide"><b>채점 방법</b><br>
1. 기준마다 <b>“이 기준이 말하는 내용이 챗봇 답에 있는가”</b>만 봅니다. 의학적으로 맞는지는 판단하지 않습니다.<br>
2. 표현이 달라도 같은 뜻이면 <span class="ok">충족</span>, 빠졌거나 흐릿하거나 반대로 말하면 <span class="no">미충족</span>입니다.<br>
3. 다른 사람과 상의하지 말고 혼자 채점합니다. 모든 기준을 고르면 맨 아래에 결과 코드가 나옵니다. 코드 한 줄을 복사해 보내 주세요.</div>
{''.join(rows)}
<div id="out"><label>채점자 이니셜 <input id="who" maxlength="4" size="5"></label> <span id="cnt"></span>
<textarea id="code" rows="2" readonly placeholder="모든 기준을 고르면 결과 코드가 나옵니다"></textarea> <button id="cp">코드 복사</button></div>
<script>
const ORDER={json.dumps(order)}, VER="{ver}";
function upd(){{let b="",n=0;for(const k of ORDER){{const e=document.querySelector('input[name="'+k+'"]:checked');if(e){{b+=e.value;n++}}else b+="?"}}
document.getElementById("cnt").textContent=n+" / "+ORDER.length+" 골랐음";
const w=(document.getElementById("who").value||"").replace(/[^A-Za-z가-힣0-9]/g,"");
document.getElementById("code").value=(n===ORDER.length&&w)?("V8S|"+VER+"|"+w+"|"+b):"";}}
document.addEventListener("change",upd);document.getElementById("who").addEventListener("input",upd);
document.getElementById("cp").onclick=()=>{{const t=document.getElementById("code");if(t.value){{t.select();try{{navigator.clipboard.writeText(t.value)}}catch(e){{document.execCommand("copy")}}}}}};upd();
</script></body></html>'''
os.makedirs(f'{J}/sample', exist_ok=True); os.chmod(f'{J}/sample', 0o700)
open(f'{J}/sample/student_sample_{ver}.html', 'w', encoding='utf-8').write(page)
json.dump({'ver': ver, 'order': order, 'map': mapping}, open(f'{J}/sample/sample_key_{ver}.json', 'w'), ensure_ascii=False, indent=1)
print('ver', ver, 'items', len(pick), 'criteria', len(order), 'split_items', len(has_split), 'split_criteria_in_sample', crit_split, '/', len(split))
print('areas', dict(collections.Counter(it['area'] for it in pick)))
print('bytes', len(page.encode()))
