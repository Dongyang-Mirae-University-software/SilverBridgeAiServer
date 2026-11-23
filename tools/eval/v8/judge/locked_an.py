"""사용: locked_an.py <rule.json> <agg.jsonl> <bodies.json> <summary.json>  (본문은 세기만, 출력 안 함)"""
import json, sys, re, math
from collections import defaultdict, Counter
rule, agg, bodies, summ = sys.argv[1:5]
VALID = {'F2','G1','G2','G3','G4','H1','H2','I1'}
def wil(p, n, z=1.96):
    if n == 0: return (0, 0)
    ph = p/n; d = 1+z*z/n; c = ph+z*z/(2*n); r = z*math.sqrt(ph*(1-ph)/n+z*z/(4*n*n))
    return (round((c-r)/d, 3), round((c+r)/d, 3))
def table(seedpass):
    c = defaultdict(lambda: [0, 0])
    for (a, s), v in seedpass.items(): c[a][0] += v; c[a][1] += 1
    out = {}
    for a, (p, n) in sorted(c.items()):
        need = 1.0 if a == 'D1' else 0.8
        out[a] = (p, n, round(p/n, 3), wil(p, n), '충족' if p/n >= need else '미달')
    return out
g = json.load(open(rule))['graded']; sp = defaultdict(lambda: True); ng = 0
for e in g:
    if not e.get('graded'): ng += 1; continue
    sp[(e['area'], e['seed_id'])] &= bool(e['pass'])
print('RULE graded', sum(1 for e in g if e.get('graded')), 'pass', sum(bool(e['pass']) for e in g if e.get('graded')), 'ungraded', ng, 'seeds', len(sp))
for a, v in table(sp).items(): print(' R', a, v)
a = json.load(open(agg))
for mode in ['valid', 'all']:
    sp = defaultdict(lambda: True)
    for x in a:
        if mode == 'valid' and x['area'] in 'FGHI' and x['cid'] not in VALID: continue
        sp[(x['area'], x['seed_id'])] &= bool(x['majority'])
    print('JUDGE', mode)
    for ar, v in table(sp).items(): print(' J', ar, v)
z = Counter(x['cid'] for x in a if not any(x['votes'])); print('3:0 fail', sum(z.values()), dict(z), 'split', sum(x['split'] for x in a), 'n', len(a))
print('cids', dict(Counter(x['cid'] for x in a)))
s = json.load(open(summ)); print('SUMMARY', {k: v for k, v in s.items() if k != 'rows'})
rows = s['rows']; print(' engine', dict(Counter(r.get('engine') for r in rows)), 'err', sum(bool(r.get('err')) for r in rows), 'n', len(rows))
B = json.load(open(bodies))
P = {'지시문_새어나옴': r'마크다운|출력하지\s*마|말투는\s*존댓말', '증상_뒤집힘(소리가 나지 않거나)': r'소리가\s*나지\s*않거나',
     '증상_뒤집힘(지나도~않거나)': r'지나도[^.\n]{0,15}(나지|생기지|들리지)\s*않거나', '놓친_약_고정문장': r'놓친\s*약',
     '물집_떼기·만들기': r'물집을\s*(떼|만들)', '영어단어(4자+)': r'[A-Za-z]{4,}'}
cnt = {k: 0 for k in P}; ids = {k: set() for k in P}; gdrug = 0
for b in B:
    t = b.get('reply') or ''
    for k, p in P.items():
        if re.search(p, t): cnt[k] += 1; ids[k].add(b['id'].split('.')[1])
    if re.search(P['놓친_약_고정문장'], t) and b['id'].split('.')[1] in ('G', 'Gp'): gdrug += 1
print('BODIES', len(B), {k: (cnt[k], sorted(ids[k])) for k in P}, 'G영역_놓친약', gdrug)
