#!/usr/bin/env python3
"""학생 결과 코드 → κ. 사용: score_sample.py <코드파일(한 줄에 코드 하나)> [--selftest]"""
import json, os, sys, random
H = os.path.expanduser; J = H('~/chat-eval/judge'); O = H('~/chat-eval/out')

def kappa(a, b):
    n = len(a)
    if n == 0: return None
    po = sum(x == y for x, y in zip(a, b)) / n
    pa, pb = sum(a) / n, sum(b) / n
    pe = pa * pb + (1 - pa) * (1 - pb)
    return 1.0 if pe == 1 else round((po - pe) / (1 - pe), 3)

def load_codes(lines, ver, n):
    out = {}
    for ln in lines:
        ln = ln.strip()
        if not ln: continue
        p = ln.split('|')
        if len(p) != 4 or p[0] != 'V8S': sys.exit(f'형식 오류: {ln[:30]}')
        if p[1] != ver: sys.exit(f'판 불일치 {p[1]} != {ver}')
        if len(p[3]) != n or set(p[3]) - {'0', '1'}: sys.exit(f'{p[2]}: 길이/문자 오류 {len(p[3])}/{n}')
        out[p[2]] = [int(c) for c in p[3]]
    return out

def main():
    key = json.load(open(J + '/sample/sample_key_613ba5c8.json'))
    ver, mp = key['ver'], key['map']
    agg = {(r['id'], r['cid']): r for r in json.load(open(O + '/ha_agg.jsonl'))}
    model = [int(bool(agg[(m['id'], m['cid'])]['majority'])) for m in mp]
    split = [m['split'] for m in mp]
    if '--selftest' in sys.argv:
        rnd = random.Random(1)
        mk = lambda flip: ''.join(str(x ^ (rnd.random() < flip)) for x in model)
        lines = [f'V8S|{ver}|AA|{mk(0.1)}', f'V8S|{ver}|BB|{mk(0.2)}']
    else:
        lines = open(sys.argv[1], encoding='utf-8').read().splitlines()
    st = load_codes(lines, ver, len(mp))
    names = list(st)
    res = {'ver': ver, 'criteria': len(mp), 'graders': names}
    if len(names) >= 2:
        a, b = st[names[0]], st[names[1]]
        res['student_kappa'] = kappa(a, b)
        res['student_agree'] = round(sum(x == y for x, y in zip(a, b)) / len(a), 3)
        agree_idx = [i for i in range(len(a)) if a[i] == b[i]]
        res['consensus_n'] = len(agree_idx)
        res['consensus_vs_model_kappa'] = kappa([a[i] for i in agree_idx], [model[i] for i in agree_idx])
        si = [i for i in agree_idx if split[i]]
        res['split_consensus_n'] = len(si)
        res['split_consensus_vs_model_agree'] = round(sum(a[i] == model[i] for i in si) / len(si), 3) if si else None
    for nm in names:
        res[f'{nm}_vs_model_kappa'] = kappa(st[nm], model)
    res['pass_rule'] = '학생 간 κ ≥0.61 이면 학생 합의를 기준으로 모델 채점 신뢰도 판단(모델-합의 κ ≥0.61 자동 채점 인정)'
    print(json.dumps(res, ensure_ascii=False, indent=1))
    if '--selftest' not in sys.argv:
        json.dump(res, open(J + f'/sample/kappa_{ver}.json', 'w'), ensure_ascii=False, indent=1)

main()
