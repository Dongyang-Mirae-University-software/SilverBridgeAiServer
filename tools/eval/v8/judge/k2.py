import json, os, sys
from collections import defaultdict
sys.path.insert(0, os.path.expanduser('~/chat-eval/judge'))
H=os.path.expanduser; J=H('~/chat-eval/judge'); O=H('~/chat-eval/out')
def kappa(a,b):
    n=len(a)
    if n==0: return None
    po=sum(x==y for x,y in zip(a,b))/n; pa,pb=sum(a)/n,sum(b)/n; pe=pa*pb+(1-pa)*(1-pb)
    return None if pe==1 else round((po-pe)/(1-pe),3)
key=json.load(open(J+'/sample/sample_key_613ba5c8.json')); mp=key['map']
code=open(J+'/sample/code_claude.txt',encoding='utf-8').read().strip().split('|')
assert code[0]=='V8S' and code[1]==key['ver'] and len(code[3])==len(mp)
c=[int(x) for x in code[3]]
agg={(r['id'],r['cid']):r for r in json.load(open(O+'/ha_agg.jsonl'))}
M={'gpt-oss':lambda v:v[0],'qwen3':lambda v:v[1],'exaone':lambda v:v[2],'다수결':lambda v:int(sum(v)>=2),'셋다충족':lambda v:int(all(v)),'하나라도':lambda v:int(any(v))}
rows=[(mp[i]['cid'],c[i],[int(bool(x)) for x in agg[(mp[i]['id'],mp[i]['cid'])]['votes']],mp[i]['split']) for i in range(len(mp))]
res={'n':len(rows),'claude_met':sum(c),'by_cid':{}, 'all':{}}
for nm,f in M.items():
    res['all'][nm]={'kappa':kappa([r[1] for r in rows],[f(r[2]) for r in rows]),'agree':round(sum(r[1]==f(r[2]) for r in rows)/len(rows),3)}
by=defaultdict(list)
for r in rows: by[r[0]].append(r)
for cid,rs in sorted(by.items()):
    d={'n':len(rs),'claude_met':sum(r[1] for r in rs),'split':sum(r[3] for r in rs)}
    for nm,f in M.items():
        d[nm]=(kappa([r[1] for r in rs],[f(r[2]) for r in rs]), round(sum(r[1]==f(r[2]) for r in rs)/len(rs),2))
    res['by_cid'][cid]=d
json.dump(res,open(J+'/sample/kappa_claude_613ba5c8.json','w'),ensure_ascii=False,indent=1)
print('ALL',json.dumps(res['all'],ensure_ascii=False))
for cid,d in res['by_cid'].items(): print(cid,json.dumps(d,ensure_ascii=False))
# 4차 F: 방식별 상황 통과
a=json.load(open(O+'/ha4_agg.jsonl'))
for nm,f in M.items():
    s=defaultdict(lambda:True)
    for x in a:
        if x['area']=='F':
            met=f([int(bool(v)) for v in x['votes']]) if x['cid']=='F1' else int(bool(x['majority']))
            s[x['seed_id']]&=bool(met)
    print('HA4_F',nm,f'{sum(s.values())}/{len(s)}')
