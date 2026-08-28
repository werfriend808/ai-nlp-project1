"""baseline(full) vs Context D2 지연 A/B 교차 측정 — 워밍업 후 번갈아 실행."""
import json, os, sys, time, statistics as st
sys.path.insert(0,'benchmark/search_experiment2')
from retrievers import connect, dense_tables, vec_literal

ev=json.load(open('benchmark/search_experiment/eval_set.json'))
q=json.load(open('benchmark/search_experiment2/queries.json'))
INS=("Given a Korean news claim sentence, retrieve the KOSIS statistical table "
     "description that best matches it")
from sentence_transformers import SentenceTransformer
m=SentenceTransformer("Qwen/Qwen3-Embedding-4B", truncate_dim=2560)

texts={v:[q[r['claim_id']][v] for r in ev] for v in ('full','ctx_D2')}
enc_ms={}
vecs={}
for v,ts in texts.items():
    inp=[f"Instruct: {INS}\nQuery: {t}" for t in ts]
    m.encode(inp[:4], batch_size=4, normalize_embeddings=True)      # 워밍업
    t0=time.perf_counter()
    e=m.encode(inp, batch_size=1, normalize_embeddings=True)
    enc_ms[v]=(time.perf_counter()-t0)*1000/len(inp)
    vecs[v]=[vec_literal(x) for x in e]
    print(f"  [{v}] 평균 글자수 {sum(len(t) for t in ts)/len(ts):.0f}자, 인코딩 {enc_ms[v]:.0f}ms/건")
del m
import gc, torch; gc.collect(); torch.cuda.empty_cache()

conn=connect(); cur=conn.cursor()
for i in range(20):                                                  # DB 워밍업
    dense_tables(cur, vecs['full'][i%len(ev)], 100)
    dense_tables(cur, vecs['ctx_D2'][i%len(ev)], 100)

db={'full':[], 'ctx_D2':[]}
for rep in range(3):
    for i in range(len(ev)):
        order = ('full','ctx_D2') if (i+rep)%2==0 else ('ctx_D2','full')   # 순서 편향 제거
        for v in order:
            t0=time.perf_counter(); dense_tables(cur, vecs[v][i], 100)
            db[v].append((time.perf_counter()-t0)*1000)

print(f"\n{'':12}{'DB검색 평균':>12}{'중앙':>9}{'p95':>9}{'인코딩':>10}{'종단 합계':>12}")
for v,label in (('full','Baseline'), ('ctx_D2','Context D2')):
    d=sorted(db[v]); p95=d[int(len(d)*0.95)]
    print(f"  {label:<10}{st.mean(d):>10.1f}ms{st.median(d):>8.1f}ms{p95:>8.1f}ms"
          f"{enc_ms[v]:>9.0f}ms{st.mean(d)+enc_ms[v]:>10.0f}ms")
print(f"\n  측정 횟수: 각 {len(db['full'])}회 (워밍업 후 교차 실행)")
