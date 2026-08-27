"""검색 개선이 리랭커를 거쳐 최종 순위까지 전달되는지 진단한다 (읽기 전용).

리랭커 코드/모델은 일절 건드리지 않는다. agent/mapping/reranker.rerank()를 그대로 호출한다.

설계가 2x2인 이유: "현재 운영 그대로"는 재현이 불가능하다 —
table_embeddings_cache.json이 2개 항목/30차원짜리 스텁이라 embedding_search 경로가 죽어 있고,
keyword_search는 64개 수동 카탈로그만 본다. 그걸 끼워 넣으면 검색 변경과 후보소스 변경이
동시에 일어나 원인 분리가 안 된다. 그래서 후보 소스를 dense 하나로 고정하고
(검색 질의) x (리랭커 유무)만 교차시킨다.

  A = dense_full + 리랭커      B = ctxD2 + 리랭커
  D = dense_full 검색순서      C = ctxD2 검색순서
"""
from __future__ import annotations

import json, math, os, sys, time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from agent.interfaces import Claim, TableCandidate
from agent.mapping.reranker import rerank

import psycopg2

HERE = Path(__file__).parent
ROOT = Path(__file__).resolve().parents[2]
POOL = 100          # 리랭커에 넣는 후보 수
K_LIST = (1, 3, 5, 10)


def metrics(ranked: list[str], gold: set[str]) -> dict:
    pos = {t: i for i, t in enumerate(ranked)}
    first = min((pos[g] + 1 for g in gold if g in pos), default=None)
    out = {"rank": first}
    for k in K_LIST:
        out[f"recall@{k}"] = sum(1 for g in gold if pos.get(g, 10**9) < k) / len(gold)
    out["mrr"] = 1.0 / first if first else 0.0
    rels = [1.0 if t in gold else 0.0 for t in ranked[:10]]
    dcg = sum(r / math.log2(i + 2) for i, r in enumerate(rels))
    idcg = sum(1.0 / math.log2(i + 2) for i in range(min(len(gold), 10)))
    out["ndcg@10"] = dcg / idcg if idcg else 0.0
    return out


def agg(rows: list[dict]) -> dict:
    n = len(rows)
    o = {f"recall@{k}": sum(r[f"recall@{k}"] for r in rows) / n for k in K_LIST}
    o["mrr"] = sum(r["mrr"] for r in rows) / n
    o["ndcg@10"] = sum(r["ndcg@10"] for r in rows) / n
    rk = [r["rank"] for r in rows if r["rank"]]
    o["median_rank"] = sorted(rk)[len(rk) // 2] if rk else None
    o["miss_rate"] = 1 - len(rk) / n
    return o


def main() -> None:
    ev = json.loads((ROOT / "benchmark/search_experiment/eval_set.json").read_text(encoding="utf-8"))
    runs = json.loads((HERE / "runs.json").read_text(encoding="utf-8"))
    gold = {r["claim_id"]: set(r["gold"]) for r in ev}

    ids = sorted({t for k in ("dense_full", "dense_ctxD2") for l in runs[k].values() for t in l[:POOL]})
    conn = psycopg2.connect(os.environ["SUPABASE_DB_URL"]); cur = conn.cursor()
    cur.execute("select table_id, table_name, embedding_text from kosis_vdb_tables_qwen "
                "where table_id = any(%s)", (ids,))
    name, text = {}, {}
    for t, n, x in cur.fetchall():
        name[t], text[t] = n, x
    conn.close()
    print(f"후보 표 메타 {len(name):,}건 로드", flush=True)

    ARMS = {"A (dense_full + 리랭커)": ("dense_full", True),
            "B (Context D2 + 리랭커)": ("dense_ctxD2", True),
            "C (Context D2, 리랭커 X)": ("dense_ctxD2", False),
            "D (dense_full, 리랭커 X)": ("dense_full", False)}

    per, final_rank = {a: {} for a in ARMS}, {a: {} for a in ARMS}
    for arm, (retr, use_rr) in ARMS.items():
        t0 = time.time()
        for r in ev:
            cid = r["claim_id"]
            cands_ids = runs[retr][cid][:POOL]
            if use_rr:
                cands = [TableCandidate(table_id=t, table_name=name.get(t, t),
                                        score=1.0 / (i + 1), required_slots=[],
                                        source_meta=f"vdb_rank={i + 1}")
                         for i, t in enumerate(cands_ids)]
                claim = Claim(sentence=r["sentence"], claim_type="규모")
                out = rerank(claim, cands, top_k=POOL,
                             document_texts={t: text.get(t, name.get(t, t)) for t in cands_ids})
                ranked = [c.table_id for c in out]
            else:
                ranked = cands_ids
            final_rank[arm][cid] = ranked
            per[arm][cid] = metrics(ranked, gold[cid])
        print(f"  [{arm}] 완료 {time.time()-t0:.0f}s", flush=True)

    summary = {a: agg(list(p.values())) for a, p in per.items()}
    (HERE / "e2e_rerank_check.json").write_text(json.dumps(
        {"summary": summary, "per_claim": per,
         "final_top10": {a: {c: v[:10] for c, v in fr.items()} for a, fr in final_rank.items()}},
        ensure_ascii=False, indent=1), encoding="utf-8")

    print(f"\n{'조합':<26}{'R@1':>8}{'R@3':>8}{'R@5':>8}{'MRR':>8}{'NDCG@10':>10}")
    for a in ARMS:
        s = summary[a]
        print(f"  {a:<24}{s['recall@1']*100:>7.1f}%{s['recall@3']*100:>7.1f}%"
              f"{s['recall@5']*100:>7.1f}%{s['mrr']:>8.3f}{s['ndcg@10']:>10.3f}")
    print("\n저장: e2e_rerank_check.json")


if __name__ == "__main__":
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    main()
