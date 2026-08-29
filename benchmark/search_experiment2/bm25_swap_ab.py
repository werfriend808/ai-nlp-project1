"""trigram -> BM25 교체 전후 비교 (골든셋 70건).

두 가지를 잰다:
 1) 어휘 검색기 단독 — 교체 전(trigram, 운영이 실제로 보내던 질의)과 후(BM25, 새 질의)
 2) dense와 RRF로 합쳤을 때 최종 Recall — 교체가 실제 파이프라인 결과를 바꾸는지

dense 순위는 ctx_prod_ab.json에 캐시된 B(문맥+문장) 결과를 재사용한다(재인코딩 생략).
"""
from __future__ import annotations
import json, sys, time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT)); sys.path.insert(0, str(Path(__file__).parent))
from dotenv import load_dotenv; load_dotenv(ROOT / ".env")

from agent.interfaces import Claim
from agent.preprocessing.claim_extractor import attach_sentence_context
from agent.mapping.reranker import build_retrieval_query, build_lexical_query
from agent.kosis.bm25_search import bm25_query_vdb
from agent.kosis.query_vdb import lexical_query_vdb
import queries as Q

DEPTH = 200
_STOP = {"전국", "전체", "국내", "없음", "nan", "", "-", "KOSIS"}


def clean(v):
    v = (v or "").strip()
    return "" if v.lower() in {s.lower() for s in _STOP} else v


def rrf(rankings: list[list[str]], k: int = 60, top_k: int = DEPTH) -> list[str]:
    scores: dict[str, float] = {}
    for lst in rankings:
        for i, tid in enumerate(lst):
            scores[tid] = scores.get(tid, 0.0) + 1.0 / (k + i + 1)
    return sorted(scores, key=lambda t: -scores[t])[:top_k]


def main() -> None:
    ev = json.loads((ROOT / "benchmark/search_experiment/eval_set.json").read_text(encoding="utf-8"))
    slots = json.loads((ROOT / "benchmark/search_experiment/claim_slots.json").read_text(encoding="utf-8"))
    cached = json.loads((Path(__file__).parent / "ctx_prod_ab.json").read_text(encoding="utf-8"))
    dense = cached["results"]["B_문맥+문장(실험D2)"]
    arts, cmap = Q.load_articles(), Q.load_claim_article_map()

    by_art: dict[str, list[dict]] = {}
    for r in ev:
        by_art.setdefault(cmap.get(r["claim_id"], ""), []).append(r)
    claims: dict[str, Claim] = {}
    for aid, rows in by_art.items():
        objs = [Claim(sentence=r["sentence"], claim_type="규모",
                      statistic_expression=clean(slots.get(r["claim_id"], {}).get("statistic_expression")) or None,
                      population=clean(slots.get(r["claim_id"], {}).get("population")) or None,
                      region=clean(slots.get(r["claim_id"], {}).get("region")) or None,
                      source_org=clean(slots.get(r["claim_id"], {}).get("source_org")) or None)
                for r in rows]
        attach_sentence_context(arts.get(aid, ""), objs)
        for r, c in zip(rows, objs):
            claims[r["claim_id"]] = c

    ids = [r["claim_id"] for r in ev]
    gold = {r["claim_id"]: set(r["gold"]) for r in ev}

    print("BM25 인덱스 로딩...", flush=True)
    t0 = time.time()
    bm25_res, bm25_ms = {}, []
    for cid in ids:
        q = build_lexical_query(claims[cid])
        t = time.time()
        bm25_res[cid] = [c.table_id for c in bm25_query_vdb(q, top_k=DEPTH)]
        bm25_ms.append((time.time() - t) * 1000)
    print(f"BM25 완료 {time.time()-t0:.0f}s (첫 호출에 인덱스 로딩 포함)", flush=True)

    print("trigram(교체 전) 측정 — 느립니다...", flush=True)
    t0 = time.time()
    trgm_res, trgm_ms = {}, []
    for n, cid in enumerate(ids, 1):
        q = build_retrieval_query(claims[cid])   # 교체 전 운영이 보내던 질의
        t = time.time()
        try:
            trgm_res[cid] = [c.table_id for c in lexical_query_vdb(q, top_k=DEPTH)]
        except Exception as e:
            print(f"  {cid} 실패: {e}"); trgm_res[cid] = []
        trgm_ms.append((time.time() - t) * 1000)
        if n % 20 == 0:
            print(f"  {n}/{len(ids)} ({time.time()-t0:.0f}s)", flush=True)

    def hit(res, k):
        return sum(1 for c in ids if gold[c] & set(res[c][:k])) / len(ids)

    print(f"\n{'어휘 검색기 단독':<22}{'hit@10':>9}{'hit@30':>9}{'hit@100':>9}{'평균지연':>11}")
    for name, res, ms in [("trigram (교체 전)", trgm_res, trgm_ms), ("BM25 (교체 후)", bm25_res, bm25_ms)]:
        print(f"{name:<22}" + "".join(f"{hit(res,k):>8.1%} " for k in (10, 30, 100))
              + f"{sum(ms)/len(ms):>9.0f}ms")

    dense_only = {c: dense[c][:DEPTH] for c in ids}
    fused_trgm = {c: rrf([dense[c], trgm_res[c]]) for c in ids}
    fused_bm25 = {c: rrf([dense[c], bm25_res[c]]) for c in ids}

    print(f"\n{'dense와 융합(RRF)':<22}{'R@10':>9}{'R@100':>9}{'R@200':>9}")
    for name, res in [("dense 단독", dense_only), ("dense + trigram", fused_trgm), ("dense + BM25", fused_bm25)]:
        print(f"{name:<22}" + "".join(f"{hit(res,k):>8.1%} " for k in (10, 100, 200)))

    for base, new, label in [(fused_trgm, fused_bm25, "dense+BM25 vs dense+trigram"),
                             (dense_only, fused_bm25, "dense+BM25 vs dense 단독")]:
        win = [c for c in ids if not gold[c] & set(base[c][:100]) and gold[c] & set(new[c][:100])]
        loss = [c for c in ids if gold[c] & set(base[c][:100]) and not gold[c] & set(new[c][:100])]
        print(f"\n{label} (R@100): +{len(win)} / -{len(loss)} → 순증 {len(win)-len(loss):+d}")
        if loss:
            print(f"  손해: {loss}")


if __name__ == "__main__":
    main()
