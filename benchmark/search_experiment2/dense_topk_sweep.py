"""DENSE_TOP_K 스윕 — dense가 몇 개를 리랭커에 넘겨야 최종 top-1이 최대인가.

깔때기 분석에서 dense는 100위 안에 53/70건을 찾는데 파이프라인이 상위 10개만 넘겨
34/70건만 리랭커에 도달하는 것이 확인됐다(11건이 11~200위에 있는 채로 잘림).
올리면 리랭커가 채점할 후보가 늘어 느려지므로, 어디가 균형점인지 잰다.
"""
from __future__ import annotations
import json, os, sys, time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT)); sys.path.insert(0, str(Path(__file__).parent))
from dotenv import load_dotenv; load_dotenv(ROOT / ".env")

from agent.interfaces import Claim
from agent.preprocessing.claim_extractor import attach_sentence_context
from agent.mapping.reranker import search_and_rerank, build_retrieval_query, build_lexical_query
from agent.mapping.embedding_search import embedding_search, build_table_embedding_cache
from agent.mapping.keyword_search import keyword_search
from agent.kosis.query_vdb import batch_query_vdb, VdbUnavailableError
from agent.kosis.bm25_search import bm25_query_vdb, BM25_TOP_K
from agent.pipeline.batch_runner import _load_table_catalog_by_id
import queries as Q

SWEEP = [10, 30, 50, 100]
TOP_K = 10
_STOP = {"전국", "전체", "국내", "없음", "nan", "", "-", "KOSIS"}


def clean(v):
    v = (v or "").strip()
    return "" if v.lower() in {s.lower() for s in _STOP} else v


def main() -> None:
    ev = json.loads((ROOT / "benchmark/search_experiment/eval_set.json").read_text(encoding="utf-8"))
    slots = json.loads((ROOT / "benchmark/search_experiment/claim_slots.json").read_text(encoding="utf-8"))
    arts, cmap = Q.load_articles(), Q.load_claim_article_map()

    by_art: dict[str, list[dict]] = {}
    for r in ev:
        by_art.setdefault(cmap.get(r["claim_id"], ""), []).append(r)
    claims: dict[str, Claim] = {}
    for aid, rows in by_art.items():
        objs = []
        for r in rows:
            s = slots.get(r["claim_id"], {})
            objs.append(Claim(sentence=r["sentence"], claim_type="규모",
                              statistic_expression=clean(s.get("statistic_expression")) or None,
                              population=clean(s.get("population")) or None,
                              region=clean(s.get("region")) or None,
                              source_org=clean(s.get("source_org")) or None))
        attach_sentence_context(arts.get(aid, ""), objs)
        for r, c in zip(rows, objs):
            claims[r["claim_id"]] = c

    ids = [r["claim_id"] for r in ev]
    gold = {r["claim_id"]: set(r["gold"]) for r in ev}
    catalog_by_id = _load_table_catalog_by_id()
    document_texts = {tid: t["embedding_text"] for tid, t in catalog_by_id.items()}
    embedding_cache = build_table_embedding_cache()

    from sentence_transformers import SentenceTransformer
    print("Qwen3-Embedding-4B 로딩 중...", flush=True)
    model = SentenceTransformer("Qwen/Qwen3-Embedding-4B", truncate_dim=2560, device="cuda")
    instruction = ("Given a Korean news claim sentence, retrieve the KOSIS statistical table "
                   "description that best matches it")
    os.environ["KOSIS_QUERY_CONTEXT"] = "1"

    # 질의 벡터는 dense_top_k와 무관하므로 한 번만 인코딩해서 재사용한다.
    print("질의 인코딩...", flush=True)
    vecs = {}
    for cid in ids:
        text = f"Instruct: {instruction}\nQuery: {build_retrieval_query(claims[cid])}"
        vecs[cid] = model.encode([text], convert_to_numpy=True, normalize_embeddings=True)[0].tolist()

    def bm25_fn(claim):
        try:
            return bm25_query_vdb(build_lexical_query(claim), top_k=BM25_TOP_K)
        except VdbUnavailableError:
            return []

    out = {}
    for dk in SWEEP:
        cur_cid = {"v": None}

        def vdb_fn(claim, _dk=dk):
            try:
                return batch_query_vdb([vecs[cur_cid["v"]]], top_k=_dk)[0]
            except VdbUnavailableError:
                return []

        print(f"\n=== DENSE_TOP_K={dk} ===", flush=True)
        ranks, t0 = {}, time.time()
        for n, cid in enumerate(ids, 1):
            cur_cid["v"] = cid
            try:
                cands = search_and_rerank(
                    claims[cid], keyword_fn=keyword_search,
                    embedding_fn=lambda c: embedding_search(c, cache=embedding_cache),
                    vdb_fn=vdb_fn, bm25_fn=bm25_fn,
                    top_k=TOP_K, document_texts=document_texts,
                )
                got = [c.table_id for c in cands]
                ranks[cid] = next((i + 1 for i, t in enumerate(got) if t in gold[cid]), None)
            except Exception as e:  # noqa: BLE001
                print(f"  {cid} 실패: {type(e).__name__}: {e}", flush=True)
                ranks[cid] = None
            if n % 35 == 0:
                print(f"  {n}/{len(ids)} ({time.time()-t0:.0f}s)", flush=True)
        out[dk] = {"ranks": ranks, "sec": time.time() - t0}
        print(f"  완료 {out[dk]['sec']:.0f}s", flush=True)

    def acc(ranks, k):
        return sum(1 for c in ids if ranks[c] and ranks[c] <= k) / len(ids)

    def mrr(ranks):
        return sum(1.0 / ranks[c] for c in ids if ranks[c]) / len(ids)

    print(f"\n{'DENSE_TOP_K':<14}{'top-1':>9}{'top-5':>9}{'top-10':>9}{'MRR':>9}{'claim당':>10}")
    for dk in SWEEP:
        r = out[dk]["ranks"]
        print(f"{dk:<14}" + "".join(f"{acc(r,k):>8.1%} " for k in (1, 5, 10))
              + f"{mrr(r):>8.3f}" + f"{out[dk]['sec']/len(ids):>9.2f}s")

    base = out[SWEEP[0]]["ranks"]
    for dk in SWEEP[1:]:
        r = out[dk]["ranks"]
        win = [c for c in ids if base[c] != 1 and r[c] == 1]
        loss = [c for c in ids if base[c] == 1 and r[c] != 1]
        print(f"\nTOP_K={dk} vs {SWEEP[0]} (top-1): +{len(win)} / -{len(loss)} → 순증 {len(win)-len(loss):+d}")
        if loss:
            print(f"  손해: {loss}")

    (Path(__file__).parent / "dense_topk_sweep.json").write_text(
        json.dumps({str(k): v for k, v in out.items()}, ensure_ascii=False, indent=1), encoding="utf-8")


if __name__ == "__main__":
    main()
