"""리랭커 질의 A/B — cross-encoder에 문맥(앞 문장)을 같이 넣으면 나아지는가.

검색기 쪽은 2026-08-28에 문맥을 붙였는데 리랭커는 claim.sentence 원문만 본다.
지표명이 앞 문장에만 있는 claim에서 리랭커가 판단할 근거가 없는 상태 —
그게 실제로 최종 순위에 영향을 주는지 잰다. 검색 단계는 두 설정에서 동일하다.
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
from agent.kosis.query_vdb import batch_query_vdb, VDB_TOP_K, VdbUnavailableError
from agent.kosis.bm25_search import bm25_query_vdb, BM25_TOP_K
from agent.pipeline.batch_runner import _load_table_catalog_by_id
import queries as Q

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
    n_ctx = sum(1 for c in claims.values() if c.context_before)
    print(f"claim {len(ids)}건 / 문맥 있는 claim {n_ctx}건", flush=True)

    catalog_by_id = _load_table_catalog_by_id()
    document_texts = {tid: t["embedding_text"] for tid, t in catalog_by_id.items()}
    embedding_cache = build_table_embedding_cache()

    from sentence_transformers import SentenceTransformer
    print("Qwen3-Embedding-4B 로딩 중...", flush=True)
    model = SentenceTransformer("Qwen/Qwen3-Embedding-4B", truncate_dim=2560, device="cuda")
    instruction = ("Given a Korean news claim sentence, retrieve the KOSIS statistical table "
                   "description that best matches it")
    os.environ["KOSIS_QUERY_CONTEXT"] = "1"

    # 검색 벡터는 두 설정에서 동일 — 한 번만 인코딩한다.
    print("질의 인코딩...", flush=True)
    vecs = {cid: model.encode(
        [f"Instruct: {instruction}\nQuery: {build_retrieval_query(claims[cid])}"],
        convert_to_numpy=True, normalize_embeddings=True)[0].tolist() for cid in ids}

    cur = {"cid": None}

    def vdb_fn(claim):
        try:
            return batch_query_vdb([vecs[cur["cid"]]], top_k=VDB_TOP_K)[0]
        except VdbUnavailableError:
            return []

    def bm25_fn(claim):
        try:
            return bm25_query_vdb(build_lexical_query(claim), top_k=BM25_TOP_K)
        except VdbUnavailableError:
            return []

    out = {}
    for name, flag in [("리랭커 질의 = 문장만 (현재)", "0"), ("리랭커 질의 = 문맥+문장", "1")]:
        os.environ["KOSIS_RERANK_CONTEXT"] = flag
        print(f"\n=== {name} ===", flush=True)
        ranks, t0 = {}, time.time()
        for n, cid in enumerate(ids, 1):
            cur["cid"] = cid
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
        out[name] = ranks
        print(f"  완료 {time.time()-t0:.0f}s", flush=True)

    def acc(r, k):
        return sum(1 for c in ids if r[c] and r[c] <= k) / len(ids)

    def mrr(r):
        return sum(1.0 / r[c] for c in ids if r[c]) / len(ids)

    print(f"\n{'설정':<28}{'top-1':>9}{'top-5':>9}{'top-10':>9}{'MRR':>9}")
    for name in out:
        r = out[name]
        print(f"{name:<28}" + "".join(f"{acc(r,k):>8.1%} " for k in (1, 5, 10)) + f"{mrr(r):>8.3f}")

    a, b = list(out.values())
    for k in (1, 5, 10):
        win = [c for c in ids if not (a[c] and a[c] <= k) and (b[c] and b[c] <= k)]
        loss = [c for c in ids if (a[c] and a[c] <= k) and not (b[c] and b[c] <= k)]
        print(f"\ntop-{k}: +{len(win)} / -{len(loss)} → 순증 {len(win)-len(loss):+d}")
        if win:  print(f"  이득: {win}")
        if loss: print(f"  손해: {loss}")

    (Path(__file__).parent / "rerank_query_ab.json").write_text(
        json.dumps(out, ensure_ascii=False, indent=1), encoding="utf-8")


if __name__ == "__main__":
    main()
