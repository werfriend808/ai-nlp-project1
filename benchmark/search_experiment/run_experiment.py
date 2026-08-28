"""KOSIS 검색 전략 4종 비교 실험 (읽기 전용).

운영 코드/DB/임베딩을 일절 수정하지 않는다. 운영 상수(질의 instruction, RRF_K,
ef_search 등)는 운영 모듈에서 그대로 import 해서 조건을 맞춘다.

전략:
  A baseline    claim 문장 -> 표 임베딩 dense 검색            (agent/kosis/query_vdb.batch_query_vdb와 동일 쿼리)
  B item        claim 지표어 -> item 임베딩 검색 -> 표로 승격
  C item_axis   B로 넓게 뽑은 뒤 axis_value 조건으로 걸러냄
  D hybrid_rrf  dense + item + lexical + axis 랭킹을 RRF로 융합 (운영 RRF_K 사용)
"""
from __future__ import annotations

import json
import os
import re
import sys
import time
from collections import defaultdict
from pathlib import Path

import numpy as np
import psycopg2

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from agent.mapping.reranker import RRF_K                      # noqa: E402  운영 RRF 상수
from agent.pipeline.rerank_local import VDB_QUERY_INSTRUCTION  # noqa: E402  운영 질의 지시문
from agent.kosis.query_vdb import HNSW_EF_SEARCH               # noqa: E402

HERE = Path(__file__).parent
DB_URL = os.environ["SUPABASE_DB_URL"]
DIM = 2560
POOL = 100          # 모든 전략이 동일하게 최대 100개 후보를 낸다
ITEM_FETCH = 1000   # item 검색 시 끌어올 item 행 수(표로 접으면 훨씬 줄어듦)
TABLE_EF = 400      # 표 dense 검색용 ef_search (LIMIT 100보다 충분히 크게)
ITEM_EF = 1000      # item 검색용 ef_search (ITEM_FETCH 이상. pgvector 상한이 1000이라 여기가 최대)
K_LIST = (1, 10, 100)


# ── 공통 유틸 ────────────────────────────────────────────────────────────
def vec_literal(v) -> str:
    return "[" + ",".join(f"{x:.7f}" for x in v) + "]"


def set_ef(cur, n: int) -> None:
    """pgvector HNSW는 ef_search 개수를 넘겨 반환하지 않는다 — LIMIT보다 반드시 크게 잡아야
    한다(2026-08-27 실측: ef_search=100인 채 limit 1500을 걸었더니 131행만 돌아왔다).
    전략별로 요구 depth가 다르므로 매 쿼리 직전에 맞춰준다."""
    cur.execute(f"set hnsw.ef_search = {n}")


def connect():
    c = psycopg2.connect(DB_URL)
    c.autocommit = True
    with c.cursor() as cur:
        cur.execute(f"set hnsw.ef_search = {HNSW_EF_SEARCH}")
        cur.execute("set pg_trgm.similarity_threshold = 0.1")
    return c


# ── 조건어(axis 조건) 추출 ────────────────────────────────────────────────
# claim 슬롯(population/region)에서 분류 조건 후보를 만든다. 운영 파이프라인에서
# claim_extractor가 채우는 필드와 같은 것들이라 실험용으로 새로 만든 정보가 아니다.
_COND_STOP = {"전국", "전체", "국내", "우리나라", "한국", "없음", "nan", "", "-"}
_AGE_RE = re.compile(r"(\d+\s*세\s*(?:이상|이하|미만)?|\d+\s*[~-]\s*\d+\s*세|[가-힣]+대)")


def condition_terms(row: dict) -> list[str]:
    terms: list[str] = []
    for key in ("population", "region"):
        raw = str(row.get(key) or "").strip()
        if raw.lower() in _COND_STOP:
            continue
        for part in re.split(r"[,/·]| 및 ", raw):
            part = part.strip()
            if len(part) >= 2 and part not in _COND_STOP:
                terms.append(part)
    sent = row.get("sentence", "")
    terms += _AGE_RE.findall(sent)
    for g in ("여성", "남성"):
        if g in sent:
            terms.append(g)
    seen, out = set(), []
    for t in terms:
        if t not in seen:
            seen.add(t)
            out.append(t)
    return out[:6]


# ── 전략 A: baseline (운영 dense 검색) ────────────────────────────────────
def strat_baseline(cur, ctx) -> tuple[list[str], dict]:
    v = ctx["v_sentence"]
    set_ef(cur, TABLE_EF)
    cur.execute(
        """select table_id, 1-(embedding::halfvec(%s) <=> %s::halfvec(%s)) as sim
           from kosis_vdb_tables_qwen
           order by embedding::halfvec(%s) <=> %s::halfvec(%s)
           limit %s""",
        (DIM, v, DIM, DIM, v, DIM, POOL),
    )
    rows = cur.fetchall()
    return [r[0] for r in rows], {"funnel": {"dense_tables": len(rows)}}


# ── 전략 B: item 중심 ─────────────────────────────────────────────────────
def strat_item(cur, ctx, pool: int = POOL) -> tuple[list[str], dict]:
    v = ctx["v_item"]
    set_ef(cur, ITEM_EF)
    cur.execute(
        """select table_id, 1-(embedding::halfvec(%s) <=> %s::halfvec(%s)) as sim
           from kosis_vdb_items_qwen
           order by embedding::halfvec(%s) <=> %s::halfvec(%s)
           limit %s""",
        (DIM, v, DIM, DIM, v, DIM, ITEM_FETCH),
    )
    item_rows = cur.fetchall()
    best: dict[str, float] = {}
    for tid, sim in item_rows:
        if tid not in best or sim > best[tid]:
            best[tid] = sim
    ranked = sorted(best, key=lambda t: -best[t])
    return ranked[:pool], {
        "funnel": {"items": len(item_rows), "tables_from_items": len(ranked),
                   "final": min(pool, len(ranked))},
        "item_sim": best,
    }


# ── 전략 C: item -> axis 계층 ─────────────────────────────────────────────
AXIS_WIDE = 400   # item 단계에서 넓게 확보할 표 수


def axis_hit_counts(cur, table_ids: list[str], terms: list[str]) -> dict[str, int]:
    """각 표가 조건어를 axis_value 이름으로 실제 보유하는지 센다(1,000만 행, 어휘 매칭)."""
    if not table_ids or not terms:
        return {}
    pattern = "|".join(re.escape(t) for t in terms)
    cur.execute(
        """select table_id, count(distinct m.term) as hits
           from kosis_vdb_axis_values_qwen v
           join lateral (
                select t.term from unnest(%s::text[]) as t(term)
                where v.value_name ilike '%%' || t.term || '%%'
           ) m on true
           where v.table_id = any(%s)
           group by table_id""",
        (terms, table_ids),
    )
    return {t: h for t, h in cur.fetchall()}


def strat_item_axis(cur, ctx) -> tuple[list[str], dict]:
    wide, meta = strat_item(cur, ctx, pool=AXIS_WIDE)
    terms = ctx["cond_terms"]
    hits = axis_hit_counts(cur, wide, terms) if terms else {}
    sim = meta["item_sim"]
    # 조건을 많이 만족하는 표를 앞으로. 조건어가 없으면 item 순위 그대로.
    ranked = sorted(wide, key=lambda t: (-hits.get(t, 0), -sim.get(t, 0.0)))
    passed = sum(1 for t in wide if hits.get(t, 0) > 0)
    return ranked[:POOL], {
        "funnel": {"items": meta["funnel"]["items"],
                   "tables_from_items": meta["funnel"]["tables_from_items"],
                   "item_wide": len(wide),
                   "axis_passed": passed,
                   "final": min(POOL, len(ranked))},
        "cond_terms": terms,
    }


# ── 전략 D: hybrid RRF ────────────────────────────────────────────────────
def strat_hybrid(cur, ctx) -> tuple[list[str], dict]:
    rankings: dict[str, list[str]] = {}

    dense, _ = strat_baseline(cur, ctx)
    rankings["dense"] = dense

    item_rank, item_meta = strat_item(cur, ctx, pool=AXIS_WIDE)
    rankings["item"] = item_rank[:POOL]

    # lexical: 운영 lexical_query_vdb와 같은 trigram 쿼리
    q = ctx["lex_query"]
    cur.execute(
        """select table_id, similarity(embedding_text, %s) as sim
           from kosis_vdb_tables_qwen
           where embedding_text %% %s
           order by sim desc limit %s""",
        (q, q, POOL),
    )
    rankings["lexical"] = [r[0] for r in cur.fetchall()]

    # axis: dense+item 합집합에 대해 조건어 충족 수로 정렬
    terms = ctx["cond_terms"]
    union = list(dict.fromkeys(dense + item_rank))
    if terms:
        hits = axis_hit_counts(cur, union, terms)
        rankings["axis"] = [t for t in sorted(hits, key=lambda t: -hits[t]) if hits[t] > 0][:POOL]
    else:
        hits = {}
        rankings["axis"] = []

    scores: dict[str, float] = defaultdict(float)
    for lst in rankings.values():
        for i, tid in enumerate(lst):
            scores[tid] += 1.0 / (RRF_K + i + 1)
    fused = sorted(scores, key=lambda t: -scores[t])[:POOL]
    return fused, {
        "funnel": {"dense": len(dense), "item": len(item_rank),
                   "lexical": len(rankings["lexical"]), "axis": len(rankings["axis"]),
                   "union": len(union), "final": len(fused)},
        "sources": {k: len(v) for k, v in rankings.items()},
    }


STRATEGIES = {
    "baseline": strat_baseline,
    "item": strat_item,
    "item_axis": strat_item_axis,
    "hybrid_rrf": strat_hybrid,
}


# ── 지표 ─────────────────────────────────────────────────────────────────
def dcg(rels) -> float:
    return sum(r / np.log2(i + 2) for i, r in enumerate(rels))


def metrics_for(ranked: list[str], gold: set[str]) -> dict:
    pos = {t: i for i, t in enumerate(ranked)}
    hit_ranks = sorted(pos[g] + 1 for g in gold if g in pos)
    first = hit_ranks[0] if hit_ranks else None
    out = {"gold_rank": first, "n_cand": len(ranked)}
    for k in K_LIST:
        found = sum(1 for g in gold if pos.get(g, 10**9) < k)
        out[f"recall@{k}"] = found / len(gold)
        out[f"hit@{k}"] = 1.0 if found else 0.0
        out[f"precision@{k}"] = found / min(k, len(ranked)) if ranked else 0.0
        rels = [1.0 if t in gold else 0.0 for t in ranked[:k]]
        ideal = [1.0] * min(len(gold), k)
        out[f"ndcg@{k}"] = (dcg(rels) / dcg(ideal)) if ideal and dcg(ideal) > 0 else 0.0
    out["mrr"] = 1.0 / first if first else 0.0
    return out


def main() -> None:
    eval_set = json.loads((HERE / "eval_set.json").read_text(encoding="utf-8"))
    slots = json.loads((HERE / "claim_slots.json").read_text(encoding="utf-8"))

    print("모델 로딩...")
    from sentence_transformers import SentenceTransformer

    model = SentenceTransformer("Qwen/Qwen3-Embedding-4B", truncate_dim=DIM)

    # 질의 벡터 준비 — 4개 전략이 같은 모델·같은 지시문을 쓴다.
    sents, item_qs = [], []
    for r in eval_set:
        s = slots.get(r["claim_id"], {})
        sents.append(f"Instruct: {VDB_QUERY_INSTRUCTION}\nQuery: {r['sentence']}")
        stat = (s.get("statistic_expression") or "").strip()
        item_q = stat if stat and stat.lower() != "nan" else r["sentence"]
        item_qs.append(f"Instruct: {VDB_QUERY_INSTRUCTION}\nQuery: {item_q}")

    print(f"질의 임베딩 {len(sents)*2}건 인코딩...")
    v_sent = model.encode(sents, batch_size=4, normalize_embeddings=True, show_progress_bar=False)
    v_item = model.encode(item_qs, batch_size=4, normalize_embeddings=True, show_progress_bar=False)
    del model
    import gc, torch
    gc.collect()
    torch.cuda.empty_cache()

    conn = connect()
    cur = conn.cursor()

    per_claim = {name: [] for name in STRATEGIES}
    details = {name: [] for name in STRATEGIES}

    for idx, r in enumerate(eval_set):
        s = slots.get(r["claim_id"], {})
        stat = (s.get("statistic_expression") or "").strip()
        lex_parts = [p for p in [stat, str(s.get("source_org") or ""), str(s.get("population") or "")]
                     if p and p.lower() != "nan"]
        ctx = {
            "v_sentence": vec_literal(v_sent[idx]),
            "v_item": vec_literal(v_item[idx]),
            "cond_terms": condition_terms({**s, "sentence": r["sentence"]}),
            "lex_query": " ".join(lex_parts) or r["sentence"],
        }
        gold = set(r["gold"])
        for name, fn in STRATEGIES.items():
            t0 = time.perf_counter()
            ranked, meta = fn(cur, ctx)
            ms = (time.perf_counter() - t0) * 1000
            m = metrics_for(ranked, gold)
            m["latency_ms"] = ms
            m["claim_id"] = r["claim_id"]
            per_claim[name].append(m)
            details[name].append({
                "claim_id": r["claim_id"], "sentence": r["sentence"], "gold": sorted(gold),
                "gold_rank": m["gold_rank"], "top10": ranked[:10],
                "funnel": meta.get("funnel", {}), "cond_terms": ctx["cond_terms"],
            })
        if (idx + 1) % 10 == 0:
            print(f"  {idx+1}/{len(eval_set)}")

    (HERE / "per_claim.json").write_text(json.dumps(per_claim, ensure_ascii=False, indent=1),
                                        encoding="utf-8")
    (HERE / "details.json").write_text(json.dumps(details, ensure_ascii=False, indent=1),
                                       encoding="utf-8")
    print(f"\n저장 완료: per_claim.json / details.json")
    conn.close()


if __name__ == "__main__":
    main()
