"""2단계: 캐시된 검색 결과를 전략별로 융합하고 평가한다.

dev/test 분리는 기사 단위로 한다 — 같은 기사에서 나온 claim들은 정답표를 공유해서
claim 단위로 쪼개면 누수가 생긴다.
"""
from __future__ import annotations

import json
import statistics as st
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).parent))
from retrievers import rrf  # noqa: E402

HERE = Path(__file__).parent
ROOT = Path(__file__).resolve().parents[2]
K_LIST = (1, 10, 50, 100, 200)
RRF_K = 60

# ── 전략 정의: 이름 -> (검색기 가중치 dict) ────────────────────────────────
STRATEGIES: dict[str, dict[str, float]] = {
    # 기준점
    "Baseline (dense_full)":        {"dense_full": 1},
    # 실험 A — 어휘 검색 비교
    "BM25 only":                    {"bm25_struct": 1},
    "trigram only":                 {"trgm_struct": 1},
    "Dense + trigram":              {"dense_full": 1, "trgm_struct": 1},
    "Dense + BM25":                 {"dense_full": 1, "bm25_struct": 1},
    "Dense + BM25(full+struct)":    {"dense_full": 1, "bm25_struct": 1, "bm25_full": 1},
    # 실험 B — 독립 검색기 RRF
    "Dense + Item":                 {"dense_full": 1, "item_measurement": 1},
    "Dense + Axis":                 {"dense_full": 1, "axis_condition": 1},
    "Dense + Item + Axis":          {"dense_full": 1, "item_measurement": 1, "axis_condition": 1},
    # 실험 C — 필드 분해
    "Field (equal)":                {"dense_full": 1, "dense_measurement": 1, "dense_population": 1,
                                     "dense_region": 1, "dense_condition": 1},
    "Field (dense 중심)":            {"dense_full": 3, "dense_measurement": 1, "dense_population": 1,
                                     "dense_region": 1, "dense_condition": 1},
    "Field (measurement 중심)":      {"dense_full": 1, "dense_measurement": 3, "dense_population": 1,
                                     "dense_region": 1, "dense_condition": 1},
    "Field (dense+measurement)":    {"dense_full": 2, "dense_measurement": 2},
    # 실험 D — 문맥
    "Context D2 (이전+현재)":         {"dense_ctxD2": 1},
    "Context D3 (현재+다음)":         {"dense_ctxD3": 1},
    "Context D4 (이전+현재+다음)":     {"dense_ctxD4": 1},
    "Context D5 (문단)":             {"dense_ctxD5": 1},
    "Dense + Context D4":           {"dense_full": 1, "dense_ctxD4": 1},
    "Dense + Context D5":           {"dense_full": 1, "dense_ctxD5": 1},
    "Dense + Ctx D4 + Ctx D5":      {"dense_full": 1, "dense_ctxD4": 1, "dense_ctxD5": 1},
    # 실험 F — 질의 확장
    "Query expansion (문장)":        {"dense_expanded": 1},
    "Query expansion (구조화)":       {"dense_expstruct": 1},
    "Dense + expansion":            {"dense_full": 1, "dense_expstruct": 1},
    # 최종 조합 후보
    "ALL (dense+bm25+item+axis+ctx)": {"dense_full": 1, "bm25_struct": 1, "item_measurement": 1,
                                       "axis_condition": 1, "dense_ctxD4": 1, "dense_ctxD5": 1},
    "COMBO-A (dense+bm25+ctx)":     {"dense_full": 1, "bm25_struct": 1, "dense_ctxD4": 1, "dense_ctxD5": 1},
    "COMBO-B (dense+bm25+item+ctx)": {"dense_full": 1, "bm25_struct": 1, "item_measurement": 1,
                                      "dense_ctxD4": 1, "dense_ctxD5": 1},
    "COMBO-C (+struct+expansion)":  {"dense_full": 1, "dense_struct": 1, "bm25_struct": 1,
                                     "item_measurement": 1, "dense_ctxD4": 1, "dense_ctxD5": 1,
                                     "dense_expstruct": 1},
}


# ── 지표 ─────────────────────────────────────────────────────────────────
def dcg(rels) -> float:
    return sum(r / np.log2(i + 2) for i, r in enumerate(rels))


def per_claim_metrics(ranked: list[str], gold: set[str]) -> dict:
    pos = {t: i for i, t in enumerate(ranked)}
    hits = sorted(pos[g] + 1 for g in gold if g in pos)
    first = hits[0] if hits else None
    out = {"gold_rank": first, "n_cand": len(ranked)}
    for k in K_LIST:
        found = sum(1 for g in gold if pos.get(g, 10**9) < k)
        out[f"recall@{k}"] = found / len(gold)
        rels = [1.0 if t in gold else 0.0 for t in ranked[:k]]
        ideal = [1.0] * min(len(gold), k)
        out[f"ndcg@{k}"] = (dcg(rels) / dcg(ideal)) if ideal else 0.0
    out["mrr"] = 1.0 / first if first else 0.0
    return out


def aggregate(rows: list[dict], lats: list[float]) -> dict:
    n = len(rows)
    ranks = [r["gold_rank"] for r in rows if r["gold_rank"]]
    sl = sorted(lats)
    out = {"n": n}
    for k in K_LIST:
        out[f"recall@{k}"] = sum(r[f"recall@{k}"] for r in rows) / n
    out["mrr"] = sum(r["mrr"] for r in rows) / n
    out["ndcg@10"] = sum(r["ndcg@10"] for r in rows) / n
    out["avg_candidates"] = sum(r["n_cand"] for r in rows) / n
    out["avg_latency_ms"] = sum(sl) / len(sl)
    out["p95_ms"] = sl[min(int(len(sl) * 0.95), len(sl) - 1)]
    out["avg_gold_rank"] = (sum(ranks) / len(ranks)) if ranks else None
    out["median_gold_rank"] = st.median(ranks) if ranks else None
    out["gold_miss_rate"] = 1 - (len(ranks) / n)
    return out


def split_dev_test(ev: list[dict]) -> tuple[set[str], set[str]]:
    """기사 단위 분할 — 같은 기사의 claim이 dev/test에 갈리지 않게 한다."""
    import queries as Q
    from collections import defaultdict
    cmap = Q.load_claim_article_map()
    by_art: dict[str, list[str]] = defaultdict(list)
    for r in ev:
        by_art[cmap.get(r["claim_id"], r["claim_id"])].append(r["claim_id"])
    # 기사마다 claim 수가 1~10건으로 제각각이라 단순 홀짝 분할은 7/63처럼 치우친다.
    # 큰 기사부터 claim 수가 적은 쪽에 붙이는 그리디 분배로 균형을 맞춘다.
    dev: set[str] = set()
    test: set[str] = set()
    for _, cids in sorted(by_art.items(), key=lambda kv: -len(kv[1])):
        (dev if len(dev) <= len(test) else test).update(cids)
    return dev, test


def main() -> None:
    ev = json.loads((ROOT / "benchmark/search_experiment/eval_set.json").read_text(encoding="utf-8"))
    runs = json.loads((HERE / "runs.json").read_text(encoding="utf-8"))
    lat = json.loads((HERE / "latency.json").read_text(encoding="utf-8"))
    gold = {r["claim_id"]: set(r["gold"]) for r in ev}
    dev, test = split_dev_test(ev)
    print(f"dev {len(dev)}건 / test {len(test)}건 (기사 단위 분할)")

    lat_mean = {k: sum(v) / len(v) for k, v in lat.items()}
    results, fused_all = {}, {}
    for name, weights in STRATEGIES.items():
        missing = [k for k in weights if k not in runs]
        if missing:
            print(f"  [건너뜀] {name} — 검색기 없음: {missing}")
            continue
        per, lats, fused = {}, [], {}
        for r in ev:
            cid = r["claim_id"]
            rankings = {k: runs[k].get(cid, []) for k in weights}
            f = rrf(rankings, k=RRF_K, weights=weights, top_k=max(K_LIST))
            fused[cid] = f
            per[cid] = per_claim_metrics(f, gold[cid])
            lats.append(sum(lat_mean[k] for k in weights))
        fused_all[name] = fused
        results[name] = {
            "weights": weights,
            "all": aggregate(list(per.values()), lats),
            "dev": aggregate([per[c] for c in per if c in dev], lats),
            "test": aggregate([per[c] for c in per if c in test], lats),
            "per_claim": per,
        }

    (HERE / "results.json").write_text(json.dumps(results, ensure_ascii=False), encoding="utf-8")
    (HERE / "fused.json").write_text(json.dumps(fused_all, ensure_ascii=False), encoding="utf-8")
    (HERE / "split.json").write_text(json.dumps({"dev": sorted(dev), "test": sorted(test)}), encoding="utf-8")

    # 개별 검색기 단독 성능도 남긴다(기여도 판단용)
    solo = {}
    for name, rk in runs.items():
        per = [per_claim_metrics(rk.get(r["claim_id"], []), gold[r["claim_id"]]) for r in ev]
        solo[name] = aggregate(per, lat[name])
    (HERE / "solo.json").write_text(json.dumps(solo, ensure_ascii=False), encoding="utf-8")

    print(f"\n{'검색기 단독':<24}{'R@10':>8}{'R@100':>9}{'R@200':>9}{'지연':>10}")
    for n, s in sorted(solo.items(), key=lambda x: -x[1]["recall@100"]):
        print(f"  {n:<22}{s['recall@10']*100:>7.1f}%{s['recall@100']*100:>8.1f}%"
              f"{s['recall@200']*100:>8.1f}%{s['avg_latency_ms']:>9.0f}ms")

    print(f"\n{'전략':<32}{'R@10':>8}{'R@100':>9}{'R@200':>9}{'지연':>10}")
    for n, r in sorted(results.items(), key=lambda x: -x[1]["all"]["recall@100"]):
        a = r["all"]
        print(f"  {n:<30}{a['recall@10']*100:>7.1f}%{a['recall@100']*100:>8.1f}%"
              f"{a['recall@200']*100:>8.1f}%{a['avg_latency_ms']:>9.0f}ms")


if __name__ == "__main__":
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    main()
