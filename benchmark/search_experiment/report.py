"""실험 결과 집계 -> results.csv / results.json / summary.md (읽기 전용)."""
from __future__ import annotations

import csv
import json
import statistics as st
from collections import Counter
from pathlib import Path

HERE = Path(__file__).parent
NAMES = {"baseline": "Baseline", "item": "Item",
         "item_axis": "Item → Axis", "hybrid_rrf": "Hybrid/RRF"}
K_LIST = (1, 10, 100)


def agg(rows: list[dict]) -> dict:
    n = len(rows)
    lat = sorted(r["latency_ms"] for r in rows)
    ranks = [r["gold_rank"] for r in rows if r["gold_rank"]]
    out = {"n": n}
    for k in K_LIST:
        for m in ("recall", "hit", "precision", "ndcg"):
            out[f"{m}@{k}"] = sum(r[f"{m}@{k}"] for r in rows) / n
    out["mrr"] = sum(r["mrr"] for r in rows) / n
    out["avg_candidates"] = sum(r["n_cand"] for r in rows) / n
    out["avg_latency_ms"] = sum(lat) / n
    out["p50_ms"] = lat[len(lat) // 2]
    out["p95_ms"] = lat[min(int(len(lat) * 0.95), len(lat) - 1)]
    out["avg_gold_rank"] = (sum(ranks) / len(ranks)) if ranks else None
    out["median_gold_rank"] = st.median(ranks) if ranks else None
    out["gold_miss_rate"] = 1 - (len(ranks) / n)
    return out


def main() -> None:
    per = json.loads((HERE / "per_claim.json").read_text(encoding="utf-8"))
    det = json.loads((HERE / "details.json").read_text(encoding="utf-8"))
    summary = {k: agg(v) for k, v in per.items()}

    (HERE / "results.json").write_text(json.dumps(summary, ensure_ascii=False, indent=1),
                                       encoding="utf-8")
    with (HERE / "results.csv").open("w", newline="", encoding="utf-8") as f:
        cols = ["strategy"] + list(next(iter(summary.values())).keys())
        w = csv.DictWriter(f, fieldnames=cols)
        w.writeheader()
        for k, v in summary.items():
            w.writerow({"strategy": k, **v})

    L = []
    L.append("# KOSIS 검색 전략 4종 비교 실험 결과\n")
    L.append(f"평가셋: 골든셋 {summary['baseline']['n']}건 (claim → 정답 table_id)\n")
    L.append("\n## 주요 지표\n")
    L.append("| Strategy | Recall@1 | Recall@10 | Recall@100 | MRR | NDCG@10 | Avg Cand | Avg Latency | P95 |")
    L.append("|---|--:|--:|--:|--:|--:|--:|--:|--:|")
    for k, n in NAMES.items():
        s = summary[k]
        L.append(f"| {n} | {s['recall@1']*100:.1f}% | {s['recall@10']*100:.1f}% | "
                 f"{s['recall@100']*100:.1f}% | {s['mrr']:.3f} | {s['ndcg@10']:.3f} | "
                 f"{s['avg_candidates']:.0f} | {s['avg_latency_ms']:.0f}ms | {s['p95_ms']:.0f}ms |")

    L.append("\n## 보조 지표\n")
    L.append("| Strategy | Hit@1 | Hit@10 | Hit@100 | NDCG@100 | 정답 평균 rank | 정답 중앙 rank | Gold miss |")
    L.append("|---|--:|--:|--:|--:|--:|--:|--:|")
    for k, n in NAMES.items():
        s = summary[k]
        ar = f"{s['avg_gold_rank']:.1f}" if s["avg_gold_rank"] else "-"
        mr = f"{s['median_gold_rank']:.0f}" if s["median_gold_rank"] else "-"
        L.append(f"| {n} | {s['hit@1']*100:.1f}% | {s['hit@10']*100:.1f}% | {s['hit@100']*100:.1f}% | "
                 f"{s['ndcg@100']:.3f} | {ar} | {mr} | {s['gold_miss_rate']*100:.1f}% |")

    # 후보 축소(퍼널)
    L.append("\n## 후보 축소 (Candidate Reduction, 평균)\n")
    for k, n in NAMES.items():
        keys, acc = [], {}
        for d in det[k]:
            for kk, vv in (d.get("funnel") or {}).items():
                acc.setdefault(kk, []).append(vv)
                if kk not in keys:
                    keys.append(kk)
        if acc:
            L.append(f"- **{n}**: " + " → ".join(f"{kk} {sum(acc[kk])/len(acc[kk]):.0f}" for kk in keys))

    (HERE / "summary.md").write_text("\n".join(L) + "\n", encoding="utf-8")
    print("\n".join(L))


if __name__ == "__main__":
    main()
