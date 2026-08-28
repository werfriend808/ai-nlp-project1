"""3단계: 최종 비교표, ablation, 실패 유형 분류, 전략 간 win/loss 대조."""
from __future__ import annotations

import json
import os
import re
import sys
from collections import Counter
from pathlib import Path

import psycopg2

sys.path.insert(0, str(Path(__file__).parent))
from fuse import STRATEGIES, per_claim_metrics, aggregate, split_dev_test  # noqa: E402
from retrievers import rrf  # noqa: E402

HERE = Path(__file__).parent
ROOT = Path(__file__).resolve().parents[2]
BASE = "Baseline (dense_full)"


def is_english(s: str) -> bool:
    letters = [c for c in (s or "") if c.isalpha()]
    return bool(letters) and all(ord(c) < 128 for c in letters)


def classify_failure(claim: str, gold_name: str, gold_items: int, gold_axes: int,
                     measurement: str, top_names: list[str]) -> str:
    """Recall@100 안에 못 들어온 claim의 실패 유형 — 판정 근거를 데이터에서만 가져온다."""
    if is_english(gold_name):
        return "3. 영어/한국어 문제"
    # claim 문장에 지표 표현의 핵심어가 없는가
    core = [w for w in re.split(r"[\s/]+", measurement) if len(w) >= 2]
    if core and not any(w in claim for w in core):
        return "1. 지표명이 claim에 없음"
    if gold_items == 0:
        return "5. item 정보 부족"
    if gold_axes == 0:
        return "6. axis 정보 부족"
    # 상위 후보에 정답표와 표명 핵심어를 공유하는 표가 몰려 있는가
    gk = {w for w in re.split(r"[\s()/,]+", gold_name) if len(w) >= 2}
    dup = sum(1 for n in top_names if gk & {w for w in re.split(r"[\s()/,]+", n) if len(w) >= 2})
    if dup >= 4:
        return "4. 유사 표가 너무 많음"
    if gk and not any(w in claim for w in gk):
        return "2. 표명과 claim 표현이 다름"
    return "9. table embedding 정보 부족"


def main() -> None:
    ev = json.loads((ROOT / "benchmark/search_experiment/eval_set.json").read_text(encoding="utf-8"))
    res = json.loads((HERE / "results.json").read_text(encoding="utf-8"))
    solo = json.loads((HERE / "solo.json").read_text(encoding="utf-8"))
    runs = json.loads((HERE / "runs.json").read_text(encoding="utf-8"))
    fused = json.loads((HERE / "fused.json").read_text(encoding="utf-8"))
    qmap = json.loads((HERE / "queries.json").read_text(encoding="utf-8"))
    lat = json.loads((HERE / "latency.json").read_text(encoding="utf-8"))
    lat_mean = {k: sum(v) / len(v) for k, v in lat.items()}
    gold = {r["claim_id"]: set(r["gold"]) for r in ev}
    sent = {r["claim_id"]: r["sentence"] for r in ev}
    dev, test = split_dev_test(ev)

    conn = psycopg2.connect(os.environ["SUPABASE_DB_URL"]); cur = conn.cursor()
    all_ids = {t for f in fused.values() for l in f.values() for t in l[:100]}
    all_ids |= {g for gs in gold.values() for g in gs}
    cur.execute("select table_id, table_name from kosis_vdb_tables_qwen where table_id = any(%s)",
                (list(all_ids),))
    tname = dict(cur.fetchall())
    gold_ids = [g for gs in gold.values() for g in gs]
    cur.execute("""select t.table_id,
                   (select count(*) from kosis_vdb_items_qwen i where i.table_id=t.table_id),
                   (select count(*) from kosis_vdb_axes_qwen a where a.table_id=t.table_id)
                   from kosis_vdb_tables_qwen t where t.table_id = any(%s)""", (gold_ids,))
    gmeta = {r[0]: (r[1], r[2]) for r in cur.fetchall()}
    conn.close()

    L: list[str] = ["# KOSIS Retrieval Recall 개선 실험 (실험 2차)\n"]
    L.append(f"평가셋 {len(ev)}건 — dev {len(dev)} / test {len(test)} (기사 단위 분할)\n")

    # ── 검색기 단독 ─────────────────────────────────────────────
    L.append("\n## 개별 검색기 단독 성능\n")
    L.append("| Retriever | R@1 | R@10 | R@50 | R@100 | R@200 | Avg Latency |")
    L.append("|---|--:|--:|--:|--:|--:|--:|")
    for n, s in sorted(solo.items(), key=lambda x: -x[1]["recall@100"]):
        L.append(f"| `{n}` | {s['recall@1']*100:.1f}% | {s['recall@10']*100:.1f}% | "
                 f"{s['recall@50']*100:.1f}% | **{s['recall@100']*100:.1f}%** | "
                 f"{s['recall@200']*100:.1f}% | {s['avg_latency_ms']:.0f}ms |")

    # ── 전략 종합 ───────────────────────────────────────────────
    L.append("\n## 전략 종합 (전체 70건)\n")
    L.append("| Strategy | R@1 | R@10 | R@50 | R@100 | R@200 | MRR | NDCG@10 | Avg Latency | P95 |")
    L.append("|---|--:|--:|--:|--:|--:|--:|--:|--:|--:|")
    for n, r in sorted(res.items(), key=lambda x: -x[1]["all"]["recall@100"]):
        a = r["all"]
        L.append(f"| {n} | {a['recall@1']*100:.1f}% | {a['recall@10']*100:.1f}% | "
                 f"{a['recall@50']*100:.1f}% | **{a['recall@100']*100:.1f}%** | "
                 f"{a['recall@200']*100:.1f}% | {a['mrr']:.3f} | {a['ndcg@10']:.3f} | "
                 f"{a['avg_latency_ms']:.0f}ms | {a['p95_ms']:.0f}ms |")

    # ── dev / test ──────────────────────────────────────────────
    L.append("\n## dev / test 분리 (과적합 점검)\n")
    L.append("| Strategy | dev R@100 | test R@100 | 차이 |")
    L.append("|---|--:|--:|--:|")
    for n, r in sorted(res.items(), key=lambda x: -x[1]["dev"]["recall@100"]):
        d, t = r["dev"]["recall@100"], r["test"]["recall@100"]
        L.append(f"| {n} | {d*100:.1f}% | {t*100:.1f}% | {(t-d)*100:+.1f}%p |")

    # ── Baseline 대비 win/loss ──────────────────────────────────
    L.append("\n## Baseline 대비 win / loss (Recall@100 기준)\n")
    L.append("| Strategy | Baseline 실패 → 성공 | Baseline 성공 → 실패 | 순증 | Gold miss | 정답 중앙 rank |")
    L.append("|---|--:|--:|--:|--:|--:|")
    bper = res[BASE]["per_claim"]
    for n, r in sorted(res.items(), key=lambda x: -x[1]["all"]["recall@100"]):
        p = r["per_claim"]
        win = sum(1 for c in p if not bper[c]["recall@100"] and p[c]["recall@100"])
        loss = sum(1 for c in p if bper[c]["recall@100"] and not p[c]["recall@100"])
        a = r["all"]
        mr = f"{a['median_gold_rank']:.0f}" if a["median_gold_rank"] else "-"
        L.append(f"| {n} | {win} | {loss} | {win-loss:+d} | {a['gold_miss_rate']*100:.1f}% | {mr} |")

    # ── Ablation: 최고 전략에서 검색기 하나씩 제거 ────────────────
    best = max(res, key=lambda n: res[n]["all"]["recall@100"])
    L.append(f"\n## Ablation — 최고 전략 `{best}` 에서 검색기 하나씩 제거\n")
    L.append("| 제거한 검색기 | R@100 | 변화 | 지연 |")
    L.append("|---|--:|--:|--:|")
    bw = res[best]["weights"]
    L.append(f"| (없음 — 전체) | {res[best]['all']['recall@100']*100:.1f}% | — | "
             f"{res[best]['all']['avg_latency_ms']:.0f}ms |")
    for drop in bw:
        w2 = {k: v for k, v in bw.items() if k != drop}
        if not w2:
            continue
        per = []
        for r in ev:
            cid = r["claim_id"]
            f = rrf({k: runs[k].get(cid, []) for k in w2}, k=60, weights=w2, top_k=200)
            per.append(per_claim_metrics(f, gold[cid]))
        a = aggregate(per, [sum(lat_mean[k] for k in w2)] * len(per))
        delta = (a["recall@100"] - res[best]["all"]["recall@100"]) * 100
        L.append(f"| −`{drop}` | {a['recall@100']*100:.1f}% | {delta:+.1f}%p | {a['avg_latency_ms']:.0f}ms |")

    # ── 실패 유형 ───────────────────────────────────────────────
    for label, sname in [("Baseline", BASE), (f"최고 전략 ({best})", best)]:
        p = res[sname]["per_claim"]
        misses = [c for c in p if not p[c]["recall@100"]]
        L.append(f"\n## 실패 유형 — {label} (Recall@100 미포함 {len(misses)}건)\n")
        cnt = Counter()
        rows = []
        for cid in misses:
            g = sorted(gold[cid])[0]
            gi, ga = gmeta.get(g, (0, 0))
            top_names = [tname.get(t, "") for t in fused[sname][cid][:10]]
            k = classify_failure(sent[cid], tname.get(g, ""), gi, ga,
                                 qmap[cid].get("measurement", ""), top_names)
            cnt[k] += 1
            rows.append((cid, g, k))
        L.append("| 유형 | 건수 |")
        L.append("|---|--:|")
        for k, v in sorted(cnt.items()):
            L.append(f"| {k} | {v} |")
        L.append("\n<details><summary>개별 claim</summary>\n")
        L.append("| claim | 정답표 | 유형 | Baseline rank | 최고전략 rank | gold를 찾은 검색기 |")
        L.append("|---|---|---|--:|--:|---|")
        for cid, g, k in rows[:40]:
            finders = [rn for rn, rk in runs.items() if g in rk.get(cid, [])[:100]]
            L.append(f"| {cid} | {g} | {k} | {bper[cid]['gold_rank'] or 'miss'} | "
                     f"{res[best]['per_claim'][cid]['gold_rank'] or 'miss'} | "
                     f"{', '.join(f'`{x}`' for x in finders[:4]) or '없음'} |")
        L.append("\n</details>\n")

    # ── 상한선: 모든 검색기 합집합 ────────────────────────────────
    union_hit = 0
    for r in ev:
        cid = r["claim_id"]
        u = {t for rk in runs.values() for t in rk.get(cid, [])[:100]}
        union_hit += bool(gold[cid] & u)
    L.append(f"\n## 상한선\n")
    L.append(f"- 이번에 만든 **모든 검색기의 top-100 합집합**에 정답이 들어있는 claim: "
             f"**{union_hit}/{len(ev)} = {union_hit/len(ev)*100:.1f}%**")
    L.append("- 융합 방식을 아무리 잘 골라도 이 값을 넘을 수 없다. 이 선을 올리려면 "
             "검색기 자체나 표 임베딩을 바꿔야 한다.\n")

    (HERE / "REPORT.md").write_text("\n".join(L) + "\n", encoding="utf-8")
    print("\n".join(L))


if __name__ == "__main__":
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    main()
