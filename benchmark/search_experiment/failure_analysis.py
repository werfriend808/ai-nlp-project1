"""전략별 실패/성공 사례 대조 + 근접 중복 후보 분석 (읽기 전용)."""
from __future__ import annotations

import json
import os
from pathlib import Path

import psycopg2

HERE = Path(__file__).parent
NAMES = {"baseline": "Baseline", "item": "Item",
         "item_axis": "Item → Axis", "hybrid_rrf": "Hybrid/RRF"}


def main() -> None:
    det = json.loads((HERE / "details.json").read_text(encoding="utf-8"))
    conn = psycopg2.connect(os.environ["SUPABASE_DB_URL"])
    cur = conn.cursor()

    by_claim: dict[str, dict] = {}
    for strat, rows in det.items():
        for r in rows:
            by_claim.setdefault(r["claim_id"], {"row": r})[strat] = r["gold_rank"]

    # 표 이름 조회용
    all_ids = {t for rows in det.values() for r in rows for t in r["top10"]}
    all_ids |= {g for rows in det.values() for r in rows for g in r["gold"]}
    cur.execute("select table_id, table_name, institution_name from kosis_vdb_tables_qwen "
                "where table_id = any(%s)", (list(all_ids),))
    meta = {t: (n, o) for t, n, o in cur.fetchall()}

    L = ["# 실패/성공 사례 분석\n"]

    buckets = {
        "baseline 실패 → item_axis 성공": lambda d: not d["baseline"] and bool(d["item_axis"]),
        "item 성공 → item_axis 실패": lambda d: bool(d["item"]) and not d["item_axis"],
        "hybrid_rrf 에서만 성공": lambda d: bool(d["hybrid_rrf"]) and not any(
            d[s] for s in ("baseline", "item", "item_axis")),
        "모든 전략 실패": lambda d: not any(d[s] for s in NAMES),
    }
    for title, pred in buckets.items():
        sel = [(cid, d) for cid, d in by_claim.items() if pred(d)]
        L.append(f"\n## {title} — {len(sel)}건\n")
        for cid, d in sel[:10]:
            r = d["row"]
            gold_names = ", ".join(f"{g}({meta.get(g,('?',))[0]})" for g in r["gold"])
            L.append(f"**[{cid}]** {r['sentence'][:90]}")
            L.append(f"- 정답표: {gold_names}")
            L.append(f"- 조건어: {r['cond_terms']}")
            L.append("- gold rank: " + ", ".join(
                f"{NAMES[s]}={d[s] if d[s] else 'miss'}" for s in NAMES))
            L.append("")

    # 근접 중복 분석: Baseline top10 안에서 정답표와 표명이 유사한 후보 비율
    L.append("\n## 근접 중복 후보 분석 (Baseline top-10)\n")
    L.append("정답표 이름의 핵심어를 공유하는 상위 후보가 몇 개인지 — 중복 경쟁의 크기.\n")
    L.append("| claim | gold rank | top10 중 유사표 | 예시 |")
    L.append("|---|--:|--:|---|")
    dup_ratios = []
    for r in det["baseline"][:200]:
        gold = r["gold"][0]
        gname = meta.get(gold, ("", ""))[0] or ""
        keys = {w for w in gname.replace("(", " ").replace(")", " ").split() if len(w) >= 2}
        if not keys:
            continue
        sim_cnt = 0
        example = ""
        for t in r["top10"]:
            if t == gold:
                continue
            nm = meta.get(t, ("", ""))[0] or ""
            if keys & {w for w in nm.replace("(", " ").replace(")", " ").split() if len(w) >= 2}:
                sim_cnt += 1
                example = example or f"{t} {nm[:30]}"
        dup_ratios.append(sim_cnt / 10)
        if len(dup_ratios) <= 15:
            L.append(f"| {r['claim_id']} | {r['gold_rank'] or 'miss'} | {sim_cnt}/10 | {example} |")
    if dup_ratios:
        L.append(f"\n**Baseline 평균 근접중복 비율: {sum(dup_ratios)/len(dup_ratios)*100:.1f}%** "
                 f"(top-10 중 정답표와 표명 핵심어를 공유하는 다른 표)\n")

    # 전략별 중복 비율
    L.append("\n### 전략별 근접중복 비율 (top-10)\n")
    L.append("| Strategy | 평균 중복 비율 |")
    L.append("|---|--:|")
    for s, n in NAMES.items():
        rs = []
        for r in det[s]:
            gold = r["gold"][0]
            gname = meta.get(gold, ("", ""))[0] or ""
            keys = {w for w in gname.replace("(", " ").replace(")", " ").split() if len(w) >= 2}
            if not keys:
                continue
            c = sum(1 for t in r["top10"] if t != gold and
                    keys & {w for w in (meta.get(t, ("", ""))[0] or "").replace("(", " ").replace(")", " ").split()
                            if len(w) >= 2})
            rs.append(c / 10)
        L.append(f"| {n} | {sum(rs)/len(rs)*100:.1f}% |")

    (HERE / "failures.md").write_text("\n".join(L) + "\n", encoding="utf-8")
    print("\n".join(L))
    conn.close()


if __name__ == "__main__":
    main()
