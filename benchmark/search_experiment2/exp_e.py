"""실험 E — 표 embedding_text 구성 비교.

전체 287,498개 표를 5가지 구성으로 재임베딩하면 변형 하나당 수 시간이라 현실적이지 않다.
대신 **축소 코퍼스**에서 통제 비교를 한다: 1단계에서 어떤 검색기든 후보로 올린 표 전부 +
정답표 전부를 코퍼스로 삼고, 그 안에서만 5가지 구성을 각각 임베딩해 brute-force로 검색한다.

주의: 코퍼스가 작아진 만큼 절대 Recall은 전체 코퍼스보다 높게 나온다. 구성 간 **상대 비교**
용도로만 읽어야 한다. DB에는 아무것도 쓰지 않는다(전부 메모리).
"""
from __future__ import annotations

import json
import os
import sys
import time
from pathlib import Path

import numpy as np
import psycopg2

sys.path.insert(0, str(Path(__file__).parent))
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from agent.kosis.embedding_text import VALUE_CAP  # noqa: E402

HERE = Path(__file__).parent
ROOT = Path(__file__).resolve().parents[2]
INSTRUCTION = ("Given a Korean news claim sentence, retrieve the KOSIS statistical table "
               "description that best matches it")
K_LIST = (1, 10, 50, 100, 200)

VARIANTS = {
    "E1 table_name": ["name"],
    "E2 +item": ["name", "item"],
    "E3 +axis": ["name", "item", "axis"],
    "E4 +axis_value": ["name", "item", "axis", "value"],
    "E5 +organization": ["org", "name", "item", "axis", "value"],
    # E6: E5에서 item만 뺀 것. E1->E2에서 item이 -1.4%p로 해로웠던 신호를 최종 포맷에서 검증한다.
    # 판정 기준(사전 확정): E5 대비 +5%p 이상이어야 실제 효과로 보고 전체 재임베딩을 검토한다.
    # ±5%p 이내면 70건 표본의 잡음으로 보고 현행 포맷을 유지한다.
    "E6 E5 minus item": ["org", "name", "axis", "value"],
}


def build_text(m: dict, parts: list[str]) -> str:
    lines = []
    if "org" in parts and m["org"]:
        lines.append(f"기관명: {m['org']}")
    lines.append(f"통계표명: {m['name']}")
    if "item" in parts and m["items"]:
        lines.append("항목: " + ", ".join(m["items"]))
    if "axis" in parts and m["axes"]:
        lines.append("분류축: " + ", ".join(m["axes"]))
    if "value" in parts and m["values"]:
        lines.append("분류값: " + ", ".join(m["values"][:VALUE_CAP]))
    return "\n\n".join(lines)


def main() -> None:
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--variants", default=None,
                    help="쉼표 구분. 생략하면 전체. 기존 exp_e.json에 병합된다.")
    args = ap.parse_args()
    todo = ({k: v for k, v in VARIANTS.items() if k in
             {x.strip() for x in args.variants.split(",")}} if args.variants else dict(VARIANTS))
    if not todo:
        raise SystemExit(f"해당 변형 없음. 가능한 값: {list(VARIANTS)}")
    print(f"실행할 변형: {list(todo)}", flush=True)

    runs = json.loads((HERE / "runs.json").read_text(encoding="utf-8"))
    ev = json.loads((ROOT / "benchmark/search_experiment/eval_set.json").read_text(encoding="utf-8"))
    corpus_ids = {t for rk in runs.values() for l in rk.values() for t in l}
    corpus_ids |= {g for r in ev for g in r["gold"]}
    corpus_ids = sorted(corpus_ids)
    print(f"축소 코퍼스 {len(corpus_ids):,}개 표", flush=True)

    conn = psycopg2.connect(os.environ["SUPABASE_DB_URL"])
    cur = conn.cursor()
    cur.execute("select table_id, institution_name, table_name from kosis_vdb_tables_qwen "
                "where table_id = any(%s)", (corpus_ids,))
    meta = {t: {"org": o, "name": n, "items": [], "axes": [], "values": []}
            for t, o, n in cur.fetchall()}
    cur.execute("select table_id, item_name from kosis_vdb_items_qwen where table_id = any(%s) order by id",
                (corpus_ids,))
    for t, v in cur.fetchall():
        if v and v not in meta[t]["items"]:
            meta[t]["items"].append(v)
    cur.execute("select table_id, axis_name from kosis_vdb_axes_qwen where table_id = any(%s) "
                "order by table_id, axis_order", (corpus_ids,))
    for t, v in cur.fetchall():
        if v and v not in meta[t]["axes"]:
            meta[t]["axes"].append(v)
    cur.execute("select table_id, value_name from kosis_vdb_axis_values_qwen where table_id = any(%s) order by id",
                (corpus_ids,))
    seen: dict[str, set] = {}
    for t, v in cur.fetchall():
        s = seen.setdefault(t, set())
        if v and v not in s and len(meta[t]["values"]) < VALUE_CAP * 3:
            s.add(v)
            meta[t]["values"].append(v)
    conn.close()
    print("  메타데이터 수집 완료", flush=True)

    from sentence_transformers import SentenceTransformer
    model = SentenceTransformer("Qwen/Qwen3-Embedding-4B", truncate_dim=2560)

    qmap = json.loads((HERE / "queries.json").read_text(encoding="utf-8"))
    qtexts = [f"Instruct: {INSTRUCTION}\nQuery: {qmap[r['claim_id']]['full']}" for r in ev]
    qvec = model.encode(qtexts, batch_size=4, normalize_embeddings=True).astype(np.float32)
    gold = [set(r["gold"]) for r in ev]

    out = {}
    for label, parts in todo.items():
        t0 = time.time()
        texts = [build_text(meta[t], parts) for t in corpus_ids]
        M = model.encode(texts, batch_size=16, normalize_embeddings=True,
                         show_progress_bar=False).astype(np.float32)
        sims = qvec @ M.T                      # 정규화돼 있어 내적 = 코사인
        order = np.argsort(-sims, axis=1)[:, :max(K_LIST)]
        rec = {k: 0.0 for k in K_LIST}
        mrr = 0.0
        for i, g in enumerate(gold):
            ranked = [corpus_ids[j] for j in order[i]]
            pos = {t: r for r, t in enumerate(ranked)}
            first = min((pos[x] + 1 for x in g if x in pos), default=None)
            mrr += 1.0 / first if first else 0.0
            for k in K_LIST:
                rec[k] += sum(1 for x in g if pos.get(x, 10**9) < k) / len(g)
        n = len(gold)
        out[label] = {**{f"recall@{k}": rec[k] / n for k in K_LIST}, "mrr": mrr / n,
                      "avg_len": sum(len(t) for t in texts) / len(texts)}
        print(f"  [{label:<18}] R@10 {out[label]['recall@10']*100:5.1f}%  "
              f"R@100 {out[label]['recall@100']*100:5.1f}%  평균길이 {out[label]['avg_len']:.0f}자  "
              f"({time.time()-t0:.0f}s)", flush=True)

    prev = {}
    if (HERE / "exp_e.json").exists():
        prev = json.loads((HERE / "exp_e.json").read_text(encoding="utf-8")).get("results", {})
    prev.update(out)
    (HERE / "exp_e.json").write_text(json.dumps(
        {"corpus_size": len(corpus_ids), "results": prev}, ensure_ascii=False, indent=1), encoding="utf-8")
    print(f"\n저장: exp_e.json (축소 코퍼스 {len(corpus_ids):,}개 — 절대값이 아니라 구성 간 상대 비교용)")


if __name__ == "__main__":
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    main()
