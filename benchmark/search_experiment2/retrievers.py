"""독립 검색기 모음 — 어떤 검색기도 다른 검색기의 후보를 hard filter 하지 않는다.

각 검색기는 (query) -> [table_id, ...] (순위 순)만 돌려준다. 융합은 fusion.py가 맡는다.
운영 DB는 읽기만 한다.
"""
from __future__ import annotations

import os
import pickle
import re
import time
from pathlib import Path

import numpy as np
import psycopg2
import scipy.sparse as sp
from sklearn.feature_extraction.text import CountVectorizer

HERE = Path(__file__).parent
DIM = 2560
EF_CAP = 1000          # pgvector hnsw.ef_search 상한


def connect():
    c = psycopg2.connect(os.environ["SUPABASE_DB_URL"])
    c.autocommit = True
    with c.cursor() as cur:
        cur.execute("set pg_trgm.similarity_threshold = 0.1")
    return c


def vec_literal(v) -> str:
    return "[" + ",".join(f"{x:.7f}" for x in v) + "]"


def _set_ef(cur, n: int) -> None:
    # pgvector는 ef_search보다 많은 행을 반환하지 않는다 — LIMIT보다 크게 잡아야 한다.
    cur.execute(f"set hnsw.ef_search = {min(max(n, 40), EF_CAP)}")
    # 2026-08-27 실측: 플래너가 HNSW 인덱스 비용을 과대평가해서 병렬 seq scan을 고른다.
    # kosis_vdb_axes_qwen은 그 결과 매 질의가 40초씩 걸렸다(인덱스 강제 시 0.9초).
    # 벡터 최근접 질의에서 seq scan이 이득인 경우는 없으므로 세션 단위로 꺼둔다.
    cur.execute("set enable_seqscan = off")


# ── dense 계열 ────────────────────────────────────────────────────────────
def dense_tables(cur, qvec: str, top_k: int) -> list[str]:
    _set_ef(cur, top_k * 2)
    cur.execute(
        """select table_id from kosis_vdb_tables_qwen
           order by embedding::halfvec(%s) <=> %s::halfvec(%s) limit %s""",
        (DIM, qvec, DIM, top_k),
    )
    return [r[0] for r in cur.fetchall()]


def _dense_child(cur, table: str, qvec: str, top_k: int, fetch: int) -> list[str]:
    """item/axis처럼 표에 딸린 행을 검색한 뒤 표 단위로 접는다(최고 유사도 기준)."""
    _set_ef(cur, fetch)
    cur.execute(
        f"""select table_id, 1-(embedding::halfvec(%s) <=> %s::halfvec(%s)) as sim
            from {table}
            order by embedding::halfvec(%s) <=> %s::halfvec(%s) limit %s""",
        (DIM, qvec, DIM, DIM, qvec, DIM, fetch),
    )
    best: dict[str, float] = {}
    for tid, sim in cur.fetchall():
        if tid not in best or sim > best[tid]:
            best[tid] = sim
    return sorted(best, key=lambda t: -best[t])[:top_k]


def dense_items(cur, qvec: str, top_k: int, fetch: int = EF_CAP) -> list[str]:
    return _dense_child(cur, "kosis_vdb_items_qwen", qvec, top_k, fetch)


def dense_axes(cur, qvec: str, top_k: int, fetch: int = EF_CAP) -> list[str]:
    return _dense_child(cur, "kosis_vdb_axes_qwen", qvec, top_k, fetch)


# ── 어휘 계열 ─────────────────────────────────────────────────────────────
def lexical_trgm(cur, text: str, top_k: int) -> list[str]:
    """운영 lexical_query_vdb와 동일한 trigram 쿼리(느린 쪽 — 비교 기준으로만 둔다)."""
    if not text.strip():
        return []
    cur.execute(
        """select table_id from kosis_vdb_tables_qwen
           where embedding_text %% %s
           order by similarity(embedding_text, %s) desc limit %s""",
        (text, text, top_k),
    )
    return [r[0] for r in cur.fetchall()]


class BM25:
    """디스크 캐시된 희소 BM25 인덱스. 질의는 이진 벡터와의 내적 한 번."""

    def __init__(self, path: Path = HERE / "bm25_index"):
        self.X = sp.load_npz(path / "matrix.npz").tocsc()
        with (path / "meta.pkl").open("rb") as f:
            meta = pickle.load(f)
        self.ids = np.array(meta["ids"])
        self.vocab = meta["vocabulary"]
        self._an = CountVectorizer(analyzer="char_wb", ngram_range=(2, 3),
                                   lowercase=True).build_analyzer()

    def search(self, text: str, top_k: int) -> list[str]:
        if not text.strip():
            return []
        cols = {self.vocab[g] for g in set(self._an(text)) if g in self.vocab}
        if not cols:
            return []
        scores = np.zeros(self.X.shape[0], dtype=np.float32)
        for c in cols:
            s, e = self.X.indptr[c], self.X.indptr[c + 1]
            np.add.at(scores, self.X.indices[s:e], self.X.data[s:e])
        k = min(top_k, len(scores))
        idx = np.argpartition(-scores, k - 1)[:k]
        idx = idx[np.argsort(-scores[idx])]
        idx = idx[scores[idx] > 0]
        return self.ids[idx].tolist()


# ── 융합 ─────────────────────────────────────────────────────────────────
def rrf(rankings: dict[str, list[str]], k: int = 60,
        weights: dict[str, float] | None = None, top_k: int = 200) -> list[str]:
    """Reciprocal Rank Fusion. hard filter 없이 순위만 합산한다."""
    scores: dict[str, float] = {}
    for name, lst in rankings.items():
        w = (weights or {}).get(name, 1.0)
        if w == 0:
            continue
        for i, tid in enumerate(lst):
            scores[tid] = scores.get(tid, 0.0) + w / (k + i + 1)
    return sorted(scores, key=lambda t: -scores[t])[:top_k]


class Timer:
    def __init__(self):
        self.ms = 0.0

    def __enter__(self):
        self._t = time.perf_counter()
        return self

    def __exit__(self, *a):
        self.ms = (time.perf_counter() - self._t) * 1000
