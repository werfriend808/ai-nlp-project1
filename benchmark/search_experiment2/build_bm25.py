"""kosis_vdb_tables_qwen.embedding_text 위에 BM25 희소 인덱스를 만들어 디스크에 캐시한다.

운영 DB를 건드리지 않기 위해(FTS 인덱스를 붙이지 않기 위해) Python 쪽에 인덱스를 만든다.
한국어는 형태소 분석기 없이도 문자 n-gram이 잘 먹으므로 char_wb (2,3)을 쓴다.

score(d, q) = sum_{t in q} IDF(t) * (tf * (k1+1)) / (tf + k1*(1 - b + b*|d|/avgdl))
위 식의 문서쪽 항을 전부 미리 계산해 CSR에 담아두면, 질의는 이진 벡터와의 내적 한 번이다.
"""
from __future__ import annotations

import os
import pickle
import time
from pathlib import Path

import numpy as np
import psycopg2
import scipy.sparse as sp
from sklearn.feature_extraction.text import CountVectorizer

HERE = Path(__file__).parent
OUT = HERE / "bm25_index"
K1, B = 1.5, 0.75


def main() -> None:
    OUT.mkdir(exist_ok=True)
    conn = psycopg2.connect(os.environ["SUPABASE_DB_URL"])
    cur = conn.cursor("bm25")          # 서버 커서 — 287k행을 한 번에 안 들고 온다
    cur.itersize = 20000
    cur.execute("select table_id, embedding_text from kosis_vdb_tables_qwen order by table_id")

    ids, docs = [], []
    for tid, txt in cur:
        ids.append(tid)
        docs.append(txt or "")
    conn.close()
    print(f"문서 {len(docs):,}개 로드", flush=True)

    t0 = time.time()
    vec = CountVectorizer(analyzer="char_wb", ngram_range=(2, 3),
                          min_df=5, max_features=300_000, lowercase=True)
    X = vec.fit_transform(docs)        # CSR, 정수 tf
    print(f"  vectorize {time.time()-t0:.0f}s  shape={X.shape}  nnz={X.nnz:,}", flush=True)

    n_docs = X.shape[0]
    dl = np.asarray(X.sum(axis=1)).ravel().astype(np.float32)
    avgdl = float(dl.mean())
    df = np.asarray((X > 0).sum(axis=0)).ravel().astype(np.float32)
    idf = np.log(1.0 + (n_docs - df + 0.5) / (df + 0.5)).astype(np.float32)

    # 문서쪽 BM25 가중치를 제자리에서 계산
    X = X.tocsr().astype(np.float32)
    norm = (K1 * (1 - B + B * dl / avgdl)).astype(np.float32)
    for i in range(n_docs):
        s, e = X.indptr[i], X.indptr[i + 1]
        tf = X.data[s:e]
        X.data[s:e] = tf * (K1 + 1) / (tf + norm[i])
    X = X.multiply(idf[np.newaxis, :]).tocsr().astype(np.float32)
    print(f"  BM25 가중치 계산 완료 {time.time()-t0:.0f}s", flush=True)

    sp.save_npz(OUT / "matrix.npz", X)
    with (OUT / "meta.pkl").open("wb") as f:
        pickle.dump({"ids": ids, "vocabulary": vec.vocabulary_,
                     "analyzer_params": {"analyzer": "char_wb", "ngram_range": (2, 3)}}, f)
    print(f"저장 완료: {OUT}  ({(OUT/'matrix.npz').stat().st_size/1e6:.0f}MB)")


if __name__ == "__main__":
    main()
