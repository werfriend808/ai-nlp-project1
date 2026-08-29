"""kosis_vdb_tables_qwen.embedding_text 위에 BM25 희소 인덱스를 만들어 디스크에 캐시한다.

    python -m agent.kosis.build_bm25_index

운영 DB에는 인덱스를 붙이지 않는다 — PostgreSQL에 BM25가 없어서 확장을 새로 깔아야 하는데,
읽기 전용으로 쓰는 운영 DB에 그런 변경을 넣는 것보다 Python 쪽에 캐시를 두는 편이 안전하다.
한국어는 형태소 분석기 없이도 문자 n-gram이 잘 먹으므로 char_wb (2,3)을 쓴다.

    score(d, q) = sum_{t in q} IDF(t) * (tf * (k1+1)) / (tf + k1*(1 - b + b*|d|/avgdl))

문서쪽 항을 전부 미리 계산해 행렬에 담아두므로, 질의 때는 해당 n-gram 열들을 더하기만 하면
된다(실측 2~10ms). 행렬은 열 슬라이싱만 하므로 CSC로 저장한다 — CSR로 저장하면 로딩할 때
변환 비용과 피크 메모리가 그만큼 더 든다.

표가 추가·변경되면 다시 돌려야 한다(28.7만 표 기준 약 2분). 검색 결과에 필요한
table_name/org_id도 같이 저장해서, 질의 때 DB를 다시 때리지 않는다.
"""
from __future__ import annotations

import os
import pickle
import time
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import psycopg2
import scipy.sparse as sp
from sklearn.feature_extraction.text import CountVectorizer

from agent.kosis.bm25_search import (
    ANALYZER, INDEX_DIR, MATRIX_FILE, META_FILE, NGRAM_RANGE, SOURCE_TABLE,
)

K1, B = 1.5, 0.75
MIN_DF = 5
MAX_FEATURES = 300_000


def main() -> None:
    out = INDEX_DIR
    out.mkdir(parents=True, exist_ok=True)

    conn = psycopg2.connect(os.environ["SUPABASE_DB_URL"])
    cur = conn.cursor("bm25_build")   # 서버 커서 — 28.7만 행을 한 번에 안 들고 온다
    cur.itersize = 20000
    cur.execute(
        f"select table_id, table_name, org_id, embedding_text from {SOURCE_TABLE} order by table_id"
    )
    ids, names, orgs, docs = [], [], [], []
    for tid, name, org, txt in cur:
        ids.append(tid)
        names.append(name or "")
        orgs.append(org or "")
        docs.append(txt or "")
    conn.close()
    print(f"문서 {len(docs):,}개 로드", flush=True)

    t0 = time.time()
    vec = CountVectorizer(analyzer=ANALYZER, ngram_range=NGRAM_RANGE,
                          min_df=MIN_DF, max_features=MAX_FEATURES, lowercase=True)
    X = vec.fit_transform(docs)       # CSR, 정수 tf
    print(f"  vectorize {time.time()-t0:.0f}s  shape={X.shape}  nnz={X.nnz:,}", flush=True)

    n_docs = X.shape[0]
    dl = np.asarray(X.sum(axis=1)).ravel().astype(np.float32)
    avgdl = float(dl.mean())
    df = np.asarray((X > 0).sum(axis=0)).ravel().astype(np.float32)
    idf = np.log(1.0 + (n_docs - df + 0.5) / (df + 0.5)).astype(np.float32)

    X = X.tocsr().astype(np.float32)
    norm = (K1 * (1 - B + B * dl / avgdl)).astype(np.float32)
    for i in range(n_docs):
        s, e = X.indptr[i], X.indptr[i + 1]
        tf = X.data[s:e]
        X.data[s:e] = tf * (K1 + 1) / (tf + norm[i])
    X = X.multiply(idf[np.newaxis, :]).tocsc().astype(np.float32)
    print(f"  BM25 가중치 계산 완료 {time.time()-t0:.0f}s", flush=True)

    sp.save_npz(out / MATRIX_FILE.name, X)
    with (out / META_FILE.name).open("wb") as f:
        pickle.dump(
            {
                "ids": ids, "names": names, "orgs": orgs,
                "vocabulary": vec.vocabulary_,
                "analyzer_params": {"analyzer": ANALYZER, "ngram_range": NGRAM_RANGE},
                "built_at": datetime.now(timezone.utc).isoformat(),
                "source_table": SOURCE_TABLE, "n_docs": n_docs,
                "bm25_params": {"k1": K1, "b": B, "min_df": MIN_DF, "max_features": MAX_FEATURES},
            },
            f,
        )
    size = (out / MATRIX_FILE.name).stat().st_size / 1e6
    print(f"저장 완료: {out}  ({size:.0f}MB, 문서 {n_docs:,}개)")


if __name__ == "__main__":
    main()
