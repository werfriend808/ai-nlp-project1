"""agent/kosis/build_bm25_index.py -- kosis_vdb_tables_qwen.embedding_text 위에 BM25
희소 인덱스를 만들어 디스크에 캐시한다 (agent/kosis/bm25_index/).

배경: 운영이 지금까지 쓰던 lexical_query_vdb(pg_trgm 부분일치)는 287,498건 전체를
ORDER BY similarity()로 스캔해야 해서 쿼리당 6.9초가 걸렸다(embedding_text가 길어져서
GIN 인덱스 선택도가 무너짐, 2026-08-27 EXPLAIN ANALYZE 실측). benchmark/search_experiment2
에서 검증된 방식(디스크 캐시 BM25 희소행렬, 쿼리는 이진벡터 내적 한 번) 그대로 옮긴 것 —
같은 파라미터(K1=1.5, B=0.75, char_wb 2-3gram)를 그대로 쓴다. 이 파라미터 자체를 튜닝하는
건 3순위 작업(실험 B)이고, 여기서는 "trigram -> BM25 교체"만 한다.

운영 DB(FTS 인덱스 등)는 건드리지 않는다 -- 인덱스는 이 프로세스 로컬 파일로만 존재한다.

재생성이 필요한 시점: kosis_vdb_tables_qwen.embedding_text가 바뀔 때마다(예: v2
재임베딩, 표 신규/삭제 반영 후). 약 2분 소요(2026-08-28 실측, 287k행 기준).

사용법:
    python -m agent.kosis.build_bm25_index
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


def _load_env():
    env_path = HERE.parent.parent / ".env"
    if env_path.exists():
        for line in env_path.read_text(encoding="utf-8").splitlines():
            line = line.strip()
            if line and not line.startswith("#") and "=" in line:
                k, v = line.split("=", 1)
                os.environ.setdefault(k.strip(), v.strip())


def main() -> None:
    _load_env()
    OUT.mkdir(exist_ok=True)
    conn = psycopg2.connect(os.environ["SUPABASE_DB_URL"])
    cur = conn.cursor("bm25_build")     # 서버 커서 -- 287k행을 한 번에 안 들고 온다
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
                     "analyzer_params": {"analyzer": "char_wb", "ngram_range": (2, 3)},
                     "k1": K1, "b": B, "built_at": time.strftime("%Y-%m-%dT%H:%M:%S")}, f)
    print(f"저장 완료: {OUT}  ({(OUT/'matrix.npz').stat().st_size/1e6:.0f}MB)  "
          f"elapsed={time.time()-t0:.0f}s")


if __name__ == "__main__":
    main()
