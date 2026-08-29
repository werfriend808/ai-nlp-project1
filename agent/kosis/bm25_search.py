"""BM25 어휘 검색 — 3단계 리트리버 중 lexical 자리.

기존 trigram(pg_trgm, query_vdb.lexical_query_vdb)을 대체한다. 골든셋 70건 실측
(benchmark/search_experiment2/REPORT.md):

    trgm_struct    Recall@100  4.3%   지연 5,649ms
    bm25_struct    Recall@100 32.1%   지연     2ms

trigram은 단어 경계를 모른 채 글자 3개씩만 겹쳐보는 방식이라 긴 설명문 검색에는 맞지
않았고, 28.7만 행 전수 비교라 느리기까지 했다. 짧은 분류값 매칭(kosis_vdb_axis_values_qwen)
쪽은 trigram이 여전히 적절하므로 그대로 둔다 — 여기서 바꾸는 건 표 검색 자리 하나다.

인덱스는 운영 DB가 아니라 디스크에 캐시된 희소 행렬이다(agent/kosis/build_bm25_index.py로
생성, 28.7만 표 기준 약 2분 / 약 340MB). 표가 추가·변경되면 다시 만들어야 한다.

메모리 주의: 로딩하면 프로세스 RSS가 약 1.6GB 늘어난다(실측). Qwen 임베딩·리랭커와 같은
프로세스에 올릴 때는 이 몫을 감안해야 한다. 로딩은 첫 질의 때까지 미룬다(lazy).
"""
from __future__ import annotations

import os
import pickle
import threading
from pathlib import Path
from typing import Optional

from agent.kosis.query_vdb import TABLE_NAME as _VDB_TABLE
from agent.kosis.query_vdb import TEXT_COL as _VDB_TEXT_COL
from agent.kosis.query_vdb import VdbUnavailableError, _get_connection

try:
    from agent.interfaces import TableCandidate
except ImportError:  # 단독 실행/테스트용 폴백
    from dataclasses import dataclass, field

    @dataclass
    class TableCandidate:  # type: ignore[no-redef]
        table_id: str
        table_name: str
        score: float
        required_slots: list = field(default_factory=list)
        source_meta: Optional[str] = None
        org_id: Optional[str] = None


SOURCE_TABLE = "kosis_vdb_tables_qwen"
ANALYZER = "char_wb"
NGRAM_RANGE = (2, 3)

_ROOT = Path(__file__).resolve().parents[2]
INDEX_DIR = Path(os.environ.get("KOSIS_BM25_INDEX_DIR", _ROOT / "data" / "bm25_index"))
MATRIX_FILE = INDEX_DIR / "matrix.npz"
META_FILE = INDEX_DIR / "meta.pkl"

# trigram 시절 이름을 그대로 쓴다 — 배포에서 BM25_TOP_K로 조정하던 관행을 깨지 않기 위함.
BM25_TOP_K = int(os.environ.get("BM25_TOP_K", "30"))

_index = None
_index_lock = threading.Lock()
_load_failed: Optional[str] = None


class _Bm25Index:
    """디스크에 캐시된 희소 BM25 인덱스. 질의는 해당 n-gram 열들의 합 한 번."""

    def __init__(self, path: Path):
        import numpy as np
        import scipy.sparse as sp
        from sklearn.feature_extraction.text import CountVectorizer

        self._np = np
        # 열 슬라이싱만 하므로 CSC. 빌더가 CSC로 저장하므로 보통 이 변환은 무비용이고,
        # CSR로 저장된 옛 인덱스가 들어와도 여기서 한 번 변환해 그대로 쓴다.
        self.X = sp.load_npz(path / MATRIX_FILE.name).tocsc()
        with (path / META_FILE.name).open("rb") as f:
            meta = pickle.load(f)
        self.ids = np.array(meta["ids"])
        # names/orgs는 이 모듈이 도입되며 추가됐다 — 옛 인덱스에는 없으므로 빈 값으로 둔다.
        self.names = np.array(meta.get("names") or [""] * len(self.ids))
        self.orgs = np.array(meta.get("orgs") or [""] * len(self.ids))
        self.vocab = meta["vocabulary"]
        self.n_docs = meta.get("n_docs", len(self.ids))
        self.built_at = meta.get("built_at")
        params = meta.get("analyzer_params") or {}
        self._analyze = CountVectorizer(
            analyzer=params.get("analyzer", ANALYZER),
            ngram_range=tuple(params.get("ngram_range", NGRAM_RANGE)),
            lowercase=True,
        ).build_analyzer()

    def search(self, text: str, top_k: int) -> list[tuple[str, str, str, float]]:
        np = self._np
        if not text.strip():
            return []
        cols = {self.vocab[g] for g in set(self._analyze(text)) if g in self.vocab}
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
        return [
            (str(self.ids[i]), str(self.names[i]), str(self.orgs[i]), float(scores[i]))
            for i in idx
        ]


def _get_index() -> "_Bm25Index":
    """인덱스를 lazy 로딩해서 프로세스 전체에서 재사용한다.

    한 번 실패하면 이유를 기억해뒀다가 이후 호출에서 재시도 없이 같은 예외를 올린다 —
    매 claim마다 340MB 로딩을 다시 시도하면 배치가 사실상 멈춘다.
    """
    global _index, _load_failed
    if _index is not None:
        return _index
    with _index_lock:
        if _index is not None:
            return _index
        if _load_failed is not None:
            raise VdbUnavailableError(_load_failed)
        if not MATRIX_FILE.exists() or not META_FILE.exists():
            _load_failed = (
                f"BM25 인덱스가 없습니다({INDEX_DIR}). "
                f"`python -m agent.kosis.build_bm25_index`로 먼저 생성하세요."
            )
            raise VdbUnavailableError(_load_failed)
        try:
            _index = _Bm25Index(INDEX_DIR)
        except Exception as e:  # noqa: BLE001 - 손상된 캐시/의존성 누락 모두 같은 폴백
            _load_failed = f"BM25 인덱스 로딩 실패({type(e).__name__}: {e})"
            raise VdbUnavailableError(_load_failed) from e
        return _index


def _fetch_embedding_texts(table_ids: list[str]) -> dict[str, str]:
    """반환할 후보들의 embedding_text를 DB에서 한 번에 가져온다.

    리랭커(reranker.rerank)는 document_texts에 없는 후보를 candidate.table_name으로
    판단한다. dense 경로(query_vdb.batch_query_vdb)와 옛 trigram 경로는 그 필드에
    embedding_text(평균 303자)를 담아 보내므로, BM25만 짧은 표명(평균 16자)을 담으면
    같은 리랭커 앞에서 BM25 후보만 불리해진다 — 후보 풀의 과반이 BM25 몫이라 영향이 크다.

    인덱스에 텍스트를 통째로 넣지 않고 조회 시점에 가져오는 이유: 인덱스 크기가 27MB에서
    110MB대로 커지는 걸 피하고, embedding_text가 갱신되면(excluded 표 복구 등) 인덱스를
    다시 만들지 않아도 최신 텍스트가 따라오게 하기 위함이다. 기본키 조회라 비용은 작다.

    실패하면 빈 dict를 돌려준다 — 호출부가 인덱스에 저장된 표명으로 폴백한다.
    """
    if not table_ids:
        return {}
    try:
        conn = _get_connection()
        with conn.cursor() as cur:
            cur.execute(
                f"select {'table_id'}, {_VDB_TEXT_COL} from {_VDB_TABLE} where table_id = any(%s)",
                (table_ids,),
            )
            return {tid: txt for tid, txt in cur.fetchall() if txt}
    except Exception as e:  # noqa: BLE001 - 조회 실패는 폴백 대상, BM25 자체는 살린다
        print(f"[BM25] embedding_text 조회 실패({type(e).__name__}) — 표명으로 폴백")
        return {}


def bm25_query_vdb(query_text: str, *, top_k: int = BM25_TOP_K) -> list[TableCandidate]:
    """BM25로 표를 검색한다. query_vdb.lexical_query_vdb와 같은 자리에 꽂히는 대체품.

    query_text는 호출부가 준비한다 — BM25는 dense와 달리 짧은 구조화 질의에서 성능이
    훨씬 좋으므로(실측: 구조화 47.1% vs 문장 전체 17.9% vs 문맥+문장 12.1%),
    reranker.build_lexical_query()로 만든 문자열을 넘길 것.

    인덱스가 없거나 손상되면 VdbUnavailableError를 올린다 — 호출부는 이걸 잡아서
    dense/keyword만으로 계속 진행한다(trigram 시절과 동일한 실패 처리).
    """
    if not query_text or not query_text.strip():
        return []
    rows = _get_index().search(query_text, top_k)
    # 리랭커가 판단 근거로 쓰는 텍스트 — dense 경로와 같은 embedding_text를 담는다.
    texts = _fetch_embedding_texts([tid for tid, _, _, _ in rows])
    return [
        TableCandidate(
            table_id=tid,
            table_name=texts.get(tid) or name,
            score=score,
            required_slots=[],
            source_meta="kosis_vdb_lexical(bm25)",
            org_id=org or None,
        )
        for tid, name, org, score in rows
    ]


def index_info() -> dict:
    """운영 점검용 — 인덱스가 있는지, 언제 만든 건지."""
    if not MATRIX_FILE.exists():
        return {"available": False, "dir": str(INDEX_DIR)}
    return {
        "available": True,
        "dir": str(INDEX_DIR),
        "size_mb": round(MATRIX_FILE.stat().st_size / 1e6, 1),
        "loaded": _index is not None,
        **({"built_at": _index.built_at, "n_docs": _index.n_docs} if _index else {}),
    }
