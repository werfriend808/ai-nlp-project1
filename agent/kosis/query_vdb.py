"""
agent/kosis/query_vdb.py — KOSIS 표 28만7천여 개 VDB(Supabase/pgvector)를 미리 계산된
쿼리 벡터로 조회한다.

2026-08-18: Chroma(로컬 서버 전용)에서 Supabase(pgvector)로 옮겼다 — 코랩 등 외부에서도
접근 가능해서, 이전엔 로컬 Chroma 서버에 못 붙어 VDB를 건너뛰어야 했던 배치 경로에서도
이제 VDB를 정상적으로 쓸 수 있다. .env의 SUPABASE_DB_URL로 접속한다.

이 모듈은 임베딩 모델을 직접 불러오지 않는다 — 쿼리 벡터는 이미 다른 곳(예:
embedding_search.embed_sentences_batch, 64개 카탈로그 매칭과 같은 모델의 배치 호출)에서
만들어 온 것을 받기만 한다. 그래야 임베딩 모델 로딩/호출이 한 곳(배치)으로만 몰려서
반복 호출로 인한 세그폴트 위험을 피한다. 쿼리 벡터 쪽 모델(KOSIS_EMBEDDING_MODEL,
.env)과 VDB에 적재된 벡터의 모델(vdb_embedding_colab.ipynb)이 반드시 같아야 한다 —
다르면 벡터 공간이 안 맞아서 유사도 값 자체가 무의미해진다.
"""

from __future__ import annotations

import os
from typing import Optional

import psycopg2

try:
    from dotenv import find_dotenv, load_dotenv

    load_dotenv(find_dotenv(usecwd=True))
except ImportError:
    pass

try:
    from agent.interfaces import TableCandidate
except ImportError:
    from dataclasses import dataclass, field

    @dataclass
    class TableCandidate:  # type: ignore[no-redef]
        table_id: str
        table_name: str
        score: float
        required_slots: list = field(default_factory=list)
        source_meta: Optional[str] = None

TABLE_NAME = "kosis_vdb_tables"

# 28만여 개 중 상위 몇 개까지 후보로 볼지.
# 2026-08-17 변경(3 -> 10): 근사 최근접 이웃 검색(HNSW)이 실제로 진짜 정답을 상위 3등
# 안에 못 넣는 사례가 실측 확인됐다. VDB 후보는 어차피 "unverified"로 표시돼서 리랭커가
# 다시 평가하므로(_merge_candidates 참고), 여기서 늘려도 잘못된 표가 부당하게 승격될
# 위험은 없다 — 그냥 리랭커가 볼 후보 풀이 넓어질 뿐이다.
VDB_TOP_K = 10
# 절대 유사도 하한선. "28만개 중 그나마 제일 비슷한 것"이어도 이 밑이면 사실상 무관하다고
# 보고 후보에서 제외한다 — top-k 개수 제한만으로는 못 거르는 노이즈를 추가로 막는다.
VDB_MIN_SIMILARITY = 0.75

_SOURCE_MODEL = os.environ.get("KOSIS_EMBEDDING_MODEL", "intfloat/multilingual-e5-small")

_conn = None  # 커넥션은 프로세스당 한 번만 만들어 재사용


def _get_connection():
    global _conn
    if _conn is None or _conn.closed:
        db_url = os.environ.get("SUPABASE_DB_URL")
        if not db_url:
            raise VdbUnavailableError(
                "SUPABASE_DB_URL이 없습니다. .env에 Supabase 프로젝트의 Postgres 연결 문자열을 넣으세요."
            )
        _conn = psycopg2.connect(db_url)
    return _conn


class VdbUnavailableError(RuntimeError):
    """Supabase(pgvector)에 연결할 수 없음 (.env의 SUPABASE_DB_URL 확인 필요)."""


def batch_query_vdb(
    query_vectors: list[list[float]], *, top_k: int = VDB_TOP_K
) -> list[list[TableCandidate]]:
    """쿼리 벡터 여러 개를 하나씩 pgvector에 조회한다(claim 개수만큼의 후보 리스트 반환).

    Supabase에 연결할 수 없으면 VdbUnavailableError를 낸다 — 이 경우 호출부가 VDB
    없이(64개 카탈로그만으로) 계속 진행할지 말지 판단해야 한다."""
    if not query_vectors:
        return []

    try:
        conn = _get_connection()
    except psycopg2.OperationalError as e:
        raise VdbUnavailableError(f"Supabase(pgvector)에 연결 못 했습니다: {e}") from e

    results: list[list[TableCandidate]] = []
    try:
        with conn.cursor() as cur:
            for query_vec in query_vectors:
                # pgvector의 `<=>`는 코사인 거리(1 - 코사인 유사도)를 돌려준다(정규화된
                # 벡터 기준) — Chroma 때와 같은 변환식(유사도 = 1 - 거리)을 그대로 쓴다.
                cur.execute(
                    f"""
                    select tbl_id, text, embedding <=> %s::vector as distance
                    from {TABLE_NAME}
                    order by embedding <=> %s::vector
                    limit %s;
                    """,
                    (query_vec, query_vec, top_k),
                )
                rows = cur.fetchall()

                candidates = []
                for tbl_id, text, dist in rows:
                    similarity = 1.0 - float(dist)
                    if similarity < VDB_MIN_SIMILARITY:
                        continue
                    candidates.append(
                        TableCandidate(
                            table_id=tbl_id,
                            table_name=text,
                            score=similarity,
                            required_slots=[],
                            source_meta=f"kosis_vdb model={_SOURCE_MODEL}",
                        )
                    )
                results.append(candidates)
    except psycopg2.OperationalError as e:
        raise VdbUnavailableError(f"Supabase(pgvector) 조회 중 연결 오류: {e}") from e

    return results
