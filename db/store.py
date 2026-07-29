"""
db/store.py — 검증 결과 저장소 (SQLite)

팀이 설계한 검증 레코드 스키마(29개 필드)를 그대로 테이블 컬럼으로 옮긴 것.
DB 파일 하나(data/verifications.db)로 관리 — 이 프로젝트 규모에 별도 서버형 DB는
과함. UI(대시보드/검증자 리뷰)는 아직 없지만, 나중에 붙일 때 스키마를 다시 안 바꿔도
되게 reviewer_agrees/reviewer_corrected_verdict 컬럼을 미리 nullable로 열어둠.

사용법:
    from db.store import init_db, insert_verification, fetch_all
    init_db()
    insert_verification({...})
    rows = fetch_all()
"""

from __future__ import annotations

import hashlib
import json
import sqlite3
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional

DB_PATH = Path(__file__).parent.parent / "data" / "verifications.db"

_SCHEMA = """
CREATE TABLE IF NOT EXISTS verifications (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    result_id TEXT UNIQUE,
    article_title TEXT,
    article_url TEXT,
    claim_sentence TEXT,
    claim_type TEXT,
    statistic_expression TEXT,
    normalized_statistic_name TEXT,
    statistic_category TEXT,
    value REAL,
    unit TEXT,
    comparison_operator TEXT,
    comparison_target TEXT,
    comparison_value REAL,
    time_expression TEXT,
    reference_time TEXT,
    population TEXT,
    region TEXT,
    source_org TEXT,
    source_report TEXT,
    kosis_table_id TEXT,
    kosis_table TEXT,
    kosis_item TEXT,
    kosis_dimension TEXT,
    calculation_required INTEGER,
    calculation_type TEXT,
    verification_possible TEXT,
    ambiguity_reason TEXT,
    verification_result TEXT,
    mismatch_reason TEXT,
    evidence TEXT,
    classifier_score REAL,
    reviewer_agrees INTEGER,
    reviewer_corrected_verdict TEXT,
    created_at TEXT
)
"""

# 스키마 컬럼 중 record dict에서 그대로 꺼내 쓰는 것들 (id는 자동증가라 제외)
_COLUMNS = [
    "result_id", "article_title", "article_url", "claim_sentence", "claim_type",
    "statistic_expression", "normalized_statistic_name", "statistic_category",
    "value", "unit", "comparison_operator", "comparison_target", "comparison_value",
    "time_expression", "reference_time", "population", "region",
    "source_org", "source_report", "kosis_table_id", "kosis_table",
    "kosis_item", "kosis_dimension", "calculation_required", "calculation_type",
    "verification_possible", "ambiguity_reason", "verification_result",
    "mismatch_reason", "evidence", "classifier_score",
    "reviewer_agrees", "reviewer_corrected_verdict", "created_at",
]


def make_result_id(article_title: str, claim_sentence: str) -> str:
    """기사 제목 + claim 문장으로 안정적인 id를 만듦 (같은 내용이면 재실행해도 같은 id —
    나중에 검증자 리뷰 기록을 이 id에 연결할 때 재실행 때마다 안 바뀌게 하기 위함)."""
    raw = f"{article_title}|{claim_sentence}".encode("utf-8")
    return hashlib.md5(raw).hexdigest()[:12]


def init_db(path: Path = DB_PATH) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with sqlite3.connect(path) as conn:
        conn.execute(_SCHEMA)


def insert_verification(record: dict, path: Path = DB_PATH) -> None:
    """record: _COLUMNS에 해당하는 키만 골라서 저장 (없는 키는 None으로 채움).
    result_id가 이미 있으면(재실행) 덮어씀 — INSERT OR REPLACE.
    kosis_dimension이 dict로 들어오면 JSON 문자열로 직렬화."""
    row = dict(record)
    if isinstance(row.get("kosis_dimension"), (dict, list)):
        row["kosis_dimension"] = json.dumps(row["kosis_dimension"], ensure_ascii=False)
    row.setdefault("created_at", datetime.now(timezone.utc).isoformat())

    values = [row.get(col) for col in _COLUMNS]
    placeholders = ", ".join("?" for _ in _COLUMNS)
    columns_sql = ", ".join(_COLUMNS)

    init_db(path)
    with sqlite3.connect(path) as conn:
        conn.execute(
            f"INSERT OR REPLACE INTO verifications ({columns_sql}) VALUES ({placeholders})",
            values,
        )


def fetch_all(path: Path = DB_PATH) -> list[dict]:
    init_db(path)
    with sqlite3.connect(path) as conn:
        conn.row_factory = sqlite3.Row
        rows = conn.execute("SELECT * FROM verifications ORDER BY id").fetchall()
        return [dict(r) for r in rows]


if __name__ == "__main__":
    #   python -m db.store  — 스모크 테스트 (더미 레코드 하나 넣고 다시 읽기)
    init_db()
    sample = {
        "result_id": make_result_id("샘플 기사", "샘플 주장 문장입니다."),
        "article_title": "샘플 기사",
        "claim_sentence": "샘플 주장 문장입니다.",
        "claim_type": "규모",
        "value": 55.8,
        "unit": "kg",
        "verification_result": "일치",
        "classifier_score": 0.9,
    }
    insert_verification(sample)
    rows = fetch_all()
    print(f"저장된 레코드 수: {len(rows)}")
    print(rows[-1])
    assert rows[-1]["result_id"] == sample["result_id"]
    assert rows[-1]["value"] == 55.8
    print("✅ db/store.py 스모크 테스트 통과")
