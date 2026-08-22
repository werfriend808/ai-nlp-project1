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
import threading
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional

DB_PATH = Path(__file__).parent.parent / "data" / "verifications.db"

# 2026-08-22: run_article()의 주장(claim) 처리를 스레드풀로 병렬화하면서 추가 — 여러
# 스레드가 동시에 sqlite3.connect()로 각자 커넥션을 열고 INSERT하면 기본 저널 모드에서
# "database is locked"(SQLITE_BUSY)가 날 수 있다. 쓰기 자체는 매우 짧아 락 경합 비용이
# 미미하므로, 파일 락 재시도에 맡기지 않고 아예 이 프로세스 안에서 직렬화한다.
_WRITE_LOCK = threading.Lock()

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
    created_at TEXT,
    published_date TEXT
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
    # 2026-08-21 추가: 기사 실제 발행일(YYYY-MM-DD). 예전엔 이 컬럼이 없어서 실시간 URL
    # 검증(agent/api/server.py) 기사는 정확한 날짜를 뽑고도 저장할 곳이 없어 프론트가
    # "검증 실행 시각"으로 잘못 폴백했다 — db/export_json.py의 export_article_dates() 참고.
    "published_date",
]


def make_result_id(article_title: str, claim_sentence: str) -> str:
    """기사 제목 + claim 문장으로 안정적인 id를 만듦 (같은 내용이면 재실행해도 같은 id —
    나중에 검증자 리뷰 기록을 이 id에 연결할 때 재실행 때마다 안 바뀌게 하기 위함)."""
    raw = f"{article_title}|{claim_sentence}".encode("utf-8")
    return hashlib.md5(raw).hexdigest()[:12]


def init_db(path: Path = DB_PATH) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    # sqlite3.Connection을 "with conn:"으로 쓰면 커밋/롤백만 해줄 뿐 연결 자체는 닫지
    # 않는다 — 매 호출마다 커넥션이 열린 채로 남아, Windows에서는 파일이 계속 "사용 중"으로
    # 잠겨서(예: 테스트가 임시 DB 파일을 지우려 할 때 WinError 32) 문제가 된다. try/finally로
    # close()를 명시해서 커밋 동작은 그대로 유지하면서 연결은 확실히 반환한다.
    conn = sqlite3.connect(path)
    try:
        conn.execute(_SCHEMA)
        # CREATE TABLE IF NOT EXISTS는 이미 만들어진 기존 테이블엔 새 컬럼을 추가해주지
        # 않는다 — published_date 컬럼 도입(2026-08-21) 전에 이미 존재하던 verifications.db
        # 파일들을 위해 없으면 추가한다. 여러 프로세스가 동시에 init_db()를 부를 수 있어서
        # (예: 배치 실행 중 API 서버도 기동) "이미 있음" 에러는 조용히 무시한다.
        existing_cols = {row[1] for row in conn.execute("PRAGMA table_info(verifications)")}
        if "published_date" not in existing_cols:
            try:
                conn.execute("ALTER TABLE verifications ADD COLUMN published_date TEXT")
            except sqlite3.OperationalError:
                pass
        conn.commit()
    finally:
        conn.close()


def insert_verification(record: dict, path: Path = DB_PATH) -> None:
    """record: _COLUMNS에 해당하는 키만 골라서 저장 (없는 키는 None으로 채움).
    result_id가 이미 있으면(재실행) 덮어씀 — INSERT OR REPLACE.

    SQLite는 list/dict를 못 담아서 JSON 문자열로 직렬화한다. kosis_dimension이 대표
    사례지만, 2단계 claim_extractor가 (예: "대구·전남·울산 등 늘었다"처럼 지역이 여러 개인
    문장에서) region 등 원래 문자열이어야 할 필드를 배열로 반환하는 경우가 실제로 있어서
    (실측: sqlite3.ProgrammingError로 배치 전체가 죽던 버그) 모든 필드에 방어적으로 적용한다.
    """
    row = dict(record)
    for key, value in row.items():
        if isinstance(value, (dict, list)):
            row[key] = json.dumps(value, ensure_ascii=False)
    row.setdefault("created_at", datetime.now(timezone.utc).isoformat())

    values = [row.get(col) for col in _COLUMNS]
    placeholders = ", ".join("?" for _ in _COLUMNS)
    columns_sql = ", ".join(_COLUMNS)

    init_db(path)
    with _WRITE_LOCK:
        conn = sqlite3.connect(path)
        try:
            conn.execute(
                f"INSERT OR REPLACE INTO verifications ({columns_sql}) VALUES ({placeholders})",
                values,
            )
            conn.commit()
        finally:
            conn.close()


def fetch_all(path: Path = DB_PATH) -> list[dict]:
    init_db(path)
    conn = sqlite3.connect(path)
    try:
        conn.row_factory = sqlite3.Row
        rows = conn.execute("SELECT * FROM verifications ORDER BY id").fetchall()
        return [dict(r) for r in rows]
    finally:
        conn.close()


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
