"""
db/export_json.py — verifications.db(SQLite) → JSON export

프론트엔드(React 대시보드)는 SQLite 파일을 직접 열 수 없어서, 배치 파이프라인이
data/verifications.db에 쌓은 검증 결과를 프론트가 바로 fetch해서 쓸 수 있는 정적 JSON
배열로 내보내는 스크립트. db/store.py의 스키마(_COLUMNS)는 건드리지 않고 그대로 읽기만
한다 — 파이프라인 쪽 산출물은 원본 그대로 두고, export만 별도로 담당.

실행 (프로젝트 루트에서):
    python -m db.export_json
    python -m db.export_json --out data/verifications_export.json
"""

from __future__ import annotations

import argparse
import csv
import io
import json
from pathlib import Path

from db.fetch_article_text import fetch_clean_article_text
from db.store import DB_PATH, fetch_all

DEFAULT_OUT_PATH = Path(__file__).parent.parent / "data" / "verifications_export.json"
DEFAULT_ARTICLES_OUT_PATH = Path(__file__).parent.parent / "data" / "articles_export.json"
DEFAULT_DATES_OUT_PATH = Path(__file__).parent.parent / "data" / "article_dates_export.json"
DEFAULT_ORG_IDS_OUT_PATH = Path(__file__).parent.parent / "data" / "table_org_ids_export.json"
DATA_CSV_PATH = Path(__file__).parent.parent / "data" / "data_set.csv"
TABLE_CATALOG_PATH = Path(__file__).parent.parent / "agent" / "mapping" / "table_catalog.json"


def export_to_json(out_path: Path = DEFAULT_OUT_PATH, db_path: Path = DB_PATH) -> int:
    """verifications.db의 모든 레코드를 JSON 배열로 out_path에 저장하고, 저장한 건수를 반환한다."""
    rows = fetch_all(db_path)

    for row in rows:
        # store.py의 insert_verification이 kosis_dimension을 JSON 문자열로 직렬화해서
        # 저장하므로(dict를 그대로 SQLite TEXT 컬럼에 넣을 수 없어서), 프론트가 바로 객체로
        # 쓸 수 있게 여기서 다시 파싱해서 되돌린다.
        dim = row.get("kosis_dimension")
        if isinstance(dim, str) and dim:
            try:
                row["kosis_dimension"] = json.loads(dim)
            except json.JSONDecodeError:
                pass  # 파싱 안 되면 원래 문자열 그대로 둔다 (알 수 없는 형식, 데이터 보존 우선)

    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(rows, ensure_ascii=False, indent=2), encoding="utf-8")
    return len(rows)


def _load_csv_article_texts(csv_path: Path = DATA_CSV_PATH) -> dict[str, str]:
    """data_set.csv에서 기사제목 -> 정제된 본문을 매핑한다. agent/pipeline/batch_runner.py의
    _clean_scraped_article_text를 그대로 재사용해서, 실제 배치 실행이 claim_extractor에
    넘겼던 것과 동일한 텍스트를 재현한다(프론트에서 "2단계가 이 본문에서 주장을 다 뽑았는지"
    눈으로 확인하려면 실제로 모델이 본 텍스트여야 의미가 있다).

    batch_runner.py의 BOM 처리(2026-08-10 수정)와 동일하게, 파일 중간에 낀 BOM까지 제거한다.
    """
    from agent.pipeline.batch_runner import _clean_scraped_article_text

    if not csv_path.exists():
        return {}

    with open(csv_path, encoding="utf-8-sig") as f:
        text = f.read().replace("﻿", "")

    result: dict[str, str] = {}
    for row in csv.DictReader(io.StringIO(text)):
        title = row.get("기사제목", "")
        if not title:
            continue
        result[title] = _clean_scraped_article_text(title, row.get("기사 본문 전체", ""))
    return result


def _load_csv_article_dates(csv_path: Path = DATA_CSV_PATH) -> dict[str, str]:
    """data_set.csv에서 기사제목 -> 작성일("YYYY-MM-DD")을 매핑한다. db/store.py 스키마에는
    기사 작성일이 저장되지 않아서(검증 실행 시각인 created_at만 있음), 원문 텍스트와
    마찬가지로 CSV에서 직접 다시 읽어와야 한다. 날짜 형식이 이상하면(파싱 실패) 그 기사는
    건너뛴다 — 프론트가 latestCreatedAt 등으로 폴백한다."""
    if not csv_path.exists():
        return {}

    with open(csv_path, encoding="utf-8-sig") as f:
        text = f.read().replace("﻿", "")

    result: dict[str, str] = {}
    for row in csv.DictReader(io.StringIO(text)):
        title = row.get("기사제목", "")
        raw_date = row.get("작성일", "")
        if not title or not raw_date:
            continue
        try:
            y, m, d = (int(v) for v in raw_date.split("-"))
            result[title] = f"{y:04d}-{m:02d}-{d:02d}"
        except ValueError:
            continue
    return result


def export_article_dates(
    out_path: Path = DEFAULT_DATES_OUT_PATH,
    db_path: Path = DB_PATH,
    csv_path: Path = DATA_CSV_PATH,
) -> int:
    """verifications.db에 있는 article_title들에 대해서만 작성일을 골라 {제목: "YYYY-MM-DD"}
    JSON으로 내보낸다."""
    rows = fetch_all(db_path)
    titles_needed = {r["article_title"] for r in rows}

    all_dates = _load_csv_article_dates(csv_path)
    dates = {title: all_dates[title] for title in titles_needed if title in all_dates}

    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(dates, ensure_ascii=False, indent=2), encoding="utf-8")
    return len(dates)


def export_table_org_ids(
    out_path: Path = DEFAULT_ORG_IDS_OUT_PATH,
    db_path: Path = DB_PATH,
    catalog_path: Path = TABLE_CATALOG_PATH,
) -> int:
    """verifications.db에서 실제로 매칭된 kosis_table_id들에 대해서만 orgId를 골라
    {tblId: orgId} JSON으로 내보낸다. db/store.py 스키마엔 orgId가 없어서(kosis_table_id/
    kosis_table만 저장), KOSIS 표 상세 페이지로 바로 연결되는 딥링크
    (statHtml.do?orgId=...&tblId=...)를 만들려면 agent/mapping/table_catalog.json에서
    orgId를 따로 가져와야 한다."""
    rows = fetch_all(db_path)
    tbl_ids_needed = {r["kosis_table_id"] for r in rows if r.get("kosis_table_id")}

    if not catalog_path.exists():
        org_ids: dict[str, str] = {}
    else:
        catalog = json.loads(catalog_path.read_text(encoding="utf-8"))
        org_id_by_tbl = {t["tblId"]: t["orgId"] for t in catalog.get("tables", [])}
        org_ids = {tbl_id: org_id_by_tbl[tbl_id] for tbl_id in tbl_ids_needed if tbl_id in org_id_by_tbl}

    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(org_ids, ensure_ascii=False, indent=2), encoding="utf-8")
    return len(org_ids)


def export_article_texts(
    out_path: Path = DEFAULT_ARTICLES_OUT_PATH,
    db_path: Path = DB_PATH,
    csv_path: Path = DATA_CSV_PATH,
    *,
    fetch_live: bool = True,
) -> int:
    """verifications.db에 있는 article_title들에 대해서만 원문을 골라 {제목: 본문} JSON으로
    내보낸다 (data_set.csv 전체를 다 내보내면 낭비라 실제 검증된 기사만 추린다).

    fetch_live=True(기본값)면 article_url로 실제 페이지에 접속해 db/fetch_article_text.py로
    광고/내비게이션 잡음 없는 본문을 가져오는 걸 먼저 시도한다 — data_set.csv의 스크랩
    본문보다 훨씬 깨끗하다(Arc XP CMS의 구조화된 콘텐츠 JSON을 직접 읽기 때문). URL이 없거나
    접속/파싱에 실패하면(네트워크 오류, 다른 CMS 구조 등) data_set.csv 기반 텍스트로
    조용히 폴백한다 — 실패해도 export 전체가 중단되지 않는다.
    """
    rows = fetch_all(db_path)
    titles_needed = {r["article_title"]: r.get("article_url") for r in rows}

    all_texts = _load_csv_article_texts(csv_path)
    articles: dict[str, str] = {}

    for title, url in titles_needed.items():
        live_text = fetch_clean_article_text(url) if (fetch_live and url) else None
        if live_text:
            articles[title] = live_text
        elif title in all_texts:
            articles[title] = all_texts[title]

    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(articles, ensure_ascii=False, indent=2), encoding="utf-8")
    return len(articles)


def main() -> None:
    parser = argparse.ArgumentParser(description="verifications.db를 JSON으로 export")
    parser.add_argument("--out", type=Path, default=DEFAULT_OUT_PATH, help="출력 JSON 경로")
    parser.add_argument(
        "--articles-out",
        type=Path,
        default=DEFAULT_ARTICLES_OUT_PATH,
        help="기사 원문(제목→본문) JSON 출력 경로",
    )
    parser.add_argument(
        "--dates-out",
        type=Path,
        default=DEFAULT_DATES_OUT_PATH,
        help="기사 작성일(제목→YYYY-MM-DD) JSON 출력 경로",
    )
    parser.add_argument(
        "--org-ids-out",
        type=Path,
        default=DEFAULT_ORG_IDS_OUT_PATH,
        help="KOSIS 표 orgId(tblId→orgId) JSON 출력 경로",
    )
    parser.add_argument(
        "--no-live-fetch",
        action="store_true",
        help="URL 재접속 없이 data_set.csv 스크랩 본문만 사용 (오프라인/네트워크 없을 때)",
    )
    args = parser.parse_args()

    count = export_to_json(args.out)
    print(f"[export] {count}건을 {args.out}에 저장했습니다.")

    article_count = export_article_texts(args.articles_out, fetch_live=not args.no_live_fetch)
    print(f"[export] 기사 원문 {article_count}건을 {args.articles_out}에 저장했습니다.")

    date_count = export_article_dates(args.dates_out)
    print(f"[export] 기사 작성일 {date_count}건을 {args.dates_out}에 저장했습니다.")

    org_id_count = export_table_org_ids(args.org_ids_out)
    print(f"[export] KOSIS 표 orgId {org_id_count}건을 {args.org_ids_out}에 저장했습니다.")


if __name__ == "__main__":
    main()
