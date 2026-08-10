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
import json
from pathlib import Path

from db.store import DB_PATH, fetch_all

DEFAULT_OUT_PATH = Path(__file__).parent.parent / "data" / "verifications_export.json"


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


def main() -> None:
    parser = argparse.ArgumentParser(description="verifications.db를 JSON으로 export")
    parser.add_argument("--out", type=Path, default=DEFAULT_OUT_PATH, help="출력 JSON 경로")
    args = parser.parse_args()

    count = export_to_json(args.out)
    print(f"[export] {count}건을 {args.out}에 저장했습니다.")


if __name__ == "__main__":
    main()
