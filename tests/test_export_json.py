"""
tests/test_export_json.py — db/export_json.py 회귀 테스트

verifications.db(SQLite) 레코드가 JSON 배열로 정확히 변환되는지, 특히
kosis_dimension처럼 store.py가 저장 시 JSON 문자열로 직렬화해두는 필드가 export 시
다시 객체(dict)로 파싱되어 나오는지 확인한다. 실제 data/verifications.db를 건드리지
않도록 매 케이스마다 임시 디렉터리에 별도 DB/출력 파일을 만든다.

실행 (프로젝트 루트에서):
    python -m tests.test_export_json
"""

from __future__ import annotations

import json
import tempfile
from pathlib import Path

from db import store
from db.export_json import export_to_json


def case_01_export_produces_valid_json_array():
    """insert한 레코드가 그대로 JSON 배열 파일로 나오는지 확인."""
    with tempfile.TemporaryDirectory() as tmp:
        db_path = Path(tmp) / "test_verifications.db"
        out_path = Path(tmp) / "export.json"

        store.insert_verification(
            {
                "result_id": "abc123",
                "article_title": "테스트 기사",
                "claim_sentence": "테스트 주장 문장입니다.",
                "claim_type": "규모",
                "value": 55.8,
                "unit": "kg",
                "verification_result": "일치",
                "classifier_score": 0.9,
            },
            path=db_path,
        )

        count = export_to_json(out_path, db_path=db_path)
        assert count == 1, f"export 건수가 다름 (기대 1, 실제 {count})"

        data = json.loads(out_path.read_text(encoding="utf-8"))
        assert len(data) == 1, data
        assert data[0]["result_id"] == "abc123", data[0]
        assert data[0]["claim_sentence"] == "테스트 주장 문장입니다.", data[0]


def case_02_kosis_dimension_json_string_is_parsed_back_to_object():
    """store.py가 kosis_dimension을 JSON 문자열로 직렬화해 저장하는데, export 시
    프론트가 바로 쓸 수 있게 다시 객체(dict)로 파싱되는지 확인."""
    with tempfile.TemporaryDirectory() as tmp:
        db_path = Path(tmp) / "test_verifications.db"
        out_path = Path(tmp) / "export.json"

        store.insert_verification(
            {
                "result_id": "dim001",
                "article_title": "테스트 기사2",
                "claim_sentence": "지역별 인구 이동 관련 주장입니다.",
                "claim_type": "규모",
                "kosis_dimension": {"region": "전국", "period": "2024"},
            },
            path=db_path,
        )

        export_to_json(out_path, db_path=db_path)
        data = json.loads(out_path.read_text(encoding="utf-8"))
        assert data[0]["kosis_dimension"] == {"region": "전국", "period": "2024"}, data[0]["kosis_dimension"]


def case_03_empty_db_exports_empty_array_not_error():
    """아직 검증 레코드가 하나도 없는 DB(예: 배치 실행 전)도 에러 없이 빈 배열을 내보내야 한다."""
    with tempfile.TemporaryDirectory() as tmp:
        db_path = Path(tmp) / "empty.db"
        out_path = Path(tmp) / "export.json"

        count = export_to_json(out_path, db_path=db_path)
        assert count == 0, f"빈 DB인데 건수가 0이 아님: {count}"

        data = json.loads(out_path.read_text(encoding="utf-8"))
        assert data == [], data


CASES = [
    case_01_export_produces_valid_json_array,
    case_02_kosis_dimension_json_string_is_parsed_back_to_object,
    case_03_empty_db_exports_empty_array_not_error,
]


def main() -> None:
    results = []
    for case in CASES:
        try:
            case()
            results.append((case.__name__, "PASS", ""))
        except Exception as e:  # noqa: BLE001 - 테스트 러너라 실패 원인만 보고 계속 진행
            results.append((case.__name__, "FAIL", str(e)))

    print(f"총 {len(results)}건 실행")
    print("=" * 70)
    for name, status, detail in results:
        mark = "✅" if status == "PASS" else "❌"
        print(f"{mark} {status}  {name}" + (f" — {detail}" if detail else ""))

    passed = sum(1 for _, s, _ in results if s == "PASS")
    print(f"\n{passed}/{len(results)} PASS")


if __name__ == "__main__":
    main()
