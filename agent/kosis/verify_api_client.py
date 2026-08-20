"""
agent/kosis/verify_api_client.py — table_params.json에 등록된 표 전체를 실제 KOSIS API로 스모크 테스트

C가 조사해서 table_params.json에 추가한 표들이 실제로 유효한 응답을 돌려주는지 한 번에
확인하기 위한 스크립트. 각 표를 slots={}(=표에 정의된 기본값만으로) 호출해서, 그 표 자체의
설정(orgId/tblId/dimensions의 default_value/startPrdDe·endPrdDe)이 유효한지를 본다.
3·4단계가 실제로 어떤 slots를 채워 넣든, 최소한 "이 표의 기본 설정 자체는 유효하다"는 걸
먼저 확인해두기 위한 것 — __main__에 표 1~2개만 수동 테스트하던 것을 전수로 확장한 버전.

실패하면 KosisApiError(err 코드 포함)를 그대로 보여주므로, PENDING_TABLES.md에 이미
정리된 "KOSIS에 애초에 없는 표"(err30 등)와 "설정 자체가 잘못된 표"를 구분하는 데 쓴다.

사용법 (프로젝트 루트에서):
    python -m agent.kosis.verify_api_client
"""

from __future__ import annotations

import json
import time

from agent.kosis.api_client import KosisApiClient, KosisApiError, TABLE_PARAMS_PATH

DELAY_BETWEEN_CALLS = 0.5


def main() -> None:
    with open(TABLE_PARAMS_PATH, encoding="utf-8") as f:
        table_params = json.load(f)

    client = KosisApiClient()
    table_ids = list(table_params.keys())

    print(f"검증 대상 {len(table_ids)}개 표\n")

    ok: list[str] = []
    failed: list[tuple[str, str, str]] = []

    for i, table_id in enumerate(table_ids, 1):
        meta = table_params[table_id]
        name = meta.get("_table_name", "")
        try:
            resp = client(table_id, {})
            print(
                f"[{i}/{len(table_ids)}] [OK] {table_id} {name} "
                f"-> {resp.raw_value} {resp.unit} ({resp.period})"
            )
            ok.append(table_id)
        except (KosisApiError, KeyError) as e:
            print(f"[{i}/{len(table_ids)}] [FAIL] {table_id} {name} -> {type(e).__name__}: {e}")
            failed.append((table_id, name, str(e)))
        except Exception as e:  # noqa: BLE001 - 점검 스크립트, 예상 못한 에러도 계속 진행
            print(f"[{i}/{len(table_ids)}] [ERROR] {table_id} {name} -> {type(e).__name__}: {e}")
            failed.append((table_id, name, f"{type(e).__name__}: {e}"))
        time.sleep(DELAY_BETWEEN_CALLS)

    print(f"\n=== 최종 요약 ===")
    print(f"{len(table_ids)}개 표 중 성공 {len(ok)}건, 실패 {len(failed)}건")
    if failed:
        print("\n실패 목록:")
        for table_id, name, err in failed:
            print(f"  - {table_id} {name}: {err[:150]}")


if __name__ == "__main__":
    main()
