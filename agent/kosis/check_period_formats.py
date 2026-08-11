"""
agent/kosis/check_period_formats.py — table_params.json에 등록된 표들의 startPrdDe/endPrdDe가
표의 prdSe(주기)에 맞는 형식인지 정적으로(네트워크 호출 없이) 점검.

verify_api_client.py는 실제 KOSIS API를 호출해서 값 자체가 유효한지 확인하지만, 이 스크립트는
그보다 앞단에서 "표 등록 시점에 넣은 startPrdDe/endPrdDe 문자열이 애초에 그 표의 prdSe가
요구하는 자릿수/형식과 맞는가"만 즉시 확인한다 — 표가 늘어날수록 사람이 매번 눈으로
확인해야 하는 부담을 대신하기 위한 것.

사용법 (프로젝트 루트에서):
    python -m agent.kosis.check_period_formats
"""

from __future__ import annotations

import json

from agent.kosis.api_client import TABLE_PARAMS_PATH, _validate_period_format


def main() -> None:
    with open(TABLE_PARAMS_PATH, encoding="utf-8") as f:
        table_params = json.load(f)

    print(f"점검 대상 {len(table_params)}개 표\n")

    problems: list[str] = []
    for table_id, base in table_params.items():
        prd_se = base.get("prdSe")
        for field in ("startPrdDe", "endPrdDe"):
            value = base.get(field)
            try:
                _validate_period_format(table_id, prd_se, value, param_name=field)
            except Exception as e:  # noqa: BLE001 - 점검 스크립트, 계속 진행
                problems.append(f"{table_id} ({field}={value!r}, prdSe={prd_se!r}): {e}")

    if not problems:
        print("이상 없음 — 등록된 모든 표의 startPrdDe/endPrdDe가 prdSe 형식과 일치합니다.")
    else:
        print(f"형식 불일치 {len(problems)}건 발견:\n")
        for p in problems:
            print(f"  - {p}")


if __name__ == "__main__":
    main()
