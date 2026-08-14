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
        # 2026-08-13부터 prdSe가 "지원 가능한 주기 목록"으로 바뀌어서, startPrdDe/endPrdDe에
        # 등록된 값(표 등록 당시 실제로 검증했던 시점 하나)이 그 목록 중 하나의 형식과만
        # 맞으면 정상으로 본다 — 목록 전체와 다 맞아야 하는 게 아니라, "이 값이 어느 주기
        # 형식으로도 해석이 안 되는 완전히 깨진 값인가"만 잡아내는 느슨한 점검이다.
        prd_se_options = base.get("prdSe")
        if isinstance(prd_se_options, str):
            prd_se_options = [prd_se_options]
        prd_se_options = prd_se_options or [None]

        for field in ("startPrdDe", "endPrdDe"):
            value = base.get(field)
            errors = []
            for prd_se in prd_se_options:
                try:
                    _validate_period_format(table_id, prd_se, value, param_name=field)
                    errors = []
                    break
                except Exception as e:  # noqa: BLE001 - 점검 스크립트, 계속 진행
                    errors.append(str(e))
            if errors:
                problems.append(
                    f"{table_id} ({field}={value!r})가 지원 주기 {prd_se_options} 중 "
                    f"어느 것과도 안 맞음: {errors[-1]}"
                )

    if not problems:
        print("이상 없음 — 등록된 모든 표의 startPrdDe/endPrdDe가 prdSe 형식과 일치합니다.")
    else:
        print(f"형식 불일치 {len(problems)}건 발견:\n")
        for p in problems:
            print(f"  - {p}")


if __name__ == "__main__":
    main()
