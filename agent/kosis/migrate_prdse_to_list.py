"""
agent/kosis/migrate_prdse_to_list.py — 2단계: table_params.json의 prdSe를 문자열에서
"지원 가능한 주기 목록"으로 바꾸는 1회성 마이그레이션 스크립트.

verify_multi_period_support.py(1단계)가 만든 multi_period_support_result.json을 읽어서,
각 표의 prdSe를 실측 확인된 리스트로 갱신한다.

안전장치: 원래 등록돼 있던 prdSe 값은 실측 테스트가 실패했더라도(예: 테스트에 쓴 특정
시점에 데이터가 없어서) 항상 리스트에 포함시킨다 — 이미 "_verified"로 검증된 값을 블랙박스
테스트 실패만으로 지워버리면 오히려 회귀가 생긴다. 즉 최종 리스트 = {원래 등록값} ∪
{실측으로 새로 확인된 값}.

사용법 (프로젝트 루트에서):
    python -m agent.kosis.migrate_prdse_to_list            # 실제 반영
    python -m agent.kosis.migrate_prdse_to_list --dry-run  # 미리보기만, 파일 안 바꿈
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

TABLE_PARAMS_PATH = Path(__file__).parent / "table_params.json"
RESULT_PATH = Path(__file__).parent / "multi_period_support_result.json"

# 리스트에 담을 때 항상 이 순서로 정렬 — 표시/디버깅 일관성용(선택 로직 동작과는 무관).
# verify_multi_period_support.py는 Y/Q/M 세 가지만 테스트하므로, 카탈로그에 F(반기)/D(일단위)
# 처럼 이 목록 밖의 코드로 등록된 표가 있으면 여기 없다고 조용히 빠뜨리면 안 된다(2026-08-13
# 실측 발견: DT_1SSSA022R="F", DT_731Y001="D"가 _ordered()에서 사라져 빈 리스트가 되는
# 회귀를 dry-run으로 미리 잡음) — 목록 밖 코드는 순서만 뒤에 붙여서 반드시 보존한다.
_CANONICAL_ORDER = ["Y", "Q", "M"]


def _ordered(periods: set[str]) -> list[str]:
    known = [p for p in _CANONICAL_ORDER if p in periods]
    unknown = sorted(p for p in periods if p not in _CANONICAL_ORDER)
    return known + unknown


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dry-run", action="store_true", help="파일을 실제로 바꾸지 않고 변경 내역만 출력")
    args = parser.parse_args()

    table_params = json.loads(TABLE_PARAMS_PATH.read_text(encoding="utf-8"))
    verify_result = json.loads(RESULT_PATH.read_text(encoding="utf-8"))

    changed = 0
    for table_id, base in table_params.items():
        registered = base.get("prdSe")
        if isinstance(registered, list):
            continue  # 이미 마이그레이션된 표(재실행 대비)

        result = verify_result.get(table_id)
        if result is None:
            print(f"⚠️ {table_id}: verify 결과 없음, 등록값({registered!r}) 그대로 리스트화만 함")
            base["prdSe"] = [registered] if registered else []
            changed += 1
            continue

        confirmed = set(result.get("actually_supported", []))
        if registered:
            confirmed.add(registered)  # 원래 등록값은 실측 실패해도 항상 유지(안전장치)

        new_value = _ordered(confirmed)
        if new_value != ([registered] if registered else []):
            print(f"{table_id} ({result.get('table_name', '')}): {registered!r} -> {new_value}")
        base["prdSe"] = new_value
        changed += 1

    print(f"\n총 {changed}개 표 prdSe를 리스트로 변경")

    if args.dry_run:
        print("(--dry-run이라 파일에 저장 안 함)")
        return

    TABLE_PARAMS_PATH.write_text(
        json.dumps(table_params, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print(f"저장 완료: {TABLE_PARAMS_PATH}")


if __name__ == "__main__":
    main()
