"""
agent/kosis/verify_multi_period_support.py — 1단계: 카탈로그 64개 표가 실제로 Y/M/Q 중
어떤 주기를 지원하는지 KOSIS API로 직접 확인하는 1회성 조사 스크립트.

배경: table_params.json은 표 하나당 prdSe 값을 하나만 저장하는데, 실제로는 한 표가 여러
주기를 동시에 지원하는 경우가 흔하다(예: DT_1B8000G "월.분기.연간 인구동향"). 등록 당시
한 가지 주기만 테스트해보고 그 값으로 고정해서 생긴 문제라, 다중 주기 지원 작업(2~4단계:
스키마 변경/선택 로직/검증 로직 갱신)을 하기 전에 먼저 "각 표가 실제로 뭘 지원하는지"
정확한 데이터를 확보해야 한다.

방법: 표마다 Y/M/Q 세 가지로 최소 파라미터(objL1=ALL, objL2=ALL, itmId=ALL)로 짧은 기간
하나씩 실제 조회해보고, 에러 없이 데이터가 나오면 그 주기를 지원하는 것으로 판단한다.
inspect_table_meta.py의 옵션2와 같은 방식(KOSIS 메타 API는 주기 목록을 안 주기 때문에
"블랙박스 테스트"가 유일한 신뢰 가능한 방법 — 2026-08-13 확인, getMeta 응답에 주기 정보 없음).

사용법 (프로젝트 루트에서):
    python -m agent.kosis.verify_multi_period_support           # 전체 64개
    python -m agent.kosis.verify_multi_period_support --limit 5 # 앞 5개만 (동작 확인용)
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import requests

try:
    from dotenv import find_dotenv, load_dotenv

    load_dotenv(find_dotenv(usecwd=True))
except ImportError:
    pass

import os

TABLE_PARAMS_PATH = Path(__file__).parent / "table_params.json"
OUTPUT_PATH = Path(__file__).parent / "multi_period_support_result.json"
API_URL = "https://kosis.kr/openapi/Param/statisticsParameterData.do"

# 각 주기별로 짧게 테스트할 시점(최근 데이터가 확실히 있을 법한 값 — 너무 최근이면 아직
# 미공표일 수 있어 넉넉히 몇 달/분기 전으로 잡음).
_TEST_PERIOD = {"Y": ("2023", "2023"), "M": ("202310", "202310"), "Q": ("202303", "202303")}


def _try_period(base: dict, api_key: str, prd_se: str) -> tuple[bool, str]:
    """표의 등록된 dimensions/itmId_fixed를 최대한 재사용해서 쿼리를 만든다 — batch_runner.py의
    build_kosis_slots와 같은 원칙("ALL"이 안 먹는 표가 있어서 표별 default_value 우선 사용,
    2026-08-13 5개 표 예비 테스트에서 objL이 필수인 표가 "ALL"만으로는 거부되는 걸 확인)."""
    start, end = _TEST_PERIOD[prd_se]
    params = {
        "method": "getList",
        "apiKey": api_key,
        "format": "json",
        "jsonVD": "Y",
        "orgId": base.get("orgId"),
        "tblId": base.get("tblId"),
        "itmId": base.get("itmId_fixed", base.get("itmId", "ALL")),
        "prdSe": prd_se,
        "startPrdDe": start,
        "endPrdDe": end,
    }
    dimensions = base.get("dimensions", {})
    if dimensions:
        for dim in dimensions.values():
            kosis_param = dim.get("kosis_param")
            default_value = dim.get("default_value")
            if kosis_param and default_value is not None:
                params[kosis_param] = default_value
    else:
        # 등록된 축 정보가 없는 표는 ALL로 시도(폴백) — 원래도 이런 표는 대부분 ALL이 통함.
        params["objL1"] = "ALL"
        params["objL2"] = "ALL"
    try:
        resp = requests.get(API_URL, params=params, timeout=15)
        data = resp.json()
    except (requests.RequestException, json.JSONDecodeError) as e:
        return False, f"요청/파싱 실패: {e}"

    if isinstance(data, dict) and "err" in data:
        return False, f"[{data.get('err')}] {data.get('errMsg')}"
    if not isinstance(data, list) or len(data) == 0:
        return False, "데이터 0건"
    return True, f"{len(data)}행 확인"


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--limit", type=int, default=None, help="앞에서부터 N개 표만 점검(동작 확인용)")
    args = parser.parse_args()

    api_key = os.environ.get("KOSIS_API_KEY")
    if not api_key:
        raise SystemExit("KOSIS_API_KEY가 없습니다. .env 또는 환경변수로 설정하세요.")

    table_params = json.loads(TABLE_PARAMS_PATH.read_text(encoding="utf-8"))
    items = list(table_params.items())
    if args.limit:
        items = items[: args.limit]

    results: dict[str, dict] = {}
    for i, (table_id, base) in enumerate(items, 1):
        org_id = base.get("orgId")
        registered_prd_se = base.get("prdSe")
        table_name = base.get("_table_name", "")
        print(f"[{i}/{len(items)}] {table_id} ({table_name}) — 등록된 prdSe={registered_prd_se!r}")

        supported = []
        details = {}
        for prd_se in ("Y", "Q", "M"):
            ok, detail = _try_period(base, api_key, prd_se)
            details[prd_se] = detail
            status = "지원" if ok else "미지원"
            print(f"    {prd_se}: {status} ({detail})")
            if ok:
                supported.append(prd_se)
            time.sleep(0.2)  # KOSIS API 과호출 방지

        extra = [p for p in supported if p != registered_prd_se]
        results[table_id] = {
            "table_name": table_name,
            "registered_prd_se": registered_prd_se,
            "actually_supported": supported,
            "details": details,
            "has_undiscovered_periods": bool(extra),
        }
        if extra:
            print(f"    ⚠️ 카탈로그엔 {registered_prd_se!r}만 등록돼 있는데 실제로는 {supported}도 지원함")

    OUTPUT_PATH.write_text(json.dumps(results, ensure_ascii=False, indent=2), encoding="utf-8")

    undiscovered = {k: v for k, v in results.items() if v["has_undiscovered_periods"]}
    print(f"\n{'=' * 60}")
    print(f"총 {len(results)}개 표 점검 완료 — 결과: {OUTPUT_PATH}")
    print(f"카탈로그에 안 적혀있던 추가 주기를 지원하는 표: {len(undiscovered)}개")
    for tid, v in undiscovered.items():
        print(f"  - {tid} ({v['table_name']}): 등록={v['registered_prd_se']!r} -> 실제={v['actually_supported']}")


if __name__ == "__main__":
    main()
