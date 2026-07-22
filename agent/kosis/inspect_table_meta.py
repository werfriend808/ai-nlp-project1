"""
agent/kosis/inspect_table_meta.py — 통계표 메타정보(분류/항목) 확인용 헬퍼 (개발 보조 도구)

역할: 파이프라인 코드가 아니라, C가 table_params.json을 채울 때 "이 표에 실제로
어떤 분류 축(objL1/objL2...)이 있고, 그 코드값이 뭔지"를 확인하기 위한 1회성 조사 스크립트.

KOSIS 메타정보 API(getMeta&type=TBL)를 호출해서 표의 분류/항목/단위/출처 정보를 그대로 출력합니다.
이 API는 코드 목록까지 상세히 주진 않을 수 있어서, 그럴 땐 실제 statisticsData 호출 결과에서
나온 C1/C2/ITM 값들을 눈으로 훑어보는 방식(이 파일 맨 아래 옵션 2)을 같이 쓰세요.

사용법 (프로젝트 루트에서):
    python -m agent.kosis.inspect_table_meta DT_1EA1019
"""

from __future__ import annotations

import json
import os
import sys

import requests

try:
    from dotenv import find_dotenv, load_dotenv

    load_dotenv(find_dotenv(usecwd=True))
except ImportError:
    pass


def _print_raw_on_failure(resp: requests.Response, label: str):
    """JSON 파싱이 안 될 때, 뭐가 잘못 왔는지 원본 텍스트를 그대로 보여줌 (진단용)."""
    print(f"[{label}] 응답을 JSON으로 못 읽었습니다. HTTP 상태코드: {resp.status_code}")
    print(f"[{label}] 원본 응답 내용(앞부분 500자):")
    print(resp.text[:500])


def inspect_meta(org_id: str, tbl_id: str, api_key: str) -> None:
    """옵션 1: 통계표설명(메타정보) API — 표 이름/분류/항목/단위/출처 등 요약 정보."""
    url = "https://kosis.kr/openapi/statisticsData.do"
    params = {
        "method": "getMeta",
        "type": "TBL",
        "apiKey": api_key,
        "orgId": org_id,
        "tblId": tbl_id,
        "format": "json",
    }
    print("=== [옵션 1] 통계표설명(메타정보) API ===")
    resp = requests.get(url, params=params, timeout=10)
    try:
        data = resp.json()
        print(json.dumps(data, ensure_ascii=False, indent=2))
    except json.JSONDecodeError:
        _print_raw_on_failure(resp, "옵션1")


def inspect_actual_categories(
    org_id: str,
    tbl_id: str,
    api_key: str,
    prd_se: str = "Y",
    start_prd_de: str = "2024",
    end_prd_de: str = "2024",
) -> None:
    """옵션 2: 실제 statisticsData 조회 결과에서 C1/C2/ITM 조합이 몇 종류나 나오는지 직접 확인.
    objL1, objL2를 둘 다 ALL로 보내서, 표가 분류축을 몇 개 요구하는지까지 같이 확인합니다."""
    url = "https://kosis.kr/openapi/Param/statisticsParameterData.do"
    params = {
        "method": "getList",
        "apiKey": api_key,
        "format": "json",
        "jsonVD": "Y",
        "orgId": org_id,
        "tblId": tbl_id,
        "objL1": "ALL",
        "objL2": "ALL",  # 표에 objL2가 없으면 KOSIS가 그냥 무시함 (에러 안 남)
        "itmId": "ALL",
        "prdSe": prd_se,
        "startPrdDe": start_prd_de,
        "endPrdDe": end_prd_de,
    }
    print(f"\n=== [옵션 2] 실제 조회 시도 (파라미터: {params}) ===")
    resp = requests.get(url, params=params, timeout=10)
    try:
        data = resp.json()
    except json.JSONDecodeError:
        _print_raw_on_failure(resp, "옵션2")
        return

    if isinstance(data, dict) and "err" in data:
        print(f"에러: [{data.get('err')}] {data.get('errMsg')}")
        print(
            "\n※ 이 에러가 계속 나면 prdSe(Y/M/Q)나 시점 형식이 안 맞을 수도 있습니다. "
            "예: python -m agent.kosis.inspect_table_meta DT_1DA7102S 101 M 202401 202412"
        )
        return

    if not isinstance(data, list) or len(data) == 0:
        print("응답은 정상(에러 없음)인데 데이터가 0건입니다. 시점/분류 조합을 바꿔서 다시 시도해보세요.")
        return

    c1_values = sorted({(r.get("C1"), r.get("C1_NM")) for r in data})
    c2_values = sorted({(r.get("C2"), r.get("C2_NM")) for r in data if r.get("C2") is not None})
    itm_values = sorted({(r.get("ITM_ID"), r.get("ITM_NM")) for r in data})

    print(f"\n총 {len(data)}행 응답, 예시 1행: {data[0]}")
    print(f"C1 값 목록 ({len(c1_values)}종): {c1_values[:20]}")
    print(f"C2 값 목록 ({len(c2_values)}종): {c2_values[:20]}")
    print(f"ITM 값 목록 ({len(itm_values)}종): {itm_values[:20]}")

    if not c2_values:
        print("\n※ C2 값이 안 나옵니다 — 이 표엔 2차 분류(C2) 축이 없거나, objL2가 이 표에서 안 쓰이는 파라미터라는 뜻일 수 있습니다.")


if __name__ == "__main__":
    api_key = os.environ.get("KOSIS_API_KEY")
    if not api_key:
        raise SystemExit("KOSIS_API_KEY가 없습니다. .env 또는 환경변수로 설정하세요.")

    tbl_id = sys.argv[1] if len(sys.argv) > 1 else "DT_1EA1019"
    org_id = sys.argv[2] if len(sys.argv) > 2 else "101"
    prd_se = sys.argv[3] if len(sys.argv) > 3 else "Y"
    start_prd_de = sys.argv[4] if len(sys.argv) > 4 else "2024"
    end_prd_de = sys.argv[5] if len(sys.argv) > 5 else "2024"

    try:
        inspect_meta(org_id, tbl_id, api_key)
    except Exception as e:  # 메타 API 자체가 없거나 형식이 다를 수 있어 실패해도 계속 진행
        print(f"(메타 API 조회 중 예외 발생, 옵션2로 계속 진행: {e})")

    inspect_actual_categories(org_id, tbl_id, api_key, prd_se, start_prd_de, end_prd_de)