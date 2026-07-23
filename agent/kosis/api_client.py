"""
agent/kosis/api_client.py — KOSIS Open API 호출 (5단계)

팀 계약(interfaces.py) 기준:
    입력: slots (dict) -> orgId/itmId/objL1/prdSe 등으로 매핑해서 호출
    출력: KosisApiResponse (raw_value 하나)

※ 중요: KosisApiResponse는 "값 하나"만 담습니다. 그래서 이 클라이언트는 KOSIS 조회 결과가
  정확히 1행일 때만 성공하고, 여러 행이 나오면 에러를 던집니다 (어떤 행을 골라야 할지
  api_client가 임의로 정하면 안 되므로). 합계·비율처럼 여러 값이 필요한 계산은,
  호출하는 쪽(6단계 calculator 또는 오케스트레이터)이 슬롯을 바꿔가며 이 클라이언트를
  "여러 번" 호출해서 list[KosisApiResponse]를 만든 뒤 calculator.py에 넘겨야 합니다.

사전 준비물:
  1. KOSIS Open API 인증키 — .env의 KOSIS_API_KEY (pip install python-dotenv 해두면 자동 로드)
  2. agent/kosis/table_params.json — 표별 orgId/tblId/prdSe + dimensions(분류축) 정의

이 모듈이 하지 않는 것:
  - 어떤 표를 쓸지 고르는 것 (3단계 통계표 매핑의 몫)
  - 수치 계산 (6단계 calculator.py의 몫)
"""

from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Optional

import requests

try:
    from dotenv import find_dotenv, load_dotenv

    load_dotenv(find_dotenv(usecwd=True))
except ImportError:
    pass

# 팀 계약 파일 위치가 프로젝트마다 다를 수 있어 순서대로 시도
try:
    from interfaces import KosisApiResponse, Slots
except ImportError:
    try:
        from agent.interfaces import KosisApiResponse, Slots
    except ImportError:  # 단독 실행/테스트용 폴백
        from dataclasses import dataclass

        Slots = dict  # type: ignore[assignment,misc]

        @dataclass
        class KosisApiResponse:  # type: ignore[no-redef]
            raw_value: float
            unit: str
            period: str
            org_id: str
            itm_id: str
            obj_l1: Optional[str] = None
            obj_l2: Optional[str] = None
            prd_se: Optional[str] = None


KOSIS_BASE_URL = "https://kosis.kr/openapi/Param/statisticsParameterData.do"
TABLE_PARAMS_PATH = Path(__file__).parent / "table_params.json"


class KosisApiError(RuntimeError):
    """KOSIS 에러 응답, 또는 결과가 0건/2건 이상이라 값 하나로 못 좁혀지는 경우."""


def _to_float(raw: Optional[str]) -> Optional[float]:
    if raw is None:
        return None
    text = str(raw).strip().replace(",", "")
    if text in ("", "-", "..", "X", "n/a"):
        return None
    try:
        return float(text)
    except ValueError:
        return None


def _load_table_params(path: Path = TABLE_PARAMS_PATH) -> dict:
    if not path.exists():
        raise FileNotFoundError(
            f"{path} 가 없습니다. C가 조사한 파라미터 표를 이 JSON 파일에 채워두세요."
        )
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


class KosisApiClient:
    """
    __call__(table_id, slots) -> KosisApiResponse  (값 하나)

    사용 예:
        client = KosisApiClient()
        resp = client("DT_1EA1019", {"period": "2024", "age": "20~24세"})
        # resp.raw_value == 41.0
    """

    def __init__(
        self,
        api_key: Optional[str] = None,
        table_params_path: Path = TABLE_PARAMS_PATH,
        timeout: int = 10,
    ):
        self.api_key = api_key or os.environ.get("KOSIS_API_KEY")
        if not self.api_key:
            raise RuntimeError(
                "KOSIS_API_KEY가 없습니다. .env(KOSIS_API_KEY=...)에 넣거나 "
                "KosisApiClient(api_key=...)로 직접 넘겨주세요."
            )
        self.timeout = timeout
        self._table_params = _load_table_params(table_params_path)

    def __call__(self, table_id: str, slots: Slots) -> KosisApiResponse:
        if table_id not in self._table_params:
            raise KeyError(
                f"'{table_id}'가 table_params.json에 없습니다. "
                f"C가 먼저 이 표의 파라미터를 조사해서 table_params.json에 추가해야 합니다."
            )
        base = self._table_params[table_id]

        params = {
            "method": "getList",
            "apiKey": self.api_key,
            "format": "json",
            "jsonVD": "Y",
            "orgId": base["orgId"],
            "tblId": base.get("tblId", table_id),
            "prdSe": base.get("prdSe", "Y"),
            "itmId": base.get("itmId_fixed", base.get("itmId", "ALL")),
        }

        # region/gender/age 등 "질문마다 달라지는" 축은 dimensions에 정의해두고,
        # slots(대화/기사에서 채워진 값)를 code_map으로 변환해서 반영. 값이 없으면 default_value.
        for dim_name, dim in base.get("dimensions", {}).items():
            kosis_param = dim["kosis_param"]
            user_value = slots.get(dim_name)
            code_map = dim.get("code_map", {})
            code = code_map.get(user_value, user_value) if user_value is not None else dim.get("default_value")
            if code is not None:
                params[kosis_param] = code

        period = slots.get("period")
        if period:
            params["startPrdDe"] = slots.get("start_period", period)
            params["endPrdDe"] = slots.get("end_period", period)
        else:
            params["startPrdDe"] = base.get("startPrdDe")
            params["endPrdDe"] = base.get("endPrdDe")

        response = requests.get(KOSIS_BASE_URL, params=params, timeout=self.timeout)
        response.raise_for_status()
        data = response.json()

        if isinstance(data, dict) and "err" in data:
            raise KosisApiError(f"[{data.get('err')}] {data.get('errMsg', '알 수 없는 오류')}")

        if not isinstance(data, list) or len(data) == 0:
            raise KosisApiError(
                f"'{table_id}' 조회 결과가 없습니다. 요청 파라미터: "
                f"{ {k: v for k, v in params.items() if k != 'apiKey'} }"
            )

        if len(data) > 1:
            raise KosisApiError(
                f"'{table_id}' 조회 결과가 {len(data)}건입니다. KosisApiResponse는 값 하나만 "
                f"담을 수 있어서 slots를 더 구체적으로 지정해야 합니다 (예: age를 'ALL' 대신 "
                f"특정 연령대로). 요청 파라미터: "
                f"{ {k: v for k, v in params.items() if k != 'apiKey'} }"
            )

        row = data[0]
        raw_value = _to_float(row.get("DT"))
        if raw_value is None:
            raise KosisApiError(f"'{table_id}' 값이 결측치입니다: {row}")

        # 엣지케이스 방어 1: PRD_DE(시점) 필드 자체가 없으면 빈 문자열로 조용히 넘기지 않고 에러.
        actual_period = row.get("PRD_DE")
        if not actual_period:
            raise KosisApiError(f"'{table_id}' 응답에 PRD_DE(시점) 필드가 없습니다: {row}")

        # 엣지케이스 방어 2: 요청한 시점과 실제 응답 시점이 다르면(KOSIS가 다른 시점 데이터로
        # 대체해서 줄 가능성 대비) 조용히 넘기지 않고 에러. 실제로 DT_1DA7102S에서는 KOSIS가
        # 데이터 없는 시점을 요청하면 스스로 에러를 내서 여기까지 안 오는 걸 확인했지만, 다른
        # 표/prdSe 조합에서도 그렇다는 보장이 없어 예방적으로 검증.
        requested_period = params.get("endPrdDe")
        if requested_period and str(actual_period) != str(requested_period):
            raise KosisApiError(
                f"'{table_id}' 요청 시점({requested_period})과 응답 시점({actual_period})이 "
                f"다릅니다. 요청 파라미터: "
                f"{ {k: v for k, v in params.items() if k != 'apiKey'} }"
            )

        return KosisApiResponse(
            raw_value=raw_value,
            unit=row.get("UNIT_NM", ""),
            period=actual_period,
            org_id=str(base["orgId"]),
            itm_id=row.get("ITM_ID", ""),
            obj_l1=row.get("C1"),
            obj_l2=row.get("C2"),
            prd_se=row.get("PRD_SE"),
        )


if __name__ == "__main__":
    #   python -m agent.kosis.api_client                        → 농가표, 20~24세 인구 조회
    #   python -m agent.kosis.api_client DT_1DA7102S             → 실업률표, 여자·청년(15~29세) 조회
    import sys

    table_id_arg = sys.argv[1] if len(sys.argv) > 1 else "DT_1EA1019"

    client = KosisApiClient()
    if table_id_arg == "DT_1DA7102S":
        slots = {"period": "2024", "gender": "여자", "age": "청년(15~29세)"}
    else:
        slots = {"period": "2024", "age": "20~24세"}

    result = client(table_id=table_id_arg, slots=slots)
    print(result)