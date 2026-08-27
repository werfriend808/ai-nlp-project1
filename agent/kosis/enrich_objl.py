"""agent/kosis/enrich_objl.py — VDB(Supabase) 표의 임베딩 텍스트에 분류축 이름(objL)을
지연 보강(lazy enrichment)한다.

배경: 2026-08-20 골든셋 41건 실측 — VDB 단독 Recall@30=26.8%, MRR=0.0675로 매우 낮았다.
실패 38건 중 23건(60%)이 "같은 기관의 다른 표"에게 졌다 — 국가데이터처 하나가 인구/물가
관련 표를 수백 개 내는데, 지금 임베딩 텍스트("기관명 (연월) 표명")만으로는 그 형제 표들이
서로 너무 비슷해 보여 구분이 안 된다. 표 하나가 어떤 축(지역별/품목별/연령별 등)으로
나뉘는지를 텍스트에 추가하면 이 형제 표 구분에 직접 도움이 될 것으로 본다.

28만7천 개 표 전부를 한 번에 보강하는 건(표마다 KOSIS API 호출 1번씩) 비현실적이라(레이트
리밋, 시간), 실제 claim 매칭 파이프라인이 VDB 후보로 뽑아온 표만 그때그때 보강하는
지연 방식으로 간다 — 자주 쓰이는 표부터 점진적으로 풍부해진다.

분류항목 개수가 표 하나에 수백 개(예: 소비자물가지수의 품목별 484종)까지 나올 수 있어서,
개별 항목 이름을 전부 텍스트에 넣으면 오히려 임베딩이 희석된다(2026-08-20 실측 논의) —
그래서 개별 항목이 아니라 "분류축 이름"(예: 시도별/품목별)만 붙인다. 이걸로 부족하면
대표 항목 샘플링을 추가하는 2차 개선을 고려한다.

사용법: GPU가 있는 곳(AWS/코랩, Qwen3-Embedding-4B 이미 로드된 상태)에서만 의미가 있다 —
embed_model을 인자로 받아 재사용한다(이 모듈이 직접 로드하지 않음, 이미 로드된 모델
재사용이 원칙).
"""

from __future__ import annotations

import os
import re
from typing import Optional

import requests

try:
    from dotenv import find_dotenv, load_dotenv

    load_dotenv(find_dotenv(usecwd=True))
except ImportError:
    pass

# [DEPRECATED] 2026-08-27: 이 상수를 쓰는 enrich_table()/enrich_candidates()는 현재
# 아무 데서도 호출되지 않는다(실제 사용되는 건 fetch_period_range/fetch_table_detail뿐).
# 가리키는 kosis_vdb_tables 테이블은 데이터 소실 후 재구축에서 없어졌고(지금은
# kosis_vdb_tables_qwen), objl_enriched 컬럼도 새 스키마엔 없다. 되살리려면
# 테이블명/컬럼명(tbl_id->table_id, text->embedding_text)을 함께 고쳐야 한다.
VDB_TABLE_NAME = "kosis_vdb_tables"  # 존재하지 않는 테이블 -- 위 주석 참고
_META_URL = "https://kosis.kr/openapi/Param/statisticsParameterData.do"

# 표마다 유효한 주기(prdSe)가 다르다(연간만 되는 표, 월별만 되는 표 등) — 실측(체크·
# multi-period 관련 기존 스크립트들)상 특정 prdSe를 먼저 안다는 보장이 없어서, 흔한
# 순서대로 시도한다. 첫 성공에서 멈춘다.
#
# 2026-08-21 실측(org_id=101 무작위 15개 표): 15개 중 13개 실패, 그중 10개가
# "[30] 데이터가 존재하지 않습니다"였다. 원인 두 가지를 직접 확인:
#   1) prdSe="A"(연간, "Y"와는 다른 별도 코드)만 되는 표가 있었다(DT_2WH23144,
#      DT_2OEEG055 — "A"로 재시도하니 바로 성공). 기존엔 A를 아예 시도하지 않았다.
#   2) 좁은 기간(2023~2024)만 보던 게 문제였다 — VDB 28.7만 개 중엔 2007년에
#      끊긴 것처럼 표 제목 자체에 오래된 기준월이 박힌 표도 많아서(예: "국가데이터처
#      (2007-05) ..."), 기간을 넓혀야 한다.
# 그래도 완전히 못 살리는 표(진짜로 폐기된 시계열)는 여전히 있다 — 이 경우 "데이터가
# 존재하지 않습니다"가 정직한 답이라 재시도로 해결 안 된다.
_PRD_SE_ATTEMPTS = (
    ("M", "200501", "202612"),
    ("A", "2005", "2026"),
    ("Y", "2005", "2026"),
    ("Q", "20051", "20264"),
)

_AXIS_NAME_RE = re.compile(r"^C(\d+)_OBJ_NM$")


class ObjlFetchError(RuntimeError):
    """KOSIS API로 분류축 이름을 못 가져옴 (표 구조가 다르거나 API 실패)."""


def fetch_axis_names(org_id: str, tbl_id: str, api_key: Optional[str] = None) -> list[str]:
    """표 하나의 분류축 이름들(예: ["시도별", "품목별"])을 KOSIS API로 조회한다.

    getMeta는 분류축 코드 목록을 안정적으로 안 줘서(inspect_table_meta.py의 실측 결론),
    실제 데이터 조회(objL1~8=ALL)를 해서 응답 행에 들어있는 C{n}_OBJ_NM 필드를 모은다 —
    데이터 값 자체는 안 쓰고 필드 이름만 쓴다."""
    key = api_key or os.environ.get("KOSIS_API_KEY")
    if not key:
        raise ObjlFetchError("KOSIS_API_KEY가 없습니다.")

    # 2026-08-20 실측: objL1~8을 표가 실제로 지원하는 개수보다 많이 보내면 KOSIS가
    # "[21] 잘못된 요청 변수" 에러로 통째로 거부하는 표가 있었다(inspect_table_meta.py의
    # "안 쓰는 objL은 그냥 무시된다"는 가정이 모든 표에 적용되진 않음) — 8개부터 1개까지
    # 줄여가며 재시도한다.
    #
    # 2026-08-21 실측(60건 대량 테스트, 건당 평균 10.2초로 예상보다 훨씬 느림): 원인은
    # prd_se(기간)가 안 맞아서 나는 "[30] 데이터가 존재하지 않습니다" 에러를 axis 개수만
    # 8→4→2→1로 바꿔가며 4번이나 무의미하게 재시도하고 있었기 때문 — 기간이 안 맞으면
    # axis 개수를 바꿔봐야 결과가 똑같다(30건 전수 검사로 "period 에러가 실은 axis
    # 문제를 가리는" 경우는 0건 확인). "[21]"류(axis 개수 문제)일 때만 axis를 줄여
    # 같은 기간으로 재시도하고, 그 외 에러(기간 문제 등)는 axis 재시도 없이 바로 다음
    # prd_se로 넘어간다 — 최악의 경우 시도 횟수가 크게 줄어든다.
    #
    # 2026-08-21 실측 추가: n_axes 후보가 (8,4,2,1)만 있어서 그 사이 값(예: 3)이
    # 필요한 표는 영원히 실패하고 있었다(DT_1LC0031: n_axes=3에서만 성공 확인) — 전체
    # 범위(8~1)로 넓힌다. 또한 err="21"이 실제로는 두 가지 다른 뜻으로 쓰인다: "잘못된
    # 요청 변수"(axis 개수 문제 — 줄여서 재시도할 가치 있음)와 "해당 통계표가
    # 존재하지 않습니다"(표 자체가 없음 — axis를 아무리 줄여도 소용없음, DT_118N_SAUP31
    # 실측 확인). errMsg까지 봐서 후자면 axis 재시도 없이 바로 다음 prd_se로 넘어간다.
    last_error: Optional[str] = None
    for prd_se, start, end in _PRD_SE_ATTEMPTS:
        for n_axes in (8, 7, 6, 5, 4, 3, 2, 1):
            params = {
                "method": "getList",
                "apiKey": key,
                "format": "json",
                "jsonVD": "Y",
                "orgId": org_id,
                "tblId": tbl_id,
                "prdSe": prd_se,
                "startPrdDe": start,
                "endPrdDe": end,
                **{f"objL{i}": "ALL" for i in range(1, n_axes + 1)},
                "itmId": "ALL",
            }
            try:
                resp = requests.get(_META_URL, params=params, timeout=15)
                data = resp.json()
            except (requests.RequestException, ValueError) as e:
                last_error = str(e)
                break  # 요청 자체가 실패 — axis 개수 문제가 아니므로 다음 prd_se로.

            if isinstance(data, dict) and "err" in data:
                err_code = str(data.get("err"))
                err_msg = data.get("errMsg", "")
                last_error = f"[{err_code}] {err_msg}"
                # err=21인데 "표가 존재하지 않습니다" 류면 axis 문제가 아니라 표 자체가
                # 없는 것 — axis 재시도 무의미.
                is_axis_count_issue = err_code == "21" and "존재하지 않습니다" not in err_msg
                if not is_axis_count_issue:
                    break  # axis 개수 문제가 아닌 에러 — 재시도해도 결과 동일, 다음 prd_se로.
                continue
            if not isinstance(data, list) or not data:
                last_error = "빈 응답"
                break

            axis_names: list[str] = []
            seen = set()
            for key_name in data[0].keys():
                m = _AXIS_NAME_RE.match(key_name)
                if m:
                    name = data[0].get(key_name)
                    if name and name not in seen:
                        seen.add(name)
                        axis_names.append(name)
            if axis_names:
                return axis_names
            last_error = "응답은 있으나 분류축 필드 없음"

    raise ObjlFetchError(f"{tbl_id}: 모든 시도 실패 (마지막 오류: {last_error})")


# PHASE 6/7(2026-08-21) — VDB(28.7만 개)에서만 찾은 표는 지금까지 build_kosis_slots()가
# table_params.json(64개 수동 카탈로그)에 없다는 이유로 무조건 None을 반환해서 실제 값
# 조회 자체가 불가능했다. fetch_axis_names()는 축 "이름"만 뽑았는데(임베딩 텍스트 보강
# 용도라 개별 항목까지 넣으면 희석됨), 값 조회를 하려면 각 축의 "코드"(예: 시도별 축의
# "서울"->"11")까지 필요하다 — 어차피 같은 API 응답(objL=ALL) 안에 이미 들어있는
# 정보라 API를 한 번 더 부를 필요는 없고, 응답의 첫 행(data[0])만 보던 것을 전체 행으로
# 넓혀서 같이 뽑는다. 캐시(kosis_table_detail_cache)에 한 번만 저장하면 재사용되므로,
# 응답이 좀 커져도(표 하나당 한 번뿐) 큰 비용은 아니다.
_AXIS_CODE_RE = re.compile(r"^C(\d+)$")
_META_TBL_URL = "https://kosis.kr/openapi/statisticsData.do"


# 2026-08-22: 3단계가 "같은 통계의 구판/신판"(예: DT_1DA7E06S 2013.01~2024.12 vs
# DT_1DA7E06S_NEW 2013.01~2026.07처럼 KOSIS가 표를 갈아치운 경우) 중 무엇을 골라야
# 하는지 판단할 근거가 없어서, 점수만 비슷하면 구판이 뽑히기도 하는 문제가 실측
# 확인됐다. fetch_table_detail()(getList 기반, 데이터 40,000셀 제한에 걸리는 큰 표는
# 실패함 — 자살률 표 사례)과 달리, 이 함수는 getMeta&type=PRD 전용 엔드포인트라
# 가볍고(응답에 실제 데이터 없이 수록기간만 옴) 그 제한에 걸리지 않는다.
_PRD_ROW_RE = re.compile(
    r'PRD_SE:"([^"]*)"\s*,\s*STRT_PRD_DE:"([^"]*)"\s*,\s*END_PRD_DE:"([^"]*)"'
)


def fetch_period_range(org_id: str, tbl_id: str, api_key: Optional[str] = None) -> Optional[dict]:
    """표 하나의 주기별 수록기간(시작~종료)을 가져온다.
    반환: {"월": ("2013.01", "2026.07"), "분기": (...), "년": (...)} 또는 실패 시 None.
    (월/분기/년 중 표가 지원하는 주기만 키로 들어있음)

    ⚠️ 이 엔드포인트 응답은 키가 따옴표로 안 감싸인 형태로 온다(예: {PRD_SE:"월",...}) —
    표준 JSON이 아니라서 resp.json()이 조용히 실패한다(실측 확인, 2026-08-22). 정규식으로
    직접 파싱한다."""
    key = api_key or os.environ.get("KOSIS_API_KEY")
    if not key:
        return None
    try:
        resp = requests.get(
            _META_TBL_URL,
            params={
                "method": "getMeta", "type": "PRD", "apiKey": key,
                "orgId": org_id, "tblId": tbl_id, "format": "json",
            },
            timeout=10,
        )
    except requests.RequestException:
        return None

    result: dict[str, tuple[str, str]] = {}
    for prd_se, start, end in _PRD_ROW_RE.findall(resp.text):
        if prd_se and start and end:
            result[prd_se] = (start, end)
    return result or None


def fetch_table_detail(org_id: str, tbl_id: str, api_key: Optional[str] = None) -> dict:
    """표 하나의 분류축 이름 + 값 코드맵을 한 번의 API 성공 응답에서 같이 뽑는다.
    반환: {"axis_names": [...], "code_maps": {axis_name: {label: code}}, "prd_se": "M"}
    fetch_axis_names()와 재시도 로직은 동일 — 실패하면 ObjlFetchError."""
    key = api_key or os.environ.get("KOSIS_API_KEY")
    if not key:
        raise ObjlFetchError("KOSIS_API_KEY가 없습니다.")

    last_error: Optional[str] = None
    for prd_se, start, end in _PRD_SE_ATTEMPTS:
        for n_axes in (8, 7, 6, 5, 4, 3, 2, 1):
            params = {
                "method": "getList", "apiKey": key, "format": "json", "jsonVD": "Y",
                "orgId": org_id, "tblId": tbl_id, "prdSe": prd_se,
                "startPrdDe": start, "endPrdDe": end,
                **{f"objL{i}": "ALL" for i in range(1, n_axes + 1)}, "itmId": "ALL",
            }
            try:
                resp = requests.get(_META_URL, params=params, timeout=15)
                data = resp.json()
            except (requests.RequestException, ValueError) as e:
                last_error = str(e)
                break

            if isinstance(data, dict) and "err" in data:
                err_code = str(data.get("err"))
                err_msg = data.get("errMsg", "")
                last_error = f"[{err_code}] {err_msg}"
                if not (err_code == "21" and "존재하지 않습니다" not in err_msg):
                    break
                continue
            if not isinstance(data, list) or not data:
                last_error = "빈 응답"
                break

            axis_names: list[str] = []
            seen_axis = set()
            for key_name in data[0].keys():
                m = _AXIS_NAME_RE.match(key_name)
                if m:
                    name = data[0].get(key_name)
                    if name and name not in seen_axis:
                        seen_axis.add(name)
                        axis_names.append(name)
            if not axis_names:
                last_error = "응답은 있으나 분류축 필드 없음"
                continue

            # 축 번호(n) -> 축 이름 매핑을 먼저 만들고(예: "1"->"시도별"), 전체 행을 훑어서
            # 각 축의 (라벨->코드) 페어를 모은다. 응답이 커도(최대 40,000셀) 표 하나당
            # 한 번만 하는 일이라 감당 가능하다.
            axis_num_to_name: dict[str, str] = {}
            for key_name in data[0].keys():
                m = _AXIS_NAME_RE.match(key_name)
                if m:
                    axis_num_to_name[m.group(1)] = data[0][key_name]

            code_maps: dict[str, dict[str, str]] = {name: {} for name in axis_names}
            for row in data:
                for key_name, value in row.items():
                    m = _AXIS_CODE_RE.match(key_name)
                    if not m:
                        continue
                    n = m.group(1)
                    axis_name = axis_num_to_name.get(n)
                    label = row.get(f"C{n}_NM")
                    if axis_name and label and value:
                        code_maps[axis_name][label] = value

            # axis_num_to_name(2026-08-21 추가, PHASE 7): "행정구역별" 같은 축 이름만으론
            # KOSIS API 파라미터(objL1/objL2/...)를 못 만든다 — 축 번호(n, "C{n}" 패턴)가
            # 곧 objL{n}이므로 이름->번호 역매핑에 쓸 수 있게 같이 반환한다(agent/kosis/
            # dynamic_slot_mapping.py가 resolve한 axis_name을 실제 API 호출로 잇는 다리).
            return {
                "axis_names": axis_names,
                "code_maps": code_maps,
                "prd_se": prd_se,
                "axis_num_to_name": axis_num_to_name,
            }

    raise ObjlFetchError(f"{tbl_id}: 모든 시도 실패 (마지막 오류: {last_error})")


def build_enriched_text(base_text: str, axis_names: list[str]) -> str:
    """기존 텍스트("기관명 (연월) 표명")에 분류축 이름을 붙인다.

    개별 분류항목(수백 개일 수 있음)이 아니라 축 이름만 붙인다 — 희석 방지(모듈 docstring
    참고). 이미 축 이름이 붙어있으면(재실행 등) 중복으로 안 붙인다."""
    if not axis_names:
        return base_text
    suffix = " — " + ", ".join(axis_names)
    if suffix in base_text:
        return base_text
    return base_text + suffix


def enrich_table(
    tbl_id: str,
    org_id: str,
    base_text: str,
    embed_model,
    *,
    api_key: Optional[str] = None,
    truncate_dim: int = 1024,
) -> Optional[tuple[str, list[float]]]:
    """표 하나를 보강한다. 성공하면 (새 텍스트, 새 임베딩) 반환, 실패하면 None
    (호출부가 조용히 건너뛰도록 — 보강 실패가 배치 전체를 막으면 안 됨)."""
    try:
        axis_names = fetch_axis_names(org_id, tbl_id, api_key)
    except ObjlFetchError:
        return None

    new_text = build_enriched_text(base_text, axis_names)
    if new_text == base_text:
        return None

    vec = embed_model.encode([new_text], convert_to_numpy=True, normalize_embeddings=True)[0]
    return new_text, vec.tolist()


def enrich_candidates(
    tbl_ids: list[str],
    embed_model,
    *,
    conn,
    api_key: Optional[str] = None,
    delay_sec: float = 0.3,
) -> int:
    """claim 매칭에서 VDB 후보로 나온 tbl_id들 중, 아직 objl_enriched가 안 된 것만 골라
    보강하고 Supabase에 반영한다. 반환값: 실제로 보강된 표 개수.

    delay_sec: KOSIS API 레이트리밋 대비 호출 간 지연(patch_english_table_names.py와
    동일한 관례)."""
    import time

    if not tbl_ids:
        return 0

    enriched_count = 0
    with conn.cursor() as cur:
        cur.execute(
            f"""
            select tbl_id, org_id, text from {VDB_TABLE_NAME}
            where tbl_id = any(%s) and (objl_enriched is null or objl_enriched = false);
            """,
            (tbl_ids,),
        )
        pending = cur.fetchall()

    for tbl_id, org_id, base_text in pending:
        result = enrich_table(tbl_id, org_id, base_text, embed_model, api_key=api_key)
        with conn.cursor() as cur:
            if result is None:
                # 이번엔 실패했어도 다음 배치에서 매번 재시도하지 않도록 시도했다는 표시만 남김
                cur.execute(
                    f"update {VDB_TABLE_NAME} set objl_enriched = true where tbl_id = %s;",
                    (tbl_id,),
                )
            else:
                new_text, new_vec = result
                cur.execute(
                    f"""
                    update {VDB_TABLE_NAME}
                    set text = %s, embedding = %s::vector, objl_enriched = true
                    where tbl_id = %s;
                    """,
                    (new_text, new_vec, tbl_id),
                )
                enriched_count += 1
        conn.commit()
        time.sleep(delay_sec)

    return enriched_count
