"""agent/kosis/reembed_worker.py — KOSIS 전체 재임베딩(TABLE/ITEM/AXIS/AXIS VALUE) 워커.

담당 partition(SERVER_A=앞쪽 절반 / SERVER_B=뒤쪽 절반, tables.jsonl 실제 줄 수 기준)의
표를 순서대로 처리한다: KOSIS API 메타데이터 enrichment -> Qwen3-Embedding-4B(2560d)
임베딩 -> PostgreSQL(kosis_db, 로컬 pgvector) upsert -> checkpoint 갱신.

체크포인트(kosis_reembed_checkpoint_qwen)로 resume 가능 — 이미 status='success'인
table_id는 다시 처리하지 않는다. tmux 등으로 Claude 세션과 분리해 장시간 실행하도록
설계됨(자세한 사용법은 이 모듈의 실행부 참고).

사용법 (프로젝트 루트, venv 활성화 후):
    python -m agent.kosis.reembed_worker SERVER_A [--limit N] [--concurrency 10]
"""
from __future__ import annotations

import argparse
import json
import os
import random
import re
import sys
import threading
import time
from concurrent.futures import ThreadPoolExecutor, as_completed

import psycopg2
import psycopg2.extras
import requests
import torch

TABLES_PATH = "agent/kosis/crawl_output/tables.jsonl"
ORG_WHITELIST_PATH = "agent/preprocessing/kosis_org_whitelist.json"
META_URL = "https://kosis.kr/openapi/Param/statisticsParameterData.do"

# KOSIS API는 API 키당 "분당 200건" 하드 리밋을 건다 — 초과 시 HTTP 429가 아니라
# 200 OK 안에 {err:"40", errMsg:"1분간 호출가능건수(200건)을 초과하였습니다"}로 온다
# (2026-08-24 실측). 이걸 감지 못 하면 rate-limit에 걸린 정상 표가 "데이터 없음/기타
# 오류"로 영구 오분류된다 — 아래에서 별도 처리한다.
#
# 2026-08-24 추가 실측: 슬라이딩 윈도우(카운트 기반) 리미터는 스레드 20개가 한꺼번에
# 버스트로 몰려서 쿼터를 순식간에 소진한 뒤 다같이 멈췄다 깨어나는 thundering herd를
# 유발해 오히려 처리량이 떨어졌다(0.40건/초까지 하락) — 초 단위 최소 간격을 강제하는
# 단일 직렬 게이트(min-interval)로 교체한다. 모든 스레드가 이 락 하나를 공유해서
# 호출이 항상 균등한 간격으로만 나가므로 버스트 자체가 발생하지 않는다.
KOSIS_RATE_LIMIT_ERR_CODE = "40"
KOSIS_MIN_CALL_INTERVAL_SEC = 0.32  # ~3.1건/초 = ~187건/분 (200/분 한도보다 여유, 0.35초보다 살짝 타이트)


class MinIntervalRateLimiter:
    """호출 사이 최소 간격을 강제하는 직렬 게이트 (여러 스레드 공유)."""

    def __init__(self, min_interval: float):
        self._min_interval = min_interval
        self._lock = threading.Lock()
        self._last_call = 0.0

    def acquire(self) -> None:
        with self._lock:
            now = time.monotonic()
            wait = self._min_interval - (now - self._last_call)
            if wait > 0:
                time.sleep(wait)
                now = time.monotonic()
            self._last_call = now


# 2026-08-25: KOSIS의 분당 200건 한도는 API 키 단위다 — 키를 여러 개 쓰면 각 키가
# 독립적으로 자기 한도를 갖는다. 키마다 별도의 리미터를 두면 키 개수만큼 처리량이
# 그대로 늘어난다(팀원 API 키 추가 시 --api-keys로 바로 확장 가능하도록 미리 대비).
_rate_limiters: dict[str, MinIntervalRateLimiter] = {}
_rate_limiters_lock = threading.Lock()


def _get_rate_limiter(api_key: str) -> MinIntervalRateLimiter:
    limiter = _rate_limiters.get(api_key)
    if limiter is None:
        with _rate_limiters_lock:
            limiter = _rate_limiters.get(api_key)
            if limiter is None:
                limiter = MinIntervalRateLimiter(KOSIS_MIN_CALL_INTERVAL_SEC)
                _rate_limiters[api_key] = limiter
    return limiter


def _rate_limited_get(url: str, params: dict, timeout: int, api_key: str) -> requests.Response:
    _get_rate_limiter(api_key).acquire()
    return requests.get(url, params=params, timeout=timeout)

EMBED_MODEL_NAME = "Qwen/Qwen3-Embedding-4B"
EMBED_DIM = 2560

AXIS_NAME_RE = re.compile(r"^C(\d+)_OBJ_NM$")
AXIS_CODE_RE = re.compile(r"^C(\d+)$")

PRD_SE_ATTEMPTS = (
    # 2026-08-25: 시작점이 "2005"였던 게 period_start의 진짜 원인이었다(실측 27건 샘플 중
    # 5건, 2005년 이전부터 실제 데이터가 있는 표는 getList 응답 자체가 2005년 이전 행을
    # 안 돌려줬음 — getMeta&type=PRD는 이 윈도우와 무관해서 몰랐던 문제). 실측 확인
    # (DT_2AT037 등 5건 + item/axis 수가 많은 밀집 표 5건, 총 10건): 시작점을 "1960"대로
    # 넓혀도 40,000셀 제한에 새로 걸리는 표가 없었고(n_axes_used/n_calls 동일), 5건 전부
    # period_start가 기존 getMeta(PRD) 값과 정확히 일치하도록 회복됐다. 주기(M/A/Y/Q)와
    # startPrdDe/endPrdDe의 의미(월/분기/년 형식)는 그대로 유지 — 시작 연도만 확장.
    ("M", "190001", "202612"),
    # 2026-08-25 실측: KOSIS 공식 prdSe 코드는 D/M/Q/H/Y/F/IR 7종인데, 기존엔 M/A/Y/Q
    # 4종만 시도해서 F(2/3/4/5/10년 등 다년 주기 전부 이 코드 하나)가 필요한 표를 전부
    # "[30] 데이터가 존재하지 않습니다"(error_no_data)로 잘못 분류하고 있었다. 실제
    # error_no_data 무작위 40건 표본 검증: F 추가만으로 33/40(82.5%)이 진짜 데이터
    # 있는 표로 복구됨(축 개수까지 맞춰 재시도 필요했던 8건 포함) — D/H/IR은 이 표본에서
    # 추가 복구 0건이었지만 커버리지를 위해 낮은 우선순위로 유지한다. M 다음으로 흔한
    # F를 두 번째로 배치.
    #
    # 2026-08-25 추가 실측: 시작 연도 "1960"도 여전히 부족했다 — "광복이전통계
    # (1908~1943)" 카테고리(전체 2,314개 표, org_id 999 계열 3,411개)가 통째로 범위
    # 밖이라 전부 놓치고 있었다(CS069001942 "삼림수입" 등, getMeta&type=PRD로 등록된
    # 실제 기간이 "1933~1943"으로 확인됨). catalog 전체에서 가장 오래된 연도가 1908이라
    # "1900"으로 넉넉하게 확장 — 2005→1960 확장 때와 동일하게, 실제 데이터가 없는
    # 표는 창을 넓혀도 반환 행 수/응답 시간에 영향이 없었다(밀집 표 스트레스 테스트로
    # 이미 확인됨).
    ("F", "1900", "2026"),
    ("A", "1900", "2026"),
    ("Y", "1900", "2026"),
    ("Q", "19001", "20264"),
    ("H", "1900", "2026"),
    ("D", "19000101", "20261231"),
    ("IR", "1900", "2026"),
)

CHUNK_SIZE = 200


def _load_env():
    env_path = os.path.join(os.path.dirname(__file__), "..", "..", ".env")
    if os.path.exists(env_path):
        with open(env_path) as f:
            for line in f:
                line = line.strip()
                if line and not line.startswith("#") and "=" in line:
                    k, v = line.split("=", 1)
                    os.environ.setdefault(k.strip(), v.strip())


def load_org_whitelist() -> dict:
    with open(ORG_WHITELIST_PATH, encoding="utf-8") as f:
        return json.load(f)


# ------------------------------------------------------------------
# KOSIS API enrichment (표당 getList 1회 — period_start/end도 이 응답의 PRD_DE에서
# 뽑아 쓰므로 getMeta&type=PRD를 별도로 부르지 않는다, fetch_table_enrichment 참고)
# ------------------------------------------------------------------

RATE_LIMIT_MAX_RETRIES = 5
RATE_LIMIT_BACKOFF_SEC = 2.0


def _backoff_sleep(attempt: int) -> None:
    """[40] 재시도 대기 — 스레드마다 흩어지도록 jitter를 더한다(동시 재시도 방지)."""
    time.sleep(RATE_LIMIT_BACKOFF_SEC * attempt + random.uniform(0.5, 2.0))


CELL_LIMIT_ERR_CODE = "31"
# 2026-08-25 실측: depth=6(최대 64조각)은 1900~2026(1,524개월) 범위를 조각당 평균
# 24개월까지만 쪼개서, 그 정도로도 여전히 40,000셀을 넘는 표(DT_1J22001 등, 단일 1개월
# 요청은 정상 성공 확인됨)를 못 살렸다. 단일 월 단위까지 확실히 쪼갤 수 있도록
# 2^12=4096조각(1,524개월보다 넉넉히 많음)까지 허용 — 이진분할이라 실제로는 필요한
# 만큼만 쪼개지고 성공하는 즉시 멈추므로, 상한을 넉넉히 잡아도 정상 표에는 비용이 없다.
CELL_SPLIT_MAX_DEPTH = 12


def _split_period_range(prd_se: str, start: str, end: str):
    """기간 범위를 반으로 쪼갠다. 더 못 쪼개면 None. 주기별 시점 형식(YYYY/YYYYMM)은
    그대로 유지 — 형식 자체를 바꾸지 않고 같은 형식 안에서만 절반으로 나눈다."""
    if prd_se in ("F", "A", "Y", "H", "IR"):
        s, e = int(start), int(end)
        if s >= e:
            return None
        mid = (s + e) // 2
        return (str(s), str(mid)), (str(mid + 1), str(e))
    if prd_se == "M":
        def to_months(yyyymm: str) -> int:
            y, m = int(yyyymm[:4]), int(yyyymm[4:6])
            return y * 12 + (m - 1)

        def from_months(total: int) -> str:
            y, m = divmod(total, 12)
            return f"{y:04d}{m + 1:02d}"

        s, e = to_months(start), to_months(end)
        if s >= e:
            return None
        mid = (s + e) // 2
        return (from_months(s), from_months(mid)), (from_months(mid + 1), from_months(e))
    # Q/D는 실측상 성공 사례가 없어 분할 미지원 — 기존처럼 즉시 포기
    return None


def _fetch_rows_with_cell_split(org_id: str, tbl_id: str, api_key: str, prd_se: str,
                                 start: str, end: str, n_axes: int, depth: int = 0) -> list | None:
    """2026-08-25: [31](40,000셀 초과)이 뜨면 그 즉시 이 표를 포기하고 있었다 — 실제로는
    n_axes/prd_se가 정확히 맞았는데 응답이 너무 커서 KOSIS가 거부한 것뿐이라, 재처리
    still_error 표본(4,600건) 중 96%(3,069건)가 이 이유였다. 기간을 반으로 쪼개 여러 번
    나눠 받은 뒤 합치면 같은 축/항목 구조를 그대로 유지하면서 셀 수만 줄어든다."""
    params = {
        "method": "getList", "apiKey": api_key, "format": "json", "jsonVD": "Y",
        "orgId": org_id, "tblId": tbl_id, "prdSe": prd_se,
        "startPrdDe": start, "endPrdDe": end,
        **{f"objL{i}": "ALL" for i in range(1, n_axes + 1)}, "itmId": "ALL",
    }
    rate_limit_retries = 0
    while True:
        try:
            resp = _rate_limited_get(META_URL, params=params, timeout=15, api_key=api_key)
            data = resp.json()
        except (requests.RequestException, ValueError):
            return None
        if isinstance(data, dict) and str(data.get("err")) == KOSIS_RATE_LIMIT_ERR_CODE:
            rate_limit_retries += 1
            if rate_limit_retries > RATE_LIMIT_MAX_RETRIES:
                return None
            _backoff_sleep(rate_limit_retries)
            continue
        break

    if isinstance(data, dict) and "err" in data:
        err_code = str(data.get("err"))
        if err_code == CELL_LIMIT_ERR_CODE and depth < CELL_SPLIT_MAX_DEPTH:
            split = _split_period_range(prd_se, start, end)
            if split:
                (s1, e1), (s2, e2) = split
                left = _fetch_rows_with_cell_split(org_id, tbl_id, api_key, prd_se, s1, e1, n_axes, depth + 1)
                right = _fetch_rows_with_cell_split(org_id, tbl_id, api_key, prd_se, s2, e2, n_axes, depth + 1)
                if left is None and right is None:
                    return None
                return (left or []) + (right or [])
        return None
    if not isinstance(data, list) or not data:
        return None
    return data


def fetch_table_enrichment(org_id: str, tbl_id: str, api_key: str) -> dict:
    """표 하나의 axis/item/unit을 getList(objL=ALL, itmId=ALL) 1회 호출로 뽑는다.
    반환 status: success / error_no_data / error_other."""
    last_error = None
    # 2026-08-25: [30](데이터 없음)이 뜨면 그 즉시 함수를 반환해서 다른 prd_se(특히
    # F=격년 등 다년 주기)를 아예 시도조차 안 하고 있었다 — no_data 무작위 40건 실측
    # 검증에서 33건(82.5%)이 실은 F 등 다른 주기로 존재하는 데이터였다. 이제 [30]은
    # "이 prd_se는 포기하고 다음 prd_se로"만 의미하고, 모든 prd_se를 다 시도한 뒤에만
    # 최종적으로 error_no_data/error_other를 가른다.
    saw_no_data = False
    for prd_se, start, end in PRD_SE_ATTEMPTS:
        # 2026-08-25 실측(150건 표본): 성공 표의 100%가 prd_se="M"에서, 94%가
        # n_axes<=2에서 성공했다(내림차순 8→1이라 매번 8~3을 헛수고로 거쳐야 했음 —
        # 표당 평균 7.2회 호출). 오름차순(1→8)으로 뒤집고 err="20"(objL 필수값 누락,
        # 축을 실제보다 적게 요청했을 때 뜨는 코드 — 기존엔 "21"만 재시도 대상이라
        # 이 경우 조용히 오답 처리될 뻔했다, 70건 대조 검증으로 확인)까지 재시도
        # 대상에 포함시키면 표당 평균 호출이 5.93→1.67회로 준다(3.55배). 70건 old/new
        # 대조 검증에서 axis_names/item_count/unit 불일치 0건 — 추출 결과는 동일.
        for n_axes in (1, 2, 3, 4, 5, 6, 7, 8):
            params = {
                "method": "getList", "apiKey": api_key, "format": "json", "jsonVD": "Y",
                "orgId": org_id, "tblId": tbl_id, "prdSe": prd_se,
                "startPrdDe": start, "endPrdDe": end,
                **{f"objL{i}": "ALL" for i in range(1, n_axes + 1)}, "itmId": "ALL",
            }
            rate_limit_retries = 0
            while True:
                try:
                    resp = _rate_limited_get(META_URL, params=params, timeout=15, api_key=api_key)
                    data = resp.json()
                except (requests.RequestException, ValueError) as e:
                    last_error = f"request_error:{e}"
                    data = None
                    break
                if isinstance(data, dict) and str(data.get("err")) == KOSIS_RATE_LIMIT_ERR_CODE:
                    rate_limit_retries += 1
                    if rate_limit_retries > RATE_LIMIT_MAX_RETRIES:
                        last_error = f"[40] rate limited after {RATE_LIMIT_MAX_RETRIES} retries"
                        return {"status": "error_other", "error": last_error}
                    _backoff_sleep(rate_limit_retries)
                    continue
                break
            if data is None:
                break
            if isinstance(data, dict) and "err" in data:
                err_code = str(data.get("err"))
                err_msg = data.get("errMsg", "")
                last_error = f"[{err_code}] {err_msg}"
                is_axis_issue = (
                    (err_code == "21" and "존재하지 않습니다" not in err_msg)
                    or (err_code == "20" and "objL" in err_msg)
                )
                is_no_data = err_code == "30"
                if is_no_data:
                    saw_no_data = True
                    break  # 이 prd_se만 포기 — 바깥 루프가 다음 prd_se로 계속 진행
                if err_code == CELL_LIMIT_ERR_CODE:
                    split_data = _fetch_rows_with_cell_split(org_id, tbl_id, api_key, prd_se, start, end, n_axes)
                    if not split_data:
                        return {"status": "error_other", "error": last_error}
                    data = split_data
                elif not is_axis_issue:
                    return {"status": "error_other", "error": last_error}
                else:
                    continue
            if not isinstance(data, list) or not data:
                last_error = "empty_response"
                return {"status": "error_other", "error": last_error}

            axis_names, seen_axis, axis_num_to_name = [], set(), {}
            for k in data[0].keys():
                m = AXIS_NAME_RE.match(k)
                if m:
                    name = data[0].get(k)
                    axis_num_to_name[m.group(1)] = name
                    if name and name not in seen_axis:
                        seen_axis.add(name)
                        axis_names.append(name)
            if not axis_names:
                last_error = "no_axis_fields"
                continue

            code_maps = {n: {} for n in axis_names}
            item_pairs = {}
            unit_name = None
            prd_de_values = []
            for row in data:
                iid, inm = row.get("ITM_ID"), row.get("ITM_NM")
                if iid is not None:
                    item_pairs[iid] = inm
                if unit_name is None and row.get("UNIT_NM"):
                    unit_name = row.get("UNIT_NM")
                prd_de = row.get("PRD_DE")
                if prd_de:
                    prd_de_values.append(prd_de)
                for k, v in row.items():
                    m = AXIS_CODE_RE.match(k)
                    if not m:
                        continue
                    axis_name = axis_num_to_name.get(m.group(1))
                    label = row.get(f"C{m.group(1)}_NM")
                    if axis_name and label and v is not None:
                        code_maps[axis_name][label] = v

            # 2026-08-25: 기간(period_start/end)을 별도 API 호출(getMeta&type=PRD)로
            # 다시 물어보지 않는다 — 이미 방금 받은 이 응답의 PRD_DE(행마다 있는 실제
            # 데이터 시점)에서 min/max를 뽑으면 표당 API 호출이 1회로 줄어든다(전에는
            # 성공 표마다 2회 필요해서 분당 200건 한도 안에서 처리량이 반토막났었다).
            period_start = min(prd_de_values) if prd_de_values else None
            period_end = max(prd_de_values) if prd_de_values else None

            return {
                "status": "success",
                "axis_names": axis_names,
                "axis_num_to_name": axis_num_to_name,
                "code_maps": code_maps,
                "item_pairs": item_pairs,
                "unit": unit_name,
                "prd_se": prd_se,
                "period_start": period_start,
                "period_end": period_end,
            }
    if saw_no_data:
        return {"status": "error_no_data", "error": last_error or "unknown"}
    return {"status": "error_other", "error": last_error or "unknown"}


def process_one(line_no: int, org_id: str, tbl_id: str, stat_id: str, tbl_nm: str,
                 send_de: str, api_key: str) -> dict:
    enrichment = fetch_table_enrichment(org_id, tbl_id, api_key)
    return {
        "line_no": line_no, "org_id": org_id, "tbl_id": tbl_id, "stat_id": stat_id,
        "tbl_nm": tbl_nm, "send_de": send_de, "enrichment": enrichment,
    }


# ------------------------------------------------------------------
# Embedding text 조립 (section 17/18/19)
# ------------------------------------------------------------------

def build_table_text(institution, table_name, topic=None, classification=None,
                      survey_name=None, description=None) -> str:
    lines = [f"기관명: {institution}"] if institution else []
    lines.append(f"통계표명: {table_name}")
    if topic:
        lines.append(f"통계주제: {topic}")
    if classification:
        lines.append(f"통계분류: {classification}")
    if survey_name:
        lines.append(f"조사명: {survey_name}")
    if description:
        lines.append(f"통계설명: {description}")
    return "\n\n".join(lines)


def build_item_text(table_name, axis_name, item_name, parent_item_name=None, item_path=None) -> str:
    lines = [f"통계표: {table_name}"]
    if axis_name:
        lines.append(f"분류축: {axis_name}")
    lines.append(f"항목: {item_name}")
    if parent_item_name:
        lines.append(f"상위항목: {parent_item_name}")
    if item_path:
        lines.append(f"항목경로: {item_path}")
    return "\n\n".join(lines)


def build_axis_text(table_name, axis_name, axis_description=None) -> str:
    lines = [f"통계표: {table_name}", f"분류축: {axis_name}"]
    if axis_description:
        lines.append(f"축 설명: {axis_description}")
    return "\n\n".join(lines)


# ------------------------------------------------------------------
# DB
# ------------------------------------------------------------------

def get_connection():
    return psycopg2.connect(os.environ["SUPABASE_DB_URL"])


def load_pending(conn, role: str, limit: int | None):
    with conn.cursor() as cur:
        q = """
            select table_id, line_no from kosis_reembed_checkpoint_qwen
            where server_role = %s and status != 'success'
            order by line_no
        """
        if limit:
            q += " limit %s"
            cur.execute(q, (role, limit))
        else:
            cur.execute(q, (role,))
        return cur.fetchall()


def mark_checkpoint(conn, table_id: str, status: str, error: str | None = None):
    with conn.cursor() as cur:
        cur.execute(
            """
            update kosis_reembed_checkpoint_qwen
            set status = %s, error_message = %s, attempts = attempts + 1, updated_at = now()
            where table_id = %s;
            """,
            (status, error, table_id),
        )


def upsert_table_row(cur, row: dict):
    cur.execute(
        """
        insert into kosis_vdb_tables_qwen
            (table_id, stat_id, org_id, table_name, institution_name, unit,
             period_start, period_end, send_date, metadata_status,
             embedding_text, embedding, embedding_model, embedding_dimension, updated_at)
        values (%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s::vector,%s,%s, now())
        on conflict (table_id) do update set
            stat_id=excluded.stat_id, org_id=excluded.org_id, table_name=excluded.table_name,
            institution_name=excluded.institution_name, unit=excluded.unit,
            period_start=excluded.period_start, period_end=excluded.period_end,
            send_date=excluded.send_date, metadata_status=excluded.metadata_status,
            embedding_text=excluded.embedding_text, embedding=excluded.embedding,
            embedding_model=excluded.embedding_model, embedding_dimension=excluded.embedding_dimension,
            updated_at=now();
        """,
        (
            row["table_id"], row["stat_id"], row["org_id"], row["table_name"],
            row["institution_name"], row["unit"], row["period_start"], row["period_end"],
            row["send_date"], row["metadata_status"], row["embedding_text"], row["embedding"],
            EMBED_MODEL_NAME, EMBED_DIM,
        ),
    )


def insert_item_rows(cur, rows: list[dict]):
    if not rows:
        return
    values = [
        (r["table_id"], r["item_id"], r["item_name"], r["axis_id"], r["metadata_status"],
         r["embedding_text"], r["embedding"], EMBED_MODEL_NAME, EMBED_DIM)
        for r in rows
    ]
    psycopg2.extras.execute_values(
        cur,
        """
        insert into kosis_vdb_items_qwen
            (table_id, item_id, item_name, axis_id, metadata_status,
             embedding_text, embedding, embedding_model, embedding_dimension)
        values %s
        on conflict (table_id, item_id) do nothing;
        """,
        values,
        template="(%s,%s,%s,%s,%s,%s,%s::vector,%s,%s)",
    )


def insert_axis_rows(cur, rows: list[dict]):
    if not rows:
        return
    values = [
        (r["table_id"], r["axis_id"], r["axis_name"], r["axis_order"], r["metadata_status"],
         r["embedding_text"], r["embedding"], EMBED_MODEL_NAME, EMBED_DIM)
        for r in rows
    ]
    psycopg2.extras.execute_values(
        cur,
        """
        insert into kosis_vdb_axes_qwen
            (table_id, axis_id, axis_name, axis_order, metadata_status,
             embedding_text, embedding, embedding_model, embedding_dimension)
        values %s
        on conflict (table_id, axis_id) do nothing;
        """,
        values,
        template="(%s,%s,%s,%s,%s,%s,%s::vector,%s,%s)",
    )


def insert_axis_value_rows(cur, rows: list[dict]):
    if not rows:
        return
    values = [
        (r["table_id"], r["axis_id"], r["value_id"], r["value_name"], r["code"], r["metadata_status"])
        for r in rows
    ]
    psycopg2.extras.execute_values(
        cur,
        """
        insert into kosis_vdb_axis_values_qwen
            (table_id, axis_id, value_id, value_name, code, metadata_status)
        values %s
        on conflict (table_id, axis_id, value_id) do nothing;
        """,
        values,
    )


# ------------------------------------------------------------------
# Main
# ------------------------------------------------------------------

def flush_batch(conn, model, org_whitelist, results, role):
    """모아둔 결과(results)를 embed + DB insert + checkpoint까지 한 번에 처리한다.
    (n_success, n_failed) 반환. producer(API 호출 스레드)와 독립적으로 동작 — 이 함수가
    도는 동안에도 ThreadPoolExecutor의 다른 스레드는 계속 다음 표를 API로 조회한다."""
    table_texts, item_texts, axis_texts = [], [], []
    table_meta, item_meta, axis_meta, axis_value_meta = [], [], [], []

    for r in results:
        tbl_id = r["tbl_id"]
        org_id = r["org_id"]
        tbl_nm = r["tbl_nm"]
        institution = org_whitelist.get(org_id) if org_id else None
        enrichment = r["enrichment"]
        status = enrichment["status"]

        table_text = build_table_text(institution, tbl_nm)
        table_texts.append(table_text)
        table_meta.append({
            "table_id": tbl_id, "stat_id": r["stat_id"], "org_id": org_id,
            "table_name": tbl_nm, "institution_name": institution,
            "unit": enrichment.get("unit"), "period_start": enrichment.get("period_start"),
            "period_end": enrichment.get("period_end"), "send_date": r["send_de"],
            "metadata_status": status, "embedding_text": table_text,
        })

        if status == "success":
            for axis_order, axis_name in enumerate(enrichment["axis_names"], start=1):
                axis_num = next((n for n, nm in enrichment["axis_num_to_name"].items() if nm == axis_name), None)
                axis_text = build_axis_text(tbl_nm, axis_name)
                axis_texts.append(axis_text)
                axis_meta.append({
                    "table_id": tbl_id, "axis_id": axis_num, "axis_name": axis_name,
                    "axis_order": axis_order, "metadata_status": "success",
                    "embedding_text": axis_text,
                })
                for label, code in enrichment["code_maps"].get(axis_name, {}).items():
                    axis_value_meta.append({
                        "table_id": tbl_id, "axis_id": axis_num, "value_id": code,
                        "value_name": label, "code": code, "metadata_status": "success",
                    })

            # item(itmId) 축은 objL 체계와 별개라 axis_id는 채우지 않는다.
            for item_id, item_name in enrichment["item_pairs"].items():
                item_text = build_item_text(tbl_nm, None, item_name)
                item_texts.append(item_text)
                item_meta.append({
                    "table_id": tbl_id, "item_id": item_id, "item_name": item_name,
                    "axis_id": None, "metadata_status": "success", "embedding_text": item_text,
                })

    all_texts = table_texts + item_texts + axis_texts
    if all_texts:
        try:
            vecs = model.encode(all_texts, batch_size=32, convert_to_numpy=True,
                                 normalize_embeddings=True, show_progress_bar=False)
        except torch.cuda.OutOfMemoryError:
            torch.cuda.empty_cache()
            print("  [경고] batch_size=32에서 CUDA OOM — batch_size=16으로 낮춰 재시도", flush=True)
            vecs = model.encode(all_texts, batch_size=16, convert_to_numpy=True,
                                 normalize_embeddings=True, show_progress_bar=False)
    else:
        vecs = []

    n_t, n_i = len(table_texts), len(item_texts)
    for i, m in enumerate(table_meta):
        m["embedding"] = vecs[i].tolist()
    for i, m in enumerate(item_meta):
        m["embedding"] = vecs[n_t + i].tolist()
    for i, m in enumerate(axis_meta):
        m["embedding"] = vecs[n_t + n_i + i].tolist()

    try:
        with conn.cursor() as cur:
            for m in table_meta:
                upsert_table_row(cur, m)
            insert_axis_rows(cur, axis_meta)
            insert_item_rows(cur, item_meta)
            insert_axis_value_rows(cur, axis_value_meta)
        conn.commit()
        for m in table_meta:
            mark_checkpoint(conn, m["table_id"], "success")
        conn.commit()
        return len(table_meta), 0
    except Exception as e:
        conn.rollback()
        print(f"  [오류] batch insert 실패: {e}", flush=True)
        for m in table_meta:
            mark_checkpoint(conn, m["table_id"], "failed", str(e)[:500])
        conn.commit()
        return 0, len(table_meta)


def main():
    _load_env()
    ap = argparse.ArgumentParser()
    ap.add_argument("role", choices=["SERVER_A", "SERVER_B"])
    ap.add_argument("--limit", type=int, default=None, help="처리할 표 개수 상한(테스트/sanity용)")
    ap.add_argument("--concurrency", type=int, default=10)
    ap.add_argument("--api-keys", type=str, default=None,
                     help="콤마로 구분한 KOSIS API 키 여러 개(팀원 키 추가 시). "
                          "생략하면 .env의 KOSIS_API_KEY 하나만 쓴다. 키마다 분당 200건이 "
                          "독립이라 개수만큼 처리량이 늘어난다.")
    args = ap.parse_args()

    if args.api_keys:
        api_keys = [k.strip() for k in args.api_keys.split(",") if k.strip()]
    else:
        api_keys = [os.environ["KOSIS_API_KEY"]]
    print(f"KOSIS API 키 {len(api_keys)}개 사용 (키마다 독립적으로 분당 {int(60/KOSIS_MIN_CALL_INTERVAL_SEC)}건 한도)", flush=True)
    org_whitelist = load_org_whitelist()

    print("모델 로딩 중 (Qwen3-Embedding-4B, dim=2560)...", flush=True)
    os.environ.setdefault("HF_HOME", "/home/ubuntu/data/hf_cache")
    from sentence_transformers import SentenceTransformer
    model = SentenceTransformer(EMBED_MODEL_NAME, truncate_dim=EMBED_DIM, device="cuda")
    print("모델 로딩 완료.", flush=True)

    conn = get_connection()
    pending = load_pending(conn, args.role, args.limit)
    print(f"[{args.role}] 처리 대상 {len(pending)}건", flush=True)
    if not pending:
        print("처리할 표가 없습니다 (전부 success 이거나 partition이 비어 있음).")
        return

    with open(TABLES_PATH, encoding="utf-8") as f:
        all_lines = f.readlines()

    # 전체 대상을 'processing'으로 일괄 표시 (재시작 시 status != 'success'는 다시 집힘)
    with conn.cursor() as cur:
        table_ids = [t for t, _ in pending]
        for b in range(0, len(table_ids), 5000):
            cur.execute(
                "update kosis_reembed_checkpoint_qwen set status='processing', updated_at=now() where table_id = any(%s)",
                (table_ids[b:b + 5000],),
            )
    conn.commit()

    # producer: 장수명 ThreadPoolExecutor 하나에 전체를 미리 제출 — 이러면 컨슈머(임베딩+
    # DB insert, 메인 스레드)가 배치를 처리하는 동안에도 다른 워커 스레드들은 계속
    # API를 호출한다(2026-08-24: 매 청크마다 executor를 새로 만들고 전부 끝날 때까지
    # 기다리던 이전 구조는 청크 사이에 API 호출이 완전히 멈추는 배리어였다).
    t0 = time.time()
    n_done = n_success = n_failed = 0
    consecutive_errors = 0
    batch = []

    ex = ThreadPoolExecutor(max_workers=args.concurrency)
    aborted = False
    try:
        futs = {}
        for i, (table_id, line_no) in enumerate(pending):
            rec = json.loads(all_lines[line_no])
            key_for_table = api_keys[i % len(api_keys)]  # 라운드로빈 — 표마다 키가 고정되므로 재시도도 같은 키로 간다
            fut = ex.submit(process_one, line_no, rec.get("ORG_ID"), rec.get("TBL_ID"),
                             rec.get("STAT_ID"), rec.get("TBL_NM"), rec.get("SEND_DE"), key_for_table)
            futs[fut] = table_id

        n_fetched = 0
        for fut in as_completed(futs):
            table_id = futs[fut]
            try:
                batch.append(fut.result())
            except Exception as e:
                batch.append({"tbl_id": table_id, "enrichment": {"status": "error_other", "error": str(e)},
                               "line_no": None, "org_id": None, "stat_id": None,
                               "tbl_nm": None, "send_de": None})

            n_fetched += 1
            if n_fetched % 200 == 0:
                print(f"  [{args.role}] enrichment 진행: {n_fetched}건 API 조회 완료 "
                      f"(배치 누적 {len(batch)}/{CHUNK_SIZE}, elapsed={time.time()-t0:.0f}s)", flush=True)

            if len(batch) >= CHUNK_SIZE:
                s, f = flush_batch(conn, model, org_whitelist, batch, args.role)
                n_success += s
                n_failed += f
                consecutive_errors = 0 if f == 0 else consecutive_errors + 1
                batch = []

                n_done += s + f
                elapsed = time.time() - t0
                rate = n_done / elapsed if elapsed > 0 else 0
                remaining = len(pending) - n_done
                eta_sec = remaining / rate if rate > 0 else float("inf")
                print(
                    f"[{args.role}] {n_done}/{len(pending)} success={n_success} failed={n_failed} "
                    f"rate={rate:.2f}/s ETA={eta_sec/3600:.1f}h concurrency={args.concurrency}",
                    flush=True,
                )
                if consecutive_errors >= 5:
                    print("연속 5회 batch 실패 — 안전하게 중단합니다.", flush=True)
                    aborted = True
                    break

        if batch and not aborted:
            s, f = flush_batch(conn, model, org_whitelist, batch, args.role)
            n_success += s
            n_failed += f
            n_done += s + f
    finally:
        # cancel_futures=True: 아직 시작 안 한 작업은 즉시 취소 — 안 그러면 abort 시에도
        # 남은 수십만 건이 rate limiter를 거쳐 전부 끝날 때까지 shutdown이 블록된다.
        ex.shutdown(wait=False, cancel_futures=True)

    conn.close()
    elapsed = time.time() - t0
    print(f"[{args.role}] 완료. 총 {n_done}건 처리, success={n_success}, failed={n_failed}, "
          f"elapsed={elapsed/3600:.2f}h", flush=True)


if __name__ == "__main__":
    main()
