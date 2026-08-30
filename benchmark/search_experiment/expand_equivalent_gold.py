"""benchmark/search_experiment/expand_equivalent_gold.py

C 트랙 인수인계 "1. 동등 표를 정답으로 인정하도록 채점 고치기" 작업.

지금 6개 실험 스크립트(fuse.py/reranker_model_ab.py/rrf_weight_ab.py/gate_ab.py/
title_ab.py/ctx_prod_ab.py)는 전부 아래 패턴으로 채점한다:

    gold = {r["claim_id"]: set(r["gold"]) for r in ev}
    ... t in gold[claim_id] ...

즉 "정답 여러 개를 인정하는 기능" 자체는 이미 있다(gold가 원래 리스트다,
eval_set.json의 19-09처럼 실제로 정답 2개인 claim도 있음). 그래서 채점 로직을
따로 고칠 필요가 없다 — eval_set.json의 gold 리스트에 "동등 표"를 미리 채워
넣기만 하면, 기존 6개 스크립트를 한 글자도 안 고쳐도 전부 그 정답을 인정하게 된다.

이 스크립트가 하는 일:
  1. eval_set.json의 gold 표들에 대해 DB(kosis_vdb_tables_qwen)에서
     (stat_id, table_name)을 조회한다.
  2. 같은 (stat_id, table_name) 쌍을 가진 다른 table_id("형제 표" — 버전이
     갈라진 것, 예: DT_1DA7E06S/DT_1DA7E06S_NEW)를 전부 찾는다.
     -- 왜 stat_id 단독이 아니라 (stat_id, table_name) 조합인가: 2026-08-25
     실측(다른 트랙, agent/kosis/version_meta.py 참고)에 따르면 stat_id 단독은
     같은 설문조사 전체(예: 경제활동인구조사)를 묶어버려서 너무 굵다(그룹당
     평균 200개 표). (stat_id, table_name) 조합이라야 진짜 버전 쌍만 깔끔하게
     묶인다.
  3. 그 형제 표의 수록기간(period_start~period_end)이 claim이 필요로 하는
     기간을 포함하면, gold 리스트에 추가한다(기존 값은 유지 — replace 아님).
     "이 기간엔 둘 다 정답"이라는 뜻이지 "원래 정답이 틀렸다"는 뜻이 아니기
     때문이다(반대로 원래 정답 자체가 그 기간 데이터를 아예 못 주는 경우는
     "추가"가 아니라 "교체"가 맞다 — 그건 이미 별도로 처리함, git log의
     "eval_set.json 골든셋에 누락된 표 버전 수정 반영" 커밋 참고).

기본은 --dry-run(아무것도 안 씀, 뭘 추가할지만 출력) — 확인하고 --apply로 실행.

사용법 (AWS 서버, SUPABASE_DB_URL이 .env에 있어야 함):
    .venv/bin/python -m benchmark.search_experiment.expand_equivalent_gold
    .venv/bin/python -m benchmark.search_experiment.expand_equivalent_gold --apply
"""
from __future__ import annotations

import argparse
import json
import os
import re
from pathlib import Path
from typing import Optional

ROOT = Path(__file__).resolve().parents[2]
EVAL_SET_PATH = ROOT / "benchmark" / "search_experiment" / "eval_set.json"


# ---------------------------------------------------------------------------
# 기간 파싱 — DB의 period_start/end는 KOSIS PRD_DE 원본이라 보통 "YYYYMM"(월별)
# 또는 "YYYY"(연별) 형태다. claim의 period 필드는 사람이 적은 자유 텍스트라
# "2025-06" / "2025년 5월" / "2025년 3~5월" 등 형태가 섞여있다. 둘 다
# "YYYYMM" 6자리 문자열로 정규화해서 문자열 비교만으로 기간 포함 여부를 판단한다
# (사전순 비교가 그대로 시간순 비교와 같아지므로 — "202412" < "202506").
#
# 못 알아듣는 형식(분기/반기 등)은 None을 반환해서 호출부가 그 claim은 그냥
# 건너뛰게 한다(fail-open) — "모르면 추가 안 함"이 "모르는데 추가함"보다 훨씬
# 안전하다. 골든셋을 잘못 넓히면 향후 모든 실험이 조용히 오염된다.
# ---------------------------------------------------------------------------
_YMD_RANGE_KO = re.compile(r"(\d{4})년\s*(\d{1,2})\s*[~\-]\s*(\d{1,2})\s*월")
_YM_LIST_KO = re.compile(r"(\d{4})년\s*((?:\d{1,2}\s*,\s*)+\d{1,2})\s*월")
_YM_KO = re.compile(r"(\d{4})년\s*(\d{1,2})\s*월")
_QUARTER_KO = re.compile(r"(\d{4})\s*년?\s*(\d)\s*분기")
_YEAR_PAREN_KO = re.compile(r"\((\d{4})년\)")
_YM_ISO = re.compile(r"^(\d{4})-(\d{1,2})$")
_YM_DOT = re.compile(r"^(\d{4})\.(\d{1,2})$")
_YM_SHORT_DOT = re.compile(r"^(\d{2})\.(\d{1,2})$")  # "25.05" -> 2025-05 (20XX 가정)
_DATETIME = re.compile(r"^(\d{4})-(\d{2})-\d{2}[ T]")
_YEAR_ONLY = re.compile(r"^(\d{4})$")

_QUARTER_MONTHS = {1: (1, 3), 2: (4, 6), 3: (7, 9), 4: (10, 12)}


def parse_claim_period(period: Optional[str]) -> Optional[tuple[str, str]]:
    """claim의 period 문자열 -> (YYYYMM 시작, YYYYMM 끝). 못 알아들으면 None.

    2026-08-30 실측: eval_set.json의 실제 period 값 16종 중 처음엔 7종만
    파싱됐다("2024년 말"처럼 진짜 애매한 것만 남기고) — datetime 문자열
    ("2025-04-01 00:00:00"), 점 표기("2025.10"/"25.05"), 분기("2025 1분기"),
    콤마로 나열된 월("2025년 3, 4, 5월"), 괄호 안 연도("작년(2024년)") 패턴을
    추가로 인식하게 넓혔다. "말"처럼 정확한 달을 특정 못 하는 건 그대로 None —
    틀리게 넓히는 것보다 못 넓히는 게 낫다."""
    if not period:
        return None
    s = period.strip()

    m = _YMD_RANGE_KO.search(s)
    if m:
        y, m1, m2 = m.groups()
        return f"{y}{int(m1):02d}", f"{y}{int(m2):02d}"

    m = _YM_LIST_KO.search(s)
    if m:
        y, months_str = m.groups()
        months = [int(x) for x in months_str.split(",")]
        return f"{y}{min(months):02d}", f"{y}{max(months):02d}"

    m = _QUARTER_KO.search(s)
    if m:
        y, q = m.groups()
        start_mo, end_mo = _QUARTER_MONTHS[int(q)]
        return f"{y}{start_mo:02d}", f"{y}{end_mo:02d}"

    m = _YM_KO.search(s)
    if m:
        y, mo = m.groups()
        ym = f"{y}{int(mo):02d}"
        return ym, ym

    m = _DATETIME.match(s)
    if m:
        y, mo = m.groups()
        ym = f"{y}{int(mo):02d}"
        return ym, ym

    m = _YM_ISO.match(s)
    if m:
        y, mo = m.groups()
        ym = f"{y}{int(mo):02d}"
        return ym, ym

    m = _YM_DOT.match(s)
    if m:
        y, mo = m.groups()
        ym = f"{y}{int(mo):02d}"
        return ym, ym

    m = _YM_SHORT_DOT.match(s)
    if m:
        yy, mo = m.groups()
        ym = f"20{yy}{int(mo):02d}"
        return ym, ym

    m = _YEAR_PAREN_KO.search(s)
    if m:
        y = m.group(1)
        return f"{y}01", f"{y}12"

    m = _YEAR_ONLY.match(s)
    if m:
        y = m.group(1)
        return f"{y}01", f"{y}12"

    return None


def _normalize_db_period(value: Optional[str]) -> Optional[str]:
    """DB의 period_start/end를 YYYYMM으로 맞춘다. 연도만(YYYY)이면 그 해의
    첫/끝 달로 못 늘린다(방향을 모르므로) — 그대로 못 씀 처리(None)하지 않고
    호출부에서 시작/끝 각각 다르게 다룬다(아래 is_period_contained 참고)."""
    if not value:
        return None
    v = value.strip()
    if re.match(r"^\d{6}$", v):
        return v
    if re.match(r"^\d{4}$", v):
        return v  # 연도만 있는 경우 — 호출부가 처리
    return None


def is_period_contained(
    claim_start: str, claim_end: str, table_start: Optional[str], table_end: Optional[str]
) -> bool:
    """claim이 필요로 하는 [claim_start, claim_end](YYYYMM)가 표의 수록기간
    [table_start, table_end] 안에 완전히 들어가는지. 표 쪽 값이 연도만(YYYY,
    4자리)이면 시작은 그 해 1월, 끝은 그 해 12월로 보수적으로 늘려 비교한다
    (표가 실제로 더 넓게 커버할 수는 있어도 좁게 볼 일은 없다는 뜻이라 안전).
    표 기간 정보가 아예 없으면 False(모르면 인정 안 함, fail-closed — 잘못된
    정답 추가보다 놓치는 게 낫다)."""
    ts = _normalize_db_period(table_start)
    te = _normalize_db_period(table_end)
    if not ts or not te:
        return False
    if len(ts) == 4:
        ts = f"{ts}01"
    if len(te) == 4:
        te = f"{te}12"
    return ts <= claim_start and claim_end <= te


# ---------------------------------------------------------------------------
# DB 조회 (AWS 서버에서만 동작 — SUPABASE_DB_URL 필요)
# ---------------------------------------------------------------------------
def _connect():
    import psycopg2

    try:
        from dotenv import find_dotenv, load_dotenv

        load_dotenv(find_dotenv(usecwd=True))
    except ImportError:
        pass

    db_url = os.environ.get("SUPABASE_DB_URL")
    if not db_url:
        raise RuntimeError("SUPABASE_DB_URL이 .env에 없음 — AWS 서버에서 실행해야 함")
    conn = psycopg2.connect(db_url)
    conn.autocommit = True
    return conn


def find_sibling_candidates(gold_ids: set[str]) -> dict[str, list[dict]]:
    """gold_ids 각각 -> [{table_id, period_start, period_end}, ...] (자기 자신
    제외, 같은 (stat_id, table_name) 클러스터에 속한 다른 표들). DB 조회 2단계:
    먼저 gold_ids의 (stat_id, table_name)을 얻고, 그다음 그 조합으로 형제들을
    찾는다."""
    import psycopg2.extras

    conn = _connect()
    with conn.cursor(cursor_factory=psycopg2.extras.RealDictCursor) as cur:
        cur.execute(
            """select table_id, stat_id, table_name
               from kosis_vdb_tables_qwen where table_id = ANY(%s)""",
            (list(gold_ids),),
        )
        gold_rows = {r["table_id"]: r for r in cur.fetchall()}

        clusters: dict[tuple[str, str], list[str]] = {}
        for gid, row in gold_rows.items():
            stat_id, tbl_nm = row.get("stat_id"), row.get("table_name")
            if not stat_id or not tbl_nm:
                continue  # 메타데이터 자체가 없으면(예: excluded_too_large) 클러스터링 불가
            clusters.setdefault((stat_id, tbl_nm), []).append(gid)

        result: dict[str, list[dict]] = {gid: [] for gid in gold_ids}
        for (stat_id, tbl_nm), members in clusters.items():
            cur.execute(
                """select table_id, period_start, period_end
                   from kosis_vdb_tables_qwen where stat_id = %s and table_name = %s""",
                (stat_id, tbl_nm),
            )
            siblings = cur.fetchall()
            for gid in members:
                result[gid] = [dict(s) for s in siblings if s["table_id"] != gid]
    conn.close()
    return result


# ---------------------------------------------------------------------------
# 메인 — eval_set.json을 읽어서 확장 후보를 계산하고, --apply일 때만 저장한다.
# ---------------------------------------------------------------------------
def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--apply", action="store_true", help="실제로 eval_set.json에 반영(기본은 미리보기만)")
    args = ap.parse_args()

    data = json.loads(EVAL_SET_PATH.read_text(encoding="utf-8"))

    all_gold_ids = {tid for r in data for tid in r["gold"]}
    siblings_by_gold = find_sibling_candidates(all_gold_ids)

    additions: list[tuple[str, str, str]] = []  # (claim_id, 추가되는 table_id, 근거)
    unparsed_periods: list[str] = []

    for r in data:
        claim_period = parse_claim_period(r.get("period"))
        if claim_period is None:
            if r.get("period"):
                unparsed_periods.append(f"{r['claim_id']} ({r['period']!r})")
            continue
        cstart, cend = claim_period

        existing = set(r["gold"])
        new_ids: list[str] = []
        for gid in r["gold"]:
            for sib in siblings_by_gold.get(gid, []):
                sid = sib["table_id"]
                if sid in existing or sid in new_ids:
                    continue
                if is_period_contained(cstart, cend, sib.get("period_start"), sib.get("period_end")):
                    new_ids.append(sid)
                    additions.append((r["claim_id"], sid, f"{gid}의 형제 표, 기간 {cstart}~{cend} 포함"))

        if new_ids:
            r["gold"].extend(new_ids)
            r["gold_status"].extend(["success"] * len(new_ids))

    print(f"기간 못 읽은 claim {len(unparsed_periods)}건(건너뜀): {unparsed_periods[:10]}")
    print(f"\n추가 대상 {len(additions)}건:")
    for claim_id, tid, reason in additions:
        print(f"  [{claim_id}] + {tid}  ({reason})")

    if not additions:
        print("\n추가할 게 없음 — 종료.")
        return

    if args.apply:
        EVAL_SET_PATH.write_text(
            json.dumps(data, ensure_ascii=False, indent=1) + "\n", encoding="utf-8"
        )
        print(f"\n적용 완료 — {EVAL_SET_PATH}")
    else:
        print("\n[미리보기 모드] 저장 안 함 — 확인 후 --apply로 다시 실행하세요.")


if __name__ == "__main__":
    main()
