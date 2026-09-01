"""agent/kosis/recover_excluded.py -- excluded_too_large로 제외된 표를 기간 분할로 복구한다.

배경: 전체 재구축(2026-08-26~27)에서는 속도를 위해 err=31(40,000셀 초과) 표를 기간 분할
없이 즉시 excluded_too_large로 제외했다. 그런데 그렇게 빠진 7,839건 중에 뉴스에서 자주
인용되는 큰 표들이 다수 포함돼 있었다(골든셋 정답표 19개 중 5개 포함 -- 사망원인,
경제활동인구, 광공업생산지수 등). 이 스크립트는 그중 지정한 표만 골라, reembed_worker에
그대로 남아있는 _fetch_rows_with_cell_split(기간을 반씩 쪼개 여러 번 받아 합치는 로직)을
써서 다시 수집한다.

전체 재구축 워커와 같은 저장 경로(flush_batch)를 쓰므로 TABLE/ITEM/AXIS/AXIS VALUE 적재와
item_axis_value_capped embedding_text 생성/임베딩까지 동일하게 처리된다.

사용법:
    python -m agent.kosis.recover_excluded --ids-file recover_ids.json
    python -m agent.kosis.recover_excluded --ids DT_1DA7002S,DT_1F02001
    python -m agent.kosis.recover_excluded --golden        # 골든셋 정답표 중 제외분만
"""
from __future__ import annotations

import argparse
import json
import os
import re
import sys
import time
from concurrent.futures import ThreadPoolExecutor, as_completed

import psycopg2
import psycopg2.extras

from agent.kosis import reembed_worker as W
from agent.kosis.recover_by_axis_code import candidate_codes, fetch_by_axis_code

TABLES_PATH = W.TABLES_PATH


def fetch_with_split(org_id: str, tbl_id: str, api_key: str) -> dict:
    """reembed_worker.fetch_table_enrichment와 동일하되, err=31에서 즉시 포기하지 않고
    _fetch_rows_with_cell_split으로 기간을 쪼개 받아온다(재구축 이전의 원래 동작)."""
    last_error = None
    saw_no_data = False
    for prd_se, start, end in W.PRD_SE_ATTEMPTS:
        for n_axes in (2, 1, 3, 4, 5, 6, 7, 8):
            params = {
                "method": "getList", "apiKey": api_key, "format": "json", "jsonVD": "Y",
                "orgId": org_id, "tblId": tbl_id, "prdSe": prd_se,
                "startPrdDe": start, "endPrdDe": end,
                **{f"objL{i}": "ALL" for i in range(1, n_axes + 1)}, "itmId": "ALL",
            }
            rate_limit_retries = 0
            while True:
                try:
                    resp = W._rate_limited_get(W.META_URL, params=params, timeout=15, api_key=api_key)
                    data = resp.json()
                except Exception as e:
                    last_error = f"request_error:{e}"
                    data = None
                    break
                if isinstance(data, dict) and str(data.get("err")) == W.KOSIS_RATE_LIMIT_ERR_CODE:
                    rate_limit_retries += 1
                    if rate_limit_retries > W.RATE_LIMIT_MAX_RETRIES:
                        return {"status": "error_other", "error": "[40] rate limited"}
                    W._backoff_sleep(rate_limit_retries)
                    continue
                break
            if data is None:
                break

            if isinstance(data, dict) and "err" in data:
                err_code = str(data.get("err"))
                err_msg = data.get("errMsg", "")
                last_error = f"[{err_code}] {err_msg}"
                is_axis_issue = ((err_code == "21" and "존재하지 않습니다" not in err_msg)
                                 or (err_code == "20" and "objL" in err_msg))
                if err_code == "30":
                    saw_no_data = True
                    break
                if err_code == W.CELL_LIMIT_ERR_CODE:
                    # 여기가 재구축 워커와 다른 유일한 지점 -- 포기 대신 기간 분할
                    split_data = W._fetch_rows_with_cell_split(
                        org_id, tbl_id, api_key, prd_se, start, end, n_axes)
                    if not split_data:
                        return {"status": "excluded_too_large", "error": last_error}
                    data = split_data
                elif not is_axis_issue:
                    return {"status": "error_other", "error": last_error}
                else:
                    continue

            if not isinstance(data, list) or not data:
                return {"status": "error_other", "error": "empty_response"}

            axis_names, seen_axis, axis_num_to_name = [], set(), {}
            for k in data[0].keys():
                m = W.AXIS_NAME_RE.match(k)
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
            item_pairs, unit_name, prd_de_values = {}, None, []
            for row in data:
                iid, inm = row.get("ITM_ID"), row.get("ITM_NM")
                if iid is not None:
                    item_pairs[iid] = inm
                if unit_name is None and row.get("UNIT_NM"):
                    unit_name = row.get("UNIT_NM")
                if row.get("PRD_DE"):
                    prd_de_values.append(row["PRD_DE"])
                for k, v in row.items():
                    m = W.AXIS_CODE_RE.match(k)
                    if not m:
                        continue
                    axis_name = axis_num_to_name.get(m.group(1))
                    label = row.get(f"C{m.group(1)}_NM")
                    if axis_name and label and v is not None:
                        code_maps[axis_name][label] = v

            return {
                "status": "success", "axis_names": axis_names,
                "axis_num_to_name": axis_num_to_name, "code_maps": code_maps,
                "item_pairs": item_pairs, "unit": unit_name, "prd_se": prd_se,
                "period_start": min(prd_de_values) if prd_de_values else None,
                "period_end": max(prd_de_values) if prd_de_values else None,
            }
    if saw_no_data:
        return {"status": "error_no_data", "error": last_error or "unknown"}
    return {"status": "error_other", "error": last_error or "unknown"}


ITEM_META_URL = "https://kosis.kr/openapi/statisticsData.do"
ITEM_SLICES = 4      # 항목을 몇 개까지 따로 받아 합칠지(분류값 50개 cap이라 4개면 충분)


def fetch_item_ids(org_id: str, tbl_id: str, api_key: str) -> list[str]:
    """getMeta&type=ITM으로 항목 코드 목록을 받는다.

    getMeta&type=OBJ(분류축)는 이런 표에서 err=30을 주지만 type=ITM은 안정적으로 응답한다
    (2026-08-27 실측)."""
    params = {"method": "getMeta", "type": "ITM", "apiKey": api_key,
              "format": "json", "jsonVD": "Y", "orgId": org_id, "tblId": tbl_id}
    try:
        data = W._rate_limited_get(ITEM_META_URL, params=params, timeout=25, api_key=api_key).json()
    except Exception:
        return []
    if not isinstance(data, list):
        return []
    return [d["ITM_ID"] for d in data if d.get("ITM_ID")]


def fetch_by_item_split(org_id: str, tbl_id: str, api_key: str) -> dict:
    """항목(itmId)을 하나씩 지정해 받아 합친다.

    err=31로 막히는 표의 상당수는 축이 아니라 '항목 수'가 원인이다(예: DT_1AG1013은 항목이
    414개라 itmId=ALL이면 셀이 414배가 된다). 항목을 1개로 좁히면 통과하고, 항목 몇 개를
    돌며 받아 합치면 항목명도 그만큼 확보된다(2026-08-27 실측으로 확인)."""
    itm_ids = fetch_item_ids(org_id, tbl_id, api_key)
    if not itm_ids:
        return {"status": "excluded_too_large", "error": "getMeta ITM 실패"}

    merged, n_ok = None, 0
    for prd_se, start, end in W.PRD_SE_ATTEMPTS[:5]:
        for n_axes in (2, 3, 1, 4, 5):
            for itm in itm_ids[:ITEM_SLICES]:
                params = {
                    "method": "getList", "apiKey": api_key, "format": "json", "jsonVD": "Y",
                    "orgId": org_id, "tblId": tbl_id, "prdSe": prd_se,
                    "startPrdDe": start, "endPrdDe": end,
                    **{f"objL{i}": "ALL" for i in range(1, n_axes + 1)}, "itmId": itm,
                }
                try:
                    data = W._rate_limited_get(W.META_URL, params=params, timeout=40,
                                                api_key=api_key).json()
                except Exception:
                    continue
                if isinstance(data, dict) or not data:
                    continue
                parsed = _parse_rows(data)
                if not parsed["axis_names"]:
                    continue
                n_ok += 1
                if merged is None:
                    merged = parsed
                else:
                    for an, mp in parsed["code_maps"].items():
                        merged["code_maps"].setdefault(an, {}).update(mp)
                    merged["item_pairs"].update(parsed["item_pairs"])
            if merged:
                break
        if merged:
            break

    if not merged:
        return {"status": "excluded_too_large", "error": f"항목 분할 실패(항목 {len(itm_ids)}개)"}
    # 항목명은 getMeta로 받은 전체 목록으로 보강(요청한 몇 개 말고 전부 이름이 있으므로)
    merged["status"] = "success"
    merged["prd_se"] = None
    merged["_item_slices"] = n_ok
    return merged


def _parse_rows(data: list) -> dict:
    """getList 응답에서 축/항목/분류값/기간을 뽑는다(reembed_worker와 동일 규칙)."""
    axis_names, seen, num2name = [], set(), {}
    for k in data[0].keys():
        m = W.AXIS_NAME_RE.match(k)
        if m:
            nm = data[0].get(k)
            num2name[m.group(1)] = nm
            if nm and nm not in seen:
                seen.add(nm)
                axis_names.append(nm)
    code_maps = {n: {} for n in axis_names}
    items, unit, prds = {}, None, []
    for row in data:
        if row.get("ITM_ID") is not None:
            items[row["ITM_ID"]] = row.get("ITM_NM")
        if unit is None and row.get("UNIT_NM"):
            unit = row["UNIT_NM"]
        if row.get("PRD_DE"):
            prds.append(row["PRD_DE"])
        for k, v in row.items():
            m = W.AXIS_CODE_RE.match(k)
            if not m:
                continue
            an = num2name.get(m.group(1))
            lb = row.get(f"C{m.group(1)}_NM")
            if an and lb and v is not None:
                code_maps[an][lb] = v
    return {"axis_names": axis_names, "axis_num_to_name": num2name,
            "code_maps": code_maps, "item_pairs": items, "unit": unit,
            "period_start": min(prds) if prds else None,
            "period_end": max(prds) if prds else None}


def recover_one(codes, org_id: str, tbl_id: str, api_key: str) -> dict:
    """3단계 복구: 기간 분할 -> 항목 분할 -> 축코드 고정 순으로 자동 시도.

    실측(2026-08-27, 국가데이터처 표본): 기간 분할만으로는 약 1/3만 살아난다. 나머지는
    축 조합 자체가 한계를 넘는 표라 기간을 아무리 쪼개도 소용이 없고, 축 하나를 구체
    코드로 고정해야 응답이 온다. 두 방식을 순서대로 자동 적용한다."""
    enr = fetch_with_split(org_id, tbl_id, api_key)
    if enr.get("status") != "excluded_too_large":
        enr["_via"] = "period_split"
        return enr

    # 2단계: 항목 수가 원인인 경우(실측상 이쪽이 가장 많다)
    enr2 = fetch_by_item_split(org_id, tbl_id, api_key)
    if enr2.get("status") == "success":
        enr2["_via"] = "item_split"
        return enr2

    # 3단계: 축 조합 자체가 한계를 넘는 경우
    enr3 = fetch_by_axis_code(codes, org_id, tbl_id, api_key)
    if enr3.get("status") == "success":
        enr3["_via"] = "axis_code"
        return enr3

    enr["_via"] = "all_failed"
    return enr


def load_target_ids(args, conn) -> list[str]:
    if args.ids:
        return [x.strip() for x in args.ids.split(",") if x.strip()]
    if args.ids_file:
        return json.loads(open(args.ids_file, encoding="utf-8").read())
    if args.org:
        # 이미 success가 된 표는 빠지므로, 같은 명령을 다시 실행하면 그대로 이어하기가 된다
        with conn.cursor() as cur:
            q = """select table_id from kosis_vdb_tables_qwen
                   where metadata_status='excluded_too_large' and org_id = %s
                   order by table_id"""
            params = [args.org]
            if args.limit:
                q += " limit %s"
                params.append(args.limit)
            cur.execute(q, params)
            return [r[0] for r in cur.fetchall()]
    if args.golden:
        import pandas as pd
        df = pd.read_excel("notebooks/골든셋_통합.xlsx", sheet_name="7단계_판정목록")
        gold = set()
        for v in df["matched_table_id(3단계)"].dropna():
            v = str(v).strip()
            if v and v != "없음":
                for p in re.split(r"[,\s]+", v):
                    if p.startswith(("DT_", "TX_", "CD", "CS")):
                        gold.add(p)
        with conn.cursor() as cur:
            cur.execute("""select table_id from kosis_vdb_tables_qwen
                           where table_id = any(%s) and metadata_status='excluded_too_large'""",
                        (sorted(gold),))
            return [r[0] for r in cur.fetchall()]
    raise SystemExit("--ids / --ids-file / --golden 중 하나가 필요합니다")


def main():
    W._load_env()
    ap = argparse.ArgumentParser()
    ap.add_argument("--ids", help="콤마로 구분한 table_id")
    ap.add_argument("--ids-file", help="table_id 배열이 담긴 JSON 파일")
    ap.add_argument("--golden", action="store_true", help="골든셋 정답표 중 제외분만")
    ap.add_argument("--org", help="기관 코드(org_id)의 제외 표 전체 (예: 101=국가데이터처). "
                                  "이미 복구된 건 자동 제외되므로 같은 명령으로 이어하기 가능")
    ap.add_argument("--limit", type=int, default=None, help="처리 개수 상한")
    ap.add_argument("--concurrency", type=int, default=2,
                    help="기간 분할은 표당 호출이 많아 낮게 잡는다")
    ap.add_argument("--api-keys", type=str, default=None)
    args = ap.parse_args()

    api_keys = ([k.strip() for k in args.api_keys.split(",") if k.strip()]
                if args.api_keys else [os.environ["KOSIS_API_KEY"]])

    conn = W.get_connection()
    targets = load_target_ids(args, conn)
    print(f"복구 대상: {len(targets)}건", flush=True)
    if not targets:
        print("대상 없음.")
        return

    with open(TABLES_PATH, encoding="utf-8") as f:
        catalog = {}
        for line in f:
            d = json.loads(line)
            if d.get("TBL_ID") in set(targets):
                catalog[d["TBL_ID"]] = d
    print(f"원본 카탈로그 매칭: {len(catalog)}/{len(targets)}", flush=True)

    org_whitelist = W.load_org_whitelist()
    print("모델 로딩 중...", flush=True)
    os.environ.setdefault("HF_HOME", "/home/ubuntu/data/hf_cache")
    from sentence_transformers import SentenceTransformer
    model = SentenceTransformer(W.EMBED_MODEL_NAME, truncate_dim=W.EMBED_DIM, device="cuda")
    print("모델 로딩 완료.", flush=True)

    t0 = time.time()
    results = []
    ex = ThreadPoolExecutor(max_workers=args.concurrency)
    futs = {}
    # 축코드 후보는 병렬 진입 전에 기관별로 한 번만 조회한다(커넥션 스레드 공유 방지)
    orgs = {catalog[t].get("ORG_ID") for t in targets if t in catalog}
    codes_by_org = {o: candidate_codes(conn, o) for o in orgs if o}
    print(f"축코드 후보 준비: {len(codes_by_org)}개 기관", flush=True)

    for i, tid in enumerate(targets):
        rec = catalog.get(tid)
        if not rec:
            print(f"  [건너뜀] {tid}: 원본 카탈로그에 없음", flush=True)
            continue
        key = api_keys[i % len(api_keys)]
        fut = ex.submit(recover_one, codes_by_org.get(rec.get("ORG_ID"), []),
                        rec.get("ORG_ID"), tid, key)
        futs[fut] = (tid, rec)

    # 2026-08-27: 예전엔 전부 끝난 뒤 한 번만 flush해서, 중간에 멈추면 그때까지 한
    # API 호출이 통째로 날아갔다. FLUSH_EVERY건마다 저장해 중단에 안전하게 만든다.
    # (이미 success가 된 표는 다음 실행 때 대상에서 자동으로 빠지므로 이어하기가 된다)
    # 2026-08-30: 100 -> 20으로 축소. 국가데이터처(org_id=101) 4,011건 복구 중 세션/환경
    # 재시작으로 프로세스가 통째로 죽어(84건 처리, 0건 반영) 전량 유실된 게 실측 확인돼서,
    # 중단 시 최대 손실을 99건에서 19건으로 낮춘다(DB round-trip이 늘지만 이 정도 빈도는
    # 무시할 수준).
    FLUSH_EVERY = 20
    pending, n_done, n_ok_total = [], 0, 0
    n_status, n_via = {}, {}

    def flush(buf):
        """성공분은 정상 적재하고, 3단계 모두 실패한 표는 상태를 바꿔 다음 실행 때
        다시 시도하지 않게 한다.

        2026-08-27: 실패해도 metadata_status가 excluded_too_large 그대로라, 재시작할 때마다
        같은 표를 또 시도해 앞부분 수백 건을 헛돌았다(OOM으로 재시작이 잦아 특히 손해였다).
        recover_failed로 따로 표시해 대상에서 빠지게 한다 -- 나중에 새 방법이 생기면
        이 상태만 골라 다시 돌리면 된다."""
        nonlocal n_ok_total
        ok = [r for r in buf if r["enrichment"]["status"] == "success"]
        failed_ids = [r["tbl_id"] for r in buf if r["enrichment"]["status"] != "success"]

        if ok:
            s_, f_ = W.flush_batch(conn, model, org_whitelist, ok, "RECOVER")
            n_ok_total += s_
            print(f"    -> DB 반영 {s_}건 (실패 {f_}) / 누적 {n_ok_total}건", flush=True)

        if failed_ids:
            with conn.cursor() as c:
                c.execute("""update kosis_vdb_tables_qwen
                             set metadata_status = 'recover_failed', updated_at = now()
                             where table_id = any(%s) and metadata_status = 'excluded_too_large'""",
                          (failed_ids,))
            conn.commit()
            print(f"    -> 복구 불가 {len(failed_ids)}건은 recover_failed로 표시(재시도 제외)", flush=True)

        buf.clear()

    try:
        for fut in as_completed(futs):
            tid, rec = futs[fut]
            try:
                enr = fut.result()
            except Exception as e:
                enr = {"status": "error_other", "error": str(e)}
            via = enr.get("_via", "?")
            n_done += 1
            print(f"  [{n_done}/{len(futs)}] {tid:<20} -> {enr['status']:<20} [{via}]"
                  f"{'' if enr['status']=='success' else ' ' + str(enr.get('error'))[:40]}", flush=True)
            rowd = {
                "line_no": None, "org_id": rec.get("ORG_ID"), "tbl_id": tid,
                "stat_id": rec.get("STAT_ID"), "tbl_nm": rec.get("TBL_NM"),
                "send_de": rec.get("SEND_DE"), "enrichment": enr,
                "rec_tbl_se": rec.get("REC_TBL_SE"), "vw_cd": rec.get("VW_CD"),
            }
            # 2026-08-27 OOM: 결과 객체(enrichment에 분류값 수천 개 포함)를 전부 리스트에
            # 쌓다가 12.3GB까지 불어나 커널에 강제 종료됐다. flush한 뒤에는 버리고
            # 카운터만 유지한다.
            n_status[enr["status"]] = n_status.get(enr["status"], 0) + 1
            n_via[via] = n_via.get(via, 0) + 1
            pending.append(rowd)
            if len(pending) >= FLUSH_EVERY:
                flush(pending)
                pending.clear()
    except KeyboardInterrupt:
        print("\n[중단 요청] 지금까지 모은 결과를 저장하고 종료합니다...", flush=True)
    finally:
        if pending:
            flush(pending)
        ex.shutdown(wait=False, cancel_futures=True)

    print(f"\n수집 성공 {n_status.get('success', 0)}/{n_done}건, DB 반영 누적 {n_ok_total}건", flush=True)
    print(f"  상태별: {n_status}", flush=True)
    print(f"  방식별: {n_via}", flush=True)

    conn.close()
    print(f"완료. elapsed={time.time()-t0:.0f}s", flush=True)


if __name__ == "__main__":
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    main()
