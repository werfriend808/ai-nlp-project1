"""agent/kosis/recover_by_axis_code.py -- 기간 분할로도 못 살린 초대형 표를 축코드 고정으로 복구.

배경: DT_1B34E13(시군구 x 사망원인50 x 성별) 같은 표는 축 조합만으로 40,000셀을 넘어서,
기간을 1년/1개월로 쪼개도 err=31이 계속 난다. 그런데 축 하나를 ALL 대신 구체적 코드로
고정하면 셀 수가 급감해 정상 응답이 온다(2026-08-27 실측: objL1=11로 3,465행 수신,
축 3개 이름과 분류값 모두 확보).

문제는 그 코드를 알 방법이다 -- getMeta&type=OBJ는 이런 표에서 err=30을 준다(실측).
그래서 이미 수집 완료된 27.9만 건 DB에서 "같은 기관이 쓰는 분류 코드"를 후보로 뽑아
차례로 시도한다. 코드 하나가 통하면 그 응답에서 축 이름 + (그 축을 뺀) 나머지 축의
분류값이 전부 나온다. 여러 코드를 시도해 고정한 축의 값도 조금씩 모은다.

한계: 고정한 축의 분류값은 시도한 코드 수만큼만 얻는다(전부는 못 얻음). embedding_text는
분류값을 50개로 자르므로 실무상 충분하지만, 완전한 수집은 아니라는 점을 명시해 둔다.

사용법:
    python -m agent.kosis.recover_by_axis_code --ids DT_1B34E13
    python -m agent.kosis.recover_by_axis_code --all-failed   # excluded 중 분할 실패분 전체
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time

import psycopg2
import psycopg2.extras

from agent.kosis import reembed_worker as W

MAX_CODE_TRIES = 12      # 후보 코드를 몇 개까지 시도할지
COLLECT_SUCCESS = 6      # 성공한 코드를 몇 개까지 모아 값을 합칠지


def candidate_codes(conn, org_id: str, limit: int = 60) -> list[str]:
    """이미 수집된 표들에서 이 기관이 실제로 쓰는 분류 코드를 빈도순으로 뽑는다."""
    with conn.cursor() as cur:
        cur.execute("""
            select v.code, count(*) as n
            from kosis_vdb_axis_values_qwen v
            join kosis_vdb_tables_qwen t on t.table_id = v.table_id
            where t.org_id = %s and v.code is not null and length(v.code) between 1 and 8
            group by v.code order by n desc limit %s
        """, (org_id, limit))
        codes = [r[0] for r in cur.fetchall()]
    # 흔한 형태를 앞쪽에 섞어 준다(기관 이력이 없을 때 대비)
    generic = ["11", "00", "01", "1", "0", "T1", "A", "10"]
    seen, out = set(), []
    for c in generic + codes:
        if c not in seen:
            seen.add(c)
            out.append(c)
    return out


def _parse(data: list) -> dict:
    """getList 응답에서 축/항목/분류값을 뽑는다(reembed_worker와 동일 규칙)."""
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


def fetch_by_axis_code(codes, org_id: str, tbl_id: str, api_key: str) -> dict:
    """축 하나를 구체 코드로 고정해 가며 메타데이터를 모은다.

    codes: candidate_codes()로 미리 뽑아둔 후보 코드 목록(스레드 안전을 위해 DB를
    직접 만지지 않는다)."""
    url = W.META_URL
    merged = None
    n_ok = 0
    tried = 0

    for prd_se, start, end in W.PRD_SE_ATTEMPTS[:4]:      # M/F/A/Y 정도면 충분
        for n_axes in (3, 2, 4, 1, 5):
            for code in codes:
                if tried >= MAX_CODE_TRIES or n_ok >= COLLECT_SUCCESS:
                    break
                tried += 1
                params = {
                    "method": "getList", "apiKey": api_key, "format": "json", "jsonVD": "Y",
                    "orgId": org_id, "tblId": tbl_id, "prdSe": prd_se,
                    "startPrdDe": start, "endPrdDe": end,
                    "objL1": code,
                    **{f"objL{i}": "ALL" for i in range(2, n_axes + 1)},
                    "itmId": "ALL",
                }
                try:
                    data = W._rate_limited_get(url, params=params, timeout=30, api_key=api_key).json()
                except Exception:
                    continue
                if isinstance(data, dict) or not data:
                    continue

                parsed = _parse(data)
                if not parsed["axis_names"]:
                    continue
                n_ok += 1
                if merged is None:
                    merged = parsed
                else:
                    # 같은 표의 다른 슬라이스 -- 분류값/항목만 합친다
                    for an, mp in parsed["code_maps"].items():
                        merged["code_maps"].setdefault(an, {}).update(mp)
                    merged["item_pairs"].update(parsed["item_pairs"])
                    if parsed["period_start"]:
                        merged["period_start"] = min(
                            filter(None, [merged["period_start"], parsed["period_start"]]))
                    if parsed["period_end"]:
                        merged["period_end"] = max(
                            filter(None, [merged["period_end"], parsed["period_end"]]))
            if merged and n_ok >= COLLECT_SUCCESS:
                break
        if merged:
            break

    if not merged:
        return {"status": "excluded_too_large",
                "error": f"축코드 {tried}회 시도 모두 실패"}
    merged["status"] = "success"
    merged["prd_se"] = None
    merged["_partial"] = True   # 고정한 축의 값은 일부만 수집됨
    merged["_codes_used"] = n_ok
    return merged


def main():
    W._load_env()
    ap = argparse.ArgumentParser()
    ap.add_argument("--ids", help="콤마로 구분한 table_id")
    ap.add_argument("--all-failed", action="store_true",
                    help="excluded_too_large로 남아있는 표 전체")
    ap.add_argument("--limit", type=int, default=None)
    ap.add_argument("--api-keys", type=str, default=None)
    args = ap.parse_args()

    api_keys = ([k.strip() for k in args.api_keys.split(",") if k.strip()]
                if args.api_keys else [os.environ["KOSIS_API_KEY"]])
    conn = W.get_connection()

    if args.ids:
        targets = [x.strip() for x in args.ids.split(",") if x.strip()]
    elif args.all_failed:
        with conn.cursor() as cur:
            q = ("select table_id from kosis_vdb_tables_qwen "
                 "where metadata_status='excluded_too_large' order by table_id")
            if args.limit:
                q += f" limit {args.limit}"
            cur.execute(q)
            targets = [r[0] for r in cur.fetchall()]
    else:
        raise SystemExit("--ids 또는 --all-failed 필요")

    print(f"대상 {len(targets)}건", flush=True)
    with open(W.TABLES_PATH, encoding="utf-8") as f:
        want = set(targets)
        catalog = {}
        for line in f:
            d = json.loads(line)
            if d.get("TBL_ID") in want:
                catalog[d["TBL_ID"]] = d

    org_whitelist = W.load_org_whitelist()
    print("모델 로딩 중...", flush=True)
    os.environ.setdefault("HF_HOME", "/home/ubuntu/data/hf_cache")
    from sentence_transformers import SentenceTransformer
    model = SentenceTransformer(W.EMBED_MODEL_NAME, truncate_dim=W.EMBED_DIM, device="cuda")
    print("모델 로딩 완료.\n", flush=True)

    codes_cache = {}
    results, t0 = [], time.time()
    for i, tid in enumerate(targets, 1):
        rec = catalog.get(tid)
        if not rec:
            print(f"  [{i}] {tid}: 카탈로그에 없음 -- 건너뜀", flush=True)
            continue
        key = api_keys[i % len(api_keys)]
        org = rec.get("ORG_ID")
        if org not in codes_cache:
            codes_cache[org] = candidate_codes(conn, org)
        enr = fetch_by_axis_code(codes_cache[org], org, tid, key)
        note = (f" (코드 {enr.get('_codes_used')}개 슬라이스, 축값 일부)"
                if enr["status"] == "success" else f" ({enr.get('error')})")
        print(f"  [{i}/{len(targets)}] {tid:<20} -> {enr['status']}{note}", flush=True)
        results.append({
            "line_no": None, "org_id": rec.get("ORG_ID"), "tbl_id": tid,
            "stat_id": rec.get("STAT_ID"), "tbl_nm": rec.get("TBL_NM"),
            "send_de": rec.get("SEND_DE"), "enrichment": enr,
            "rec_tbl_se": rec.get("REC_TBL_SE"), "vw_cd": rec.get("VW_CD"),
        })

    ok = [r for r in results if r["enrichment"]["status"] == "success"]
    print(f"\n성공 {len(ok)}/{len(results)}건 -- DB 반영 중...", flush=True)
    if ok:
        s, f = W.flush_batch(conn, model, org_whitelist, ok, "RECOVER_AXIS")
        print(f"DB 반영: success={s} failed={f}", flush=True)
    conn.close()
    print(f"완료. elapsed={time.time()-t0:.0f}s", flush=True)


if __name__ == "__main__":
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    main()
