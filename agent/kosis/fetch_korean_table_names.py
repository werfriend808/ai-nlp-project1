"""영문으로 저장된 표명을 KOSIS getMeta(type=TBL)로 한글명을 받아 DB에 반영한다.

배경: 크롤러가 MT_ETITLE(영문 KOSIS) 트리를 먼저 훑는 바람에 일부 표가 영문 이름으로
저장됐다(2026-08-18 최초 발견, agent/kosis/patch_english_table_names.py 참고). 그 스크립트는
구 파이프라인의 data/vdb_pending.jsonl을 고치는 것이라, 2026-08-27 재구축한
kosis_vdb_tables_qwen에는 같은 문제가 그대로 남아 있다.

이 스크립트는 표명만 갱신한다. embedding_text 재생성과 재임베딩은
reembed_v2_worker.py가 담당한다(표명을 바꾼 뒤 그 워커를 돌리면 된다).

되돌리려면 kosis_english_name_backup 테이블을 쓴다.
"""
from __future__ import annotations

import argparse
import json
import os
import re
import sys
import threading
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import psycopg2
import requests

META_URL = "https://kosis.kr/openapi/statisticsData.do"
# getMeta&type=TBL 응답은 표준 JSON이 아니다(키에 따옴표 없음) — TBL_NM만 정규식으로 뽑는다.
# "TBL_NM_ENG:"와 구분하려고 콜론이 TBL_NM 바로 뒤에 오는 경우만 잡는다.
_TBL_NM_RE = re.compile(r'TBL_NM:"([^"]+)"')

_session = requests.Session()
_lock = threading.Lock()
_last_call: dict[str, float] = {}
MIN_INTERVAL = 60.0 / 195  # 키당 분당 200회 제한 아래로


def _rate_limited_get(params: dict, api_key: str):
    with _lock:
        now = time.monotonic()
        wait = _last_call.get(api_key, 0.0) + MIN_INTERVAL - now
        if wait > 0:
            time.sleep(wait)
        _last_call[api_key] = time.monotonic()
    return _session.get(META_URL, params=params, timeout=20)


def looks_english(text: str) -> bool:
    letters = [c for c in (text or "") if c.isalpha()]
    return bool(letters) and all(ord(c) < 128 for c in letters)


def fetch_korean_name(org_id: str, tbl_id: str, api_key: str) -> str | None:
    params = {"method": "getMeta", "type": "TBL", "apiKey": api_key,
              "format": "json", "orgId": org_id, "tblId": tbl_id}
    try:
        resp = _rate_limited_get(params, api_key)
    except requests.RequestException:
        return None
    m = _TBL_NM_RE.search(resp.text)
    return m.group(1) if m else None


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--ids-file", required=True, help="[{table_id, org_id, table_name}] JSON")
    ap.add_argument("--api-keys", required=True, help="쉼표 구분")
    ap.add_argument("--concurrency", type=int, default=3)
    ap.add_argument("--flush-every", type=int, default=100)
    args = ap.parse_args()

    keys = [k.strip() for k in args.api_keys.split(",") if k.strip()]
    targets = json.loads(Path(args.ids_file).read_text(encoding="utf-8"))

    conn = psycopg2.connect(os.environ["SUPABASE_DB_URL"])
    cur = conn.cursor()
    # 이미 한글로 바뀐 건 건너뛴다(재실행 안전)
    cur.execute("select table_id, table_name from kosis_vdb_tables_qwen where table_id = any(%s)",
                ([t["table_id"] for t in targets],))
    current = dict(cur.fetchall())
    todo = [t for t in targets if looks_english(current.get(t["table_id"], ""))]
    print(f"대상 {len(targets):,}건 중 아직 영문인 것 {len(todo):,}건")

    results: list[tuple[str, str]] = []
    stats = {"ok": 0, "no_name": 0, "still_english": 0, "fail": 0}

    def work(i_t):
        i, t = i_t
        key = keys[i % len(keys)]
        name = fetch_korean_name(t["org_id"], t["table_id"], key)
        return t, name

    def flush():
        if not results:
            return
        cur.executemany(
            "update kosis_vdb_tables_qwen set table_name = %s, updated_at = now() where table_id = %s",
            [(n, t) for t, n in results],
        )
        conn.commit()
        print(f"    -> DB 반영 {len(results)}건 / 누적 ok={stats['ok']}")
        results.clear()

    with ThreadPoolExecutor(max_workers=args.concurrency) as ex:
        for n_done, (t, name) in enumerate(ex.map(work, enumerate(todo)), 1):
            tid = t["table_id"]
            if name is None:
                stats["fail"] += 1
                mark = "실패"
            elif looks_english(name):
                stats["still_english"] += 1
                mark = f"여전히 영문: {name[:40]}"
            elif name == t["table_name"]:
                stats["no_name"] += 1
                mark = "변화 없음"
            else:
                stats["ok"] += 1
                results.append((tid, name))
                mark = f"-> {name[:44]}"
            if n_done <= 20 or n_done % 100 == 0:
                print(f"  [{n_done}/{len(todo)}] {tid:<18} {mark}")
            if len(results) >= args.flush_every:
                flush()
    flush()
    print(f"\n완료: {stats}")
    conn.close()


if __name__ == "__main__":
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    main()
