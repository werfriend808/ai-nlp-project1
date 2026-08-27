"""agent/kosis/consolidate_excluded.py -- excluded_too_large 표 목록을 하나로 통합한다.

배경: 재구축 중 excluded_too_large(40,000셀 초과로 제외) 표를 서버별 JSONL에 남기게
했는데, (a) 로깅 기능을 도중에 추가해서 그 이전 처리분이 파일에 없고, (b) SERVER_B/
Colab의 파일은 각자 다른 파일시스템에 있어 7-1에서 모을 수 없다. DB의
kosis_vdb_tables_qwen.metadata_status='excluded_too_large'가 유일하게 완전한 원본이므로
DB를 기준으로 목록을 재구성하고, 로컬에 남아있는 JSONL이 있으면 거기서만 얻을 수 있는
정보(reason_detail, detected_at)를 덧붙인다. rec_tbl_se/vw_cd는 DB에 없는 컬럼이라
원본 tables.jsonl에서 table_id로 찾아 채운다.

산출물: backup/excluded_too_large_ALL.jsonl (나중에 여유 생기면 이 목록만 재처리 가능)

사용법:
    python -m agent.kosis.consolidate_excluded
"""
from __future__ import annotations

import json
import os
from pathlib import Path

import psycopg2
import psycopg2.extras

PROJECT_DIR = Path(__file__).resolve().parent.parent.parent
TABLES_PATH = PROJECT_DIR / "agent/kosis/crawl_output/tables.jsonl"
BACKUP_DIR = PROJECT_DIR / "backup"
OUT_PATH = BACKUP_DIR / "excluded_too_large_ALL.jsonl"


def _load_env():
    env_path = PROJECT_DIR / ".env"
    if env_path.exists():
        with open(env_path) as f:
            for line in f:
                line = line.strip()
                if line and not line.startswith("#") and "=" in line:
                    k, v = line.split("=", 1)
                    os.environ.setdefault(k.strip(), v.strip())


def main():
    _load_env()
    conn = psycopg2.connect(os.environ["SUPABASE_DB_URL"])
    cur = conn.cursor(cursor_factory=psycopg2.extras.RealDictCursor)

    # 1) DB = 완전한 원본
    cur.execute("""
        select t.table_id, t.org_id, t.table_name, t.stat_id, t.send_date,
               t.institution_name, t.metadata_status, t.updated_at,
               c.server_role
        from kosis_vdb_tables_qwen t
        left join kosis_reembed_checkpoint_qwen c on c.table_id = t.table_id
        where t.metadata_status = 'excluded_too_large'
        order by t.table_id
    """)
    db_rows = cur.fetchall()
    print(f"DB의 excluded_too_large: {len(db_rows):,}건")

    # 2) 로컬에 남아있는 JSONL에서 reason_detail/detected_at 보강
    extra = {}
    for p in sorted(BACKUP_DIR.glob("excluded_too_large_SERVER_*.jsonl")) + \
             sorted(BACKUP_DIR.glob("excluded_too_large_COLAB*.jsonl")):
        n = 0
        with open(p, encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                try:
                    d = json.loads(line)
                except json.JSONDecodeError:
                    continue
                if d.get("TBL_ID"):
                    extra[d["TBL_ID"]] = d
                    n += 1
        print(f"  보강 소스: {p.name} ({n:,}건)")
    print(f"  -> 상세정보 확보: {len(extra):,}건")

    # 3) rec_tbl_se / vw_cd 는 DB에 없는 컬럼이라 원본 카탈로그에서 가져온다
    need = {r["table_id"] for r in db_rows}
    catalog = {}
    with open(TABLES_PATH, encoding="utf-8") as f:
        for line in f:
            d = json.loads(line)
            tid = d.get("TBL_ID")
            if tid in need:
                catalog[tid] = d
    print(f"  원본 카탈로그 매칭: {len(catalog):,}/{len(need):,}건")

    BACKUP_DIR.mkdir(exist_ok=True)
    written = 0
    with open(OUT_PATH, "w", encoding="utf-8") as out:
        for r in db_rows:
            tid = r["table_id"]
            cat = catalog.get(tid, {})
            ex = extra.get(tid, {})
            entry = {
                "TBL_ID": tid,
                "ORG_ID": r["org_id"],
                "TBL_NM": r["table_name"],
                "STAT_ID": r["stat_id"],
                "SEND_DE": r["send_date"],
                "REC_TBL_SE": cat.get("REC_TBL_SE"),
                "VW_CD": cat.get("VW_CD"),
                "institution_name": r["institution_name"],
                "reason": "excluded_too_large",
                # KOSIS err=31(40,000셀 초과) 시 기간분할 없이 즉시 제외한 표.
                "reason_detail": ex.get("reason_detail", "[31] 40,000셀 초과 (상세 미기록)"),
                "detected_at": ex.get("detected_at") or (
                    r["updated_at"].isoformat() if r["updated_at"] else None),
                "server": ex.get("server") or r["server_role"],
                "detail_source": "jsonl" if tid in extra else "db_only",
            }
            out.write(json.dumps(entry, ensure_ascii=False) + "\n")
            written += 1

    print(f"\n생성 완료: {OUT_PATH} ({written:,}건)")
    n_jsonl = sum(1 for r in db_rows if r["table_id"] in extra)
    print(f"  상세정보 있음(jsonl): {n_jsonl:,}건 / DB만: {written - n_jsonl:,}건")

    conn.close()


if __name__ == "__main__":
    main()
