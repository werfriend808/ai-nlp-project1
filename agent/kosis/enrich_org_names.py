"""
agent/kosis/enrich_org_names.py — data/vdb_pending.jsonl의 표 이름 앞에 기관명+연도를 붙인다.

배경: 골든셋 실측 평가 중 KOSIS 표 이름이 기관(지자체)마다 완전히 겹치는 경우가 많다는 걸
발견했다(2026-08-18) — 예: "공원"이라는 이름의 표가 336개(서로 다른 org_id 210개, 즉 시군구
마다 하나씩). 표 이름만 임베딩하면 이런 표들은 벡터가 완전히 동일해져서 임베딩 모델
성능과 무관하게 절대 구분이 안 된다(전체 표의 57.6%, 16만5천여 개가 이런 중복 이름
그룹에 속함 — 실측 확인). 기관명을 표 이름 앞에 붙이면("서울특별시 종로구 공원") 서로
다른 문자열이 되어 임베딩 벡터도 달라진다.

2026-08-18 추가: 기관명만 붙여도(57.6% -> 34.0%로 개선 확인) "같은 기관이 시점만 다르게
여러 번 발행한 표"(예: DT_769001_I000007 vs DT_76901_I001017, 둘 다 org_id=769, TBL_NM=
"공원"이지만 SEND_DE가 2025-12-08/2024-10-31로 다름)는 여전히 못 가른다 — 기관명까지
똑같으니까. tables.jsonl(크롤링 원본)에 있는 SEND_DE(자료 전송일자)에서 연도만 뽑아
같이 붙인다("경상북도 문경시 (2025) 공원" vs "경상북도 문경시 (2024) 공원"). 완벽한 해법은
아니다(같은 해에 여러 번 갱신된 경우까지는 못 가름) — KOSIS 데이터 구조 자체의 한계다.

data/vdb_pending.jsonl에 등장하는 org_id 389개가 agent/preprocessing/kosis_org_whitelist.json
(build_org_whitelist.py가 이미 만들어둔 캐시)에 전부 이미 있어서, 기관명은 새로 API를
호출할 필요 없이 그 캐시만 읽어서 붙인다. 연도도 크롤링 원본에 이미 있는 필드라 추가
호출이 필요 없다.

⚠️ 실행 순서 주의: patch_english_table_names.py(영문→한글 표 이름 패치)보다 반드시
나중에 실행해야 한다 — 두 스크립트 다 vdb_pending.jsonl을 제자리에서 덮어쓰는데, 순서가
바뀌면 먼저 한 작업이 나중 작업에 덮어써져서 사라진다.

사용법 (프로젝트 루트에서):
    python -m agent.kosis.enrich_org_names
    -> data/vdb_pending.jsonl을 제자리에서 덮어씀 (직전 사본을 .bak2로 남김)
"""

from __future__ import annotations

import json
from pathlib import Path

PENDING_PATH = Path(__file__).parent.parent.parent / "data" / "vdb_pending.jsonl"
BACKUP_PATH = PENDING_PATH.with_suffix(".jsonl.bak2")
ORG_WHITELIST_PATH = Path(__file__).parent.parent / "preprocessing" / "kosis_org_whitelist.json"
CRAWL_OUTPUT_PATH = Path(__file__).parent / "crawl_output" / "tables.jsonl"


def _load_send_years() -> dict[str, str]:
    """크롤링 원본에서 tbl_id -> SEND_DE 연도(4자리)만 뽑는다."""
    years: dict[str, str] = {}
    if not CRAWL_OUTPUT_PATH.exists():
        return years
    with CRAWL_OUTPUT_PATH.open(encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            row = json.loads(line)
            tbl_id = row.get("TBL_ID")
            send_de = row.get("SEND_DE")
            if tbl_id and send_de and len(send_de) >= 4:
                years[tbl_id] = send_de[:4]
    return years


def main() -> None:
    if not PENDING_PATH.exists():
        raise SystemExit(f"{PENDING_PATH}가 없습니다.")
    if not ORG_WHITELIST_PATH.exists():
        raise SystemExit(
            f"{ORG_WHITELIST_PATH}가 없습니다. 먼저 python -m agent.kosis.build_org_whitelist 실행하세요."
        )

    org_names: dict[str, str] = json.loads(ORG_WHITELIST_PATH.read_text(encoding="utf-8"))
    print(f"기관명 캐시 {len(org_names)}개 로드")

    send_years = _load_send_years()
    print(f"연도 정보 {len(send_years)}개 로드 (tables.jsonl 기준)")

    rows = []
    with PENDING_PATH.open(encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                rows.append(json.loads(line))

    enriched = 0
    org_only = 0
    no_org_name = 0
    already_prefixed = 0
    for row in rows:
        org_id = row.get("org_id")
        tbl_id = row.get("tbl_id")
        text = row.get("text", "")
        org_name = org_names.get(org_id) if org_id else None
        if not org_name:
            no_org_name += 1
            continue
        if text.startswith(org_name):
            already_prefixed += 1
            continue
        year = send_years.get(tbl_id)
        if year:
            row["text"] = f"{org_name} ({year}) {text}"
            enriched += 1
        else:
            row["text"] = f"{org_name} {text}"
            org_only += 1

    PENDING_PATH.rename(BACKUP_PATH)
    with PENDING_PATH.open("w", encoding="utf-8") as out_f:
        for row in rows:
            out_f.write(json.dumps(row, ensure_ascii=False) + "\n")

    print(f"\n완료: {enriched}건에 기관명+연도 추가, {org_only}건은 연도 없어 기관명만 추가")
    print(f"  기관명 캐시에 없어서 건너뜀: {no_org_name}건")
    print(f"  이미 기관명이 붙어있어서 건너뜀: {already_prefixed}건")
    print(f"패치 전 원본은 {BACKUP_PATH}에 백업했습니다.")
    print(f"다음: {PENDING_PATH}를 구글 드라이브에 업로드해서 vdb_embedding_colab.ipynb로 재임베딩하세요.")


if __name__ == "__main__":
    main()
