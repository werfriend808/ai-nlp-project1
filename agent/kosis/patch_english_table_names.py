"""
agent/kosis/patch_english_table_names.py — data/vdb_pending.jsonl에서 영문 제목으로 남은
표들만 KOSIS getMeta(type=TBL) API로 한글 이름을 다시 조회해서 덮어쓴다.

배경: crawl_table_catalog.py가 12개 vwCd를 스택(queue.pop())으로 순회하는데,
ROOT_VW_CODES 리스트 맨 끝에 있던 "MT_ETITLE"(영문KOSIS)이 스택 특성상 실제로는 제일
먼저 크롤링됐다. 크롤러 자체의 seen_tbl_ids 중복 제거 때문에, 영문 버전이 먼저 기록된
표는 나중에 MT_OTITLE(한글) 트리에서 같은 표를 다시 만나도 "이미 있음"으로 버려져서
애초에 한글 이름이 tables.jsonl에 도달하지도 못했다(2026-08-18 골든셋 실측 평가 중
VDB Recall@5가 0%로 나와서 발견 — 정답 표 15개 중 10개, 67%가 영문으로 저장돼 있었음).

전체 28만7천여 개를 처음부터 다시 크롤링하는 건 "수 시간~며칠" 걸려서 부담이 크므로,
이미 완료된 크롤링 결과는 그대로 두고 영문으로 남은 표(약 2,827개)만 KOSIS
getMeta(type=TBL) API로 한글 이름(TBL_NM)을 다시 조회해서 그 표만 패치한다.

getMeta&type=TBL 응답이 표준 JSON이 아니라서(키에 따옴표 없음, build_org_whitelist.py의
getMeta&type=ORG와 동일한 이유) 정규식으로 TBL_NM만 뽑는다.

사용법 (프로젝트 루트에서):
    python -m agent.kosis.patch_english_table_names
    -> data/vdb_pending.jsonl을 제자리에서 덮어씀 (패치 전 사본을 .bak으로 남김)
"""

from __future__ import annotations

import json
import os
import re
import time
from pathlib import Path

import requests

try:
    from dotenv import find_dotenv, load_dotenv

    load_dotenv(find_dotenv(usecwd=True))
except ImportError:
    pass

PENDING_PATH = Path(__file__).parent.parent.parent / "data" / "vdb_pending.jsonl"
BACKUP_PATH = PENDING_PATH.with_suffix(".jsonl.bak")
META_URL = "https://kosis.kr/openapi/statisticsData.do"
REQUEST_DELAY_SEC = 0.3

# getMeta&type=ORG(build_org_whitelist.py)와 동일한 이유로 정규식 파싱. "TBL_NM_ENG:"와
# 구분하기 위해 콜론이 "TBL_NM" 바로 뒤에 오는 경우만 잡는다(TBL_NM_ENG는 "_ENG"가 껴서
# 이 패턴에 안 걸림).
_TBL_NM_RE = re.compile(r'TBL_NM:"([^"]+)"')


def _looks_english(text: str) -> bool:
    """알파벳이 하나라도 있으면 전부 ASCII인지로 대략 판정 — 한글이 섞여 있으면 False."""
    letters = [c for c in text if c.isalpha()]
    return bool(letters) and all(ord(c) < 128 for c in letters)


def _fetch_korean_name(org_id: str, tbl_id: str, api_key: str) -> str | None:
    params = {
        "method": "getMeta", "type": "TBL", "apiKey": api_key,
        "format": "json", "orgId": org_id, "tblId": tbl_id,
    }
    try:
        resp = requests.get(META_URL, params=params, timeout=15)
    except requests.RequestException:
        return None
    match = _TBL_NM_RE.search(resp.text)
    return match.group(1) if match else None


def main() -> None:
    api_key = os.environ.get("KOSIS_API_KEY")
    if not api_key:
        raise SystemExit("KOSIS_API_KEY가 없습니다. .env에 설정하세요.")
    if not PENDING_PATH.exists():
        raise SystemExit(f"{PENDING_PATH}가 없습니다. prepare_vdb_export.py를 먼저 실행하세요.")

    rows = []
    with PENDING_PATH.open(encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                rows.append(json.loads(line))

    targets = [r for r in rows if _looks_english(r["text"])]
    print(f"전체 {len(rows)}건 중 영문으로 보이는 표 {len(targets)}건 패치 시도")

    patched = 0
    still_english = 0
    for i, row in enumerate(targets):
        korean_name = _fetch_korean_name(row["org_id"], row["tbl_id"], api_key)
        if korean_name and not _looks_english(korean_name):
            row["text"] = korean_name
            patched += 1
        else:
            still_english += 1
        if (i + 1) % 100 == 0:
            print(f"  [{i + 1}/{len(targets)}] 진행 중... (성공 {patched}, 실패 {still_english})")
        time.sleep(REQUEST_DELAY_SEC)

    PENDING_PATH.rename(BACKUP_PATH)
    with PENDING_PATH.open("w", encoding="utf-8") as out_f:
        for row in rows:
            out_f.write(json.dumps(row, ensure_ascii=False) + "\n")

    print(f"\n완료: {patched}건 한글로 패치, {still_english}건은 한글 이름을 못 찾아 영문 그대로 남음")
    print(f"패치 전 원본은 {BACKUP_PATH}에 백업했습니다.")
    print(f"다음: {PENDING_PATH}를 구글 드라이브에 업로드해서 vdb_embedding_colab.ipynb로 재임베딩하세요.")


if __name__ == "__main__":
    main()
