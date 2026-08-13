"""
agent/kosis/build_org_whitelist.py — table_params.json에 실제로 등장하는 orgId 전부를
KOSIS getMeta(type=ORG) API로 조회해서, 기관명 화이트리스트 캐시 파일을 만든다.

배경: source_filter.py의 KOSIS_VERIFIED_ORGS는 사람이 손으로 채운 목록이라, 카탈로그에
새 표를 추가할 때 그 표의 발행 기관을 화이트리스트에 넣는 걸 깜빡하면 조용히 "uncertain"
으로 걸러지는 문제가 있었다(2026-08-13 실측: 과학기술정보통신부/기획예산처 2개 누락 발견).
이 스크립트는 "카탈로그에 실제 등장하는 기관"을 API로 직접 물어봐서 정답을 만들어두고,
source_filter.py가 이 파일을 읽어서 수동 목록과 합쳐 쓴다 — 표가 추가될 때마다 이 스크립트를
다시 돌리면 화이트리스트가 항상 카탈로그와 동기화된다.

배치 파이프라인(batch_runner.py) 실행 중에는 네트워크 호출을 안 하려고, 결과를 JSON
캐시 파일로 저장해두고 source_filter.py는 그 파일만 읽는다(런타임에 KOSIS API를 매번
부르지 않음).

사용법 (프로젝트 루트에서):
    python -m agent.kosis.build_org_whitelist
"""

from __future__ import annotations

import json
import os
import re
from pathlib import Path

import requests

# getMeta&type=ORG 응답이 표준 JSON이 아니다(키에 따옴표가 없음, 예:
# `[{ORG_NM_ENG:"...",ORG_NM:"국가데이터처"}]`) — 2026-08-13 실측 확인, json.loads()가
# 그대로 실패해서 정규식으로 직접 뽑는다.
_ORG_NM_RE = re.compile(r'ORG_NM:"([^"]+)"')

try:
    from dotenv import find_dotenv, load_dotenv

    load_dotenv(find_dotenv(usecwd=True))
except ImportError:
    pass

TABLE_PARAMS_PATH = Path(__file__).parent / "table_params.json"
OUTPUT_PATH = Path(__file__).parent.parent / "preprocessing" / "kosis_org_whitelist.json"
META_URL = "https://kosis.kr/openapi/statisticsData.do"


def _fetch_org_name(org_id: str, api_key: str) -> str | None:
    params = {"method": "getMeta", "type": "ORG", "apiKey": api_key, "format": "json", "orgId": org_id}
    resp = requests.get(META_URL, params=params, timeout=15)
    match = _ORG_NM_RE.search(resp.text)
    return match.group(1) if match else None


def main() -> None:
    api_key = os.environ.get("KOSIS_API_KEY")
    if not api_key:
        raise SystemExit("KOSIS_API_KEY가 없습니다. .env 또는 환경변수로 설정하세요.")

    table_params = json.loads(TABLE_PARAMS_PATH.read_text(encoding="utf-8"))
    org_ids = sorted({base.get("orgId") for base in table_params.values() if base.get("orgId")})
    print(f"카탈로그에 등장하는 고유 orgId {len(org_ids)}개 조회 중...")

    org_names: dict[str, str] = {}
    for org_id in org_ids:
        name = _fetch_org_name(org_id, api_key)
        print(f"  {org_id} -> {name!r}")
        if name:
            org_names[org_id] = name

    OUTPUT_PATH.write_text(json.dumps(org_names, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"\n{len(org_names)}개 기관명 저장 완료: {OUTPUT_PATH}")


if __name__ == "__main__":
    main()
