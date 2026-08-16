"""
agent/kosis/crawl_table_catalog.py — KOSIS 통계표 목록 전체(약 28만개)를 BFS로 크롤링해서
표 ID/이름/기관 메타데이터를 로컬에 수집한다 (벡터DB 인덱스 구축의 1단계: 원자재 수집).

배경: kosis_list류 "목록조회" API는 트리 구조라 한 번에 대량 조회가 안 된다. 최상위
12개 vwCd(국내통계 주제별/기관별, e-지방지표, 국제통계, 북한통계 등)에서 시작해서
parentListId로 계속 드릴다운해야 하고, 실측 확인 결과 리프(진짜 표)까지 5~7단계가
걸린다. 이 스크립트는 그 트리 전체를 BFS로 순회하면서, 리프 노드(TBL_ID가 있는 응답
항목)를 만나면 기록하고, 카테고리 노드(LIST_ID만 있는 항목)는 큐에 넣어 계속 내려간다.

실측 확인한 응답 스키마 (2026-08-14, https://kosis.kr/openapi/statisticsList.do):
  - 카테고리 노드: {"LIST_ID": ..., "LIST_NM": ..., "VW_CD": ..., "VW_NM": ...}
  - 리프(표) 노드: {"TBL_ID": ..., "TBL_NM": ..., "ORG_ID": ..., "STAT_ID": ...,
                    "SEND_DE": ..., "REC_TBL_SE": ..., "VW_CD": ...}
  "TBL_ID" 키의 존재 여부로 리프인지 카테고리인지 구분한다.

전체 표가 약 28만개로 추정되고 API 호출도 수만 건 필요할 것으로 보여서, 오래 걸리는
작업이다(수 시간~며칠). 그래서:
  - 진행 상황(방문한 parentListId 집합 + 남은 큐)을 주기적으로 체크포인트 파일에 저장해서,
    중간에 끊겨도(네트워크 오류, 프로세스 종료 등) 이어서 할 수 있게 한다.
  - 수집한 리프 표는 JSONL로 한 줄씩 즉시 append해서, 크롤링이 중간에 멈춰도 그때까지
    모은 데이터는 안전하게 남는다.
  - 호출 사이에 딜레이를 둬서 공공 API에 과부하를 주지 않는다(정확한 rate limit 문서를
    못 찾아서 보수적으로 REQUEST_DELAY_SEC=0.3으로 시작 — 429/오류가 잦으면 늘릴 것).

사용법 (프로젝트 루트에서):
    python -m agent.kosis.crawl_table_catalog
    (중간에 멈췄다가 다시 실행하면 체크포인트에서 이어서 진행됨)
"""

from __future__ import annotations

import json
import os
import time
from pathlib import Path
from typing import Optional

import requests

try:
    from dotenv import find_dotenv, load_dotenv

    load_dotenv(find_dotenv(usecwd=True))
except ImportError:
    pass

LIST_URL = "https://kosis.kr/openapi/statisticsList.do"

# 국내통계(주제별/기관별) + e-지방지표(주제별/지역별) + 국제통계 + 북한통계 + 대상별/이슈별 +
# 광복이전통계 + 대한민국통계연감 + 작성중지 + 영문KOSIS — kosis_list 도구 설명에 나열된 12개
# 전체. 이 중 일부는 서로 겹치는 표를 가리킬 수 있어(같은 표가 여러 분류 체계에 동시에
# 속함) 최종적으로 TBL_ID 기준 중복 제거가 필요하다(아래 _write_leaf에서 처리).
ROOT_VW_CODES = [
    "MT_ZTITLE",  # 국내통계(주제별)
    "MT_OTITLE",  # 국내통계(기관별)
    "MT_GTITLE01",  # e-지방지표(주제별)
    "MT_GTITLE02",  # e-지방지표(지역별)
    "MT_RTITLE",  # 국제통계(국제통계연감·OECD·세계)
    "MT_BUKHAN",  # 북한통계
    "MT_TM1_TITLE",  # 대상별
    "MT_TM2_TITLE",  # 이슈별
    "MT_CHOSUN_TITLE",  # 광복이전통계
    "MT_HANKUK_TITLE",  # 대한민국통계연감
    "MT_STOP_TITLE",  # 작성중지
    "MT_ETITLE",  # 영문KOSIS
]

OUT_DIR = Path(__file__).parent / "crawl_output"
LEAVES_PATH = OUT_DIR / "tables.jsonl"
CHECKPOINT_PATH = OUT_DIR / "checkpoint.json"

REQUEST_DELAY_SEC = 0.3
MAX_RETRIES = 3
CHECKPOINT_EVERY = 20  # 이 횟수만큼 노드 처리할 때마다 체크포인트 저장


def _load_checkpoint() -> tuple[list[tuple[str, Optional[str]]], set[str]]:
    """체크포인트 파일에서 (남은 큐, 방문한 노드 집합)을 복원한다.
    없으면 처음부터 시작 — 큐를 모든 vwCd의 최상위(parentListId=None)로 초기화."""
    if CHECKPOINT_PATH.exists():
        data = json.loads(CHECKPOINT_PATH.read_text(encoding="utf-8"))
        queue = [tuple(x) for x in data["queue"]]
        visited = set(data["visited"])
        print(f"[재개] 체크포인트에서 이어서 시작 — 남은 큐 {len(queue)}건, 방문 완료 {len(visited)}건")
        return queue, visited
    queue = [(vw, None) for vw in ROOT_VW_CODES]
    return queue, set()


def _save_checkpoint(queue: list[tuple[str, Optional[str]]], visited: set[str]) -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    CHECKPOINT_PATH.write_text(
        json.dumps({"queue": list(queue), "visited": list(visited)}, ensure_ascii=False),
        encoding="utf-8",
    )


def _fetch_list(vw_cd: str, parent_list_id: Optional[str], api_key: str) -> list[dict]:
    params = {"method": "getList", "vwCd": vw_cd, "format": "json", "jsonVD": "Y", "apiKey": api_key}
    if parent_list_id:
        params["parentListId"] = parent_list_id

    for attempt in range(1, MAX_RETRIES + 1):
        try:
            resp = requests.get(LIST_URL, params=params, timeout=15)
            resp.raise_for_status()
            data = json.loads(resp.text)
            if isinstance(data, dict):
                # 자식이 없는 노드(진짜 리프)는 빈 배열이 아니라 에러성 dict를 반환하는
                # 경우가 실측 확인됨(예: {"err": "...", "errMsg": "조회할 데이터가 없습니다."})
                # — 이건 크롤링 실패가 아니라 "여기가 끝"이라는 신호이므로 빈 목록으로 처리.
                return []
            return data
        except (requests.RequestException, json.JSONDecodeError) as e:
            wait = 2 ** attempt
            print(f"  [재시도 {attempt}/{MAX_RETRIES}] {vw_cd}/{parent_list_id}: {e} -> {wait}초 대기")
            time.sleep(wait)
    print(f"  [실패, 포기] {vw_cd}/{parent_list_id} — {MAX_RETRIES}회 재시도 후 스킵")
    return []


def _write_leaves(leaves: list[dict], seen_tbl_ids: set[str], out_f) -> int:
    written = 0
    for item in leaves:
        tbl_id = item.get("TBL_ID")
        if not tbl_id or tbl_id in seen_tbl_ids:
            continue
        seen_tbl_ids.add(tbl_id)
        out_f.write(json.dumps(item, ensure_ascii=False) + "\n")
        written += 1
    return written


def main() -> None:
    api_key = os.environ.get("KOSIS_API_KEY")
    if not api_key:
        raise SystemExit("KOSIS_API_KEY가 없습니다. .env에 설정하세요.")

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    queue, visited = _load_checkpoint()

    # 이미 수집된 TBL_ID는 재기록하지 않도록 기존 파일에서 미리 읽어둔다.
    seen_tbl_ids: set[str] = set()
    if LEAVES_PATH.exists():
        with open(LEAVES_PATH, encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if line:
                    seen_tbl_ids.add(json.loads(line)["TBL_ID"])
        print(f"[재개] 기존에 수집된 표 {len(seen_tbl_ids)}개 확인")

    processed_since_checkpoint = 0
    total_leaves = len(seen_tbl_ids)

    # buffering=1(line-buffered): 프로세스가 중간에 죽어도(SIGTERM/Ctrl-C) 이미 쓴 줄이
    # 버퍼에 남아있다 유실되지 않고 바로바로 디스크에 반영되게 한다 — 20초 스모크
    # 테스트에서 기본 버퍼링으로는 파일이 비어있던 문제 실측 확인(2026-08-14).
    with open(LEAVES_PATH, "a", encoding="utf-8", buffering=1) as out_f:
        while queue:
            # DFS(스택처럼 끝에서 pop) — 순수 BFS로 앞에서 pop하면, 리프(진짜 표)가 5~7단계
            # 밑에 있어서 얕은 단계 카테고리를 전부 다 훑기 전까진 표가 하나도 안 나온다
            # (실측: 20초 스모크 테스트에서 리프 0개, 체크포인트도 안 찍힘 — 얕은 단계
            # 노드 수가 너무 많아서). DFS로 한 가지를 끝까지 파고들면 훨씬 빨리 실제
            # 데이터가 쌓이기 시작하고, 중간에 끊겨도 최소한 일부 분기는 완결된 상태로
            # 남는다. 최종적으로 전체 노드를 다 도는 건 BFS든 DFS든 동일하다.
            vw_cd, parent_list_id = queue.pop()
            node_key = f"{vw_cd}::{parent_list_id}"
            if node_key in visited:
                continue

            items = _fetch_list(vw_cd, parent_list_id, api_key)
            visited.add(node_key)
            time.sleep(REQUEST_DELAY_SEC)

            leaves = [it for it in items if "TBL_ID" in it]
            categories = [it for it in items if "LIST_ID" in it and "TBL_ID" not in it]

            written = _write_leaves(leaves, seen_tbl_ids, out_f)
            total_leaves += written

            for cat in categories:
                child_key = f"{vw_cd}::{cat['LIST_ID']}"
                if child_key not in visited:
                    queue.append((vw_cd, cat["LIST_ID"]))

            processed_since_checkpoint += 1
            if processed_since_checkpoint >= CHECKPOINT_EVERY:
                out_f.flush()
                _save_checkpoint(queue, visited)
                print(
                    f"[진행] 방문 {len(visited)}건 | 큐에 남음 {len(queue)}건 | "
                    f"수집된 표 {total_leaves}개 (누적)"
                )
                processed_since_checkpoint = 0

    _save_checkpoint(queue, visited)
    print(f"\n[완료] 총 방문 노드 {len(visited)}건, 수집된 표 {total_leaves}개 -> {LEAVES_PATH}")


if __name__ == "__main__":
    main()
