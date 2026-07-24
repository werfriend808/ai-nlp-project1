import os
import sys
import json
import time
import requests
from dotenv import load_dotenv

load_dotenv(dotenv_path=os.path.join(os.getcwd(), ".env"))
KEY = os.environ["KOSIS_API_KEY"]
BASE = "https://kosis.kr/openapi/statisticsList.do"

MAX_PER_TOPIC = 200  # 주제별 크롤링 상한 (안전장치, 균형 있게 분배)
MAX_DEPTH = 8

TOPICS = {
    "A": "인구",
    "D": "노동",
    "P2": "물가",
    "Q": "국민계정",
    "S2": "무역ㆍ국제수지",
    "I1": "주거",
}


def fetch(parent_list_id: str):
    params = {
        "method": "getList",
        "apiKey": KEY,
        "vwCd": "MT_ZTITLE",
        "parentListId": parent_list_id,
        "format": "json",
        "jsonVD": "Y",
    }
    r = requests.get(BASE, params=params, timeout=15)
    try:
        data = r.json()
        return data if isinstance(data, list) else []
    except Exception:
        return []


def crawl(list_id: str, tables: list, visited: set, depth: int = 0):
    if depth > MAX_DEPTH or list_id in visited or len(tables) >= MAX_PER_TOPIC:
        return
    visited.add(list_id)
    data = fetch(list_id)
    for row in data:
        if len(tables) >= MAX_PER_TOPIC:
            return
        if "TBL_ID" in row:
            tables.append({"tblId": row["TBL_ID"], "title": row.get("TBL_NM", ""), "orgId": row.get("ORG_ID", "")})
        elif "LIST_ID" in row:
            crawl(row["LIST_ID"], tables, visited, depth + 1)


def main():
    results = {}
    t0 = time.time()
    for tid, name in TOPICS.items():
        tables = []
        visited = set()
        crawl(tid, tables, visited)
        results[tid] = tables
        elapsed = time.time() - t0
        print(f"{tid}({name}): {len(tables)}개 표  (누적 {elapsed:.0f}s)", flush=True)

    total = sum(len(v) for v in results.values())
    print(f"\n총 표 개수: {total}", flush=True)

    out_path = "notebooks/kosis_full_table_crawl_trial.json"
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(results, f, ensure_ascii=False, indent=1)
    print(f"저장 완료: {out_path}", flush=True)


if __name__ == "__main__":
    main()
