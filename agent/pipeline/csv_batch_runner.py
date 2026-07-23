# 1→2→3단계, 실제 CSV(data/data_set.csv) 대상 배치 실행기
"""
agent/pipeline/csv_batch_runner.py — 1~3단계 CSV 배치 처리 안정화

batch_runner.py(1~6단계, 하드코딩된 10개 시나리오)와는 별도로, 실제 회사 제공
CSV(data/data_set.csv, 기사제목/작성일/URL/기사 본문 전체/검색 구분 레이블)를 대상으로
1~3단계(classifier → claim_extractor → keyword/embedding search + rerank)만 순회한다.

확인하려는 것: 기사 1건당 HCX API 호출이 정확히 "1단계 1회 + (관련 기사면) 2단계 1회"로
유지되는지. 3단계(keyword_search/embedding_search/rerank)는 아직 실제 HCX 호출이 없으므로
집계 대상에서 제외한다. call_hcx를 카운팅 래퍼로 monkeypatch해서 기사마다 실제 호출 수를
재고, 예상(1 또는 2)을 벗어나면 그 기사를 CALL COUNT MISMATCH로 표시한다.

실행 (프로젝트 루트에서):
    python -m agent.pipeline.csv_batch_runner --limit 25
    python -m agent.pipeline.csv_batch_runner --limit 25 --out out.json

사전 준비물: .env에 HCX_API_KEY 필요 (1·2단계가 실제로 호출함).

⚠️ 25건 샘플 실행에서 확인된 것 (2단계 claim_extractor.py에 반영 완료):
    - 장문 기사(1만3천자 이상)는 HCX-003 컨텍스트 초과(40003) → 2단계 호출 전
      MAX_CLAIM_EXTRACT_CHARS(4000자)로 잘라서 보냄.
    - 30초 타임아웃에 가끔 걸림 → hcx_client.call_hcx 기본 timeout 60초로 상향.
    - HCX가 JSON 문자열 구분자로 스마트 쿼트(“ ”)를 섞어 쓰거나, maxTokens에 걸려
      배열이 중간에 끊기는 경우가 있음 → claim_extractor에 스마트 쿼트 보정 +
      개별 객체 단위 구제(salvage) 파싱 추가, maxTokens도 1024→2048로 상향.
    - 남은 known issue(미해결): classifier.py도 드물게(25건 중 1건) 같은 유형의
      깨진 JSON(끝에 `}` 중복 등)을 응답할 수 있음 — 이번 작업 범위에서는 손대지
      않음. 전체 CSV 실행 시 1단계에서도 소수 기사가 파싱 실패로 스킵될 수 있다는
      점을 감안할 것.
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")

import pandas as pd

from agent.preprocessing import classifier as classifier_module
from agent.preprocessing import claim_extractor as claim_extractor_module
from agent.mapping.keyword_search import keyword_search
from agent.mapping.embedding_search import embedding_search, build_table_embedding_cache
from agent.mapping.reranker import search_and_rerank

CSV_PATH = Path(__file__).parent.parent.parent / "data" / "data_set.csv"

# HCX-003(2단계 claim_extractor, legacy v1 모델)은 few-shot 프롬프트 자체가 이미
# 컨텍스트를 상당히 차지해서, 기사 본문이 길면 "40003 Context length exceeded"로
# 400이 난다(실측: 25건 샘플 중 13000자 이상 기사는 전부 실패, 4400자는 통과).
# 여유를 두고 4000자에서 자른다 — 한국 뉴스는 두괄식이라 앞부분에 핵심 수치가
# 나오는 경우가 많아 완전한 손실은 아니지만, 뒷부분 주장은 놓칠 수 있는 임시 조치.
MAX_CLAIM_EXTRACT_CHARS = 4000


def _make_counting_call_hcx(real_call_hcx):
    """호출 횟수를 세는 call_hcx 래퍼. (원본 함수, 카운터 dict)를 반환한다."""
    counters = {"count": 0}

    def wrapped(*args, **kwargs):
        counters["count"] += 1
        return real_call_hcx(*args, **kwargs)

    return wrapped, counters


def load_articles(csv_path: Path, limit: int | None) -> pd.DataFrame:
    df = pd.read_csv(csv_path, encoding="utf-8")
    if limit is not None:
        df = df.head(limit)
    return df


def run_article(title: str, published_date, article_text: str, embedding_cache: dict) -> dict:
    result: dict = {"title": title, "date": str(published_date), "hcx_calls": 0}

    # --- 1단계: classifier (호출 1회 기대) ---
    original_call_hcx = classifier_module.call_hcx
    wrapped, counters = _make_counting_call_hcx(original_call_hcx)
    classifier_module.call_hcx = wrapped
    try:
        cls_result = classifier_module.classify(article_text)
    except Exception as e:
        result["error"] = f"[1단계] {type(e).__name__}: {e}"
        result["stage1_calls"] = counters["count"]
        return result
    finally:
        classifier_module.call_hcx = original_call_hcx

    stage1_calls = counters["count"]
    result["stage1"] = {
        "label": cls_result.label,
        "score": cls_result.score,
        "reason": cls_result.reason,
    }
    result["stage1_calls"] = stage1_calls
    result["hcx_calls"] += stage1_calls

    if not cls_result.label:
        return result

    # --- 2단계: claim_extractor (호출 1회 기대) ---
    claim_input = article_text
    if len(claim_input) > MAX_CLAIM_EXTRACT_CHARS:
        claim_input = claim_input[:MAX_CLAIM_EXTRACT_CHARS]
        result["stage2_truncated"] = {"original_length": len(article_text), "used_length": len(claim_input)}

    original_call_hcx2 = claim_extractor_module.call_hcx
    wrapped2, counters2 = _make_counting_call_hcx(original_call_hcx2)
    claim_extractor_module.call_hcx = wrapped2
    try:
        claims = claim_extractor_module.extract_claims(claim_input)
    except Exception as e:
        result["error"] = f"[2단계] {type(e).__name__}: {e}"
        result["stage2_calls"] = counters2["count"]
        result["hcx_calls"] += counters2["count"]
        return result
    finally:
        claim_extractor_module.call_hcx = original_call_hcx2

    stage2_calls = counters2["count"]
    result["stage2_calls"] = stage2_calls
    result["hcx_calls"] += stage2_calls
    result["claims"] = [
        {"sentence": c.sentence, "claim_type": c.claim_type} for c in claims
    ]

    # --- 3단계: keyword_search + embedding_search + rerank (HCX 호출 없음) ---
    stage3 = []
    for claim in claims:
        try:
            candidates = search_and_rerank(
                claim,
                keyword_fn=keyword_search,
                embedding_fn=lambda c: embedding_search(c, cache=embedding_cache),
            )
        except Exception as e:
            stage3.append({"claim_sentence": claim.sentence, "error": f"{type(e).__name__}: {e}"})
            continue

        stage3.append({
            "claim_sentence": claim.sentence,
            "top_candidates": [
                {"table_id": c.table_id, "table_name": c.table_name, "score": round(c.score, 3)}
                for c in candidates[:3]
            ],
        })
    result["stage3"] = stage3

    return result


def main() -> None:
    parser = argparse.ArgumentParser(description="1~3단계 CSV 배치 처리 안정화 확인")
    parser.add_argument("--limit", type=int, default=25, help="처리할 기사 수 (기본 25, 전체는 --limit 0)")
    parser.add_argument("--out", type=str, default=None, help="결과 JSON 저장 경로")
    args = parser.parse_args()

    limit = None if args.limit == 0 else args.limit
    df = load_articles(CSV_PATH, limit)
    embedding_cache = build_table_embedding_cache()

    results = []
    mismatches = []
    total_calls = 0
    start = time.time()

    for i, row in df.iterrows():
        title = row["기사제목"]
        print(f"\n[{i + 1}/{len(df)}] {title}")

        res = run_article(title, row["작성일"], row["기사 본문 전체"], embedding_cache)
        results.append(res)
        total_calls += res["hcx_calls"]

        stage1_calls = res.get("stage1_calls", 0)
        stage2_calls = res.get("stage2_calls", 0)
        expected = 1 if "stage2_calls" not in res else 2
        actual = stage1_calls + stage2_calls
        if "error" not in res and actual != expected:
            mismatches.append({"title": title, "expected": expected, "actual": actual})
            print(f"  ⚠️ 호출 수 불일치: 기대 {expected}회, 실제 {actual}회")
        elif "error" in res:
            print(f"  [오류] {res['error']}")
        else:
            trunc = res.get("stage2_truncated")
            trunc_note = f" (본문 {trunc['original_length']}자 → {trunc['used_length']}자로 잘라서 2단계 호출)" if trunc else ""
            print(f"  1단계 label={res['stage1']['label']} | 호출 {stage1_calls}+{stage2_calls}회{trunc_note}")

    elapsed = time.time() - start

    summary = {
        "total_articles": len(df),
        "total_hcx_calls": total_calls,
        "avg_calls_per_article": round(total_calls / len(df), 3) if len(df) else 0,
        "call_count_mismatches": mismatches,
        "elapsed_seconds": round(elapsed, 1),
    }

    print(f"\n{'=' * 60}")
    print(f"기사 {len(df)}건 처리 완료, 총 HCX 호출 {total_calls}회 "
          f"(기사당 평균 {summary['avg_calls_per_article']}회), {elapsed:.1f}초")
    if mismatches:
        print(f"⚠️ 호출 수 불일치 {len(mismatches)}건 발견 (요약 참고)")
    else:
        print("✅ 모든 기사에서 '기사당 1회 호출' 구조 확인됨 (1단계 1회 + 관련 기사면 2단계 1회)")

    if args.out:
        out_path = Path(args.out)
        out_path.write_text(
            json.dumps({"summary": summary, "results": results}, ensure_ascii=False, indent=2),
            encoding="utf-8",
        )
        print(f"결과 저장: {out_path}")


if __name__ == "__main__":
    main()
