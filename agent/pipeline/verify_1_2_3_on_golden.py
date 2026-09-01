"""
agent/pipeline/verify_1_2_3_on_golden.py — 1→2→3단계 전체를 골든셋 기준으로 실행·채점

agent/preprocessing/verify_classifier_on_golden.py(1단계)와
agent/preprocessing/verify_claim_extractor_on_golden.py(2단계)를 하나의 파이프라인으로
이어붙이고, 거기에 3단계(표 매핑)까지 실제로 실행해서 채점한다. 즉 정답 문장을 미리
주지 않고 원본 기사부터 시작해서, 우리 classify() → extract_claims() → search_and_rerank()를
그대로 실행한 결과를 골든셋과 비교한다 (기존 score_golden_matching.py는 이미 저장된
DB를 재사용했는데, 이번엔 병합된 팀 최신 코드로 처음부터 다시 실행해야 해서 새로 만듦).

- 1단계: article_id 단위로 claim 유무 = 정답 True/False (verify_classifier_on_golden.py와 동일 기준)
- 2단계: gold claim_sentence를 추출 결과와 매칭 (verify_claim_extractor_on_golden.py의
  문장매칭 + 숫자재매칭 로직 재사용)
- 3단계: 매칭된 gold claim에 한해 search_and_rerank() top-1 table_id를 gold_table_id와 비교
  (agent/mapping/golden_set.py의 TABLE_ID_OVERRIDES 적용, match_status가 "매칭 실패"/"미완료"인
  건 3단계 채점에서 제외 — 애초에 KOSIS에 정답이 없는 케이스)

사용법 (프로젝트 루트에서):
    python -m agent.pipeline.verify_1_2_3_on_golden
"""

from __future__ import annotations

import socket
import time
from pathlib import Path

# HF Hub CDN(CloudFront)으로의 연결이 SSL 소켓 connect() 단계에서 무한 대기하는 현상을
# 실측으로 확인함(2026-08-05). curl로 직접 테스트한 결과 IPv4로는 0.07초만에 연결되는데
# IPv6로는 아예 연결이 안 됨(로컬 네트워크의 IPv6 라우팅 문제로 추정) — macOS가 IPv6를
# 먼저 시도하다가 거기서 멈추는 것으로 보인다. socket.getaddrinfo가 IPv4 주소만 반환하도록
# 패치해서 이 문제를 우회한다. socket.setdefaulttimeout은 방어적으로 유지(원인이 다른
# 네트워크 지연이더라도 최소한 타임아웃 예외로 끝나게).
_original_getaddrinfo = socket.getaddrinfo


def _getaddrinfo_ipv4_only(host, port, family=0, type=0, proto=0, flags=0):
    return _original_getaddrinfo(host, port, socket.AF_INET, type, proto, flags)


socket.getaddrinfo = _getaddrinfo_ipv4_only
socket.setdefaulttimeout(30)

import pandas as pd

from agent.mapping.embedding_search import build_table_embedding_cache, embedding_search
from agent.mapping.golden_set import TABLE_ID_OVERRIDES, _NO_ANSWER_STATUSES
from agent.mapping.keyword_search import keyword_search
from agent.mapping.reranker import load_document_texts, search_and_rerank
from agent.pipeline.batch_runner import _clean_scraped_article_text
from agent.preprocessing.claim_extractor import ClaimExtractorError, extract_claims
from agent.preprocessing.classifier import ClassifierError, classify
from agent.preprocessing.verify_claim_extractor_on_golden import (
    _extract_distinctive_numbers,
    _has_digit,
    _match_gold_to_extracted,
    _rematch_by_number_overlap,
)

NOTEBOOKS_DIR = Path(__file__).parent.parent.parent / "notebooks"
CLAIMS_XLSX = NOTEBOOKS_DIR / "추출 골든셋 단위 분리.xlsx"
MAPPING_XLSX = NOTEBOOKS_DIR / "매핑 골든셋 ord 추가.xlsx"
DATA_CSV = Path(__file__).parent.parent.parent / "data" / "data_set.csv"

# verify_classifier_on_golden.py / verify_claim_extractor_on_golden.py와 동일한 이유로 제외
# (article_id=A001: 서울 인구/배스킨라빈스 두 기사가 같은 article_url을 공유하는 원본 오타).
KNOWN_BAD_ARTICLE_IDS = {"A001"}

CLEAN_MAX_LEN = 3000  # verify_claim_extractor_on_golden.py 실측 결과 채택된 값

MAX_RETRIES = 4
RETRY_WAIT_SECONDS = (5, 10, 15, 20)
DELAY_BETWEEN_CALLS = 1.2


def _normalize_url(url: object) -> str:
    return str(url).strip().rstrip("/")


def build_gold() -> pd.DataFrame:
    claims = pd.read_excel(CLAIMS_XLSX)
    claims = claims[~claims["article_id"].isin(KNOWN_BAD_ARTICLE_IDS)]
    mapping = pd.read_excel(MAPPING_XLSX)[["claim_id", "kosis_table_id", "match_status"]]

    rows = []
    for article_id, g in claims.groupby("article_id"):
        has_claim = g["claim_sentence"].notna().any()
        rep = g[g["claim_sentence"].notna()].iloc[0] if has_claim else g.iloc[0]

        gold_claims = []
        if has_claim:
            for _, row in g[g["claim_sentence"].notna()].iterrows():
                m = mapping[mapping["claim_id"] == row["claim_id"]]
                status = m.iloc[0]["match_status"] if len(m) else None
                raw_table = m.iloc[0]["kosis_table_id"] if len(m) else None
                gold_table = (
                    TABLE_ID_OVERRIDES.get(raw_table, raw_table) if pd.notna(raw_table) else None
                )
                gold_claims.append(
                    {
                        "claim_id": row["claim_id"],
                        "sentence": row["claim_sentence"],
                        "gold_table_id": gold_table,
                        "match_status": status,
                    }
                )

        rows.append(
            {
                "article_id": article_id,
                "article_title": rep["article_title"],
                "article_url": rep["article_url"],
                "gold_label": bool(has_claim),
                "gold_claims": gold_claims,
            }
        )
    return pd.DataFrame(rows)


def _call_with_retry(fn, *args):
    last_err = None
    for attempt in range(MAX_RETRIES):
        try:
            return fn(*args)
        except Exception as e:  # noqa: BLE001 - 점검 스크립트, 계속 진행
            last_err = e
            if attempt < MAX_RETRIES - 1:
                time.sleep(RETRY_WAIT_SECONDS[attempt])
    raise last_err  # type: ignore[misc]


def main() -> None:
    gold = build_gold()

    data = pd.read_csv(DATA_CSV)
    data["_url_norm"] = data["URL"].map(_normalize_url)
    gold["_url_norm"] = gold["article_url"].map(_normalize_url)

    merged = gold.merge(data[["_url_norm", "기사 본문 전체"]], on="_url_norm", how="left")
    missing_body = merged[merged["기사 본문 전체"].isna()]
    merged = merged.dropna(subset=["기사 본문 전체"]).reset_index(drop=True)

    print(f"검증 대상 {len(merged)}개 기사 (본문 못 찾음 {len(missing_body)}건 제외)")

    print("표 임베딩 캐시/문서 로딩 중...")
    embedding_cache = build_table_embedding_cache()
    document_texts = load_document_texts()

    # ---- 집계 변수 ----
    n1_total = n1_correct = 0
    n2_gold_total = n2_found_total = 0
    n3_total = n3_correct = 0  # 전체 대비 (매칭 실패한 gold claim 포함)
    n3_extracted_denom = n3_extracted_correct = 0  # 2단계가 찾은 것만(순수 3단계)

    stage1_mismatches = []
    stage3_mismatches = []
    failed = []

    for i, row in merged.iterrows():
        title = row["article_title"]
        body_raw = row["기사 본문 전체"]

        # ---- 1단계 ----
        try:
            cls_result = _call_with_retry(classify, body_raw)
        except Exception as e:  # noqa: BLE001
            print(f"[{i + 1}/{len(merged)}] [1단계 FAIL] {title[:40]} -> {e}")
            failed.append(title)
            time.sleep(DELAY_BETWEEN_CALLS)
            continue

        n1_total += 1
        pred_label = cls_result.label
        ok1 = pred_label == row["gold_label"]
        if ok1:
            n1_correct += 1
        else:
            stage1_mismatches.append((title, row["gold_label"], pred_label, cls_result.score, cls_result.reason))

        tag1 = "OK" if ok1 else "MISMATCH"
        print(
            f"[{i + 1}/{len(merged)}] [1단계:{tag1}] gold={row['gold_label']} pred={pred_label} "
            f"score={cls_result.score:.2f} | {title[:40]}"
        )
        time.sleep(DELAY_BETWEEN_CALLS)

        # 골든셋 정답이 False(claim 없음)인 기사는 2·3단계 채점 대상이 아님 (비교할 gold claim이 없음)
        if not row["gold_label"]:
            continue

        # ---- 2단계 ----
        body_clean = _clean_scraped_article_text(title, body_raw, max_len=CLEAN_MAX_LEN)
        try:
            claims = _call_with_retry(extract_claims, body_clean)
        except Exception as e:  # noqa: BLE001
            print(f"    [2단계 FAIL] -> {e}")
            failed.append(title)
            time.sleep(DELAY_BETWEEN_CALLS)
            continue

        extracted_sentences = [c.sentence for c in claims]
        matched_extracted: dict[str, object] = {}  # extracted_sentence -> Claim
        gold_to_extracted: dict[str, object] = {}  # claim_id -> Claim (매칭된 것만)

        gold_claims = row["gold_claims"]
        missed_gold_ids = []
        for gc in gold_claims:
            found = _match_gold_to_extracted(gc["sentence"], extracted_sentences)
            if found:
                claim_obj = next(c for c in claims if c.sentence == found)
                matched_extracted[found] = claim_obj
                gold_to_extracted[gc["claim_id"]] = claim_obj
            else:
                missed_gold_ids.append(gc["claim_id"])

        # 숫자 재매칭 보정 (verify_claim_extractor_on_golden.py와 동일 로직)
        still_missed = []
        for cid in missed_gold_ids:
            gc = next(g for g in gold_claims if g["claim_id"] == cid)
            gs_nums = _extract_distinctive_numbers(gc["sentence"])
            hit = None
            if gs_nums:
                for c in claims:
                    if c.sentence in matched_extracted:
                        continue
                    if gs_nums & _extract_distinctive_numbers(c.sentence):
                        hit = c
                        break
            if hit:
                matched_extracted[hit.sentence] = hit
                gold_to_extracted[cid] = hit
            else:
                still_missed.append(cid)

        n2_gold_total += len(gold_claims)
        n2_found_total += len(gold_to_extracted)
        print(
            f"    [2단계] gold={len(gold_claims)} 발견={len(gold_to_extracted)} 추출총={len(claims)}"
        )
        time.sleep(DELAY_BETWEEN_CALLS)

        # ---- 3단계: 2단계가 찾아준 claim에 한해 표 매핑 실행 ----
        for gc in gold_claims:
            status = gc["match_status"]
            if status in _NO_ANSWER_STATUSES:
                continue  # KOSIS에 애초에 정답 없는 케이스, 3단계 채점 제외

            n3_total += 1
            matched_claim = gold_to_extracted.get(gc["claim_id"])
            if matched_claim is None:
                # 2단계가 못 찾았으니 3단계도 당연히 실패 처리 (전체 대비 정확도에 포함)
                continue

            try:
                candidates = _call_with_retry(
                    lambda: search_and_rerank(
                        matched_claim,
                        keyword_fn=keyword_search,
                        embedding_fn=lambda c: embedding_search(c, cache=embedding_cache),
                        top_k=3,
                        document_texts=document_texts,
                    )
                )
            except Exception as e:  # noqa: BLE001
                print(f"    [3단계 FAIL] {gc['claim_id']} -> {e}")
                continue

            n3_extracted_denom += 1
            pred_table_id = candidates[0].table_id if candidates else None
            ok3 = pred_table_id == gc["gold_table_id"]
            if ok3:
                n3_correct += 1
                n3_extracted_correct += 1
            else:
                stage3_mismatches.append(
                    (title, gc["claim_id"], gc["sentence"][:50], gc["gold_table_id"], pred_table_id)
                )
            time.sleep(DELAY_BETWEEN_CALLS)

    # ---- 결과 요약 ----
    print(f"\n{'=' * 70}")
    print("=== 1~3단계 전체 파이프라인 골든셋 채점 결과 ===")
    print(f"{'=' * 70}")
    print(f"실패(API 등): {len(failed)}건")

    print(f"\n[1단계] 정확도: {n1_correct}/{n1_total} ({n1_correct / n1_total:.1%})")
    for title, gold, pred, score, reason in stage1_mismatches:
        print(f"  불일치: gold={gold} pred={pred} score={score:.2f} | {title[:40]}")
        print(f"    근거: {reason}")

    if n2_gold_total:
        print(f"\n[2단계] recall: {n2_found_total}/{n2_gold_total} ({n2_found_total / n2_gold_total:.1%})")

    if n3_total:
        print(f"\n[3단계] 표 매핑 정확도 (전체 대비, 2단계 누락 포함): "
              f"{n3_correct}/{n3_total} ({n3_correct / n3_total:.1%})")
    if n3_extracted_denom:
        print(f"[3단계] 표 매핑 정확도 (2단계가 찾은 것만, 순수 3단계): "
              f"{n3_extracted_correct}/{n3_extracted_denom} ({n3_extracted_correct / n3_extracted_denom:.1%})")
        for title, cid, sent, gold_t, pred_t in stage3_mismatches:
            print(f"  불일치: [{cid}] gold={gold_t} pred={pred_t} | {sent} ({title[:30]})")


if __name__ == "__main__":
    main()
