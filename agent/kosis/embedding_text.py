"""agent/kosis/embedding_text.py -- KOSIS 표의 embedding_text 생성 규칙(단일 원본).

원래 benchmark/metadata_embedding_experiment.py 안에 있었는데, 운영 워커
(reembed_worker.py)가 벤치마크 스크립트를 import하는 역방향 의존이 생겨
순환 import가 났다(2026-08-27). 텍스트 규칙 자체는 실험이 아니라 운영 규격이므로
여기로 옮기고, 벤치마크 쪽이 이 모듈을 가져다 쓰도록 방향을 바로잡았다.

mode 설명:
  baseline               기관명 + 통계표명 (재구축 이전의 옛 포맷)
  item                   + 항목
  item_axis              + 분류축
  item_axis_value_full   + 분류값 전체
  item_axis_value_capped + 분류값 최대 50개  <- 운영 채택 포맷
  full_metadata          + 조사명

item_axis_value_capped을 쓰는 이유: 70건 골든셋 실측에서 Candidate Recall 40.0% ->
52.9%, Recall@10 27.1% -> 34.3%로 개선됐다. 분류값을 전부 넣으면(full) 표당 수백 개까지
가서 임베딩이 희석되므로 50개에서 자른다.
"""
from __future__ import annotations

# 분류값 상한. 실험으로 확정된 값이라 임의로 바꾸지 않는다.
VALUE_CAP = 50


def build_experimental_text(meta: dict, mode: str, survey_name: str | None) -> str:
    lines = []
    if meta["institution_name"]:
        lines.append(f"기관명: {meta['institution_name']}")
    lines.append(f"통계표명: {meta['table_name']}")

    if mode == "baseline":
        return "\n\n".join(lines)

    if mode in ("item", "item_axis", "item_axis_value_full", "item_axis_value_capped", "full_metadata"):
        if meta["items"]:
            lines.append("항목: " + ", ".join(meta["items"]))

    if mode in ("item_axis", "item_axis_value_full", "item_axis_value_capped", "full_metadata"):
        if meta["axes"]:
            lines.append("분류축: " + ", ".join(meta["axes"]))

    if mode == "item_axis_value_full":
        if meta["values_dedup"]:
            lines.append("분류값: " + ", ".join(meta["values_dedup"]))

    if mode in ("item_axis_value_capped", "full_metadata"):
        capped_values = meta["values_dedup"][:VALUE_CAP]
        if capped_values:
            lines.append("분류값: " + ", ".join(capped_values))

    if mode == "full_metadata" and survey_name:
        lines.append(f"조사명: {survey_name}")

    return "\n\n".join(lines)
