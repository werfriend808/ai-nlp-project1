# Embedding 기반 Semantic Retrieval 전환 설계안

keyword_search 단독 매칭에서 embedding 기반 semantic retrieval(+keyword와의 hybrid)로
전환하기 위한 현황 분석과 설계. **이 문서는 설계 단계 산출물이며, 이번 라운드에서
table_catalog.json/table_params.json은 수정하지 않았다.**

---

## ① table_catalog.json 필드 역할 분석

| 필드 | 실제 소비 주체 | 역할 |
|---|---|---|
| `title` | `keyword_search.py`(제목 앞 6글자 일치 시 보너스 매칭), `embedding_text`의 첫 문장 | 공식 명칭. 사람이 표를 식별하는 1차 근거이자, keyword_search에서 유일하게 "매칭된 keyword 목록"과 별도로 취급되는 보너스 신호 |
| `description` | `embedding_text`에 포함되어 임베딩 입력으로만 소비됨 (코드가 직접 읽지 않음) | 사람이 표를 검증/구분하기 위한 자연어 설명. 기관명·API 검증값·다른 표와의 차이점·한계를 담음 |
| `keywords` | `keyword_search.py`의 유일한 실질 매칭 필드 (`_score_table`이 이 리스트만 순회) | substring 매칭 대상. `SYNONYMS` 사전을 통해 확장되기도 함 |
| `embedding_text` | `embedding_search.py`의 유일한 입력 (table 쪽 벡터화 대상) | `title. keywords(쉼표구분). description.` 규칙으로 생성되는 합성 텍스트 |
| `related_tblId` | 현재 코드가 읽지 않음 — 순수 사람 문서 | 키워드/개념이 겹치는 인접 표 사이의 선택 기준을 documentaion으로 남김. **retrieval 로직엔 관여하지 않지만, 향후 rerank 이후 후처리(disambiguation) 단계의 잠재 입력으로 쓸 수 있음** |

**핵심 관찰**: `keyword_search`와 `embedding_search`는 서로 다른 필드를 본다 — 키워드 검색은 `keywords`(+`title` 일부)만, 임베딩 검색은 `embedding_text`(title+keywords+description 합성) 전체를 본다. 즉 **description을 아무리 잘 써도 keyword_search 성능엔 영향이 없고, 오직 embedding_search 품질에만 기여한다.** 이번 전환 작업의 핵심 레버는 `description`/`embedding_text`다.

---

## ② embedding_text 품질 점검 결과 (54개 표 전수 확인)

실제로 스크립트로 길이 분포와 코드성 잡음을 확인했다.

- **길이 분포**: 최소 64자(`DT_2KAA809`) ~ 최대 505자(`DT_1L9U103`), 평균 154자.
- **짧은 표(< 80자) 10개**: `DT_2KAA809`, `DT_1G18007`, `DT_102006_001`, `DT_1F70011`, `DT_404Y014`, `DT_402Y014`, `DT_1B26006_A01`, `DT_102N_AD01`, `DT_311Y001`, `DT_1EA1019` — 전부 keyword 2~4개, description 1문장 수준. **임베딩 모델 입장에서 의미 신호가 상대적으로 빈약할 가능성이 있음** — keyword_search는 정확히 그 단어가 있어야만 매칭되지만, embedding_search는 "비슷한 뜻의 다른 표현"까지 잡아주는 게 강점인데, 원문 자체가 짧으면 그 강점이 덜 발휘됨.
- **코드성 잡음 포함 표 6개** (`objL1=`, `itmId=`, `C1=` 등이 embedding_text에 그대로 들어간 경우): `DT_1DA7001S`, `DT_1JH20202`, `DT_1C8015`, `DT_1L9U103`, `DT_1J22112`, `DT_1KE10071`. keyword_search에선 이런 코드가 안 매칭돼도 해가 없지만, **임베딩 모델에게는 의미 없는 토큰 잡음**이라 벡터의 중심을 흐릴 수 있음. 다만 전체 embedding_text 대비 비중이 작아(예: "objL1='1C'로 조회 가능" 한 문장 정도) 치명적이진 않아 보임.
- **중복 표현**: `title. keywords. description.` 규칙상 keywords가 description 안에서 다시 언급되는 경우가 흔함(예: "물가상승률"이 keywords에도, description 문장에도 등장) — 이건 설계상 의도된 반복이라 문제로 보지 않음(임베딩 모델은 반복 등장 자체를 오히려 강한 신호로 활용하는 경향이 있음).

**결론**: 전면 재작성이 필요할 정도로 심각한 문제는 없음. 다만 **하위 10개 표는 description을 1~2문장 더 늘려서(기관명 외에 "관련 용어"·"발표 주기 표현" 추가) 임베딩 입력을 보강할 가치가 있음** — 단, 이번 라운드는 "점검"까지만 지시받았으므로 실제 수정은 하지 않았고, 다음 라운드 작업 후보로 남긴다.

---

## ③ Embedding Retrieval 설계안

### 이미 구현되어 있는 부분 (재사용)

`embedding_search.py`/`reranker.py`에 이미 아래가 구현돼 있고, 이번 설계는 이를 그대로 채택한다 — 새로 설계할 필요가 없다:

- **모델**: Qwen3-Embedding-4B (50건 라벨링 비교실험에서 top-1 70%로 1위, notebooks/embedding_model_comparison.ipynb)
- **비대칭 인코딩**: query(claim 문장) 쪽에만 instruction 프리픽스(`"Given a Korean news claim sentence, retrieve..."`)를 붙이고, document(표) 쪽은 `embedding_text`를 그대로 인코딩 — Qwen3-Embedding 권장 사용법 그대로 반영됨
- **table embedding 생성 방식**: `build_table_embedding_cache()`가 배치로 1회 생성 후 `table_embeddings_cache.json`에 캐시. 모델명이 바뀌거나 표 개수가 달라지면 자동 무효화·재생성
- **유사도**: 코사인 유사도
- **하이브리드**: `search_and_rerank()` = `keyword_search` + `embedding_search` 후보를 table_id 기준 병합(`_merge_candidates`) → `rerank()`. 리랭커(Qwen3-Reranker-4B) 사용 가능하면 그 점수로 최종 정렬, 불가능하면 "keyword 검증 후보 우선 + 그 안에서 score 내림차순"이라는 항등 폴백으로 안전하게 전환

### query(claim) preprocessing

현재는 `claim.sentence` 원문을 그대로 인코딩한다. 제안: 전처리 자체는 최소화 유지 — Qwen3-Embedding 같은 최신 임베딩 모델은 원문 그대로 넣는 게 권장 사용법과 일치하고, 과도한 전처리(불용어 제거 등)가 오히려 성능을 깎는 사례가 많음. **다만 claim_extractor가 만든 구조화 필드(`period`, `unit`, `population`)를 문장 뒤에 부가 정보로 짧게 덧붙이는 실험은 해볼 가치가 있음** — 예: `"{sentence} [population: {population}]"`. 이건 실제 벤치마크로 효과를 확인한 뒤 채택 여부를 정해야 하는 실험 항목으로 남긴다(지금 확정하지 않음).

### hybrid scoring 방식 — 현재 구조 평가

현재 `_merge_candidates`의 설계 원칙("keyword 검증 후보는 신뢰, embedding-only 후보는 unverified로 표시해 리랭커가 재평가하게 함")은 **임베딩이 아직 로컬에서 검증 안 된 지금 상황에 맞는 안전한 설계**다. 실제 임베딩 모델이 정상 작동하기 시작하면(현재 `KOSIS_DISABLE_EMBEDDING=1`로 하드웨어 문제 때문에 꺼져있음), 다음을 재검토해야 한다:

- 지금은 embedding 점수(cosine) 크기를 keyword 점수와 직접 비교하지 않는다(스케일이 다르기 때문). 리랭커가 정상 작동하면 이 문제는 리랭커 점수로 흡수되므로 괜찮음.
- 리랭커가 꺼진 상태에서 embedding-only 후보가 keyword 후보보다 항상 밀리는 지금 설계는, **keyword가 전혀 못 찾고 embedding만 찾은 진짜 정답 표가 있어도 순위에서 밀릴 수 있다**는 의미 — 실제 임베딩 검증이 끝나기 전까지는 감수할 수밖에 없는 트레이드오프로 문서화해둔다.

---

## ④ keyword retrieval baseline 유지 확인

`table_catalog.json`/`table_params.json`을 변경하지 않았으므로 기존 회귀 결과가 그대로 유지된다.

## ⑤ Evaluation Pipeline 설계안 (claim_id 기반 gold set 연동)

### 현재 상태의 한계

`test_mapping.py`의 `TEST_CASES`는 `(문장, claim_type, 정답 tblId, 카테고리)` 튜플 12개로, **claim_id가 없다** — 결과가 매번 stdout에 출력만 되고 파일로 남지 않아, 모델을 바꾸기 전/후 결과를 비교하거나 실패 케이스를 추적하기 어렵다. 또한 12건이 29개 카테고리 중 7개만 커버해서, 카탈로그가 54개 표로 늘어난 지금은 대표성이 부족하다.

### 제안하는 Gold Set 스키마

```json
{
  "claim_id": "map-0007",
  "sentence": "지난달 청년 실업률이 6%에 육박했다",
  "claim_type": "규모",
  "expected_tblId": "DT_1DA7102S",
  "category": "고용/노동",
  "notes": "DT_1DA7001S(전체 실업률)와 혼동 주의 사례"
}
```

- `claim_id`: 안정적인 식별자(순번 또는 해시) — 결과를 run마다 diff하거나, 향후 Extraction Gold Set/Slot Filling Gold Set과 조인할 때 조인 키로 쓸 수 있게 미리 확보.
- 나머지 필드는 기존 `TEST_CASES` 구조를 그대로 계승.

### 평가 지표 제안

- **Top-1 accuracy** (현재 유지): keyword/embedding/rerank 각각.
- **Top-3, Top-5 accuracy 추가**: embedding은 top-1에서 약해도 top-3 안에 정답이 있으면 리랭커가 구제할 여지가 있음 — 이 구간을 분리해서 보지 않으면 "embedding이 쓸모없다"는 성급한 결론으로 이어질 위험이 있음(이전에 이미 한번 겪은 논쟁).
- **결과를 파일로 저장**: 현재처럼 stdout print만 하지 말고, `{claim_id, method, predicted_tblId, score, correct}` 레코드를 JSONL로 남겨서 모델 교체 전후 비교(regression diff)가 가능하게 함.

### 실행 방식 제안

`test_mapping.py`를 다음처럼 확장(제안, 이번 라운드에서 구현하지 않음):
1. `TEST_CASES`를 `gold_set_mapping.json`(claim_id 포함)으로 분리
2. 각 실행마다 `results_{timestamp}.jsonl`로 결과 저장
3. 카테고리 커버리지를 12→29개 전체로 확장 (현재 7개 카테고리만 커버 중인 gap 해소)

---

## 다음 단계 (이번 라운드에서 진행하지 않음)

1. 하위 10개 표의 description 보강 (실제 편집)
2. gold set을 `claim_id` 포함 스키마로 마이그레이션 + 카테고리 커버리지 29개로 확장
3. 실제 임베딩 모델(Qwen3-Embedding-4B)을 로컬/서버 환경에서 재시도 (하드웨어 이슈 재확인)
4. 임베딩이 정상 작동하면 hybrid scoring 재검토 (③ 참고)