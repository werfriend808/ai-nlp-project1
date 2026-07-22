# B 담당(3단계) 모듈 단위 테스트 기록

- 대상: `agent/mapping/keyword_search.py`, `agent/mapping/embedding_search.py`, `agent/mapping/reranker.py`
- 결과: **12/12 PASS** (최종 판정 기준은 `reranker` top-1. 아래 "발견된 이슈" 참고)

## 실행 방법

프로젝트 루트에서 실행:

```
python -m agent.mapping.test_mapping
```

Windows 콘솔(cmd/PowerShell)에서 한글 결과가 깨져 보이면 UTF-8을 강제해서 실행:

```
# cmd
set PYTHONUTF8=1 && python -m agent.mapping.test_mapping

# PowerShell
$env:PYTHONUTF8=1; python -m agent.mapping.test_mapping
```

맨 아래 3줄(keyword/embedding/reranker top-1 정답률) 중 **reranker 줄이 최종 판정**이다. keyword/embedding 개별 줄은 왜 맞았는지/틀렸는지 진단용 참고 자료.
(`cd agent/mapping && python3 test_mapping.py`처럼 파일 경로로 직접 실행하면 다른 모듈 `__main__` 블록들과 실행 관례가 안 맞으므로 꼭 `-m agent.mapping.test_mapping` 형태로 실행할 것.)

## 케이스 목록 (7개 카테고리 전부 커버)

| # | 카테고리 | 문장 | 기대 table_id | keyword | embedding | rerank(최종) |
|---|---|---|---|---|---|---|
| 1 | 고용/노동 | 지난달 청년 실업률이 6%에 육박했다 | DT_1DA7001S | ✅ | ❌ | ✅ |
| 2 | 고용/노동 | 고용률이 역대 최고치를 기록했다 | DT_1DA7001S | ✅ | ✅ | ✅ |
| 3 | 고용/노동 | 취업자 수가 46개월 만에 감소 전환했다 | DT_1DA7001S | ✅ | ❌ | ✅ |
| 4 | 물가/CPI | 지난달 소비자물가가 전년 동월 대비 2.2% 올랐다 | DT_1J22003 | ✅ | ✅ | ✅ |
| 5 | 물가/CPI | 생활물가가 5개월 연속 올랐다 | DT_1J22003 | ✅ | ✅ | ✅ |
| 6 | 인구 | 전국 주민등록인구가 5000만명 아래로 떨어졌다 | DT_1B04005N | ✅ | ✅ | ✅ |
| 7 | 경제성장 | 한국 경제성장률이 3개 분기 연속 0%대에 머물렀다 | DT_200Y102 | ✅ | ❌ | ✅ |
| 8 | 무역/수출입 | 지난해 수출이 6838억달러로 역대 최대를 기록했다 | DT_1R11006_FRM101 | ✅ | ✅ | ✅ |
| 9 | 무역/수출입 | 무역수지가 3년 만에 흑자로 전환했다 | DT_1R11006_FRM101 | ✅ | ✅ | ✅ |
| 10 | 부동산/주택 | 전국 집값이 하락세를 보였다 | DT_30404_B012 | ✅ | ❌ | ✅ |
| 11 | 출생/사망/혼인 | 혼인 건수가 역대 최저를 기록했다 | DT_1B8000G | ❌ (없음) | ✅ | ✅ |
| 12 | 출생/사망/혼인 | 합계출산율이 0.7명대로 떨어졌다 | DT_1B8000G | ✅ | ❌ | ✅ |

- keyword_search top-1 정답률: 11/12
- embedding_search top-1 정답률: 7/12 (임베딩 API 미확정 폴백이라 낮게 나오는 게 정상 — 아래 이슈 1 참고)
- reranker(최종) top-1 정답률: 12/12

## 발견된 이슈

### 1. (설계상 정상) `embedding_search`는 아직 실제 임베딩 API가 없어 노이즈에 가깝다
`embed_texts()`가 실제 API 엔드포인트 확정 전까지 해시 기반 더미 벡터(글자 단위 bag-of-characters)로 폴백한다. 이 벡터는 모든 성분이 0 이상이라 코사인 유사도가 관련 여부와 무관하게 구조적으로 0.5~0.8대에 몰려서 나온다. 그래서 embedding 단독 정답률(7/12)은 낮은 게 정상이며, 실제 임베딩 API가 붙으면 `embed_texts()`의 TODO 부분만 교체하면 된다.

### 2. (수정 완료) `reranker.py` `__main__` import 경로 오류
`from keyword_search import keyword_search` 처럼 상대 모듈명으로 되어 있어 `python -m agent.mapping.reranker`로 실행하면 `ModuleNotFoundError`가 났음. `agent.mapping.keyword_search` 형태의 절대 경로로 수정.

### 3. (수정 완료) `_merge_candidates`가 원본 candidate 객체를 in-place로 변형함
같은 `table_id`가 keyword/embedding 양쪽에 있을 때 `cand.source_meta = ...`로 직접 대입해서, 호출자가 들고 있던 원본 리스트의 객체까지 조용히 바뀌는 부작용이 있었음. `dataclasses.replace`로 복사본을 만들도록 수정.

### 4. (수정 완료, 가장 중요) 스코어 스케일이 다른 두 점수를 크기로만 비교해서 노이즈가 정답을 밀어냄
`keyword_search`의 score(0/0.33/0.67/1.0, 확신도 기반)와 `embedding_search`의 더미 코사인 유사도(0.5~0.8대, 이슈 1의 이유로 구조적으로 높게 나옴)를 `rerank()`의 identity fallback이 그냥 크기로 비교해 정렬했음. 그 결과 keyword가 맞힌 정답이 embedding 노이즈에 밀려 최종 순위에서 사라지는 사례가 실제로 나왔음(예: "실업률" 케이스). `_merge_candidates`에서 embedding-only 후보를 `"(embedding-only, unverified)"`로 표시하고, `rerank()`의 identity fallback을 `(unverified, -score)` 기준으로 정렬해서 keyword로 검증된 후보가 항상 먼저 오도록 수정. 진짜 리랭커 API가 붙으면 이 폴백 자체가 안 쓰이므로 임시 조치.

### 5. (확인 완료, 미해결) `claim_type` 값이 `interfaces.py`의 `ClaimType` Literal과 어긋날 수 있음
Day2 가이드 PDF에는 claim_type 5종(규모/증감률/비교/전망/역대기록)이 언급되지만, `interfaces.py`의 `ClaimType = Literal["규모", "증감률", "비교", "전망"]`엔 "역대기록"이 없음. 테스트 케이스 11번("혼인 건수가 역대 최저를 기록했다")에 `claim_type="역대기록"`을 그대로 썼는데, dataclass는 런타임에 Literal을 검증하지 않아 에러는 안 나지만, A(`claim_extractor.py`)가 실제로 이 값을 뽑아내면 타입 체크에서 어긋날 수 있음. → A+B 연결 테스트 때 A 담당자와 확인 필요.

## 커버 안 된 것 (다음에 필요하면 추가)
- `SYNONYMS` 사전은 지금 테스트 문장에 맞춰 고른 표현만 커버함. 실제 기사/A의 `extract_claims()`가 뽑아내는 실제 문장은 사전에 없는 표현일 수 있어 재검증 필요 (→ A+B 연결 테스트에서 확인 예정).
- 실제 임베딩 API, 실제 리랭커 API 연동은 둘 다 TODO 상태 (멘토링에서 엔드포인트 확정 후 진행).
- `RerankerError`는 선언만 되어 있고 아직 어디서도 raise되지 않음 (실 API 연동 시 에러 처리에 연결 예정).
- A가 실제로 뽑은 `Claim`(진짜 HCX API 결과)을 B 파이프라인에 넣어보는 실전 연결 테스트는 아직 미실시.
