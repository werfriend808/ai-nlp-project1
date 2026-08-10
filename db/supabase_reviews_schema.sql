-- db/supabase_reviews_schema.sql
--
-- 프론트엔드 리뷰 대시보드용 Supabase 테이블 정의.
--
-- data/verifications.db(db/store.py)는 배치 파이프라인(1~8단계)의 산출물 원본이라
-- 여기서 건드리지 않는다. 사람 검증자가 AI 판정을 검토/수정한 기록만 이 reviews
-- 테이블에 별도로 쌓고, claim_id로 verifications.db의 result_id와 연결한다.
--
-- ai_verdict/ai_reason은 verifications.db 값을 그대로 복사해두는 캐시 컬럼이다 —
-- 조인 없이 프론트에서 "AI 판정 vs 검증자 판정"을 바로 나란히 보여주기 위함이며,
-- 진실의 원천(source of truth)은 여전히 verifications.db 쪽 원본이다.
--
-- 적용 (Supabase SQL Editor에서 실행):
--     이 파일 내용을 그대로 붙여넣고 실행.

create table if not exists reviews (
    id bigint generated always as identity primary key,
    claim_id text unique not null,           -- db/store.py의 result_id (article_title+claim_sentence md5 12자)
    ai_verdict text,                          -- verifications.verification_result 복사 (일치/불일치/판단불가)
    ai_reason text,                           -- verifications.evidence 복사 (AI 판정 근거)
    reviewer_verdict text,                    -- 검증자가 최종 확정한 판정 (AI와 같을 수도, 다를 수도 있음)
    reviewer_reason text,                     -- 검증자가 남기는 수정/확인 사유
    reviewer_name text,                       -- 검증한 사람 이름/닉네임
    created_at timestamptz not null default now()
);

comment on table reviews is 'AI 판정에 대한 사람 검증자의 리뷰 기록. claim_id로 verifications.db와 연결.';
comment on column reviews.claim_id is 'db/store.py make_result_id()가 만든 값과 동일 (재실행해도 안 바뀜)';
