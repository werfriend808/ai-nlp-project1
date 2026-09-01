-- item_axis_value_capped 재임베딩(TABLE embedding_text/embedding만 UPDATE) 전용 체크포인트.
-- 기존 kosis_reembed_checkpoint_qwen(초기 적재용)과 별개 -- resume 가능, 서버 2대 분할.
create table if not exists kosis_reembed_v2_checkpoint_qwen (
    table_id text primary key,
    server_role text not null,       -- SERVER_A | SERVER_B
    status text not null default 'pending',  -- pending | processing | success | skipped | failed
    error_message text,
    updated_at timestamptz not null default now()
);
create index if not exists idx_reembed_v2_ckpt_role_status
    on kosis_reembed_v2_checkpoint_qwen (server_role, status);
