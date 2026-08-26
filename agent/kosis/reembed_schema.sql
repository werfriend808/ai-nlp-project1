-- agent/kosis/reembed_schema.sql
-- KOSIS 전체 재임베딩(TABLE/ITEM/AXIS/AXIS VALUE) 신규 스키마.
-- 기존 production 테이블(kosis_vdb_tables 등, Supabase)은 별도 DB이므로 건드리지 않는다.
-- 이 스키마는 로컬 kosis_db(pgvector)에 신규 versioned 테이블로 생성한다.

create extension if not exists vector;
create extension if not exists pg_trgm;

-- ============================================================
-- TABLE
-- ============================================================
create table if not exists kosis_vdb_tables_qwen (
    table_id            text primary key,
    stat_id             text,
    org_id              text,
    table_name          text,
    institution_name    text,
    topic               text,
    classification      text,
    survey_name         text,
    description         text,
    unit                text,
    period_start        text,
    period_end          text,
    send_date           text,
    metadata_status     text not null default 'pending',
    embedding_text       text,
    embedding            vector(2560),
    embedding_model      text,
    embedding_dimension  int,
    created_at           timestamptz not null default now(),
    updated_at           timestamptz not null default now()
);

create index if not exists kosis_vdb_tables_qwen_text_trgm_idx
    on kosis_vdb_tables_qwen using gin (embedding_text gin_trgm_ops);

-- ============================================================
-- ITEM  (OBJ_ID == "ITEM" 레코드만)
-- ============================================================
create table if not exists kosis_vdb_items_qwen (
    id                    bigserial primary key,
    table_id              text not null references kosis_vdb_tables_qwen(table_id),
    item_id               text not null,
    item_name             text,
    normalized_item_name  text,
    parent_item_id        text,
    item_path             text,
    axis_id               text,
    metadata_status       text not null default 'pending',
    embedding_text        text,
    embedding             vector(2560),
    embedding_model       text,
    embedding_dimension   int,
    created_at            timestamptz not null default now(),
    unique (table_id, item_id)
);

create index if not exists kosis_vdb_items_qwen_text_trgm_idx
    on kosis_vdb_items_qwen using gin (embedding_text gin_trgm_ops);

-- ============================================================
-- AXIS
-- ============================================================
create table if not exists kosis_vdb_axes_qwen (
    id                   bigserial primary key,
    table_id             text not null references kosis_vdb_tables_qwen(table_id),
    axis_id              text not null,  -- KOSIS objL 축 번호 (예: "1","2") == obj_id
    axis_name            text,
    axis_order           int,
    axis_description      text,
    metadata_status      text not null default 'pending',
    embedding_text        text,
    embedding             vector(2560),
    embedding_model       text,
    embedding_dimension   int,
    created_at            timestamptz not null default now(),
    unique (table_id, axis_id)
);

create index if not exists kosis_vdb_axes_qwen_text_trgm_idx
    on kosis_vdb_axes_qwen using gin (embedding_text gin_trgm_ops);

-- ============================================================
-- AXIS VALUE (dense embedding 없음 — FTS/trigram/metadata 중심)
-- table_id를 같이 둔다: axis_id(축 번호)는 표마다 재사용되는 로컬 번호라 table_id 없이는
-- 어느 표의 축인지 특정할 수 없다 (AXIS와 조인하려면 반드시 필요).
-- ============================================================
create table if not exists kosis_vdb_axis_values_qwen (
    id                bigserial primary key,
    table_id          text not null references kosis_vdb_tables_qwen(table_id),
    axis_id           text not null,
    value_id          text not null,
    value_name        text,
    code              text,
    parent_value_id   text,
    hierarchy         text,
    metadata_status   text not null default 'pending',
    created_at        timestamptz not null default now(),
    unique (table_id, axis_id, value_id),
    foreign key (table_id, axis_id) references kosis_vdb_axes_qwen(table_id, axis_id)
);

create index if not exists kosis_vdb_axis_values_qwen_name_trgm_idx
    on kosis_vdb_axis_values_qwen using gin (value_name gin_trgm_ops);

-- ============================================================
-- Checkpoint / resume (section 14) — 서버별 partition 진행상황
-- ============================================================
create table if not exists kosis_reembed_checkpoint_qwen (
    table_id        text primary key,
    server_role     text not null,   -- SERVER_A / SERVER_B
    line_no         int not null,
    status          text not null default 'pending',  -- pending/processing/success/failed
    error_message   text,
    attempts        int not null default 0,
    updated_at      timestamptz not null default now()
);

create index if not exists kosis_reembed_checkpoint_qwen_status_idx
    on kosis_reembed_checkpoint_qwen (server_role, status);
