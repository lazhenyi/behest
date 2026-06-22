CREATE TABLE IF NOT EXISTS usage_records (
    id              UUID PRIMARY KEY,
    session_id      UUID NOT NULL REFERENCES sessions(id) ON DELETE CASCADE,
    message_id      UUID NOT NULL REFERENCES messages(id) ON DELETE CASCADE,
    provider        TEXT NOT NULL,
    model           TEXT NOT NULL,
    input_tokens    BIGINT NOT NULL DEFAULT 0,
    output_tokens   BIGINT NOT NULL DEFAULT 0,
    total_tokens    BIGINT NOT NULL DEFAULT 0,
    created_at      TIMESTAMPTZ NOT NULL DEFAULT NOW()
);
CREATE INDEX IF NOT EXISTS idx_usage_records_session ON usage_records(session_id, created_at);
