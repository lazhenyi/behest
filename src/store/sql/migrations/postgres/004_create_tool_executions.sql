CREATE TABLE IF NOT EXISTS tool_executions (
    id          UUID PRIMARY KEY,
    session_id  UUID NOT NULL REFERENCES sessions(id) ON DELETE CASCADE,
    message_id  UUID NOT NULL REFERENCES messages(id) ON DELETE CASCADE,
    call_id     TEXT NOT NULL,
    tool_name   TEXT NOT NULL,
    arguments   JSONB NOT NULL DEFAULT '{}',
    result      JSONB,
    status      TEXT NOT NULL DEFAULT 'pending',
    error       TEXT,
    duration_ms BIGINT NOT NULL DEFAULT 0,
    created_at  TIMESTAMPTZ NOT NULL DEFAULT NOW()
);
CREATE INDEX IF NOT EXISTS idx_tool_executions_session ON tool_executions(session_id, created_at);
CREATE INDEX IF NOT EXISTS idx_tool_executions_message ON tool_executions(message_id);
