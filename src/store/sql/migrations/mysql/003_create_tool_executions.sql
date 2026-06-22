CREATE TABLE IF NOT EXISTS tool_executions (
    id          CHAR(36) NOT NULL,
    session_id  CHAR(36) NOT NULL,
    message_id  CHAR(36) NOT NULL,
    call_id     VARCHAR(255) NOT NULL,
    tool_name   VARCHAR(255) NOT NULL,
    arguments   JSON NOT NULL DEFAULT ('{}'),
    result      JSON,
    status      VARCHAR(20) NOT NULL DEFAULT 'pending',
    error       TEXT,
    duration_ms BIGINT NOT NULL DEFAULT 0,
    created_at  DATETIME(6) NOT NULL DEFAULT CURRENT_TIMESTAMP(6),
    PRIMARY KEY (id),
    INDEX idx_tool_executions_session (session_id, created_at),
    INDEX idx_tool_executions_message (message_id),
    CONSTRAINT fk_te_session FOREIGN KEY (session_id) REFERENCES sessions(id) ON DELETE CASCADE
) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4;
