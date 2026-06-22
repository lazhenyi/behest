CREATE TABLE IF NOT EXISTS messages (
    id          CHAR(36) NOT NULL,
    session_id  CHAR(36) NOT NULL,
    role        VARCHAR(20) NOT NULL,
    content     JSON NOT NULL DEFAULT ('[]'),
    tool_calls  JSON NOT NULL DEFAULT ('[]'),
    `usage`     JSON,
    created_at  DATETIME(6) NOT NULL DEFAULT CURRENT_TIMESTAMP(6),
    PRIMARY KEY (id),
    INDEX idx_messages_session (session_id, created_at),
    CONSTRAINT fk_messages_session FOREIGN KEY (session_id) REFERENCES sessions(id) ON DELETE CASCADE
) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4;
