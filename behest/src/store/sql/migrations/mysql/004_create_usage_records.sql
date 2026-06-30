CREATE TABLE IF NOT EXISTS usage_records (
    id              CHAR(36) NOT NULL,
    session_id      CHAR(36) NOT NULL,
    message_id      CHAR(36) NOT NULL,
    provider        VARCHAR(255) NOT NULL,
    model           VARCHAR(255) NOT NULL,
    input_tokens    BIGINT NOT NULL DEFAULT 0,
    output_tokens   BIGINT NOT NULL DEFAULT 0,
    total_tokens    BIGINT NOT NULL DEFAULT 0,
    created_at      DATETIME(6) NOT NULL DEFAULT CURRENT_TIMESTAMP(6),
    PRIMARY KEY (id),
    INDEX idx_usage_records_session (session_id, created_at),
    CONSTRAINT fk_ur_session FOREIGN KEY (session_id) REFERENCES sessions(id) ON DELETE CASCADE
) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4;
