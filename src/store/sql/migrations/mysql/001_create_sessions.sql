CREATE TABLE IF NOT EXISTS sessions (
    id          CHAR(36) NOT NULL,
    title       TEXT NOT NULL,
    model       VARCHAR(255) NOT NULL,
    metadata    JSON NOT NULL DEFAULT ('{}'),
    created_at  DATETIME(6) NOT NULL DEFAULT CURRENT_TIMESTAMP(6),
    updated_at  DATETIME(6) NOT NULL DEFAULT CURRENT_TIMESTAMP(6) ON UPDATE CURRENT_TIMESTAMP(6),
    PRIMARY KEY (id),
    INDEX idx_sessions_updated (updated_at DESC)
) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4;
