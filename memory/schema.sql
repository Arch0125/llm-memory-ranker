PRAGMA foreign_keys = ON;

CREATE TABLE IF NOT EXISTS memory_item (
    memory_id TEXT PRIMARY KEY,
    user_id TEXT NOT NULL,
    type TEXT NOT NULL,
    text TEXT NOT NULL,
    source_turn_id TEXT,
    created_at TEXT NOT NULL,
    last_accessed_at TEXT NOT NULL,
    times_retrieved INTEGER NOT NULL DEFAULT 0,
    importance REAL NOT NULL DEFAULT 0.5,
    decay_score REAL NOT NULL DEFAULT 0.0,
    status TEXT NOT NULL DEFAULT 'active',
    version_group_id TEXT,
    metadata_json TEXT NOT NULL DEFAULT '{}'
);

CREATE INDEX IF NOT EXISTS idx_memory_item_user_status
    ON memory_item(user_id, status, type);

CREATE INDEX IF NOT EXISTS idx_memory_item_version_group
    ON memory_item(user_id, version_group_id);

CREATE TABLE IF NOT EXISTS memory_embedding (
    memory_id TEXT NOT NULL,
    model_name TEXT NOT NULL,
    dim INTEGER NOT NULL,
    vector_json TEXT NOT NULL,
    created_at TEXT NOT NULL,
    PRIMARY KEY(memory_id, model_name),
    FOREIGN KEY(memory_id) REFERENCES memory_item(memory_id) ON DELETE CASCADE
);

CREATE INDEX IF NOT EXISTS idx_memory_embedding_model
    ON memory_embedding(model_name);

CREATE TABLE IF NOT EXISTS memory_event (
    event_id TEXT PRIMARY KEY,
    memory_id TEXT NOT NULL,
    event_type TEXT NOT NULL,
    event_at TEXT NOT NULL,
    event_meta_json TEXT NOT NULL DEFAULT '{}',
    FOREIGN KEY(memory_id) REFERENCES memory_item(memory_id) ON DELETE CASCADE
);
