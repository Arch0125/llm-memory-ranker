import json
import os
import sqlite3
import uuid

from .types import MemoryHit, MemoryRecord
from .utils import cosine_similarity, iso_timestamp, parse_timestamp, utc_now


class SQLiteMemoryStore:
    def __init__(self, path):
        self.path = path
        directory = os.path.dirname(path)
        if directory:
            os.makedirs(directory, exist_ok=True)
        self.conn = sqlite3.connect(path)
        self.conn.row_factory = sqlite3.Row
        self.conn.execute("PRAGMA foreign_keys = ON")
        self._ensure_schema()

    def close(self):
        self.conn.close()

    def _ensure_schema(self):
        schema_path = os.path.join(os.path.dirname(__file__), "schema.sql")
        with open(schema_path, "r", encoding="utf-8") as handle:
            self.conn.executescript(handle.read())
        self.conn.commit()

    def _record_from_row(self, row):
        return MemoryRecord(
            memory_id=row["memory_id"],
            user_id=row["user_id"],
            memory_type=row["type"],
            text=row["text"],
            source_turn_id=row["source_turn_id"],
            created_at=row["created_at"],
            last_accessed_at=row["last_accessed_at"],
            times_retrieved=row["times_retrieved"],
            importance=row["importance"],
            decay_score=row["decay_score"],
            status=row["status"],
            version_group_id=row["version_group_id"],
            metadata=json.loads(row["metadata_json"] or "{}"),
        )

    def add_memory(
        self,
        user_id,
        text,
        memory_type="ephemeral",
        importance=0.5,
        embedding=None,
        embedding_model=None,
        metadata=None,
        source_turn_id=None,
        version_group_id=None,
        status="active",
        memory_id=None,
        replace_version_group=True,
    ):
        memory_id = memory_id or str(uuid.uuid4())
        timestamp = iso_timestamp()
        metadata_json = json.dumps(metadata or {}, sort_keys=True)
        if version_group_id and replace_version_group:
            self.conn.execute(
                """
                UPDATE memory_item
                SET status = 'archived'
                WHERE user_id = ? AND version_group_id = ? AND status = 'active'
                """,
                (user_id, version_group_id),
            )
        self.conn.execute(
            """
            INSERT INTO memory_item (
                memory_id, user_id, type, text, source_turn_id, created_at, last_accessed_at,
                times_retrieved, importance, decay_score, status, version_group_id, metadata_json
            ) VALUES (?, ?, ?, ?, ?, ?, ?, 0, ?, 0.0, ?, ?, ?)
            """,
            (
                memory_id,
                user_id,
                memory_type,
                text,
                source_turn_id,
                timestamp,
                timestamp,
                importance,
                status,
                version_group_id,
                metadata_json,
            ),
        )
        if embedding is not None:
            self.upsert_embedding(memory_id, embedding_model, embedding)
        self.record_event(memory_id, "created", {"type": memory_type})
        self.conn.commit()
        return self.get_memory(memory_id)

    def upsert_embedding(self, memory_id, model_name, vector):
        if model_name is None:
            raise ValueError("embedding model_name is required when persisting embeddings")
        self.conn.execute(
            """
            INSERT INTO memory_embedding (memory_id, model_name, dim, vector_json, created_at)
            VALUES (?, ?, ?, ?, ?)
            ON CONFLICT(memory_id, model_name) DO UPDATE SET
                dim = excluded.dim,
                vector_json = excluded.vector_json,
                created_at = excluded.created_at
            """,
            (
                memory_id,
                model_name,
                len(vector),
                json.dumps(vector),
                iso_timestamp(),
            ),
        )

    def get_memory(self, memory_id):
        row = self.conn.execute(
            "SELECT * FROM memory_item WHERE memory_id = ?",
            (memory_id,),
        ).fetchone()
        if row is None:
            return None
        return self._record_from_row(row)

    def list_memories(self, user_id, status=None, limit=100):
        clauses = ["user_id = ?"]
        params = [user_id]
        if status is not None:
            clauses.append("status = ?")
            params.append(status)
        params.append(limit)
        rows = self.conn.execute(
            f"""
            SELECT * FROM memory_item
            WHERE {' AND '.join(clauses)}
            ORDER BY last_accessed_at DESC, created_at DESC
            LIMIT ?
            """,
            params,
        ).fetchall()
        return [self._record_from_row(row) for row in rows]

    def delete_memory(self, memory_id, user_id=None):
        if user_id is None:
            self.conn.execute("DELETE FROM memory_item WHERE memory_id = ?", (memory_id,))
        else:
            self.conn.execute(
                "DELETE FROM memory_item WHERE memory_id = ? AND user_id = ?",
                (memory_id, user_id),
            )
        self.conn.commit()

    def archive_memory(self, memory_id, user_id=None):
        params = [memory_id]
        query = "UPDATE memory_item SET status = 'archived' WHERE memory_id = ?"
        if user_id is not None:
            query += " AND user_id = ?"
            params.append(user_id)
        self.conn.execute(query, params)
        self.record_event(memory_id, "archived", {})
        self.conn.commit()

    def record_event(self, memory_id, event_type, event_meta):
        self.conn.execute(
            """
            INSERT INTO memory_event (event_id, memory_id, event_type, event_at, event_meta_json)
            VALUES (?, ?, ?, ?, ?)
            """,
            (
                str(uuid.uuid4()),
                memory_id,
                event_type,
                iso_timestamp(),
                json.dumps(event_meta or {}, sort_keys=True),
            ),
        )

    def mark_retrieved(self, memory_id):
        now = iso_timestamp()
        self.conn.execute(
            """
            UPDATE memory_item
            SET times_retrieved = times_retrieved + 1,
                last_accessed_at = ?
            WHERE memory_id = ?
            """,
            (now, memory_id),
        )
        self.record_event(memory_id, "retrieved", {})
        self.conn.commit()

    def search(
        self,
        vector,
        user_id,
        top_k=20,
        type_allowlist=None,
        status="active",
        model_name=None,
    ):
        if model_name is None:
            raise ValueError("search requires model_name to match stored embeddings")
        clauses = ["m.user_id = ?", "m.status = ?", "e.model_name = ?"]
        params = [user_id, status, model_name]
        if type_allowlist:
            placeholders = ", ".join("?" for _ in type_allowlist)
            clauses.append(f"m.type IN ({placeholders})")
            params.extend(type_allowlist)
        rows = self.conn.execute(
            f"""
            SELECT m.*, e.model_name, e.vector_json
            FROM memory_item m
            JOIN memory_embedding e ON e.memory_id = m.memory_id
            WHERE {' AND '.join(clauses)}
            """,
            params,
        ).fetchall()

        hits = []
        now = utc_now()
        for row in rows:
            embedding = json.loads(row["vector_json"])
            score = cosine_similarity(vector, embedding)
            record = self._record_from_row(row)
            age_days = max(0, (now - parse_timestamp(record.last_accessed_at)).days)
            hits.append(
                MemoryHit(
                    record=record,
                    score=score,
                    embedding_model=row["model_name"],
                    age_days=age_days,
                )
            )

        hits.sort(key=lambda hit: hit.score, reverse=True)
        return hits[:top_k]
