"""SQLite / Turso-backed persistence for predictions and feedback.

Backends:
  Local:  plain sqlite3 at data/forensic.db (default)
  Turso:  libSQL over HTTP — set TURSO_DATABASE_URL + TURSO_AUTH_TOKEN

When Turso credentials are present (either env vars or Streamlit secrets),
all reads and writes go to the hosted database. Same schema, same API.

Schema:
  predictions: every scan the model runs (image/video/audio/text)
  feedback:    user audits attached to a prediction (correct / incorrect + true label)
"""

from __future__ import annotations

import json
import os
import sqlite3
from contextlib import contextmanager
from typing import Any, Iterable

DB_PATH = os.environ.get("FORENSIC_DB_PATH", "data/forensic.db")

SCHEMA = """
CREATE TABLE IF NOT EXISTS predictions (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    file_hash TEXT,
    file_name TEXT,
    file_type TEXT NOT NULL,
    model_prediction TEXT NOT NULL,
    confidence REAL,
    raw_result TEXT,
    model_version TEXT,
    created_at TEXT NOT NULL DEFAULT (datetime('now'))
);

CREATE TABLE IF NOT EXISTS feedback (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    prediction_id INTEGER REFERENCES predictions(id) ON DELETE CASCADE,
    file_hash TEXT,
    true_label TEXT NOT NULL,
    was_correct INTEGER NOT NULL,
    stored_media_path TEXT,
    notes TEXT,
    created_at TEXT NOT NULL DEFAULT (datetime('now'))
);

CREATE INDEX IF NOT EXISTS idx_predictions_hash ON predictions(file_hash);
CREATE INDEX IF NOT EXISTS idx_feedback_hash ON feedback(file_hash);
CREATE INDEX IF NOT EXISTS idx_feedback_label ON feedback(true_label);
"""


# ---------------------------------------------------------------------------
# Connection abstraction
# ---------------------------------------------------------------------------

def _turso_credentials() -> tuple[str | None, str | None]:
    """Return (url, token) preferring Streamlit secrets, then env vars."""
    url = os.getenv("TURSO_DATABASE_URL") or os.getenv("LIBSQL_URL")
    token = os.getenv("TURSO_AUTH_TOKEN") or os.getenv("LIBSQL_AUTH_TOKEN")
    try:
        import streamlit as st
        if hasattr(st, "secrets"):
            url = url or st.secrets.get("TURSO_DATABASE_URL")
            token = token or st.secrets.get("TURSO_AUTH_TOKEN")
    except Exception:
        pass
    return url, token


def is_turso() -> bool:
    url, token = _turso_credentials()
    return bool(url and token)


def _open_conn():
    """Open a connection to either Turso (libSQL) or local SQLite."""
    url, token = _turso_credentials()
    if url and token:
        import libsql_experimental as libsql
        # Remote-only mode: every query hits Turso directly. Simpler and durable.
        return libsql.connect(database=url, auth_token=token)

    parent = os.path.dirname(DB_PATH)
    if parent:
        os.makedirs(parent, exist_ok=True)
    return sqlite3.connect(DB_PATH)


@contextmanager
def get_conn():
    conn = _open_conn()
    try:
        yield conn
        conn.commit()
    finally:
        try:
            conn.close()
        except Exception:
            pass


def _rows_as_dicts(cursor) -> list[dict[str, Any]]:
    """Convert a cursor's results to a list of dicts using column names."""
    desc = cursor.description
    if not desc:
        return []
    cols = [d[0] for d in desc]
    return [dict(zip(cols, row)) for row in cursor.fetchall()]


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def init_db() -> None:
    with get_conn() as conn:
        conn.executescript(SCHEMA)


def log_prediction(
    file_type: str,
    model_prediction: str,
    confidence: float | None,
    raw_result: dict[str, Any] | str,
    file_hash: str | None = None,
    file_name: str | None = None,
    model_version: str | None = None,
) -> int:
    payload = json.dumps(raw_result) if not isinstance(raw_result, str) else raw_result
    with get_conn() as conn:
        cur = conn.execute(
            """
            INSERT INTO predictions
                (file_hash, file_name, file_type, model_prediction, confidence, raw_result, model_version)
            VALUES (?, ?, ?, ?, ?, ?, ?)
            """,
            (file_hash, file_name, file_type, model_prediction, confidence, payload, model_version),
        )
        return int(cur.lastrowid or 0)


def log_feedback(
    prediction_id: int | None,
    true_label: str,
    was_correct: bool,
    file_hash: str | None = None,
    stored_media_path: str | None = None,
    notes: str | None = None,
) -> int:
    with get_conn() as conn:
        cur = conn.execute(
            """
            INSERT INTO feedback
                (prediction_id, file_hash, true_label, was_correct, stored_media_path, notes)
            VALUES (?, ?, ?, ?, ?, ?)
            """,
            (prediction_id, file_hash, true_label, 1 if was_correct else 0, stored_media_path, notes),
        )
        return int(cur.lastrowid or 0)


def feedback_exists_for_hash(file_hash: str, true_label: str) -> bool:
    with get_conn() as conn:
        row = conn.execute(
            "SELECT 1 FROM feedback WHERE file_hash = ? AND true_label = ? LIMIT 1",
            (file_hash, true_label),
        ).fetchone()
    return row is not None


def stats() -> dict[str, Any]:
    with get_conn() as conn:
        total_preds = conn.execute("SELECT COUNT(*) FROM predictions").fetchone()[0]
        total_fb = conn.execute("SELECT COUNT(*) FROM feedback").fetchone()[0]
        accuracy_row = conn.execute("SELECT AVG(was_correct) FROM feedback").fetchone()
        per_type = {
            row[0]: row[1]
            for row in conn.execute(
                "SELECT file_type, COUNT(*) FROM predictions GROUP BY file_type"
            ).fetchall()
        }
        per_label = {
            row[0]: row[1]
            for row in conn.execute(
                "SELECT true_label, COUNT(*) FROM feedback GROUP BY true_label"
            ).fetchall()
        }
    return {
        "backend": "turso" if is_turso() else "sqlite-local",
        "total_predictions": total_preds,
        "total_feedback": total_fb,
        "audited_accuracy": round((accuracy_row[0] or 0) * 100, 2) if total_fb else None,
        "predictions_by_type": per_type,
        "feedback_by_label": per_label,
    }


def recent_predictions(limit: int = 20) -> list[dict[str, Any]]:
    with get_conn() as conn:
        cur = conn.execute(
            """
            SELECT p.id, p.file_name, p.file_type, p.model_prediction, p.confidence, p.created_at,
                   f.true_label, f.was_correct
            FROM predictions p
            LEFT JOIN feedback f ON f.prediction_id = p.id
            ORDER BY p.id DESC LIMIT ?
            """,
            (limit,),
        )
        return _rows_as_dicts(cur)


def training_samples() -> Iterable[dict[str, Any]]:
    """Yield image/video feedback samples that have a stored media path on disk."""
    with get_conn() as conn:
        cur = conn.execute(
            """
            SELECT f.id, f.true_label, f.stored_media_path, p.file_type
            FROM feedback f
            JOIN predictions p ON p.id = f.prediction_id
            WHERE f.stored_media_path IS NOT NULL
              AND p.file_type IN ('image', 'video')
            """
        )
        rows = _rows_as_dicts(cur)
    for d in rows:
        if d["stored_media_path"] and os.path.exists(d["stored_media_path"]):
            yield d


if __name__ == "__main__":
    init_db()
    s = stats()
    print(f"Backend: {s['backend']}")
    print(json.dumps(s, indent=2))
