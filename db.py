"""SQLite-backed persistence for predictions and feedback.

Schema:
  predictions: every scan the model runs (image/video/audio/text)
  feedback:    user audits attached to a prediction (correct / incorrect + true label)

A prediction row is created for every scan. A feedback row is created whenever the
user marks the verdict correct or incorrect. The stored_media_path on a feedback row
points at the persisted copy of the media used for retraining.
"""

from __future__ import annotations

import json
import os
import sqlite3
from contextlib import contextmanager
from datetime import datetime
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


def _ensure_parent(path: str) -> None:
    parent = os.path.dirname(path)
    if parent:
        os.makedirs(parent, exist_ok=True)


@contextmanager
def get_conn(db_path: str = DB_PATH):
    _ensure_parent(db_path)
    conn = sqlite3.connect(db_path)
    conn.row_factory = sqlite3.Row
    conn.execute("PRAGMA foreign_keys = ON")
    try:
        yield conn
        conn.commit()
    finally:
        conn.close()


def init_db(db_path: str = DB_PATH) -> None:
    with get_conn(db_path) as conn:
        conn.executescript(SCHEMA)


def log_prediction(
    file_type: str,
    model_prediction: str,
    confidence: float | None,
    raw_result: dict[str, Any] | str,
    file_hash: str | None = None,
    file_name: str | None = None,
    model_version: str | None = None,
    db_path: str = DB_PATH,
) -> int:
    """Insert a prediction row, return its id."""
    payload = json.dumps(raw_result) if not isinstance(raw_result, str) else raw_result
    with get_conn(db_path) as conn:
        cur = conn.execute(
            """
            INSERT INTO predictions
                (file_hash, file_name, file_type, model_prediction, confidence, raw_result, model_version)
            VALUES (?, ?, ?, ?, ?, ?, ?)
            """,
            (file_hash, file_name, file_type, model_prediction, confidence, payload, model_version),
        )
        return int(cur.lastrowid)


def log_feedback(
    prediction_id: int | None,
    true_label: str,
    was_correct: bool,
    file_hash: str | None = None,
    stored_media_path: str | None = None,
    notes: str | None = None,
    db_path: str = DB_PATH,
) -> int:
    with get_conn(db_path) as conn:
        cur = conn.execute(
            """
            INSERT INTO feedback
                (prediction_id, file_hash, true_label, was_correct, stored_media_path, notes)
            VALUES (?, ?, ?, ?, ?, ?)
            """,
            (prediction_id, file_hash, true_label, 1 if was_correct else 0, stored_media_path, notes),
        )
        return int(cur.lastrowid)


def feedback_exists_for_hash(file_hash: str, true_label: str, db_path: str = DB_PATH) -> bool:
    with get_conn(db_path) as conn:
        row = conn.execute(
            "SELECT 1 FROM feedback WHERE file_hash = ? AND true_label = ? LIMIT 1",
            (file_hash, true_label),
        ).fetchone()
    return row is not None


def stats(db_path: str = DB_PATH) -> dict[str, Any]:
    with get_conn(db_path) as conn:
        total_preds = conn.execute("SELECT COUNT(*) FROM predictions").fetchone()[0]
        total_fb = conn.execute("SELECT COUNT(*) FROM feedback").fetchone()[0]
        accuracy_row = conn.execute(
            "SELECT AVG(was_correct) FROM feedback"
        ).fetchone()
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
        "total_predictions": total_preds,
        "total_feedback": total_fb,
        "audited_accuracy": round((accuracy_row[0] or 0) * 100, 2) if total_fb else None,
        "predictions_by_type": per_type,
        "feedback_by_label": per_label,
    }


def recent_predictions(limit: int = 20, db_path: str = DB_PATH) -> list[dict[str, Any]]:
    with get_conn(db_path) as conn:
        rows = conn.execute(
            """
            SELECT p.id, p.file_name, p.file_type, p.model_prediction, p.confidence, p.created_at,
                   f.true_label, f.was_correct
            FROM predictions p
            LEFT JOIN feedback f ON f.prediction_id = p.id
            ORDER BY p.id DESC LIMIT ?
            """,
            (limit,),
        ).fetchall()
    return [dict(r) for r in rows]


def training_samples(db_path: str = DB_PATH) -> Iterable[dict[str, Any]]:
    """Yield image/video feedback samples that have a stored media path on disk."""
    with get_conn(db_path) as conn:
        rows = conn.execute(
            """
            SELECT f.id, f.true_label, f.stored_media_path, p.file_type
            FROM feedback f
            JOIN predictions p ON p.id = f.prediction_id
            WHERE f.stored_media_path IS NOT NULL
              AND p.file_type IN ('image', 'video')
            """
        ).fetchall()
    for r in rows:
        d = dict(r)
        if d["stored_media_path"] and os.path.exists(d["stored_media_path"]):
            yield d


if __name__ == "__main__":
    init_db()
    print(f"Initialized DB at {DB_PATH}")
    print(json.dumps(stats(), indent=2))
