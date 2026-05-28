"""Migrate existing local SQLite data to Turso.

Reads data/forensic.db (local) and inserts every row into Turso, skipping
any rows that already exist (matched by file_hash + created_at).

Prereqs:
  pip install libsql-experimental
  TURSO_DATABASE_URL and TURSO_AUTH_TOKEN exported in the shell

Run:
  python scripts/migrate_to_turso.py
"""

import os
import sqlite3
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
import db

LOCAL_DB = "data/forensic.db"


def main():
    url = os.getenv("TURSO_DATABASE_URL")
    token = os.getenv("TURSO_AUTH_TOKEN")
    if not (url and token):
        print("ERROR: set TURSO_DATABASE_URL and TURSO_AUTH_TOKEN in your shell.")
        sys.exit(1)

    if not os.path.exists(LOCAL_DB):
        print(f"ERROR: local DB not found at {LOCAL_DB}")
        sys.exit(1)

    print(f"Source:      {LOCAL_DB}")
    print(f"Destination: {url}")

    # Initialize Turso schema
    db.init_db()
    print("Turso schema ready.")

    # Read from local
    local = sqlite3.connect(LOCAL_DB)
    local.row_factory = sqlite3.Row

    preds = local.execute(
        "SELECT * FROM predictions ORDER BY id ASC"
    ).fetchall()
    fbs = local.execute(
        "SELECT * FROM feedback ORDER BY id ASC"
    ).fetchall()

    print(f"Local rows: {len(preds)} predictions, {len(fbs)} feedback")

    # Map local prediction IDs to new Turso IDs (Turso autoincrement may differ)
    id_map: dict[int, int] = {}

    with db.get_conn() as turso:
        # First check what's already in Turso so we don't duplicate
        existing_hashes = {
            row[0] for row in
            turso.execute("SELECT file_hash || '|' || created_at FROM predictions").fetchall()
            if row[0]
        }

        skipped_preds = 0
        for p in preds:
            key = f"{p['file_hash']}|{p['created_at']}"
            if key in existing_hashes:
                skipped_preds += 1
                # Try to find the existing remote id so feedback can still link
                existing = turso.execute(
                    "SELECT id FROM predictions WHERE file_hash = ? AND created_at = ? LIMIT 1",
                    (p['file_hash'], p['created_at']),
                ).fetchone()
                if existing:
                    id_map[p['id']] = int(existing[0])
                continue

            cur = turso.execute(
                """
                INSERT INTO predictions
                    (file_hash, file_name, file_type, model_prediction, confidence,
                     raw_result, model_version, created_at)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (p['file_hash'], p['file_name'], p['file_type'], p['model_prediction'],
                 p['confidence'], p['raw_result'], p['model_version'], p['created_at']),
            )
            id_map[p['id']] = int(cur.lastrowid or 0)

        # Migrate feedback, remapping prediction_id
        existing_fb_keys = {
            f"{row[0]}|{row[1]}|{row[2]}" for row in
            turso.execute(
                "SELECT file_hash, true_label, created_at FROM feedback"
            ).fetchall()
        }

        skipped_fb = 0
        for f in fbs:
            key = f"{f['file_hash']}|{f['true_label']}|{f['created_at']}"
            if key in existing_fb_keys:
                skipped_fb += 1
                continue

            new_pred_id = id_map.get(f['prediction_id']) if f['prediction_id'] else None
            turso.execute(
                """
                INSERT INTO feedback
                    (prediction_id, file_hash, true_label, was_correct,
                     stored_media_path, notes, created_at)
                VALUES (?, ?, ?, ?, ?, ?, ?)
                """,
                (new_pred_id, f['file_hash'], f['true_label'], f['was_correct'],
                 f['stored_media_path'], f['notes'], f['created_at']),
            )

    print(f"Done. Predictions: migrated={len(preds) - skipped_preds}, skipped={skipped_preds}")
    print(f"      Feedback:    migrated={len(fbs) - skipped_fb}, skipped={skipped_fb}")
    print(f"\nVerify: python -c 'import db, json; print(json.dumps(db.stats(), indent=2))'")


if __name__ == "__main__":
    main()
