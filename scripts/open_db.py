"""Initialize the forensic DB and open it in DB Browser for SQLite (macOS)."""

import os
import subprocess
import sys

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
import db

db.init_db()

db_abs = os.path.abspath(db.DB_PATH)
print(f"Database ready at: {db_abs}")
print(f"Tables: predictions, feedback")

s = db.stats()
print(f"  predictions: {s['total_predictions']}")
print(f"  feedback:    {s['total_feedback']}")

try:
    subprocess.run(["open", "-a", "DB Browser for SQLite", db_abs], check=True)
    print("Opened in DB Browser for SQLite.")
except (FileNotFoundError, subprocess.CalledProcessError):
    print("DB Browser not found in Applications. Open the file manually:")
    print(f"  {db_abs}")
