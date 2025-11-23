import sqlite3
from pathlib import Path
from datetime import datetime
from typing import List, Dict, Optional


DEFAULT_DB = Path(__file__).parent / "sentiments.db"


def _get_db_path(db_path: Optional[str] = None) -> Path:
    return Path(db_path) if db_path else DEFAULT_DB


def _ensure_table(conn: sqlite3.Connection):
    conn.execute(
        """
        CREATE TABLE IF NOT EXISTS sentiments (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            text TEXT NOT NULL,
            sentiment TEXT NOT NULL,
            timestamp TEXT NOT NULL
        )
        """
    )
    conn.commit()


def save_sentiment(text: str, sentiment: str, db_path: Optional[str] = None) -> int:
    """Save a classified text into the SQLite DB.

    Uses parameterized queries to avoid SQL injection. Returns the new row id.
    """
    db_file = _get_db_path(db_path)
    db_file.parent.mkdir(parents=True, exist_ok=True)

    ts = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

    conn = sqlite3.connect(str(db_file), timeout=10, check_same_thread=False)
    try:
        _ensure_table(conn)
        cur = conn.cursor()
        cur.execute(
            "INSERT INTO sentiments (text, sentiment, timestamp) VALUES (?, ?, ?)",
            (text, sentiment, ts),
        )
        conn.commit()
        return cur.lastrowid
    finally:
        conn.close()


def load_sentiments(limit: int = 50, db_path: Optional[str] = None) -> List[Dict[str, str]]:
    """Load the most recent `limit` sentiment records, newest first.

    Returns a list of dicts with keys: id, text, sentiment, timestamp.
    """
    db_file = _get_db_path(db_path)
    if not db_file.exists():
        return []

    conn = sqlite3.connect(str(db_file), timeout=10, check_same_thread=False)
    try:
        cur = conn.cursor()
        cur.execute(
            "SELECT id, text, sentiment, timestamp FROM sentiments ORDER BY timestamp DESC LIMIT ?",
            (limit,),
        )
        rows = cur.fetchall()
        result = []
        for r in rows:
            result.append({"id": r[0], "text": r[1], "sentiment": r[2], "timestamp": r[3]})
        return result
    finally:
        conn.close()
