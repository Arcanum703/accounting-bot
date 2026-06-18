"""SQLite store — the SOURCE OF TRUTH (CLAUDE.md #4).

Two tables:
  transactions        — every classified transaction (the data aggregation sums).
  classification_cache — keyed by normalized description, so each distinct
                         description is classified once and the same statement
                         classified twice gives identical totals (CLAUDE.md #2).

No Chroma here yet — vector storage is added in a later phase.
"""

from __future__ import annotations

import os
import re
import sqlite3

# Default store location; override with the DB_PATH env var.
_DEFAULT_DB = os.path.join(
    os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))),
    "data",
    "accounting.db",
)


def db_path() -> str:
    return os.environ.get("DB_PATH", _DEFAULT_DB)


def normalize_description(description: str) -> str:
    """Cache key: lowercased, whitespace-collapsed, trimmed."""
    return re.sub(r"\s+", " ", description.strip().lower())


def get_connection(path: str | None = None) -> sqlite3.Connection:
    path = path or db_path()
    os.makedirs(os.path.dirname(path), exist_ok=True)
    conn = sqlite3.connect(path)
    conn.row_factory = sqlite3.Row
    return conn


def init_db(conn: sqlite3.Connection) -> None:
    conn.executescript(
        """
        CREATE TABLE IF NOT EXISTS transactions (
            id            INTEGER PRIMARY KEY,
            date          TEXT NOT NULL,
            description   TEXT NOT NULL,
            amount        REAL NOT NULL,
            currency      TEXT NOT NULL,
            category      TEXT NOT NULL,
            subcategory   TEXT NOT NULL,
            confidence    REAL,
            justification TEXT,
            source_ref    TEXT UNIQUE          -- dedupe re-uploads of the same row
        );

        CREATE TABLE IF NOT EXISTS classification_cache (
            normalized_description TEXT PRIMARY KEY,
            subcategory            TEXT NOT NULL,
            category               TEXT NOT NULL,
            confidence             REAL,
            justification          TEXT
        );
        """
    )
    conn.commit()


def reset(conn: sqlite3.Connection) -> None:
    """Clear the store and the classification cache (for /reset)."""
    conn.executescript("DELETE FROM transactions; DELETE FROM classification_cache;")
    conn.commit()


def insert_transactions(conn: sqlite3.Connection, rows: list[dict]) -> int:
    """Insert classified transactions. Duplicate source_refs are ignored.

    Returns the number of rows actually inserted.
    """
    before = conn.total_changes
    conn.executemany(
        """
        INSERT OR IGNORE INTO transactions
            (date, description, amount, currency, category, subcategory,
             confidence, justification, source_ref)
        VALUES
            (:date, :description, :amount, :currency, :category, :subcategory,
             :confidence, :justification, :source_ref)
        """,
        rows,
    )
    conn.commit()
    return conn.total_changes - before


def fetch_all_transactions(conn: sqlite3.Connection) -> list[sqlite3.Row]:
    return conn.execute(
        "SELECT * FROM transactions ORDER BY date, id"
    ).fetchall()


def count_transactions(conn: sqlite3.Connection) -> int:
    return conn.execute("SELECT COUNT(*) FROM transactions").fetchone()[0]


# ---- classification cache -------------------------------------------------- #

def get_cached(conn: sqlite3.Connection, normalized: str) -> sqlite3.Row | None:
    return conn.execute(
        "SELECT subcategory, category, confidence, justification "
        "FROM classification_cache WHERE normalized_description = ?",
        (normalized,),
    ).fetchone()


def set_cached(
    conn: sqlite3.Connection,
    normalized: str,
    subcategory: str,
    category: str,
    confidence: float,
    justification: str,
) -> None:
    conn.execute(
        "INSERT OR REPLACE INTO classification_cache "
        "(normalized_description, subcategory, category, confidence, justification) "
        "VALUES (?, ?, ?, ?, ?)",
        (normalized, subcategory, category, confidence, justification),
    )
    conn.commit()
