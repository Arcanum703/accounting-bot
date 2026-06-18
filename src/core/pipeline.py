"""Phase-1 pipeline: parse -> classify -> store -> aggregate.

Wired as plain function calls for now. CLAUDE.md's target is a LangGraph graph
(parse -> classify -> store -> aggregate -> validate); that conversion and the
`validate` node come in a later phase. The function boundaries here are chosen so
each becomes a graph node with minimal change.

Run standalone to exercise the whole thing without Telegram:

    python -m src.core.pipeline data/bank_statement.csv
"""

from __future__ import annotations

import sqlite3
from dataclasses import dataclass

from . import aggregate, classify, parse, store


@dataclass
class IngestStats:
    rows_parsed: int = 0
    rows_inserted: int = 0
    cache_hits: int = 0
    currency: str = ""


def ingest(path: str, conn: sqlite3.Connection) -> IngestStats:
    """Parse a statement, classify each transaction (cache-aware), and store it."""
    txns = parse.parse_statement(path)
    stats = IngestStats(rows_parsed=len(txns))

    classified: list[dict] = []
    for t in txns:
        # Cache by normalized description (CLAUDE.md #2) so each distinct
        # description is classified once and re-runs are identical.
        norm = store.normalize_description(t.description)
        cached = store.get_cached(conn, norm)
        if cached is not None:
            stats.cache_hits += 1
            category = cached["category"]
            subcategory = cached["subcategory"]
            confidence = cached["confidence"]
            justification = cached["justification"]
        else:
            result = classify.classify(t.description, t.txn_type)
            category = result.category
            subcategory = result.subcategory
            confidence = result.confidence
            justification = result.justification
            store.set_cached(conn, norm, subcategory, category,
                             confidence, justification)

        classified.append({
            "date": t.date,
            "description": t.description,
            "amount": t.amount,           # signed, straight from pandas
            "currency": t.currency,
            "category": category,
            "subcategory": subcategory,
            "confidence": confidence,
            "justification": justification,
            "source_ref": t.source_ref,
        })

    if classified:
        stats.currency = classified[0]["currency"]
    stats.rows_inserted = store.insert_transactions(conn, classified)
    return stats


def build_summary(conn: sqlite3.Connection) -> aggregate.Summary:
    """Aggregate the complete store into a cash-basis income statement summary."""
    return aggregate.summarize(store.fetch_all_transactions(conn))


def _main() -> None:
    import sys

    path = sys.argv[1] if len(sys.argv) > 1 else "data/bank_statement.csv"
    conn = store.get_connection()
    store.init_db(conn)
    store.reset(conn)  # fresh run for the CLI demo

    stats = ingest(path, conn)
    summary = build_summary(conn)

    print(f"Parsed {stats.rows_parsed} rows, inserted {stats.rows_inserted} "
          f"(cache hits: {stats.cache_hits})\n")
    print(aggregate.render_statement_text(summary))


if __name__ == "__main__":
    _main()
