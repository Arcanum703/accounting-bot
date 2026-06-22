"""LangGraph pipeline: parse -> classify -> store -> aggregate -> validate.

This is the CLAUDE.md pipeline graph. The bot's document handler calls
`run_pipeline()`; each node is a small, testable function so the graph stays
readable.

Dependencies (the SQLite connection and the LLM) are passed through the graph
STATE rather than imported globally, which keeps the nodes pure and lets tests
inject a fake LLM. The graph is compiled without a checkpointer (single in-process
run), so non-serializable objects in state are fine.

Run standalone (needs GOOGLE_API_KEY in .env):

    python -m src.core.pipeline data/bank_statement.csv
"""

from __future__ import annotations

import sqlite3
from dataclasses import dataclass
from typing import Any, TypedDict

from langgraph.graph import END, START, StateGraph

from . import aggregate, classify, parse, store, validate
from .aggregate import Summary
from .validate import ValidationReport


# --------------------------------------------------------------------------- #
# Graph state
# --------------------------------------------------------------------------- #

class PipelineState(TypedDict, total=False):
    # inputs / dependencies
    file_path: str
    conn: Any            # sqlite3.Connection
    llm: Any             # chat model, or None -> get_llm()
    # produced along the way
    txns: list
    classified: list
    cache_hits: int
    rows_inserted: int
    summary: Summary
    validation: ValidationReport


# --------------------------------------------------------------------------- #
# Nodes
# --------------------------------------------------------------------------- #

def parse_node(state: PipelineState) -> dict:
    txns = parse.parse_statement(state["file_path"])
    return {"txns": txns}


def classify_node(state: PipelineState) -> dict:
    conn = state["conn"]
    llm = state.get("llm")
    classified: list[dict] = []
    cache_hits = 0

    for t in state["txns"]:
        result, hit = classify.classify_cached(conn, t.description, t.txn_type, llm=llm)
        cache_hits += int(hit)
        classified.append({
            "date": t.date,
            "description": t.description,
            "amount": t.amount,          # signed, straight from pandas
            "currency": t.currency,
            "category": result.category,
            "subcategory": result.subcategory,
            "confidence": result.confidence,
            "justification": result.justification,
            "source_ref": t.source_ref,
        })

    return {"classified": classified, "cache_hits": cache_hits}


def store_node(state: PipelineState) -> dict:
    conn = state["conn"]
    inserted = store.insert_transactions(conn, state["classified"])
    # Embed each stored description into the local Chroma index (id as metadata),
    # so the Q&A agent can do semantic retrieval. Idempotent upsert.
    store.index_source_refs(conn, [r["source_ref"] for r in state["classified"]])
    return {"rows_inserted": inserted}


def aggregate_node(state: PipelineState) -> dict:
    summary = aggregate.summarize(store.fetch_all_transactions(state["conn"]))
    return {"summary": summary}


def validate_node(state: PipelineState) -> dict:
    report = validate.validate(state["conn"], state["summary"])
    return {"validation": report}


# --------------------------------------------------------------------------- #
# Graph
# --------------------------------------------------------------------------- #

def build_graph():
    g = StateGraph(PipelineState)
    g.add_node("parse", parse_node)
    g.add_node("classify", classify_node)
    g.add_node("store", store_node)
    g.add_node("aggregate", aggregate_node)
    g.add_node("validate", validate_node)

    g.add_edge(START, "parse")
    g.add_edge("parse", "classify")
    g.add_edge("classify", "store")
    g.add_edge("store", "aggregate")
    g.add_edge("aggregate", "validate")
    g.add_edge("validate", END)
    return g.compile()


_GRAPH = None


def _graph():
    global _GRAPH
    if _GRAPH is None:
        _GRAPH = build_graph()
    return _GRAPH


# --------------------------------------------------------------------------- #
# Public API
# --------------------------------------------------------------------------- #

@dataclass
class PipelineResult:
    rows_parsed: int
    rows_inserted: int
    cache_hits: int
    summary: Summary
    validation: ValidationReport


def run_pipeline(path: str, conn: sqlite3.Connection, llm=None) -> PipelineResult:
    """Run the full graph on one statement file and return the result."""
    final: PipelineState = _graph().invoke(
        {"file_path": path, "conn": conn, "llm": llm}
    )
    return PipelineResult(
        rows_parsed=len(final.get("txns", [])),
        rows_inserted=final.get("rows_inserted", 0),
        cache_hits=final.get("cache_hits", 0),
        summary=final["summary"],
        validation=final["validation"],
    )


def build_summary(conn: sqlite3.Connection) -> Summary:
    """Aggregate the current store without re-ingesting (for /statement)."""
    return aggregate.summarize(store.fetch_all_transactions(conn))


def _main() -> None:
    import sys

    from dotenv import load_dotenv
    load_dotenv()  # so the CLI picks up GOOGLE_API_KEY from .env

    path = sys.argv[1] if len(sys.argv) > 1 else "data/bank_statement.csv"
    conn = store.get_connection()
    store.init_db(conn)
    store.reset(conn)  # fresh run for the CLI demo

    result = run_pipeline(path, conn)
    print(f"Parsed {result.rows_parsed} rows, inserted {result.rows_inserted} "
          f"(cache hits: {result.cache_hits})\n")
    print(aggregate.render_statement_text(result.summary))
    warnings = result.validation.warnings()
    if warnings:
        print("\nValidation:")
        for w in warnings:
            print(f"  • {w}")
    print(f"\nValidation ok: {result.validation.ok}")


if __name__ == "__main__":
    _main()
