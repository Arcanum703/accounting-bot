"""Deterministic core: parse -> classify (cached) -> store -> aggregate -> validate.

Classification comes from the recorded predictions in eval/fixtures, so these tests
make no LLM calls and need no API key. They check the project's central promise:
every figure in the income statement is computed by code from the parsed file.
"""

from __future__ import annotations

import csv

import pytest

from fixture import seed_cache
from src.core import aggregate, classify, parse, store, taxonomy, validate
from test_classification import evaluate

from conftest import ROOT

STATEMENT = ROOT / "data" / "bank_statement.csv"


class NoLLM:
    """Stands in for the chat model; any use means a cache miss reached the LLM."""

    def __getattr__(self, name):
        raise AssertionError(f"LLM was called ({name}) — expected a cache hit")


@pytest.fixture
def conn(tmp_path):
    conn = store.get_connection(str(tmp_path / "test.db"))
    store.init_db(conn)
    seed_cache(conn)
    return conn


def ingest(conn):
    txns = parse.parse_statement(str(STATEMENT))
    rows, hits = [], 0
    for t in txns:
        result, hit = classify.classify_cached(conn, t.description, t.txn_type, llm=NoLLM())
        hits += hit
        rows.append({
            "date": t.date, "description": t.description, "amount": t.amount,
            "currency": t.currency, "category": result.category,
            "subcategory": result.subcategory, "confidence": result.confidence,
            "justification": result.justification, "source_ref": t.source_ref,
        })
    store.insert_transactions(conn, rows)
    return txns, hits


def test_parsed_amounts_match_the_file_exactly():
    with STATEMENT.open(encoding="utf-8") as f:
        raw = list(csv.DictReader(f))
    txns = parse.parse_statement(str(STATEMENT))

    assert len(txns) == len(raw) == 80
    for i, (t, r) in enumerate(zip(txns, raw)):
        assert t.amount == round(float(r["Amount"]) - float(r["Fee"] or 0), 2)
        assert t.source_ref == f"bank_statement.csv:{i}"


def test_every_row_is_classified_from_cache(conn):
    txns, hits = ingest(conn)
    assert hits == len(txns)


def test_net_income_is_revenue_minus_expenses_and_ignores_excluded(conn):
    ingest(conn)
    rows = store.fetch_all_transactions(conn)
    summary = aggregate.summarize(rows)

    revenue = round(sum(r["amount"] for r in rows if r["category"] == taxonomy.REVENUE), 2)
    expenses = round(-sum(r["amount"] for r in rows if r["category"] == taxonomy.EXPENSE), 2)

    assert summary.revenue_total == revenue
    assert summary.expense_total == expenses
    assert summary.net_income == round(revenue - expenses, 2)
    assert summary.n_transactions == 80


def test_clean_statement_validates_and_reconciles(conn):
    ingest(conn)
    summary = aggregate.summarize(store.fetch_all_transactions(conn))
    report = validate.validate(conn, summary)

    assert report.ok
    assert report.reconciliation_diff == 0
    assert not report.unclassified and not report.duplicates


def test_reuploading_the_same_file_inserts_nothing(conn):
    ingest(conn)
    _, _ = ingest(conn)
    assert store.count_transactions(conn) == 80


def test_duplicate_payment_is_flagged(conn):
    ingest(conn)
    first = dict(store.fetch_all_transactions(conn)[0])
    first.pop("id")
    first["source_ref"] = "second_upload.csv:0"
    store.insert_transactions(conn, [first])

    summary = aggregate.summarize(store.fetch_all_transactions(conn))
    report = validate.validate(conn, summary)

    assert not report.ok
    assert len(report.duplicates) == 1


def test_unknown_category_is_flagged(conn):
    store.insert_transactions(conn, [{
        "date": "2025-04-01", "description": "MYSTERY", "amount": -10.0,
        "currency": "GBP", "category": "misc", "subcategory": "misc",
        "confidence": 0.9, "justification": "", "source_ref": "x.csv:0",
    }])
    summary = aggregate.summarize(store.fetch_all_transactions(conn))
    report = validate.validate(conn, summary)

    assert report.unclassified == ["x.csv:0"]
    assert not report.ok


def test_classifier_accuracy_does_not_regress(conn):
    n, cat_ok, sub_ok, _ = evaluate(conn)
    assert cat_ok / n >= 0.95
    assert sub_ok / n >= 0.90


@pytest.mark.parametrize("raw, expected", [
    ("  RENT - Westgate  Properties ", "rent - westgate properties"),
    ("INV\t0042", "inv 0042"),
])
def test_cache_key_normalization(raw, expected):
    assert store.normalize_description(raw) == expected
