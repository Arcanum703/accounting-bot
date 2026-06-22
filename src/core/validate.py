"""Validation node — runs before any statement is returned (CLAUDE.md).

Checks:
  * every transaction has a non-null, valid category;
  * the aggregated totals reconcile to the raw sum of all stored rows
    (no row silently dropped or double-counted in aggregation);
  * no duplicate rows (same date + amount + description) counted more than once;
  * low-confidence transactions are flagged for review;
  * any single transaction that is an outlier share of its category is flagged.

Structural failures (unclassified rows, reconciliation mismatch, duplicates) set
`ok = False`. Low-confidence and outliers are advisory warnings, not failures.
"""

from __future__ import annotations

import sqlite3
from collections import defaultdict
from dataclasses import dataclass, field

from . import store, taxonomy
from .aggregate import LOW_CONFIDENCE_THRESHOLD, Summary

# A transaction is an "outlier share" if it alone is at least this fraction of its
# category's total magnitude (only checked when the category has several rows).
OUTLIER_SHARE = 0.6
OUTLIER_MIN_ROWS = 3


@dataclass
class ValidationReport:
    ok: bool = True
    unclassified: list[str] = field(default_factory=list)          # source_refs
    reconciles: bool = True
    reconciliation_diff: float = 0.0
    duplicates: list[tuple] = field(default_factory=list)          # (count,date,amount,desc)
    low_confidence: list[tuple] = field(default_factory=list)      # (ref,desc,conf)
    outliers: list[tuple] = field(default_factory=list)            # (ref,desc,amount,cat,share)

    def warnings(self) -> list[str]:
        msgs: list[str] = []
        if self.unclassified:
            msgs.append(f"{len(self.unclassified)} transaction(s) not classified.")
        if not self.reconciles:
            msgs.append(
                f"Totals do not reconcile (diff {self.reconciliation_diff:+.2f})."
            )
        if self.duplicates:
            msgs.append(f"{len(self.duplicates)} possible duplicate group(s).")
        if self.low_confidence:
            msgs.append(
                f"{len(self.low_confidence)} low-confidence row(s) "
                f"(< {LOW_CONFIDENCE_THRESHOLD:.1f}) — review suggested."
            )
        if self.outliers:
            msgs.append(f"{len(self.outliers)} outlier transaction(s) in a category.")
        return msgs


def validate(
    conn: sqlite3.Connection,
    summary: Summary,
    low_confidence_threshold: float = LOW_CONFIDENCE_THRESHOLD,
) -> ValidationReport:
    rows = store.fetch_all_transactions(conn)
    report = ValidationReport()

    # --- every row classified with a valid category ------------------------ #
    for r in rows:
        if not r["category"] or r["category"] not in taxonomy.CATEGORIES:
            report.unclassified.append(r["source_ref"])

    # --- aggregation reconciles to the raw sum of all rows ----------------- #
    raw_sum = round(sum(float(r["amount"]) for r in rows), 2)
    agg_sum = round(
        summary.revenue_total - summary.expense_total + summary.excluded_total, 2
    )
    report.reconciliation_diff = round(raw_sum - agg_sum, 2)
    report.reconciles = abs(report.reconciliation_diff) < 0.01

    # --- duplicate detection (same date + amount + description) ------------ #
    seen = defaultdict(list)
    for r in rows:
        seen[(r["date"], round(float(r["amount"]), 2), r["description"])].append(r["id"])
    for (date, amount, desc), ids in seen.items():
        if len(ids) > 1:
            report.duplicates.append((len(ids), date, amount, desc))

    # --- low confidence ---------------------------------------------------- #
    for r in rows:
        if r["confidence"] is not None and r["confidence"] < low_confidence_threshold:
            report.low_confidence.append(
                (r["source_ref"], r["description"], round(float(r["confidence"]), 2))
            )

    # --- outliers (one txn dominating its category) ------------------------ #
    cat_totals = defaultdict(float)
    cat_counts = defaultdict(int)
    for r in rows:
        cat_totals[r["category"]] += abs(float(r["amount"]))
        cat_counts[r["category"]] += 1
    for r in rows:
        total = cat_totals[r["category"]]
        if cat_counts[r["category"]] >= OUTLIER_MIN_ROWS and total > 0:
            share = abs(float(r["amount"])) / total
            if share >= OUTLIER_SHARE:
                report.outliers.append(
                    (r["source_ref"], r["description"], round(float(r["amount"]), 2),
                     r["category"], round(share, 2))
                )

    report.ok = (
        not report.unclassified and report.reconciles and not report.duplicates
    )
    return report
