"""Deterministic roll-up into a cash-basis income statement.

CLAUDE.md #4: this iterates the COMPLETE set of stored rows (never vector
retrieval) and sums with plain Python — numbers come straight from the store,
which came straight from pandas.

CLAUDE.md #3 / taxonomy: the `excluded` bucket is reported as a separate total
and kept OUT of net income.

    net income (cash basis) = total revenue − total expense

Honest labeling (CLAUDE.md #7): the output is a CASH-BASIS INCOME STATEMENT.
Never a balance sheet / statement of financial position.
"""

from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass, field
from typing import Iterable, Mapping

from . import taxonomy

LOW_CONFIDENCE_THRESHOLD = 0.6


@dataclass
class Summary:
    currency: str = ""
    revenue_total: float = 0.0
    expense_by_sub: dict[str, float] = field(default_factory=dict)   # magnitudes
    expense_total: float = 0.0                                        # magnitude
    excluded_by_sub: dict[str, float] = field(default_factory=dict)   # signed net
    excluded_total: float = 0.0                                       # signed net
    net_income: float = 0.0
    n_transactions: int = 0
    n_low_confidence: int = 0

    @property
    def low_confidence_threshold(self) -> float:
        return LOW_CONFIDENCE_THRESHOLD


def summarize(rows: Iterable[Mapping]) -> Summary:
    """Aggregate stored transaction rows into a Summary.

    Each row needs: amount (signed float), category, subcategory, currency,
    confidence.
    """
    rows = list(rows)
    s = Summary()
    s.n_transactions = len(rows)

    revenue = 0.0
    expense_signed = defaultdict(float)   # signed sums (outflows are negative)
    excluded_signed = defaultdict(float)

    currencies = set()
    for r in rows:
        amount = float(r["amount"])
        cat = r["category"]
        sub = r["subcategory"]
        currencies.add(r["currency"])

        if r["confidence"] is not None and r["confidence"] < LOW_CONFIDENCE_THRESHOLD:
            s.n_low_confidence += 1

        if cat == taxonomy.REVENUE:
            revenue += amount
        elif cat == taxonomy.EXPENSE:
            expense_signed[sub] += amount
        elif cat == taxonomy.EXCLUDED:
            excluded_signed[sub] += amount

    s.currency = currencies.pop() if len(currencies) == 1 else "/".join(sorted(currencies))

    s.revenue_total = round(revenue, 2)
    # Present expenses as positive magnitudes; keep them ordered by the taxonomy.
    s.expense_by_sub = {
        sub: round(-expense_signed[sub], 2)
        for sub in taxonomy.EXPENSE_SUBCATEGORIES
        if sub in expense_signed
    }
    s.expense_total = round(sum(s.expense_by_sub.values()), 2)

    # Excluded shown as signed net cash effect (loan_in positive, transfers negative).
    s.excluded_by_sub = {
        sub: round(excluded_signed[sub], 2)
        for sub in taxonomy.EXCLUDED_SUBCATEGORIES
        if sub in excluded_signed
    }
    s.excluded_total = round(sum(s.excluded_by_sub.values()), 2)

    # net income = revenue − expense. Excluded never enters this line.
    s.net_income = round(s.revenue_total - s.expense_total, 2)
    return s


# --------------------------------------------------------------------------- #
# Rendering
# --------------------------------------------------------------------------- #

_SUB_LABELS = {
    "cogs": "Cost of goods sold",
    "rent": "Rent",
    "salaries": "Salaries",
    "utilities": "Utilities",
    "supplies": "Supplies",
    "fees": "Bank/transaction fees",
    "taxes": "Taxes",
    "other_opex": "Other operating expenses",
    "transfer": "Internal transfers",
    "loan_in": "Loan received",
    "loan_out": "Loan repayment",
    "owner_draw": "Owner draw",
    "owner_contribution": "Owner contribution",
    "capital_purchase": "Capital purchase (equipment)",
    "refund": "Refunds",
}


def _money(value: float, currency: str) -> str:
    return f"{value:>12,.2f} {currency}"


def render_statement_text(s: Summary) -> str:
    """Format the Summary as a monospace block suitable for Telegram."""
    cur = s.currency or ""
    lines: list[str] = []
    lines.append("CASH-BASIS INCOME STATEMENT")
    lines.append("(financial summary — cash movements only)")
    lines.append("=" * 44)

    lines.append("")
    lines.append("REVENUE")
    lines.append(f"  {'Sales / income received':<26}{_money(s.revenue_total, cur)}")

    lines.append("")
    lines.append("EXPENSES")
    if s.expense_by_sub:
        for sub, amount in s.expense_by_sub.items():
            lines.append(f"  {_SUB_LABELS.get(sub, sub):<26}{_money(amount, cur)}")
    else:
        lines.append("  (none)")
    lines.append(f"  {'-' * 40}")
    lines.append(f"  {'Total expenses':<26}{_money(s.expense_total, cur)}")

    lines.append("")
    lines.append("=" * 44)
    lines.append(f"  {'NET INCOME (cash basis)':<26}{_money(s.net_income, cur)}")
    lines.append("=" * 44)

    lines.append("")
    lines.append("EXCLUDED — not part of net income")
    lines.append("(transfers, loans, owner draws/contributions, capital, refunds)")
    if s.excluded_by_sub:
        for sub, amount in s.excluded_by_sub.items():
            lines.append(f"  {_SUB_LABELS.get(sub, sub):<26}{_money(amount, cur)}")
        lines.append(f"  {'-' * 40}")
        lines.append(f"  {'Net excluded cash flow':<26}{_money(s.excluded_total, cur)}")
    else:
        lines.append("  (none)")

    lines.append("")
    lines.append("-" * 44)
    note = f"{s.n_transactions} transactions"
    if s.n_low_confidence:
        note += (f" · {s.n_low_confidence} low-confidence "
                 f"(< {s.low_confidence_threshold:.1f}) — review suggested")
    lines.append(note)
    return "\n".join(lines)
