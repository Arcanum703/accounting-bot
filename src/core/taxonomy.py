"""The three-bucket taxonomy (see CLAUDE.md — this is mandatory).

  revenue  — operating income / sales received
  expense  — cogs | rent | salaries | utilities | supplies | fees | taxes | other_opex
  excluded — transfer | loan_in | loan_out | owner_draw | owner_contribution
             | capital_purchase | refund

The `excluded` bucket is the whole point of correctness: a loan received is NOT
revenue, a transfer to savings is NOT an expense, buying equipment is NOT an
operating expense. Excluded items are reported as a separate total and kept OUT
of net income.

    net income (cash basis) = total revenue − total expense   (excluded never enters)
"""

from __future__ import annotations

REVENUE = "revenue"
EXPENSE = "expense"
EXCLUDED = "excluded"

CATEGORIES = (REVENUE, EXPENSE, EXCLUDED)

# Subcategories per top-level bucket.
REVENUE_SUBCATEGORIES = ("revenue",)

EXPENSE_SUBCATEGORIES = (
    "cogs",
    "rent",
    "salaries",
    "utilities",
    "supplies",
    "fees",
    "taxes",
    "other_opex",
)

EXCLUDED_SUBCATEGORIES = (
    "transfer",
    "loan_in",
    "loan_out",
    "owner_draw",
    "owner_contribution",
    "capital_purchase",
    "refund",
)

# subcategory -> category lookup (the single mapping everything else relies on).
SUBCATEGORY_TO_CATEGORY = {
    **{s: REVENUE for s in REVENUE_SUBCATEGORIES},
    **{s: EXPENSE for s in EXPENSE_SUBCATEGORIES},
    **{s: EXCLUDED for s in EXCLUDED_SUBCATEGORIES},
}

ALL_SUBCATEGORIES = tuple(SUBCATEGORY_TO_CATEGORY.keys())

# Subcategories that count toward the cash-basis P&L (net income).
PNL_CATEGORIES = (REVENUE, EXPENSE)


def category_for(subcategory: str) -> str:
    """Return the top-level bucket for a subcategory, or raise if unknown."""
    try:
        return SUBCATEGORY_TO_CATEGORY[subcategory]
    except KeyError as exc:
        raise ValueError(f"Unknown subcategory: {subcategory!r}") from exc


def is_valid(category: str, subcategory: str) -> bool:
    """True if (category, subcategory) is a legal pair in the taxonomy."""
    return SUBCATEGORY_TO_CATEGORY.get(subcategory) == category
