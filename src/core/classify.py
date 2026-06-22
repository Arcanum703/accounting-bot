"""LLM transaction classifier (Phase 2 — replaces the temporary rule classifier).

Each DISTINCT normalized description is classified once by the LLM and the result
is cached (CLAUDE.md #2). The model runs at temperature 0 for reproducibility.

What the LLM sees: the DESCRIPTION text and the Revolut Type ONLY. It NEVER sees
amounts (CLAUDE.md #1) — amounts are parsed deterministically by pandas and must
never be produced or influenced by the LLM. Type is a categorical hint, not a
number.

The prompt is written language-agnostically: it classifies by MEANING, so the
same prompt works on Russian / Kyrgyz descriptions later with no code changes —
only different input text.

Output per description: {category, subcategory, confidence, justification}. The
justification is the audit trail (CLAUDE.md #6).
"""

from __future__ import annotations

import sqlite3
from dataclasses import dataclass
from typing import Literal

from pydantic import BaseModel, Field

from . import llm as llm_mod
from . import store, taxonomy

# Literals so the model can only emit valid taxonomy values.
_SubcategoryLiteral = Literal[
    "revenue",
    "cogs", "rent", "salaries", "utilities", "supplies", "fees", "taxes", "other_opex",
    "transfer", "loan_in", "loan_out", "owner_draw", "owner_contribution",
    "capital_purchase", "refund",
]
_CategoryLiteral = Literal["revenue", "expense", "excluded"]


class _LLMClassification(BaseModel):
    """Structured schema the model must fill (no amounts in or out)."""

    category: _CategoryLiteral = Field(
        description="Top-level bucket: revenue, expense, or excluded."
    )
    subcategory: _SubcategoryLiteral = Field(
        description="Specific subcategory within the chosen bucket."
    )
    confidence: float = Field(
        ge=0.0, le=1.0,
        description="Your confidence in this classification, 0.0–1.0.",
    )
    justification: str = Field(
        description="One short sentence explaining the choice (the audit trail)."
    )


@dataclass(frozen=True)
class Classification:
    category: str
    subcategory: str
    confidence: float
    justification: str


SYSTEM_PROMPT = """\
You categorize a single bank-statement transaction for a small business's
CASH-BASIS income statement. You will receive a transaction Type and a free-text
Description. The description may be in ANY language (e.g. English, Russian,
Kyrgyz) — classify by MEANING, not by language or keywords.

You are NOT given the amount and you must not guess or rely on one.

Assign exactly one of three top-level categories, and one subcategory:

revenue
  - revenue: money earned from selling goods/services to customers (sales,
    client invoice payments, payouts from payment processors like Stripe).

expense (operating costs that reduce profit):
  - cogs: direct cost of goods/materials sold (production materials, stock, ink/paper).
  - rent: premises rent / lease.
  - salaries: wages, payroll, salary payments to staff.
  - utilities: electricity, gas, water, internet/telecom.
  - supplies: office/consumable supplies.
  - fees: bank, card, payment-processor, or FX fees.
  - taxes: tax authority payments (VAT, payroll tax, income tax paid).
  - other_opex: any other operating cost (software subscriptions, advertising,
    postage, travel, meals).

excluded (NOT part of profit — these are financing, owner, or internal movements,
or money that is not income/expense). This bucket is the most important to get
right:
  - transfer: moving money between the owner's OWN accounts (e.g. to/from a
    savings vault), or currency exchange between own balances. NOT income/expense.
  - loan_in: loan or credit RECEIVED. A big inflow, but it is borrowed money, NOT revenue.
  - loan_out: repayment of a loan's principal. An outflow, but NOT an operating expense.
  - owner_draw: owner taking money out for personal use (drawings, personal ATM cash).
  - owner_contribution: owner putting personal money INTO the business (capital injection).
  - capital_purchase: buying durable equipment/assets (laptop, machinery, furniture).
    A business cost, but capital — NOT an operating expense.
  - refund: money refunded on a previous purchase, or a refund received. NOT revenue.

Guidance:
- A positive/incoming transaction is not automatically revenue: loans received,
  owner contributions, refunds, and inbound transfers are all 'excluded'.
- An outgoing transaction is not automatically an expense: loan repayments, owner
  draws, capital purchases, and outbound transfers are all 'excluded'.
- Use the Type as a hint (e.g. FEE → fees; EXCHANGE → transfer; TOPUP/ATM often
  relate to owner movements) but let the description's meaning decide.
- If genuinely unsure, pick your best guess and report LOW confidence.

The category and subcategory you return MUST be consistent.
"""


def _build_messages(description: str, txn_type: str):
    from langchain_core.messages import HumanMessage, SystemMessage

    human = f"Type: {txn_type or '(unknown)'}\nDescription: {description}"
    return [SystemMessage(content=SYSTEM_PROMPT), HumanMessage(content=human)]


def classify(description: str, txn_type: str = "", llm=None,
             model: str | None = None) -> Classification:
    """Classify one description via the LLM (no cache, no amounts).

    `model` only labels the justification's audit tag; if omitted it is read from
    the active provider config.
    """
    llm = llm or llm_mod.get_llm()
    model = model or llm_mod.get_active()[1]
    structured = llm.with_structured_output(_LLMClassification)
    result: _LLMClassification = structured.invoke(_build_messages(description, txn_type))

    subcategory = result.subcategory
    # Trust the subcategory; derive the category from the taxonomy so the pair is
    # always valid even if the model's category field disagrees.
    category = taxonomy.category_for(subcategory)

    return Classification(
        category=category,
        subcategory=subcategory,
        confidence=round(float(result.confidence), 2),
        justification=f"[{model}] {result.justification.strip()}",
    )


def classify_cached(
    conn: sqlite3.Connection, description: str, txn_type: str = "", llm=None
) -> tuple[Classification, bool]:
    """Classify with the cache. Returns (classification, was_cache_hit).

    Cache key is (provider, model, normalized description) (CLAUDE.md #2), so each
    distinct description costs at most one LLM call per model and re-runs are
    identical. Switching provider/model re-classifies rather than reusing another
    model's labels.
    """
    provider, model = llm_mod.get_active()
    norm = store.normalize_description(description)
    cached = store.get_cached(conn, provider, model, norm)
    if cached is not None:
        return (
            Classification(
                category=cached["category"],
                subcategory=cached["subcategory"],
                confidence=cached["confidence"],
                justification=cached["justification"],
            ),
            True,
        )

    result = classify(description, txn_type, llm=llm, model=model)
    store.set_cached(
        conn, provider, model, norm, result.subcategory, result.category,
        result.confidence, result.justification,
    )
    return result, False
