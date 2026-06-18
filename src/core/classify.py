"""Transaction classifier.

================================  TEMPORARY  ================================
This is a TEMPORARY, rule-based classifier for Phase 1 so we can see end-to-end
output (parse -> classify -> store -> aggregate -> statement) WITHOUT an LLM.

It is a keyword/Type heuristic. It WILL get some of the deliberately hard rows
wrong (that's expected — those traps exist to motivate the real classifier). It
gets REPLACED in Phase 2 by an LLM classifier that reads description text only,
runs at temperature 0, and is cached by normalized description.

Until then it conforms to the same interface the LLM version will use:
    classify(description, txn_type) -> Classification
============================================================================

Note: unlike the future LLM classifier (description text ONLY), this temporary
rule may also look at the Revolut Type, which is a strong signal for the
`excluded` bucket (FEE, EXCHANGE, TOPUP, ATM ...).
"""

from __future__ import annotations

from dataclasses import dataclass

from . import taxonomy

CLASSIFIER_NAME = "temp-rules-v1"


@dataclass(frozen=True)
class Classification:
    category: str
    subcategory: str
    confidence: float
    justification: str


def _result(subcategory: str, confidence: float, reason: str) -> Classification:
    category = taxonomy.category_for(subcategory)
    return Classification(
        category=category,
        subcategory=subcategory,
        confidence=confidence,
        justification=f"[{CLASSIFIER_NAME}] {reason}",
    )


def classify(description: str, txn_type: str = "") -> Classification:
    """Best-effort keyword/Type classification. Rule order matters."""
    d = (description or "").lower()
    t = (txn_type or "").upper()

    # --- refund first: 'refund' overrides tax/supplies/revenue keywords ----- #
    if "refund" in d:
        return _result("refund", 0.9, "description mentions 'refund'")

    # --- strong Revolut Type signals --------------------------------------- #
    if t == "FEE":
        return _result("fees", 0.95, "Type=FEE")
    if t == "EXCHANGE":
        return _result("transfer", 0.9, "Type=EXCHANGE (currency conversion)")
    if t == "ATM":
        return _result("owner_draw", 0.7, "Type=ATM cash withdrawal")
    if t == "TOPUP":
        return _result("owner_contribution", 0.7, "Type=TOPUP (funds added)")

    # --- excluded by description ------------------------------------------- #
    if "savings" in d or "vault" in d:
        return _result("transfer", 0.85, "savings/vault -> internal transfer")
    if "loan" in d and ("disbursement" in d or "received" in d):
        return _result("loan_in", 0.9, "loan disbursement received")
    if "loan" in d and ("repayment" in d or "repay" in d):
        return _result("loan_out", 0.9, "loan repayment")
    if "owner draw" in d or "drawing" in d:
        return _result("owner_draw", 0.85, "owner draw")
    if ("founder" in d or "owner" in d) and (
        "capital" in d or "injection" in d or "funds from" in d or "contribution" in d
    ):
        return _result("owner_contribution", 0.8, "owner capital contribution")
    if any(k in d for k in ("macbook", "laptop", "monitor", "apple store", "equipment")):
        return _result("capital_purchase", 0.8, "equipment purchase (capital)")

    # --- expense by description -------------------------------------------- #
    if "rent" in d:
        return _result("rent", 0.9, "rent")
    if "salary" in d or "payroll" in d or "wages" in d:
        return _result("salaries", 0.9, "salary/payroll")
    if any(k in d for k in ("hmrc", "vat", "paye", " tax")):
        return _result("taxes", 0.85, "tax authority payment")
    if any(k in d for k in (
        "gas", "water", "electric", "internet", "bt ", "utility", "broadband"
    )):
        return _result("utilities", 0.8, "utility bill")
    if any(k in d for k in (
        "print materials", "ink", "toner", "paper", "materials", "paperworld"
    )):
        return _result("cogs", 0.75, "production materials (COGS)")
    if any(k in d for k in (
        "staples", "ryman", "ikea", "stationery", "office supplies", "amzn", "amazon"
    )):
        return _result("supplies", 0.7, "office supplies")
    if any(k in d for k in (
        "adobe", "google workspace", "canva", "slack", "subscription",
        "ads", "linkedin", "facebook", "postage", "royal mail", "courier",
    )):
        return _result("other_opex", 0.7, "software/marketing/other opex")

    # --- revenue ----------------------------------------------------------- #
    if any(k in d for k in (
        "payment from", "stripe", "gumroad", "payout", "sales", "invoice", "inv"
    )):
        return _result("revenue", 0.7, "incoming customer/sales payment")
    if d.strip() == "payment":
        return _result("revenue", 0.5, "bare 'payment' — assumed inbound (low conf)")

    # --- fallback ---------------------------------------------------------- #
    return _result(
        "other_opex", 0.3, "no keyword matched — fallback (flag for review)"
    )
