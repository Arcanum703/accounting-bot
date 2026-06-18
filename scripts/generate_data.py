#!/usr/bin/env python3
"""Generate the synthetic demo bank statement + draft ground-truth labels.

Phase 0 of the Accounting Telegram Bot (see CLAUDE.md).

This is a *reproducible generator*, not a one-off dump:
  * The transactions are a fixed, curated list (no random draws), so running the
    script twice produces byte-identical files. Reproducibility is a
    Non-negotiable for this project.
  * The statement and the draft labels are emitted from ONE source of truth
    (the TRANSACTIONS list below), so a transaction can never appear in the
    statement with a different category than it has in the labels.
  * The running Balance is computed here from a fixed opening balance, so it
    always adds up exactly.

Outputs:
  data/bank_statement.csv   Revolut export layout (what the parser will consume).
  eval/labels.csv           DRAFT ground truth for the accuracy eval.

IMPORTANT about labels: these are a DRAFT only. A human must review and correct
eval/labels.csv by hand before it is used as ground truth — scoring the
classifier against machine-generated labels would make the accuracy eval
circular (we'd just be measuring how well one heuristic reproduces another).

Layout note: the statement mirrors a real Revolut CSV export
(Type, Product, Started Date, Completed Date, Description, Amount, Fee,
Currency, State, Balance) so a real export can be dropped in later with no
parser changes.

Taxonomy (must match CLAUDE.md exactly):
  revenue
  expense  -> cogs | rent | salaries | utilities | supplies | fees | taxes | other_opex
  excluded -> transfer | loan_in | loan_out | owner_draw | owner_contribution
              | capital_purchase | refund
"""

from __future__ import annotations

import csv
import datetime as dt
from pathlib import Path

# --------------------------------------------------------------------------- #
# Config
# --------------------------------------------------------------------------- #

CURRENCY = "GBP"
OPENING_BALANCE = 4000.00
PROJECT_ROOT = Path(__file__).resolve().parents[1]
STATEMENT_PATH = PROJECT_ROOT / "data" / "bank_statement.csv"
LABELS_PATH = PROJECT_ROOT / "eval" / "labels.csv"
SOURCE_FILE = "bank_statement.csv"  # used to build source_ref

# Revolut CSV column order (do not reorder — the parser maps by this layout).
STATEMENT_COLUMNS = [
    "Type",
    "Product",
    "Started Date",
    "Completed Date",
    "Description",
    "Amount",
    "Fee",
    "Currency",
    "State",
    "Balance",
]

LABEL_COLUMNS = [
    "source_ref",
    "description",
    "amount",
    "true_category",
    "true_subcategory",
]


# --------------------------------------------------------------------------- #
# Transaction model
# --------------------------------------------------------------------------- #
#
# Each entry is a dict with:
#   date        : "YYYY-MM-DD" (date the transaction completed)
#   type        : Revolut Type (CARD_PAYMENT, TRANSFER, TOPUP, EXCHANGE, FEE, ATM, ...)
#   product     : Revolut Product (Current / Savings)
#   description : raw description/memo
#   amount      : SIGNED float (+ inflow, - outflow)
#   fee         : float (almost always 0.00; non-zero on a few rows)
#   category    : revenue | expense | excluded
#   subcategory : taxonomy subcategory
#   hard        : None, or a short note explaining why the row is deliberately
#                 ambiguous / a trap for the classifier.
#
# Rows are listed in roughly chronological order; the script sorts by date and
# assigns a deterministic timestamp + running balance afterwards.


def tx(date, type_, description, amount, category, subcategory,
       product="Current", fee=0.00, hard=None):
    return {
        "date": date,
        "type": type_,
        "product": product,
        "description": description,
        "amount": round(float(amount), 2),
        "fee": round(float(fee), 2),
        "category": category,
        "subcategory": subcategory,
        "hard": hard,
    }


# --------------------------------------------------------------------------- #
# The data (the single source of truth for statement + labels)
# --------------------------------------------------------------------------- #
# ~80 transactions across Apr–Jun 2025 for a small design/print studio.
# Recurring items (rent, salaries, subscriptions, utilities) repeat monthly so
# the statement reads like a real business account.

TRANSACTIONS = [
    # ===================== APRIL 2025 =====================
    # Owner seeds the account by topping up from a personal card. TOPUP type is a
    # strong "not revenue" signal — this is owner capital, not a customer paying.
    tx("2025-04-02", "TOPUP", "Top-Up by Card *4421", +3000.00,
       "excluded", "owner_contribution"),

    tx("2025-04-03", "TRANSFER", "PAYMENT FROM ACME LTD INV-1001", +1850.00,
       "revenue", "revenue"),
    tx("2025-04-03", "TRANSFER", "RENT - WESTGATE PROPERTIES", -1200.00,
       "expense", "rent"),

    # HARD: terse, cryptic memo. No merchant words, just an invoice-looking code.
    # A human reads "INV" as a customer invoice payment (revenue); the classifier
    # has almost nothing to go on.
    tx("2025-04-05", "TRANSFER", "INV 0042", +900.00,
       "revenue", "revenue",
       hard="Terse/cryptic memo — only an invoice-like code, easy to misread."),

    tx("2025-04-06", "CARD_PAYMENT", "ADOBE SUBSCRIPTION", -49.99,
       "expense", "other_opex"),
    tx("2025-04-07", "CARD_PAYMENT", "GOOGLE WORKSPACE", -11.04,
       "expense", "other_opex"),
    tx("2025-04-08", "CARD_PAYMENT", "STAPLES OFFICE SUPPLIES", -63.80,
       "expense", "supplies"),
    tx("2025-04-09", "CARD_PAYMENT", "PRINT MATERIALS LTD", -342.00,
       "expense", "cogs"),

    # HARD: looks like income (positive, "PAYMENT FROM ..."), but it is the owner
    # moving money back from their own Revolut savings vault — an internal
    # transfer that must be EXCLUDED, not counted as revenue.
    tx("2025-04-10", "TRANSFER", "PAYMENT FROM SAVINGS VAULT", +2000.00,
       "excluded", "transfer",
       hard="Positive 'PAYMENT FROM...' that is actually an internal "
            "own-account transfer, not revenue."),

    tx("2025-04-11", "CARD_PAYMENT", "BT INTERNET", -42.00,
       "expense", "utilities"),
    tx("2025-04-14", "TRANSFER", "STRIPE PAYOUT", +640.00,
       "revenue", "revenue"),
    tx("2025-04-15", "CARD_PAYMENT", "BRITISH GAS", -95.40,
       "expense", "utilities"),

    # HARD: ambiguous merchant. "SQ *" is a Square card-reader settlement — here
    # it is a customer paying the studio (revenue), but the same prefix appears on
    # ordinary card purchases, so it's genuinely ambiguous from the text alone.
    tx("2025-04-16", "TRANSFER", "SQ *BLUE ROOM STUDIO", +480.00,
       "revenue", "revenue",
       hard="Ambiguous merchant — 'SQ *' (Square) can be either an inbound "
            "payout or an outbound purchase."),

    tx("2025-04-17", "FEE", "Revolut card delivery fee", -4.99,
       "expense", "fees"),
    tx("2025-04-18", "TRANSFER", "SALARY - J SMITH", -1600.00,
       "expense", "salaries"),
    tx("2025-04-18", "TRANSFER", "SALARY - A PATEL", -1400.00,
       "expense", "salaries"),

    # HARD: foreign (German) merchant name from a supplier trip abroad. Tests the
    # language-agnostic prompt — the bot is built for RU/KY/EN, so a non-English
    # vendor string must still classify. This is a business meal -> other_opex.
    tx("2025-04-22", "CARD_PAYMENT", "BRAUHAUS MÜNCHEN GMBH", -38.60,
       "expense", "other_opex",
       hard="Foreign-language merchant name (German); tests language-agnostic "
            "classification."),

    # Capital purchase: a laptop is equipment, NOT an operating expense. Excluded.
    tx("2025-04-23", "CARD_PAYMENT", "APPLE STORE MACBOOK PRO", -1999.00,
       "excluded", "capital_purchase"),

    tx("2025-04-24", "CARD_PAYMENT", "FACEBOOK ADS", -75.00,
       "expense", "other_opex"),
    tx("2025-04-25", "TRANSFER", "TO SAVINGS VAULT", -1500.00,
       "excluded", "transfer", product="Savings"),

    # HARD: a refund landing as a positive amount. Looks like revenue but it is
    # money coming back on a previous purchase — excluded (it isn't sales income).
    tx("2025-04-28", "CARD_PAYMENT", "REFUND - STAPLES OFFICE SUPPLIES", +63.80,
       "excluded", "refund",
       hard="Inbound refund (positive) that superficially looks like revenue."),

    tx("2025-04-12", "CARD_PAYMENT", "CANVA PRO", -12.99,
       "expense", "other_opex"),
    tx("2025-04-26", "TRANSFER", "PAYMENT FROM HARLOW PRINTS", +430.00,
       "revenue", "revenue"),
    tx("2025-04-29", "EXCHANGE", "Exchange GBP to EUR", -300.00,
       "excluded", "transfer"),
    tx("2025-04-30", "CARD_PAYMENT", "INK & TONER CO", -88.20,
       "expense", "cogs"),

    # ===================== MAY 2025 =====================
    tx("2025-05-01", "TRANSFER", "RENT - WESTGATE PROPERTIES", -1200.00,
       "expense", "rent"),
    tx("2025-05-02", "TRANSFER", "PAYMENT FROM ACME LTD INV-1002", +1850.00,
       "revenue", "revenue"),

    # Loan received from the bank. A big positive inflow that is NOT revenue —
    # this is the single most important "excluded" trap in the dataset.
    tx("2025-05-05", "TRANSFER", "LOAN DISBURSEMENT BARCLAYS BUSINESS", +5000.00,
       "excluded", "loan_in",
       hard="Large positive inflow that is a bank loan, not revenue — the core "
            "excluded-bucket trap."),

    tx("2025-05-06", "CARD_PAYMENT", "ADOBE SUBSCRIPTION", -49.99,
       "expense", "other_opex"),
    tx("2025-05-06", "CARD_PAYMENT", "GOOGLE WORKSPACE", -11.04,
       "expense", "other_opex"),
    tx("2025-05-07", "TRANSFER", "GUMROAD SALES PAYOUT", +320.00,
       "revenue", "revenue"),
    tx("2025-05-08", "CARD_PAYMENT", "THAMES WATER", -28.10,
       "expense", "utilities"),
    tx("2025-05-09", "CARD_PAYMENT", "RYMAN STATIONERY", -37.50,
       "expense", "supplies"),

    # HARD: single-word memo with no merchant or context at all.
    tx("2025-05-12", "TRANSFER", "PAYMENT", +250.00,
       "revenue", "revenue",
       hard="Maximally terse memo ('PAYMENT') — almost no signal."),

    tx("2025-05-13", "CARD_PAYMENT", "PRINT MATERIALS LTD", -410.00,
       "expense", "cogs"),
    tx("2025-05-14", "CARD_PAYMENT", "BT INTERNET", -42.00,
       "expense", "utilities"),
    tx("2025-05-15", "CARD_PAYMENT", "BRITISH GAS", -101.20,
       "expense", "utilities"),
    tx("2025-05-16", "TRANSFER", "PAYMENT FROM RIVERSIDE CAFE", +540.00,
       "revenue", "revenue"),

    # Loan repayment (first instalment). Outflow that is NOT an expense.
    tx("2025-05-16", "TRANSFER", "LOAN REPAYMENT BARCLAYS", -450.00,
       "excluded", "loan_out"),

    tx("2025-05-19", "TRANSFER", "SALARY - J SMITH", -1600.00,
       "expense", "salaries"),
    tx("2025-05-19", "TRANSFER", "SALARY - A PATEL", -1400.00,
       "expense", "salaries"),
    tx("2025-05-20", "CARD_PAYMENT", "LINKEDIN ADS", -60.00,
       "expense", "other_opex"),

    # Owner draw: the founder taking money out for personal use. Excluded
    # (it is not a salary expense and not a business cost).
    tx("2025-05-21", "TRANSFER", "OWNER DRAW - A FOUNDER", -800.00,
       "excluded", "owner_draw"),

    # ATM cash withdrawal by the owner — also a personal draw, not an expense.
    tx("2025-05-22", "ATM", "ATM WITHDRAWAL HIGH ST", -200.00,
       "excluded", "owner_draw"),

    tx("2025-05-23", "FEE", "Revolut monthly plan fee", -6.99,
       "expense", "fees"),
    tx("2025-05-26", "CARD_PAYMENT", "IKEA BUSINESS", -118.00,
       "expense", "supplies"),

    # HARD: tax refund from HMRC. Positive, mentions HMRC (which the classifier
    # may anchor to "taxes"), but money coming *back* is a refund, not revenue
    # and not a (negative) tax expense.
    tx("2025-05-27", "TRANSFER", "HMRC VAT REFUND", +180.00,
       "excluded", "refund",
       hard="HMRC inbound — 'tax' wording but it's a refund, not revenue or a "
            "tax expense."),

    tx("2025-05-28", "CARD_PAYMENT", "AMZN MKTP UK", -54.30,
       "expense", "supplies"),
    tx("2025-05-10", "CARD_PAYMENT", "ROYAL MAIL POSTAGE", -23.40,
       "expense", "other_opex"),
    tx("2025-05-25", "TRANSFER", "STRIPE PAYOUT", +510.00,
       "revenue", "revenue"),
    tx("2025-05-29", "TRANSFER", "PAYMENT FROM ACME LTD INV-1002B", +600.00,
       "revenue", "revenue"),
    tx("2025-05-30", "EXCHANGE", "Exchange EUR to GBP", +120.00,
       "excluded", "transfer"),

    # ===================== JUNE 2025 =====================
    tx("2025-06-02", "TRANSFER", "RENT - WESTGATE PROPERTIES", -1200.00,
       "expense", "rent"),
    tx("2025-06-03", "TRANSFER", "PAYMENT FROM ACME LTD INV-1003", +1850.00,
       "revenue", "revenue"),
    tx("2025-06-04", "CARD_PAYMENT", "ADOBE SUBSCRIPTION", -49.99,
       "expense", "other_opex"),
    tx("2025-06-04", "CARD_PAYMENT", "GOOGLE WORKSPACE", -11.04,
       "expense", "other_opex"),
    tx("2025-06-05", "TRANSFER", "STRIPE PAYOUT", +720.00,
       "revenue", "revenue"),
    tx("2025-06-06", "CARD_PAYMENT", "PRINT MATERIALS LTD", -298.00,
       "expense", "cogs"),
    tx("2025-06-07", "CARD_PAYMENT", "SLACK SUBSCRIPTION", -7.50,
       "expense", "other_opex"),
    tx("2025-06-09", "CARD_PAYMENT", "BT INTERNET", -42.00,
       "expense", "utilities"),
    tx("2025-06-10", "CARD_PAYMENT", "BRITISH GAS", -88.70,
       "expense", "utilities"),

    # Second-hand monitor — small capital purchase (equipment), still excluded.
    tx("2025-06-11", "CARD_PAYMENT", "DELL MONITOR REFURB", -189.00,
       "excluded", "capital_purchase"),

    tx("2025-06-12", "TRANSFER", "PAYMENT FROM ORTON DESIGN", +960.00,
       "revenue", "revenue"),

    # HARD: ambiguous PayPal merchant string. "PAYPAL *" can be a payout from a
    # customer (revenue) or a purchase from a vendor. Here it is a vendor purchase
    # of stock materials -> cogs.
    tx("2025-06-13", "CARD_PAYMENT", "PAYPAL *PAPERWORLD", -132.40,
       "expense", "cogs",
       hard="Ambiguous 'PAYPAL *' merchant — could read as inbound revenue, but "
            "is an outbound vendor purchase."),

    tx("2025-06-16", "TRANSFER", "SALARY - J SMITH", -1600.00,
       "expense", "salaries"),
    tx("2025-06-16", "TRANSFER", "SALARY - A PATEL", -1400.00,
       "expense", "salaries"),
    tx("2025-06-17", "TRANSFER", "HMRC PAYE", -420.00,
       "expense", "taxes"),
    tx("2025-06-18", "TRANSFER", "HMRC VAT", -780.00,
       "expense", "taxes"),
    tx("2025-06-19", "TRANSFER", "LOAN REPAYMENT BARCLAYS", -450.00,
       "excluded", "loan_out"),
    tx("2025-06-20", "CARD_PAYMENT", "FACEBOOK ADS", -90.00,
       "expense", "other_opex"),

    # HARD: foreign (Turkish) merchant — a working lunch with a client abroad.
    tx("2025-06-20", "CARD_PAYMENT", "ÇIGKOFTEM ISTANBUL", -22.15,
       "expense", "other_opex",
       hard="Foreign-language merchant name (Turkish); tests language-agnostic "
            "classification."),

    tx("2025-06-22", "CARD_PAYMENT", "PRINT MATERIALS LTD", -156.00,
       "expense", "cogs"),
    tx("2025-06-23", "TRANSFER", "PAYMENT FROM BENNETT & CO", +1200.00,
       "revenue", "revenue"),
    tx("2025-06-24", "FEE", "Revolut FX fee", -3.40,
       "expense", "fees"),
    tx("2025-06-25", "CARD_PAYMENT", "INK & TONER CO", -76.50,
       "expense", "cogs"),
    tx("2025-06-25", "TRANSFER", "TO SAVINGS VAULT", -1000.00,
       "excluded", "transfer", product="Savings"),

    # HARD: refund from an online retailer landing positive.
    tx("2025-06-26", "CARD_PAYMENT", "AMZN MKTP UK REFUND", +18.50,
       "excluded", "refund",
       hard="Inbound retailer refund (positive) — not revenue."),

    tx("2025-06-27", "CARD_PAYMENT", "THAMES WATER", -29.40,
       "expense", "utilities"),

    # HARD: owner injects more capital, this time as a plain bank transfer in
    # (not a TOPUP). Positive and large; looks like a big sale but is owner equity.
    tx("2025-06-27", "TRANSFER", "FUNDS FROM A FOUNDER PERSONAL", +1500.00,
       "excluded", "owner_contribution",
       hard="Owner capital arriving as a normal inbound transfer — looks like "
            "a customer payment."),

    tx("2025-06-30", "TRANSFER", "PAYMENT FROM ACME LTD INV-1004", +1850.00,
       "revenue", "revenue"),
]


# --------------------------------------------------------------------------- #
# Build
# --------------------------------------------------------------------------- #

def _completed_datetime(date_str: str, ordinal: int) -> str:
    """Deterministic timestamp within the day so rows have a stable order."""
    base = dt.datetime.strptime(date_str, "%Y-%m-%d")
    stamp = base + dt.timedelta(hours=9, minutes=(ordinal * 7) % 60,
                                seconds=(ordinal * 13) % 60)
    return stamp.strftime("%Y-%m-%d %H:%M:%S")


def _started_datetime(completed: str, type_: str) -> str:
    """Started a little before completed; transfers settle a day later."""
    c = dt.datetime.strptime(completed, "%Y-%m-%d %H:%M:%S")
    delta = dt.timedelta(days=1) if type_ in ("TRANSFER", "EXCHANGE") else dt.timedelta(minutes=2)
    return (c - delta).strftime("%Y-%m-%d %H:%M:%S")


def build_rows():
    """Return (statement_rows, label_rows, hard_rows) all sorted by date."""
    ordered = sorted(TRANSACTIONS, key=lambda t: t["date"])

    statement_rows = []
    label_rows = []
    hard_rows = []

    balance = OPENING_BALANCE
    for i, t in enumerate(ordered):
        balance = round(balance + t["amount"], 2)
        completed = _completed_datetime(t["date"], i)
        started = _started_datetime(completed, t["type"])
        source_ref = f"{SOURCE_FILE}:{i}"

        statement_rows.append({
            "Type": t["type"],
            "Product": t["product"],
            "Started Date": started,
            "Completed Date": completed,
            "Description": t["description"],
            "Amount": f"{t['amount']:.2f}",
            "Fee": f"{t['fee']:.2f}",
            "Currency": CURRENCY,
            "State": "COMPLETED",
            "Balance": f"{balance:.2f}",
        })

        label_rows.append({
            "source_ref": source_ref,
            "description": t["description"],
            "amount": f"{t['amount']:.2f}",
            "true_category": t["category"],
            "true_subcategory": t["subcategory"],
        })

        if t["hard"]:
            hard_rows.append((source_ref, t["description"],
                              f"{t['amount']:.2f}",
                              f"{t['category']}/{t['subcategory']}",
                              t["hard"]))

    return statement_rows, label_rows, hard_rows


def write_csv(path: Path, columns, rows):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=columns)
        writer.writeheader()
        writer.writerows(rows)


def main():
    statement_rows, label_rows, hard_rows = build_rows()

    write_csv(STATEMENT_PATH, STATEMENT_COLUMNS, statement_rows)
    write_csv(LABELS_PATH, LABEL_COLUMNS, label_rows)

    # ---- summary to stdout (sanity check, not part of the artifacts) ----
    closing = statement_rows[-1]["Balance"]
    inflow = sum(float(r["Amount"]) for r in statement_rows if float(r["Amount"]) > 0)
    outflow = sum(float(r["Amount"]) for r in statement_rows if float(r["Amount"]) < 0)

    by_cat = {}
    for r in label_rows:
        by_cat[r["true_category"]] = by_cat.get(r["true_category"], 0) + 1

    print(f"Wrote {len(statement_rows)} transactions -> {STATEMENT_PATH.relative_to(PROJECT_ROOT)}")
    print(f"Wrote {len(label_rows)} draft labels    -> {LABELS_PATH.relative_to(PROJECT_ROOT)}")
    print()
    print(f"Opening balance: {OPENING_BALANCE:.2f} {CURRENCY}")
    print(f"Total inflow:   +{inflow:.2f}")
    print(f"Total outflow:  {outflow:.2f}")
    print(f"Closing balance: {closing} {CURRENCY}  "
          f"(check: {OPENING_BALANCE + inflow + outflow:.2f})")
    print()
    print("Rows per category:", by_cat)
    print()
    print(f"Deliberately HARD / ambiguous rows ({len(hard_rows)}):")
    for ref, desc, amt, truth, note in hard_rows:
        print(f"  [{ref}] {desc!r} ({amt}) -> truth={truth}")
        print(f"        why: {note}")


if __name__ == "__main__":
    main()
