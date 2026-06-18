"""Deterministic parsing of a bank statement into canonical transactions.

NON-NEGOTIABLE (CLAUDE.md #1): amounts are read here by pandas and never produced
by the LLM. The classifier downstream only ever sees description text (+ the
Revolut Type, for the temporary Phase-1 rule classifier).

Different banks use different column layouts. For the demo we support the Revolut
export layout; the mapping lives in ONE config dict (REVOLUT_COLUMN_MAP) so other
banks can be added later without touching the parsing logic.
"""

from __future__ import annotations

import os
from dataclasses import dataclass

import pandas as pd

# Revolut export layout -> canonical fields. To support another bank, add a
# sibling dict with the same canonical keys; nothing else needs to change.
REVOLUT_COLUMN_MAP = {
    "date": "Completed Date",  # the date cash actually moved (cash basis)
    "description": "Description",
    "amount": "Amount",
    "fee": "Fee",
    "currency": "Currency",
    "type": "Type",
}


@dataclass(frozen=True)
class CanonicalTxn:
    """One parsed transaction in the bot's internal, bank-agnostic shape."""

    date: str          # ISO YYYY-MM-DD
    description: str    # raw memo from the statement
    amount: float       # SIGNED: + inflow, - outflow (net of fee)
    currency: str
    txn_type: str       # raw Revolut Type — used by the Phase-1 rule classifier,
                        # NOT persisted (not part of the SQLite schema)
    source_ref: str     # "<filename>:<row index>"


def _read_dataframe(path: str) -> pd.DataFrame:
    ext = os.path.splitext(path)[1].lower()
    if ext == ".csv":
        return pd.read_csv(path)
    if ext in (".xlsx", ".xls"):
        return pd.read_excel(path, engine="openpyxl")
    raise ValueError(f"Unsupported statement format: {ext!r} (expected .csv/.xlsx)")


def parse_statement(path: str, column_map: dict | None = None) -> list[CanonicalTxn]:
    """Read a statement file into canonical transactions, in file order.

    The row index in `source_ref` is the 0-based position within the file, so it
    lines up with the draft labels in eval/labels.csv.
    """
    column_map = column_map or REVOLUT_COLUMN_MAP
    df = _read_dataframe(path)

    missing = [src for src in column_map.values() if src not in df.columns]
    if missing:
        raise ValueError(f"Statement is missing expected columns: {missing}")

    filename = os.path.basename(path)
    txns: list[CanonicalTxn] = []

    # Numbers come straight from pandas — parsed, never invented.
    dates = pd.to_datetime(df[column_map["date"]]).dt.strftime("%Y-%m-%d")
    amounts = pd.to_numeric(df[column_map["amount"]], errors="raise")
    fees = pd.to_numeric(df[column_map["fee"]], errors="raise").fillna(0.0)

    for i in range(len(df)):
        # Net cash movement = Amount − Fee. A Revolut fee further reduces the
        # balance, so subtracting it keeps the running balance reconcilable for
        # real exports. In the demo data Fee is always 0, so this is a no-op.
        signed = round(float(amounts.iloc[i]) - float(fees.iloc[i]), 2)
        txns.append(
            CanonicalTxn(
                date=dates.iloc[i],
                description=str(df[column_map["description"]].iloc[i]).strip(),
                amount=signed,
                currency=str(df[column_map["currency"]].iloc[i]).strip(),
                txn_type=str(df[column_map["type"]].iloc[i]).strip(),
                source_ref=f"{filename}:{i}",
            )
        )
    return txns
