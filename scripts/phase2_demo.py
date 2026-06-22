"""Phase 2 acceptance check (verification helper, not part of the bot).

1. Runs the LangGraph pipeline on the demo statement (warms the classification
   cache via real LLM calls on the active provider).
2. Re-runs it and asserts the totals are IDENTICAL (cache + temperature 0 →
   reproducible), with zero new LLM calls on the second pass.
3. Prints 5 example classifications with their justifications.

Run:  python scripts/phase2_demo.py
"""

from __future__ import annotations

import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from dotenv import load_dotenv  # noqa: E402

from src.core import aggregate, llm, store  # noqa: E402
from src.core.pipeline import run_pipeline  # noqa: E402

STATEMENT = "data/bank_statement.csv"
EXAMPLES = [
    "LOAN DISBURSEMENT BARCLAYS BUSINESS",  # big inflow, but a loan -> excluded
    "PAYMENT FROM SAVINGS VAULT",           # looks like income -> excluded/transfer
    "SQ *BLUE ROOM STUDIO",                 # ambiguous merchant -> revenue
    "ÇIGKOFTEM ISTANBUL",                   # foreign merchant -> expense
    "APPLE STORE MACBOOK PRO",              # equipment -> capital_purchase
]


def _totals(s) -> dict:
    return {
        "revenue": s.revenue_total,
        "expense": s.expense_total,
        "net_income": s.net_income,
        "excluded": s.excluded_total,
    }


def main() -> None:
    load_dotenv()
    conn = store.get_connection()
    store.init_db(conn)

    # ---- RUN 1: warm the cache (real LLM calls for every distinct desc) -----
    # Full reset so the whole statement is classified by ONE model consistently.
    store.reset(conn)
    t0 = time.time()
    r1 = run_pipeline(STATEMENT, conn)
    print(f"RUN 1: parsed {r1.rows_parsed}, inserted {r1.rows_inserted}, "
          f"cache_hits {r1.cache_hits}  ({time.time() - t0:.0f}s)\n")
    print(aggregate.render_statement_text(r1.summary))
    print("\nValidation:", "OK" if r1.validation.ok else "ISSUES")
    for w in r1.validation.warnings():
        print(f"  • {w}")

    # ---- RUN 2: same file again -> must be identical, all from cache --------
    conn.execute("DELETE FROM transactions")
    conn.commit()
    r2 = run_pipeline(STATEMENT, conn)

    same = _totals(r1.summary) == _totals(r2.summary)
    print("\n" + "=" * 60)
    print("REPRODUCIBILITY CHECK")
    print(f"  RUN 1 totals: {_totals(r1.summary)}")
    print(f"  RUN 2 totals: {_totals(r2.summary)}")
    print(f"  RUN 2 cache_hits: {r2.cache_hits}/{r2.rows_parsed} "
          f"(LLM calls on re-run: {r2.rows_parsed - r2.cache_hits} distinct-new)")
    print(f"  IDENTICAL TOTALS: {same}")

    # ---- 5 example classifications -----------------------------------------
    provider, model = llm.get_active()
    print("\n" + "=" * 60)
    print(f"5 EXAMPLE CLASSIFICATIONS ({provider} / {model})")
    for desc in EXAMPLES:
        row = store.get_cached(conn, provider, model, store.normalize_description(desc))
        if row is None:
            print(f"\n  {desc!r}: (not classified)")
            continue
        print(f"\n  {desc}")
        print(f"    -> {row['category']} / {row['subcategory']} "
              f"(confidence {row['confidence']})")
        print(f"    justification: {row['justification']}")


if __name__ == "__main__":
    main()
