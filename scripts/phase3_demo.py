"""Phase 3 acceptance check (verification helper, not part of the bot).

Simulates the Telegram free-text handler:
  * a numeric question  -> should route to sql_query and return an exact, cited figure
  * a vague descriptive question -> should route to vector_search

Also prints a direct-SQL "ground truth" for the numeric question so you can see the
agent's figure is exact (no invented numbers).

Run:  python scripts/phase3_demo.py
"""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from dotenv import load_dotenv  # noqa: E402

from src.core import qa_agent, store  # noqa: E402
from src.core.pipeline import run_pipeline  # noqa: E402

STATEMENT = "data/bank_statement.csv"

NUMERIC_Q = "How much did I spend on utilities in May 2025?"
VAGUE_Q = "Was there anything about eating out or a restaurant while abroad?"


def main() -> None:
    load_dotenv()
    conn = store.get_connection()
    store.init_db(conn)

    # Make sure the statement is loaded AND embedded into Chroma. Re-running the
    # pipeline is cheap here: classifications are all cached (0 LLM calls) and the
    # store step upserts the embeddings.
    run_pipeline(STATEMENT, conn)
    print(f"Transactions in store: {store.count_transactions(conn)} | "
          f"vectors in Chroma: {store.get_collection().count()}\n")

    # Ground truth for the numeric question (direct SQL over every row).
    gt = conn.execute(
        "SELECT -SUM(amount) spend, GROUP_CONCAT(id) ids FROM transactions "
        "WHERE category='expense' AND subcategory='utilities' "
        "AND substr(date,1,7)='2025-05'"
    ).fetchone()
    print(f"[ground truth SQL] utilities spend in 2025-05 = {gt['spend']} "
          f"(ids {gt['ids']})\n")

    for label, q in [("NUMERIC", NUMERIC_Q), ("VAGUE", VAGUE_Q)]:
        print("=" * 64)
        print(f"USER ({label}): {q}")
        res = qa_agent.answer_question(q)
        print(f"TOOLS USED: {res['tools_used']}")
        print(f"BOT: {res['answer']}")
        print()


if __name__ == "__main__":
    main()
