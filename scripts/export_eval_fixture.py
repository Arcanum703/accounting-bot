"""Export the classification cache to eval/fixtures/classification_cache.json.

The fixture lets the classification eval and the unit tests run in CI with the
exact predictions the classifier made, without an API key or any LLM call.

Run:  python scripts/export_eval_fixture.py            (reads DB_PATH or data/accounting.db)
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from dotenv import load_dotenv  # noqa: E402

from src.core import llm, store  # noqa: E402

OUT = Path(__file__).resolve().parents[1] / "eval" / "fixtures" / "classification_cache.json"


def main() -> None:
    load_dotenv()
    conn = store.get_connection()
    provider, model = llm.get_active()
    rows = conn.execute(
        "SELECT normalized_description, category, subcategory, confidence, justification "
        "FROM classification_cache WHERE provider = ? AND model = ? "
        "ORDER BY normalized_description",
        (provider, model),
    ).fetchall()
    if not rows:
        sys.exit(f"No cached predictions for {provider}/{model}; run the pipeline first.")
    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(json.dumps({
        "provider": provider,
        "model": model,
        "predictions": [dict(r) for r in rows],
    }, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    print(f"Wrote {len(rows)} predictions for {provider}/{model} to {OUT}")


if __name__ == "__main__":
    main()
