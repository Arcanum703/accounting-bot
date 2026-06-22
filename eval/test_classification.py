"""Classification accuracy vs the human-reviewed ground truth (eval/labels.csv).

Compares the LLM classifier's output to the reviewed labels and prints accuracy at
both the category and subcategory level, then lists every mismatch so disagreements
are auditable.

The classifier is cache-backed (keyed by provider+model+description), so this runs
with zero LLM calls once the demo statement has been processed — and it scores the
SAME classifier the bot uses.

Run:  python eval/test_classification.py
"""

from __future__ import annotations

import csv
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from dotenv import load_dotenv  # noqa: E402

from src.core import classify, llm, store  # noqa: E402

LABELS = Path(__file__).resolve().parent / "labels.csv"


def main() -> None:
    load_dotenv()
    conn = store.get_connection()
    store.init_db(conn)

    provider, model = llm.get_active()
    with LABELS.open(encoding="utf-8") as f:
        rows = list(csv.DictReader(f))

    n = len(rows)
    cat_ok = sub_ok = 0
    mismatches = []

    for r in rows:
        # Cache hit (classified during ingest). Type isn't in labels.csv; the cache
        # key is the description, so the stored result (which used Type) is returned.
        pred, _ = classify.classify_cached(conn, r["description"])
        cat_hit = pred.category == r["true_category"]
        sub_hit = pred.subcategory == r["true_subcategory"]
        cat_ok += cat_hit
        sub_ok += sub_hit
        if not (cat_hit and sub_hit):
            mismatches.append((r, pred))

    print(f"Classifier: {provider} / {model}")
    print(f"Labels:     {n} rows ({LABELS})")
    print("-" * 60)
    print(f"Category accuracy:    {cat_ok}/{n} = {cat_ok / n:.1%}")
    print(f"Subcategory accuracy: {sub_ok}/{n} = {sub_ok / n:.1%}")

    if mismatches:
        print("\nMismatches (label  ->  predicted):")
        for r, pred in mismatches:
            print(f"  {r['source_ref']:<22} {r['description'][:34]:<34} "
                  f"{r['true_category']}/{r['true_subcategory']}  ->  "
                  f"{pred.category}/{pred.subcategory}")


if __name__ == "__main__":
    main()
