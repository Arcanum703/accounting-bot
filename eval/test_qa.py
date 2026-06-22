"""Q&A pass-rate eval — runs eval/qa_pairs.yaml through the agent.

For each pair it checks (whichever are specified):
  * expect_contains : all substrings present (case/comma-insensitive),
  * expect_tool     : the agent routed to that tool,
  * expect_no_data  : the agent said there is no such data (and invented no figure).

Prints per-pair PASS/FAIL and an overall pass rate.

Run:  python eval/test_qa.py
"""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import yaml  # noqa: E402
from dotenv import load_dotenv  # noqa: E402

from src.core import qa_agent, store  # noqa: E402
from src.core.pipeline import run_pipeline  # noqa: E402

PAIRS = Path(__file__).resolve().parent / "qa_pairs.yaml"
STATEMENT = "data/bank_statement.csv"

_NO_DATA_MARKERS = (
    "no ", "not ", "n't", "none", "no record", "no transaction", "does not",
    "no relevant", "no such", "cannot find", "could not find", "couldn't find",
    "i don't see", "do not include", "doesn't include", "no cryptocurrency",
    "no bitcoin", "no crypto",
)


def _norm(text: str) -> str:
    return text.lower().replace(",", "")


def main() -> None:
    load_dotenv()
    conn = store.get_connection()
    store.init_db(conn)
    # Ensure the statement is loaded + embedded (cache-warm: no LLM classify calls).
    if store.count_transactions(conn) == 0:
        run_pipeline(STATEMENT, conn)

    pairs = yaml.safe_load(PAIRS.read_text(encoding="utf-8"))
    passed = 0

    for p in pairs:
        res = qa_agent.answer_question(p["question"])
        answer = res["answer"] or ""
        tools = res["tools_used"]
        norm = _norm(answer)

        reasons = []
        for sub in p.get("expect_contains", []):
            if _norm(sub) not in norm:
                reasons.append(f"missing {sub!r}")
        if "expect_tool" in p and p["expect_tool"] not in tools:
            reasons.append(f"tool {p['expect_tool']} not used (used {tools})")
        if p.get("expect_no_data"):
            if not any(m in norm for m in _NO_DATA_MARKERS):
                reasons.append("did not state 'no data'")

        ok = not reasons
        passed += ok
        status = "PASS" if ok else "FAIL"
        print(f"[{status}] {p['id']}: {p['question']}")
        print(f"        tools={tools}  answer={answer[:140].strip()!r}")
        if reasons:
            print(f"        -> {'; '.join(reasons)}")

    n = len(pairs)
    print("-" * 60)
    print(f"QA pass rate: {passed}/{n} = {passed / n:.1%}")


if __name__ == "__main__":
    main()
