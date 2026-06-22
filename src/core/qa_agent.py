"""Q&A agent — an agentic LangGraph subgraph (CLAUDE.md), NOT a RetrievalQA chain.

The LLM is given three tools and CHOOSES between them per question:

  sql_query        — read-only SQL over the transactions table. THE DEFAULT for
                     anything involving a number/total/count/aggregate. Exact and
                     complete (iterates every matching row).
  vector_search    — Chroma semantic similarity over descriptions. ONLY for fuzzy/
                     descriptive questions where an exact filter can't be written.
                     NEVER used to compute totals (CLAUDE.md #4/#5).
  filter           — structured filter by date range / vendor / category.

Every answer must cite the source transaction id(s) it used. If the data can't
answer the question, the agent says so — it never invents figures (CLAUDE.md #6).

Built with langgraph's prebuilt tool-calling agent (a graph: agent ⇄ tools),
which is an agentic subgraph, not a single chain.
"""

from __future__ import annotations

from langchain_core.tools import tool

from . import store
from .llm import get_llm

# Tools open their own short-lived SQLite connection so the agent is safe to run
# in a worker thread (sqlite connections are not shared across threads).

_FORBIDDEN_SQL = (
    "insert", "update", "delete", "drop", "alter", "attach",
    "pragma", "create", "replace", "vacuum",
)


@tool
def sql_query(sql: str) -> str:
    """Run ONE read-only SQL SELECT against the transactions table and return rows.

    USE THIS FOR ANY QUESTION INVOLVING A NUMBER: totals, sums, counts, averages,
    "how much", "how many". It is exact and complete.

    Schema — transactions(
        id INTEGER, date TEXT 'YYYY-MM-DD', description TEXT,
        amount REAL  -- SIGNED: positive = inflow, negative = outflow,
        currency TEXT, category TEXT, subcategory TEXT,
        confidence REAL, justification TEXT, source_ref TEXT)
    category is one of: 'revenue', 'expense', 'excluded'.
    expense subcategory: cogs, rent, salaries, utilities, supplies, fees, taxes, other_opex.
    excluded subcategory: transfer, loan_in, loan_out, owner_draw, owner_contribution,
                          capital_purchase, refund.

    Tips:
      - Expenses are stored NEGATIVE. Report spend as a positive number, e.g.
        SELECT -SUM(amount) ... WHERE category='expense'.
      - Filter a month with substr(date,1,7)='2025-05'.
      - ALWAYS include the ids so the answer can cite them, e.g. add
        GROUP_CONCAT(id) to aggregate queries, or SELECT id in row queries.
    """
    s = sql.strip().rstrip(";")
    low = s.lower()
    if not (low.startswith("select") or low.startswith("with")):
        return "ERROR: only read-only SELECT queries are allowed."
    if ";" in s:
        return "ERROR: only a single statement is allowed."
    if any(f in low for f in _FORBIDDEN_SQL):
        return "ERROR: query contains a forbidden keyword; SELECT only."
    conn = store.get_connection()
    try:
        rows = conn.execute(s).fetchall()
    except Exception as exc:  # surface SQL errors back to the agent to retry
        return f"SQL error: {exc}"
    if not rows:
        return "No rows matched."
    cols = rows[0].keys()
    out = [" | ".join(cols)]
    for r in rows[:100]:
        out.append(" | ".join("" if r[c] is None else str(r[c]) for c in cols))
    return "\n".join(out)


@tool
def vector_search(query: str) -> str:
    """Semantic similarity search over transaction descriptions (meaning-based).

    USE ONLY for vague/descriptive questions where you cannot write an exact SQL
    filter (e.g. "anything about eating out abroad?"). Do NOT use it to compute
    totals — it returns only the top few matches. Returns up to 5 transactions
    with their ids.
    """
    matches = store.vector_search(query, k=5)
    if not matches:
        return "No matching transactions."
    out = []
    for m in matches:
        out.append(
            f"#{m['id']} | {m['date']} | {m['description']} | "
            f"{m['amount']} {m['currency']} | {m['category']}/{m['subcategory']} "
            f"(distance {m['distance']})"
        )
    return "\n".join(out)


@tool
def filter(date_from: str = "", date_to: str = "",
           vendor: str = "", category: str = "") -> str:
    """Structured filter over the store. All args optional; combine with AND.

    date_from / date_to: inclusive 'YYYY-MM-DD' bounds.
    vendor: case-insensitive substring matched against the description.
    category: one of 'revenue', 'expense', 'excluded'.
    Returns matching transactions (with ids). Use for listing/narrowing, not sums.
    """
    clauses, params = [], []
    if date_from:
        clauses.append("date >= ?"); params.append(date_from)
    if date_to:
        clauses.append("date <= ?"); params.append(date_to)
    if vendor:
        clauses.append("lower(description) LIKE ?"); params.append(f"%{vendor.lower()}%")
    if category:
        clauses.append("category = ?"); params.append(category)
    where = (" WHERE " + " AND ".join(clauses)) if clauses else ""
    conn = store.get_connection()
    rows = conn.execute(
        "SELECT id, date, description, amount, currency, category, subcategory "
        f"FROM transactions{where} ORDER BY date, id",
        params,
    ).fetchall()
    if not rows:
        return "No matching transactions."
    out = []
    for r in rows[:100]:
        out.append(
            f"#{r['id']} | {r['date']} | {r['description']} | "
            f"{r['amount']} {r['currency']} | {r['category']}/{r['subcategory']}"
        )
    return "\n".join(out)


TOOLS = [sql_query, vector_search, filter]

SYSTEM_PROMPT = """\
You answer questions about a small business's bank transactions, stored in a
SQLite table. You have three tools and must choose the right one:

- sql_query: USE THIS BY DEFAULT for anything numeric — totals, sums, counts,
  "how much", "how many", per-category or per-month figures. It is exact and
  iterates every matching row.
- vector_search: ONLY for vague/semantic questions where you cannot write an exact
  SQL filter (e.g. "was there anything about coffee or eating out?"). Never use it
  to compute a total.
- filter: to list/narrow transactions by date range, vendor (description
  substring), or category.

Rules:
- Amounts are SIGNED: inflows positive, outflows (expenses) negative. When asked
  "how much was spent", report a positive number.
- Use the transaction's OWN currency from the `currency` column. These are GBP, so
  format money with £ (e.g. £171.30). Never write $ unless the currency is USD.
- ALWAYS cite the source transaction id(s) you used, in the form [#id, #id].
- If the data does not contain the answer, say so plainly. NEVER invent figures.
- Be concise. Give the figure (with currency) and the citation.
"""

_agent = None


def get_agent():
    """Build (once) the prebuilt tool-calling agent graph over our three tools."""
    global _agent
    if _agent is None:
        from langgraph.prebuilt import create_react_agent

        _agent = create_react_agent(get_llm(), TOOLS, prompt=SYSTEM_PROMPT)
    return _agent


def answer_question(question: str) -> dict:
    """Run the agent on a question. Returns answer text + which tools it used."""
    result = get_agent().invoke(
        {"messages": [{"role": "user", "content": question}]}
    )
    messages = result["messages"]

    tools_used: list[str] = []
    for m in messages:
        for call in getattr(m, "tool_calls", None) or []:
            tools_used.append(call["name"])

    answer = ""
    for m in reversed(messages):
        if m.__class__.__name__ == "AIMessage" and getattr(m, "content", ""):
            answer = m.content if isinstance(m.content, str) else str(m.content)
            break

    return {"answer": answer, "tools_used": tools_used, "messages": messages}
