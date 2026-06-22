"""SQLite store — the SOURCE OF TRUTH (CLAUDE.md #4) — plus a local Chroma index.

SQLite tables:
  transactions        — every classified transaction (the data aggregation sums).
  classification_cache — keyed by (provider, model, normalized description), so each
                         distinct description is classified once per model and the
                         same statement classified twice gives identical totals.

Chroma (local, in-process): a `transactions` collection holding each description's
embedding with the SQLite transaction id as the document id/metadata. Embeddings
are LOCAL (sentence-transformers, multilingual) — never a hosted model.

IMPORTANT (CLAUDE.md #4/#5): the vector index is ONLY for semantic Q&A retrieval.
Statement totals are ALWAYS computed by SQL over every row, never from Chroma
(vector search returns top-k and would silently drop rows).
"""

from __future__ import annotations

import os
import re
import sqlite3

# Local multilingual embedding model (EN/RU/KY) — runs on-device, no API key.
EMBED_MODEL = "paraphrase-multilingual-MiniLM-L12-v2"
_CHROMA_COLLECTION = "transactions"


def chroma_path() -> str:
    return os.environ.get(
        "CHROMA_PATH",
        os.path.join(os.path.dirname(db_path()), "chroma"),
    )


def use_local_hf_cache_if_available() -> None:
    """If the embedding model is already downloaded, load it OFFLINE.

    Avoids a network round-trip (and the transient failures that come with it) on
    every embedding load, and makes runs reproducible. Call this BEFORE
    huggingface_hub is first imported (e.g. at the top of an entry point); it's a
    no-op once the model isn't cached yet (first run downloads it normally).
    """
    if os.environ.get("HF_HUB_OFFLINE"):
        return
    hf_home = os.environ.get("HF_HOME") or os.path.join(
        os.path.expanduser("~"), ".cache", "huggingface"
    )
    marker = os.path.join(
        hf_home, "hub",
        "models--sentence-transformers--" + EMBED_MODEL.replace("/", "--"),
    )
    if os.path.isdir(marker):
        os.environ["HF_HUB_OFFLINE"] = "1"
        os.environ["TRANSFORMERS_OFFLINE"] = "1"

# Default store location; override with the DB_PATH env var.
_DEFAULT_DB = os.path.join(
    os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))),
    "data",
    "accounting.db",
)


def db_path() -> str:
    return os.environ.get("DB_PATH", _DEFAULT_DB)


def normalize_description(description: str) -> str:
    """Cache key: lowercased, whitespace-collapsed, trimmed."""
    return re.sub(r"\s+", " ", description.strip().lower())


def get_connection(path: str | None = None) -> sqlite3.Connection:
    path = path or db_path()
    os.makedirs(os.path.dirname(path), exist_ok=True)
    conn = sqlite3.connect(path)
    conn.row_factory = sqlite3.Row
    return conn


# The cache is keyed by (provider, model, normalized_description) so each model's
# results are stored separately — switching provider/model re-classifies instead
# of serving another model's labels.
_CREATE_CACHE_SQL = """
    CREATE TABLE IF NOT EXISTS classification_cache (
        provider               TEXT NOT NULL,
        model                  TEXT NOT NULL,
        normalized_description TEXT NOT NULL,
        subcategory            TEXT NOT NULL,
        category               TEXT NOT NULL,
        confidence             REAL,
        justification          TEXT,
        PRIMARY KEY (provider, model, normalized_description)
    );
"""


def init_db(conn: sqlite3.Connection) -> None:
    conn.executescript(
        """
        CREATE TABLE IF NOT EXISTS transactions (
            id            INTEGER PRIMARY KEY,
            date          TEXT NOT NULL,
            description   TEXT NOT NULL,
            amount        REAL NOT NULL,
            currency      TEXT NOT NULL,
            category      TEXT NOT NULL,
            subcategory   TEXT NOT NULL,
            confidence    REAL,
            justification TEXT,
            source_ref    TEXT UNIQUE          -- dedupe re-uploads of the same row
        );
        """
    )
    _ensure_cache_schema(conn)
    conn.commit()


def _model_from_justification(justification: str | None) -> str | None:
    """Legacy entries tag the model as a '[model] ...' prefix in the justification."""
    if justification and justification.startswith("["):
        end = justification.find("]")
        if end > 1:
            return justification[1:end].strip()
    return None


def _ensure_cache_schema(conn: sqlite3.Connection) -> None:
    cols = [r[1] for r in conn.execute("PRAGMA table_info(classification_cache)").fetchall()]
    if not cols:
        conn.execute(_CREATE_CACHE_SQL)
        return
    if "provider" in cols:
        return  # already migrated

    # Migrate the old (description-only) schema WITHOUT wiping entries. All legacy
    # rows were produced by the gemini path; recover each row's model from its
    # justification tag, falling back to the configured Gemini model.
    legacy = conn.execute(
        "SELECT normalized_description, subcategory, category, confidence, justification "
        "FROM classification_cache"
    ).fetchall()
    fallback_model = os.environ.get("GEMINI_MODEL", "gemini-2.5-flash")

    conn.execute("ALTER TABLE classification_cache RENAME TO classification_cache_legacy")
    conn.execute(_CREATE_CACHE_SQL)
    for r in legacy:
        model = _model_from_justification(r["justification"]) or fallback_model
        conn.execute(
            "INSERT OR REPLACE INTO classification_cache "
            "(provider, model, normalized_description, subcategory, category, "
            " confidence, justification) VALUES (?, ?, ?, ?, ?, ?, ?)",
            ("gemini", model, r["normalized_description"], r["subcategory"],
             r["category"], r["confidence"], r["justification"]),
        )
    conn.execute("DROP TABLE classification_cache_legacy")


def reset(conn: sqlite3.Connection) -> None:
    """Clear the store, the classification cache, and the vector index (for /reset)."""
    conn.executescript("DELETE FROM transactions; DELETE FROM classification_cache;")
    conn.commit()
    reset_vectors()


def insert_transactions(conn: sqlite3.Connection, rows: list[dict]) -> int:
    """Insert classified transactions. Duplicate source_refs are ignored.

    Returns the number of rows actually inserted.
    """
    before = conn.total_changes
    conn.executemany(
        """
        INSERT OR IGNORE INTO transactions
            (date, description, amount, currency, category, subcategory,
             confidence, justification, source_ref)
        VALUES
            (:date, :description, :amount, :currency, :category, :subcategory,
             :confidence, :justification, :source_ref)
        """,
        rows,
    )
    conn.commit()
    return conn.total_changes - before


def fetch_all_transactions(conn: sqlite3.Connection) -> list[sqlite3.Row]:
    return conn.execute(
        "SELECT * FROM transactions ORDER BY date, id"
    ).fetchall()


def count_transactions(conn: sqlite3.Connection) -> int:
    return conn.execute("SELECT COUNT(*) FROM transactions").fetchone()[0]


# ---- Chroma vector index (semantic Q&A only) ------------------------------- #

_collection = None


def get_collection():
    """Return the (cached) local Chroma collection, creating it if needed.

    Chroma + sentence-transformers are imported lazily so the rest of the package
    works without the vector deps installed.
    """
    global _collection
    if _collection is not None:
        return _collection

    use_local_hf_cache_if_available()  # set offline BEFORE huggingface_hub imports
    import chromadb
    from chromadb.utils import embedding_functions

    client = chromadb.PersistentClient(path=chroma_path())
    ef = embedding_functions.SentenceTransformerEmbeddingFunction(
        model_name=EMBED_MODEL
    )
    _collection = client.get_or_create_collection(
        name=_CHROMA_COLLECTION,
        embedding_function=ef,
        metadata={"hnsw:space": "cosine"},
    )
    return _collection


def reset_vectors() -> None:
    """Drop and recreate the vector collection."""
    global _collection
    try:
        import chromadb
    except ImportError:
        return
    client = chromadb.PersistentClient(path=chroma_path())
    try:
        client.delete_collection(_CHROMA_COLLECTION)
    except Exception:
        pass  # didn't exist
    _collection = None  # force rebuild on next get_collection()


def index_source_refs(conn: sqlite3.Connection, source_refs: list[str]) -> int:
    """Embed the given transactions into Chroma (idempotent upsert).

    Called from the pipeline's store step. Uses the SQLite transaction id as the
    Chroma id so Q&A retrieval can cite the exact row.
    """
    if not source_refs:
        return 0
    placeholders = ",".join("?" * len(source_refs))
    rows = conn.execute(
        f"SELECT id, date, description, amount, currency, category, subcategory, "
        f"source_ref FROM transactions WHERE source_ref IN ({placeholders})",
        source_refs,
    ).fetchall()
    if not rows:
        return 0

    collection = get_collection()
    collection.upsert(
        ids=[str(r["id"]) for r in rows],
        documents=[r["description"] for r in rows],
        metadatas=[{
            "id": r["id"],
            "date": r["date"],
            "amount": r["amount"],
            "currency": r["currency"],
            "category": r["category"],
            "subcategory": r["subcategory"],
            "source_ref": r["source_ref"],
        } for r in rows],
    )
    return len(rows)


def vector_search(query: str, k: int = 5) -> list[dict]:
    """Semantic similarity search over descriptions. Returns up to k matches.

    For Q&A ONLY — never use this to compute statement totals (CLAUDE.md #4/#5).
    """
    collection = get_collection()
    if collection.count() == 0:
        return []
    res = collection.query(query_texts=[query], n_results=min(k, collection.count()))
    out = []
    docs = res.get("documents", [[]])[0]
    metas = res.get("metadatas", [[]])[0]
    dists = res.get("distances", [[]])[0]
    for doc, meta, dist in zip(docs, metas, dists):
        out.append({**meta, "description": doc, "distance": round(float(dist), 4)})
    return out


# ---- classification cache -------------------------------------------------- #

def get_cached(
    conn: sqlite3.Connection, provider: str, model: str, normalized: str
) -> sqlite3.Row | None:
    return conn.execute(
        "SELECT subcategory, category, confidence, justification "
        "FROM classification_cache "
        "WHERE provider = ? AND model = ? AND normalized_description = ?",
        (provider, model, normalized),
    ).fetchone()


def set_cached(
    conn: sqlite3.Connection,
    provider: str,
    model: str,
    normalized: str,
    subcategory: str,
    category: str,
    confidence: float,
    justification: str,
) -> None:
    conn.execute(
        "INSERT OR REPLACE INTO classification_cache "
        "(provider, model, normalized_description, subcategory, category, "
        " confidence, justification) VALUES (?, ?, ?, ?, ?, ?, ?)",
        (provider, model, normalized, subcategory, category, confidence, justification),
    )
    conn.commit()
