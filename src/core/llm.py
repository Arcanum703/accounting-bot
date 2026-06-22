"""LLM factory — a single place to construct the chat model (CLAUDE.md Stack).

Keeping construction behind one `get_llm()` makes the provider/model swappable and
pins the reproducibility settings (temperature 0, CLAUDE.md #2) in one spot.

Provider is chosen with the LLM_PROVIDER env var:
  * "cerebras" (default) -> ChatCerebras, model from CEREBRAS_MODEL.
  * "gemini"             -> ChatGoogleGenerativeAI, model from GEMINI_MODEL.

⚠️ Gemini branch only: do NOT use a Gemini 3.x model — langchain-google-genai
forces temperature to 1.0 on Gemini 3+, which destroys reproducibility. This
factory refuses 3.x on that branch.

Both providers support temperature 0 and `.with_structured_output(...)`, so the
classifier code is provider-agnostic.

Note: this is for the CHAT model only. Embeddings stay LOCAL
(sentence-transformers, added in the Q&A phase) — never a hosted embedding model.
"""

from __future__ import annotations

import os
from functools import lru_cache

DEFAULT_PROVIDER = "cerebras"
# Available Cerebras models are key/account-specific — list yours via the
# /v1/models endpoint. gpt-oss-120b is a current, broadly available model with
# OpenAI-style tool calling (needed for with_structured_output).
DEFAULT_CEREBRAS_MODEL = "gpt-oss-120b"
DEFAULT_GEMINI_MODEL = "gemini-2.5-flash"
DEFAULT_TEMPERATURE = 0.0

# Client-side throttle (requests/min). Only affects TIMING, never outputs, so
# reproducibility is unaffected. Gemini's free tier is very small; Cerebras is
# more generous. Override with GEMINI_RPM / CEREBRAS_RPM.
GEMINI_RPM = float(os.environ.get("GEMINI_RPM", "4.5"))
CEREBRAS_RPM = float(os.environ.get("CEREBRAS_RPM", "25"))


def get_provider() -> str:
    return os.environ.get("LLM_PROVIDER", DEFAULT_PROVIDER).strip().lower()


def get_model(provider: str | None = None) -> str:
    provider = provider or get_provider()
    if provider == "cerebras":
        return os.environ.get("CEREBRAS_MODEL", DEFAULT_CEREBRAS_MODEL)
    if provider == "gemini":
        return os.environ.get("GEMINI_MODEL", DEFAULT_GEMINI_MODEL)
    raise ValueError(f"Unknown LLM_PROVIDER {provider!r} (use 'cerebras' or 'gemini').")


def get_active() -> tuple[str, str]:
    """Return the (provider, model) currently configured — used for cache keys."""
    provider = get_provider()
    return provider, get_model(provider)


def _rate_limiter(rpm: float):
    from langchain_core.rate_limiters import InMemoryRateLimiter

    return InMemoryRateLimiter(
        requests_per_second=rpm / 60.0,
        check_every_n_seconds=0.5,
        max_bucket_size=1,  # strict drip; no bursting past the per-minute cap
    )


@lru_cache(maxsize=16)
def _build(provider: str, model: str, temperature: float):
    if provider == "cerebras":
        from langchain_cerebras import ChatCerebras

        if not os.environ.get("CEREBRAS_API_KEY"):
            raise RuntimeError(
                "CEREBRAS_API_KEY is not set. Copy .env.example to .env and add your "
                "key from https://cloud.cerebras.ai"
            )
        return ChatCerebras(
            model=model,
            temperature=temperature,
            rate_limiter=_rate_limiter(CEREBRAS_RPM),
            max_retries=5,
        )

    if provider == "gemini":
        from langchain_google_genai import ChatGoogleGenerativeAI

        if model.startswith(("gemini-3", "gemini-4")):
            raise ValueError(
                f"Refusing model {model!r}: Gemini 3+ forces temperature=1.0 in "
                "langchain-google-genai, breaking reproducibility. Use gemini-2.5-flash."
            )
        if not os.environ.get("GOOGLE_API_KEY"):
            raise RuntimeError(
                "GOOGLE_API_KEY is not set. Copy .env.example to .env and add your key "
                "from https://aistudio.google.com/apikey"
            )
        return ChatGoogleGenerativeAI(
            model=model,
            temperature=temperature,
            google_api_key=os.environ["GOOGLE_API_KEY"],
            rate_limiter=_rate_limiter(GEMINI_RPM),
            max_retries=5,
        )

    raise ValueError(f"Unknown LLM_PROVIDER {provider!r} (use 'cerebras' or 'gemini').")


def get_llm(provider: str | None = None, model: str | None = None,
            temperature: float = DEFAULT_TEMPERATURE):
    """Return a configured chat model for the active (or given) provider/model."""
    provider = provider or get_provider()
    model = model or get_model(provider)
    return _build(provider, model, temperature)
