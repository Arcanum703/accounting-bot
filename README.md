# Accounting Telegram Bot

A Telegram bot that turns a bank statement (CSV/XLSX) into a **cash-basis income
statement**. Designed for the Central Asian market (RU / KY / EN documents),
demoed on English Revolut data.

> **Scope (honest):** this produces a *cash-basis income statement* — a
> categorized summary of bank **cash movements** only. It is **not** a balance
> sheet / statement of financial position: a bank statement contains no
> asset/liability/equity data, so those cannot be produced here.

## Status

- **Phase 0** — synthetic Revolut statement + draft labels (`scripts/generate_data.py`),
  multilingual demo invoices (`scripts/generate_invoices.py`).
- **Phase 1 (current)** — runnable bot: deterministic parse → **temporary
  rule-based classifier** → SQLite → cash-basis income statement.
  The classifier is a keyword/Type heuristic and is replaced by an LLM in Phase 2.

## Setup

```bash
python -m venv .venv && . .venv/Scripts/activate   # Windows: .venv\Scripts\activate
pip install -r requirements.txt

cp .env.example .env        # then add your TELEGRAM_BOT_TOKEN from @BotFather
```

## Run the bot

```bash
python -m src.bot.main
```

Then in Telegram: send the statement file (`data/bank_statement.csv`) to your bot.
It replies with the income statement.

Commands: `/statement` (re-send), `/reset` (clear data), `/start` · `/help`.

## Try it without Telegram

Run the pipeline straight from the CLI to see the statement printed:

```bash
python -m src.core.pipeline data/bank_statement.csv
```

## Layout

```
src/core/parse.py      pandas → canonical transactions (Revolut column map)
src/core/taxonomy.py   the three buckets: revenue / expense / excluded
src/core/classify.py   TEMPORARY rule classifier (replaced by LLM in Phase 2)
src/core/store.py      SQLite: transactions + classification_cache
src/core/aggregate.py  deterministic roll-up + statement formatting
src/core/pipeline.py   parse → classify → store → aggregate
src/bot/main.py        python-telegram-bot, polling mode
```
