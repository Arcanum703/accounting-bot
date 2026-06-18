"""Telegram bot (polling mode) for the cash-basis income statement.

Phase 1 handlers:
  document (.csv/.xlsx)  -> parse -> classify -> store -> aggregate -> reply
  /statement             -> re-send the current income statement
  /reset                 -> clear the store + classification cache
  /start, /help          -> short usage text

Polling mode (CLAUDE.md Stack): no webhook / public server needed.

Run:  python -m src.bot.main      (after setting TELEGRAM_BOT_TOKEN in .env)
"""

from __future__ import annotations

import logging
import os
import tempfile

from dotenv import load_dotenv
from telegram import Update
from telegram.constants import ParseMode
from telegram.ext import (
    Application,
    CommandHandler,
    ContextTypes,
    MessageHandler,
    filters,
)

from src.core import aggregate, pipeline, store

logging.basicConfig(
    format="%(asctime)s %(name)s %(levelname)s %(message)s", level=logging.INFO
)
log = logging.getLogger("accounting-bot")

START_TEXT = (
    "📒 *Accounting bot*\n\n"
    "Send me a bank statement (CSV or XLSX, Revolut export layout) and I'll reply "
    "with a *cash-basis income statement*: revenue, expenses by category, the "
    "excluded items (transfers, loans, owner draws, capital, refunds) shown "
    "separately, and net income.\n\n"
    "Commands:\n"
    "  /statement — re-send the current statement\n"
    "  /reset — clear stored data\n"
    "  /help — this message\n\n"
    "_Note: this is a cash-basis summary of bank cash movements only — not a "
    "balance sheet._"
)


def _mono(text: str) -> str:
    """Wrap text in a Telegram monospace block."""
    return f"```\n{text}\n```"


async def start(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    await update.message.reply_text(START_TEXT, parse_mode=ParseMode.MARKDOWN)


async def reset(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    conn = context.bot_data["conn"]
    store.reset(conn)
    await update.message.reply_text("🧹 Store and classification cache cleared.")


async def statement(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    conn = context.bot_data["conn"]
    if store.count_transactions(conn) == 0:
        await update.message.reply_text(
            "No data yet — send me a statement file first."
        )
        return
    summary = pipeline.build_summary(conn)
    await update.message.reply_text(
        _mono(aggregate.render_statement_text(summary)),
        parse_mode=ParseMode.MARKDOWN,
    )


async def handle_document(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    conn = context.bot_data["conn"]
    doc = update.message.document
    ext = os.path.splitext(doc.file_name or "")[1].lower()
    if ext not in (".csv", ".xlsx", ".xls"):
        await update.message.reply_text(
            "Please send a .csv or .xlsx bank statement."
        )
        return

    await update.message.reply_text(f"📥 Got *{doc.file_name}* — processing…",
                                    parse_mode=ParseMode.MARKDOWN)

    tg_file = await doc.get_file()
    with tempfile.TemporaryDirectory() as tmp:
        local_path = os.path.join(tmp, doc.file_name or f"statement{ext}")
        await tg_file.download_to_drive(custom_path=local_path)
        try:
            stats = pipeline.ingest(local_path, conn)
        except Exception as exc:  # surface parse/classify errors to the user
            log.exception("Ingest failed")
            await update.message.reply_text(f"⚠️ Could not process the file: {exc}")
            return

    summary = pipeline.build_summary(conn)
    header = (
        f"✅ Processed {stats.rows_parsed} transactions "
        f"({stats.rows_inserted} new"
        + (f", {stats.cache_hits} cached classifications" if stats.cache_hits else "")
        + ")."
    )
    await update.message.reply_text(header)
    await update.message.reply_text(
        _mono(aggregate.render_statement_text(summary)),
        parse_mode=ParseMode.MARKDOWN,
    )


def build_application(token: str) -> Application:
    app = Application.builder().token(token).build()

    conn = store.get_connection()
    store.init_db(conn)
    app.bot_data["conn"] = conn

    app.add_handler(CommandHandler(["start", "help"], start))
    app.add_handler(CommandHandler("reset", reset))
    app.add_handler(CommandHandler("statement", statement))
    app.add_handler(MessageHandler(filters.Document.ALL, handle_document))
    return app


def main() -> None:
    load_dotenv()
    token = os.environ.get("TELEGRAM_BOT_TOKEN")
    if not token:
        raise SystemExit(
            "TELEGRAM_BOT_TOKEN is not set. Copy .env.example to .env and add your "
            "token from @BotFather."
        )
    app = build_application(token)
    log.info("Starting bot (polling)…")
    app.run_polling(allowed_updates=Update.ALL_TYPES)


if __name__ == "__main__":
    main()
