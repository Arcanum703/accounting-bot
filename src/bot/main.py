"""Telegram bot (polling mode) for the cash-basis income statement.

Handlers:
  document (.csv/.xlsx)  -> parse -> classify -> store -> aggregate -> reply
  free text              -> Q&A agent (sql_query / vector_search / filter), cited
  /statement             -> re-send the current income statement
  /reset                 -> clear the store + classification cache + vector index
  /start, /help          -> short usage text

Polling mode (CLAUDE.md Stack): no webhook / public server needed.

Run:  python -m src.bot.main      (after setting TELEGRAM_BOT_TOKEN in .env)
"""

from __future__ import annotations

import asyncio
import logging
import os
import tempfile

from dotenv import load_dotenv
from telegram import Update
from telegram.constants import ChatAction, ParseMode
from telegram.ext import (
    Application,
    CommandHandler,
    ContextTypes,
    MessageHandler,
    filters,
)

from src.core import aggregate, pipeline, qa_agent, store

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
    "Once a statement is loaded, just *ask questions* in plain language, e.g. "
    "“how much did I spend on utilities in May?” or “anything about eating out "
    "abroad?” — answers cite the exact transaction ids.\n\n"
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
            result = pipeline.run_pipeline(local_path, conn)
        except Exception as exc:  # surface parse/classify errors to the user
            log.exception("Pipeline failed")
            await update.message.reply_text(f"⚠️ Could not process the file: {exc}")
            return

    header = (
        f"✅ Processed {result.rows_parsed} transactions "
        f"({result.rows_inserted} new"
        + (f", {result.cache_hits} cached classifications" if result.cache_hits else "")
        + ")."
    )
    warnings = result.validation.warnings()
    if warnings:
        header += "\n\n⚠️ " + "\n⚠️ ".join(warnings)
    await update.message.reply_text(header)
    await update.message.reply_text(
        _mono(aggregate.render_statement_text(result.summary)),
        parse_mode=ParseMode.MARKDOWN,
    )


async def handle_text(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    """Free-text questions -> Q&A agent (sql_query / vector_search / filter)."""
    conn = context.bot_data["conn"]
    if store.count_transactions(conn) == 0:
        await update.message.reply_text(
            "No data yet — send me a statement file first, then ask away."
        )
        return

    question = update.message.text
    await update.message.chat.send_action(ChatAction.TYPING)
    try:
        # The agent runs synchronous LLM/tool calls; off-load it so the event
        # loop isn't blocked. Tools open their own SQLite connections, so this is
        # thread-safe.
        result = await asyncio.to_thread(qa_agent.answer_question, question)
    except Exception as exc:
        log.exception("Q&A failed")
        await update.message.reply_text(f"⚠️ Sorry, I couldn't answer that: {exc}")
        return

    answer = result["answer"] or "I couldn't find an answer in the data."
    if result["tools_used"]:
        answer += f"\n\n_via {', '.join(dict.fromkeys(result['tools_used']))}_"
    await update.message.reply_text(answer, parse_mode=ParseMode.MARKDOWN)


def build_application(token: str) -> Application:
    app = Application.builder().token(token).build()

    conn = store.get_connection()
    store.init_db(conn)
    app.bot_data["conn"] = conn

    app.add_handler(CommandHandler(["start", "help"], start))
    app.add_handler(CommandHandler("reset", reset))
    app.add_handler(CommandHandler("statement", statement))
    app.add_handler(MessageHandler(filters.Document.ALL, handle_document))
    # Any non-command text -> Q&A agent.
    app.add_handler(MessageHandler(filters.TEXT & ~filters.COMMAND, handle_text))
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
