#!/usr/bin/env python3
"""Generate multilingual demo invoices as clean digital PDFs (Phase 0b).

See CLAUDE.md. Produces ~6 invoices under data/invoices/:
  * 2 Russian, 2 Kyrgyz, 2 English
  * each with vendor, invoice number, date, 2-4 line items
    (description, qty, unit price), subtotal, tax, total, currency.

These PDFs carry a REAL selectable text layer (drawn with reportlab using a
Unicode TrueType font) — they are NOT scans, so no OCR is needed to read them.

IMPORTANT (single-source-of-truth rule, CLAUDE.md):
  These invoices are an ALTERNATE demo input for the multilingual posture. They
  are NOT merged with data/bank_statement.csv and must never be aggregated
  together with it — the bank statement is one input; these invoices are another.

Reproducible: fixed data, no randomness. Re-running overwrites identical PDFs.

Fonts: needs a Unicode TTF that covers Cyrillic incl. the extra Kyrgyz letters
(ң ө ү). Tries Arial (Windows) then DejaVuSans (Linux); both qualify.
"""

from __future__ import annotations

from decimal import Decimal, ROUND_HALF_UP
from pathlib import Path

from reportlab.lib import colors
from reportlab.lib.pagesizes import A4
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.lib.units import mm
from reportlab.pdfbase import pdfmetrics
from reportlab.pdfbase.ttfonts import TTFont
from reportlab.platypus import (
    SimpleDocTemplate, Paragraph, Spacer, Table, TableStyle,
)

PROJECT_ROOT = Path(__file__).resolve().parents[1]
OUT_DIR = PROJECT_ROOT / "data" / "invoices"

FONT = "InvFont"
FONT_BOLD = "InvFont-Bold"

# (regular, bold) candidates, in priority order.
_FONT_CANDIDATES = [
    (r"C:\Windows\Fonts\arial.ttf", r"C:\Windows\Fonts\arialbd.ttf"),
    ("/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf",
     "/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf"),
    ("/Library/Fonts/Arial.ttf", "/Library/Fonts/Arial Bold.ttf"),
]


def register_fonts() -> None:
    for regular, bold in _FONT_CANDIDATES:
        if Path(regular).exists() and Path(bold).exists():
            pdfmetrics.registerFont(TTFont(FONT, regular))
            pdfmetrics.registerFont(TTFont(FONT_BOLD, bold))
            return
    raise RuntimeError(
        "No Unicode TTF font found for Cyrillic/Kyrgyz text. Install DejaVu or "
        "Arial, or add a path to _FONT_CANDIDATES."
    )


# --------------------------------------------------------------------------- #
# Localized field labels (the static text on each invoice)
# --------------------------------------------------------------------------- #

LABELS = {
    "ru": {
        "invoice": "СЧЁТ-ФАКТУРА",
        "number": "Счёт №",
        "date": "Дата",
        "seller": "Поставщик",
        "buyer": "Покупатель",
        "buyer_name": "ОсОО «Дизайн-Студия Алуа»",
        "col_desc": "Наименование",
        "col_qty": "Кол-во",
        "col_price": "Цена за ед.",
        "col_amount": "Сумма",
        "subtotal": "Промежуточный итог",
        "total": "Итого к оплате",
    },
    "ky": {
        "invoice": "ЭСЕП-ФАКТУРА",
        "number": "Эсеп №",
        "date": "Күнү",
        "seller": "Сатуучу",
        "buyer": "Сатып алуучу",
        "buyer_name": "«Алуа Дизайн-Студия» ЖЧК",
        "col_desc": "Аталышы",
        "col_qty": "Саны",
        "col_price": "Бирдик баасы",
        "col_amount": "Сумма",
        "subtotal": "Аралык жыйынтык",
        "total": "Жалпы төлөө",
    },
    "en": {
        "invoice": "INVOICE",
        "number": "Invoice No.",
        "date": "Date",
        "seller": "Seller",
        "buyer": "Bill To",
        "buyer_name": "Alua Design Studio LLC",
        "col_desc": "Description",
        "col_qty": "Qty",
        "col_price": "Unit Price",
        "col_amount": "Amount",
        "subtotal": "Subtotal",
        "total": "Total Due",
    },
}


# --------------------------------------------------------------------------- #
# The invoices (single source of truth for this alternate input)
# --------------------------------------------------------------------------- #
# items: (description, qty, unit_price)

INVOICES = [
    # ----------------------------- Russian -----------------------------
    {
        "lang": "ru",
        "file": "invoice_ru_1.pdf",
        "vendor": "ОсОО «Беркут-Снаб»",
        "vendor_meta": "г. Бишкек, ул. Чүй 154 · ИНН 01234567890123",
        "number": "БС-2025-0418",
        "date": "18.04.2025",
        "currency": "сом",
        "tax_rate": Decimal("0.12"),
        "tax_label": "НДС (12%)",
        "items": [
            ("Бумага офисная A4, 80 г/м² (пачка 500 л.)", 12, Decimal("420.00")),
            ("Картридж для принтера HP 106A", 4, Decimal("2150.00")),
            ("Папка-регистратор 50 мм", 20, Decimal("185.00")),
        ],
    },
    {
        "lang": "ru",
        "file": "invoice_ru_2.pdf",
        "vendor": "ИП Жумабеков А. К.",
        "vendor_meta": "г. Ош, ул. Ленина 28 · ИНН 22109876543210",
        "number": "ЖАК-0091",
        "date": "06.05.2025",
        "currency": "сом",
        "tax_rate": Decimal("0.12"),
        "tax_label": "НДС (12%)",
        "items": [
            ("Услуги курьерской доставки по городу", 8, Decimal("250.00")),
            ("Широкоформатная печать баннера 2×1 м", 3, Decimal("1450.00")),
            ("Ламинация листов A3", 40, Decimal("35.00")),
        ],
    },
    # ----------------------------- Kyrgyz ------------------------------
    {
        "lang": "ky",
        "file": "invoice_ky_1.pdf",
        "vendor": "«Ак-Тилек Соода» ЖЧК",
        "vendor_meta": "Бишкек ш., Манас проспекти 78 · ИНН 33450912784500",
        "number": "АТ-2025-77",
        "date": "22.04.2025",
        "currency": "сом",
        "tax_rate": Decimal("0.12"),
        "tax_label": "КНС (12%)",
        "items": [
            ("Кеңсе кагазы A4 (500 барак)", 10, Decimal("415.00")),
            ("Принтер картриджи (кара)", 3, Decimal("2200.00")),
            ("Папка-файл, пластик", 25, Decimal("28.00")),
            ("Шарик калемсап, көк", 50, Decimal("12.00")),
        ],
    },
    {
        "lang": "ky",
        "file": "invoice_ky_2.pdf",
        "vendor": "«Манас Логистик» ЖЧК",
        "vendor_meta": "Бишкек ш., Жибек Жолу 412 · ИНН 33988201640077",
        "number": "МЛ-0345",
        "date": "12.06.2025",
        "currency": "сом",
        "tax_rate": Decimal("0.12"),
        "tax_label": "КНС (12%)",
        "items": [
            ("Жүк жеткирүү кызматы (шаар ичинде)", 6, Decimal("300.00")),
            ("Сактоо кампасынын ижарасы (айына)", 1, Decimal("8500.00")),
            ("Таңгактоо материалдары", 15, Decimal("75.00")),
        ],
    },
    # ----------------------------- English -----------------------------
    {
        "lang": "en",
        "file": "invoice_en_1.pdf",
        "vendor": "Tien Shan Trading LLC",
        "vendor_meta": "78 Manas Ave, Bishkek · Reg. 33450912784500",
        "number": "TST-2025-1042",
        "date": "2025-05-09",
        "currency": "USD",
        "tax_rate": Decimal("0.12"),
        "tax_label": "VAT (12%)",
        "items": [
            ("Branded tote bags, screen-printed", 100, Decimal("3.40")),
            ("Vinyl sticker sheets, A5", 250, Decimal("0.85")),
            ("Setup & artwork fee", 1, Decimal("60.00")),
        ],
    },
    {
        "lang": "en",
        "file": "invoice_en_2.pdf",
        "vendor": "Silk Road Supplies Co.",
        "vendor_meta": "28 Lenin St, Osh · Reg. 22109876543210",
        "number": "SRS-0188",
        "date": "2025-06-17",
        "currency": "USD",
        "tax_rate": Decimal("0.12"),
        "tax_label": "VAT (12%)",
        "items": [
            ("Recycled kraft paper, roll 100 m", 6, Decimal("18.50")),
            ("Eco ink cartridge, black", 4, Decimal("24.00")),
            ("Express courier, intercity", 2, Decimal("12.75")),
        ],
    },
]


# --------------------------------------------------------------------------- #
# Rendering
# --------------------------------------------------------------------------- #

def _money(value: Decimal) -> Decimal:
    return value.quantize(Decimal("0.01"), rounding=ROUND_HALF_UP)


def _fmt(value: Decimal, currency: str) -> str:
    return f"{_money(value):,.2f} {currency}"


def build_invoice(inv: dict) -> None:
    lang = inv["lang"]
    L = LABELS[lang]
    currency = inv["currency"]

    styles = getSampleStyleSheet()
    base = ParagraphStyle("base", parent=styles["Normal"],
                          fontName=FONT, fontSize=10, leading=13)
    small = ParagraphStyle("small", parent=base, fontSize=8,
                           textColor=colors.grey)
    h_vendor = ParagraphStyle("vendor", parent=base, fontName=FONT_BOLD,
                              fontSize=15, leading=18)
    h_title = ParagraphStyle("title", parent=base, fontName=FONT_BOLD,
                             fontSize=13, alignment=2, textColor=colors.HexColor("#33475b"))
    label = ParagraphStyle("label", parent=small, fontName=FONT_BOLD,
                           textColor=colors.grey)

    doc = SimpleDocTemplate(
        str(OUT_DIR / inv["file"]), pagesize=A4,
        leftMargin=20 * mm, rightMargin=20 * mm,
        topMargin=18 * mm, bottomMargin=18 * mm,
        title=f"{L['invoice']} {inv['number']}",
        author=inv["vendor"],
    )

    story = []

    # Header: vendor block (left) + INVOICE title (right)
    header = Table(
        [[Paragraph(inv["vendor"], h_vendor), Paragraph(L["invoice"], h_title)],
         [Paragraph(inv["vendor_meta"], small), ""]],
        colWidths=[100 * mm, 70 * mm],
    )
    header.setStyle(TableStyle([
        ("VALIGN", (0, 0), (-1, -1), "TOP"),
        ("SPAN", (1, 0), (1, 1)),
        ("BOTTOMPADDING", (0, 0), (-1, -1), 2),
    ]))
    story.append(header)
    story.append(Spacer(1, 8 * mm))

    # Meta: number / date / buyer
    meta = Table(
        [[Paragraph(L["number"], label), Paragraph(inv["number"], base),
          Paragraph(L["date"], label), Paragraph(inv["date"], base)],
         [Paragraph(L["buyer"], label),
          Paragraph(L["buyer_name"], base), "", ""]],
        colWidths=[28 * mm, 57 * mm, 20 * mm, 65 * mm],
    )
    meta.setStyle(TableStyle([
        ("VALIGN", (0, 0), (-1, -1), "TOP"),
        ("SPAN", (1, 1), (3, 1)),
        ("BOTTOMPADDING", (0, 0), (-1, -1), 4),
    ]))
    story.append(meta)
    story.append(Spacer(1, 6 * mm))

    # Line-item table
    head_style = ParagraphStyle("th", parent=base, fontName=FONT_BOLD,
                                fontSize=9, textColor=colors.white)
    cell = ParagraphStyle("td", parent=base, fontSize=9, leading=12)
    cell_r = ParagraphStyle("tdr", parent=cell, alignment=2)

    rows = [[
        Paragraph(L["col_desc"], head_style),
        Paragraph(L["col_qty"], head_style),
        Paragraph(L["col_price"], head_style),
        Paragraph(L["col_amount"], head_style),
    ]]

    subtotal = Decimal("0.00")
    for desc, qty, unit in inv["items"]:
        line_total = _money(unit * qty)
        subtotal += line_total
        rows.append([
            Paragraph(desc, cell),
            Paragraph(str(qty), cell_r),
            Paragraph(_fmt(unit, currency), cell_r),
            Paragraph(_fmt(line_total, currency), cell_r),
        ])

    tax = _money(subtotal * inv["tax_rate"])
    total = _money(subtotal + tax)

    item_table = Table(rows, colWidths=[88 * mm, 18 * mm, 32 * mm, 32 * mm],
                       repeatRows=1)
    item_table.setStyle(TableStyle([
        ("BACKGROUND", (0, 0), (-1, 0), colors.HexColor("#33475b")),
        ("VALIGN", (0, 0), (-1, -1), "MIDDLE"),
        ("ALIGN", (1, 0), (-1, -1), "RIGHT"),
        ("LINEBELOW", (0, 0), (-1, -1), 0.4, colors.HexColor("#cccccc")),
        ("ROWBACKGROUNDS", (0, 1), (-1, -1), [colors.white, colors.HexColor("#f5f7fa")]),
        ("TOPPADDING", (0, 0), (-1, -1), 5),
        ("BOTTOMPADDING", (0, 0), (-1, -1), 5),
        ("LEFTPADDING", (0, 0), (-1, -1), 6),
        ("RIGHTPADDING", (0, 0), (-1, -1), 6),
    ]))
    story.append(item_table)
    story.append(Spacer(1, 5 * mm))

    # Totals block (right-aligned)
    tot_label = ParagraphStyle("totlabel", parent=base, alignment=2)
    tot_value = ParagraphStyle("totval", parent=base, alignment=2)
    tot_label_b = ParagraphStyle("totlabelb", parent=base, fontName=FONT_BOLD,
                                 alignment=2, fontSize=11)
    tot_value_b = ParagraphStyle("totvalb", parent=base, fontName=FONT_BOLD,
                                 alignment=2, fontSize=11)

    totals = Table(
        [[Paragraph(L["subtotal"], tot_label), Paragraph(_fmt(subtotal, currency), tot_value)],
         [Paragraph(inv["tax_label"], tot_label), Paragraph(_fmt(tax, currency), tot_value)],
         [Paragraph(L["total"], tot_label_b), Paragraph(_fmt(total, currency), tot_value_b)]],
        colWidths=[110 * mm, 60 * mm],
    )
    totals.setStyle(TableStyle([
        ("LINEABOVE", (0, 2), (-1, 2), 0.8, colors.HexColor("#33475b")),
        ("TOPPADDING", (0, 0), (-1, -1), 3),
        ("BOTTOMPADDING", (0, 0), (-1, -1), 3),
    ]))
    story.append(totals)

    doc.build(story)


def main() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    register_fonts()
    for inv in INVOICES:
        build_invoice(inv)
        print(f"Wrote {inv['lang'].upper()} invoice {inv['number']:<14} "
              f"-> {(OUT_DIR / inv['file']).relative_to(PROJECT_ROOT)}")
    print(f"\n{len(INVOICES)} invoices written to {OUT_DIR.relative_to(PROJECT_ROOT)}")
    print("NOTE: alternate demo input — NOT merged with data/bank_statement.csv.")


if __name__ == "__main__":
    main()
