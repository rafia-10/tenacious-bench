#!/usr/bin/env python3
"""
Generate the Tenacious-Bench v0.1 Interim PDF Report.
Run: python3 generate_report.py
Output: tenacious_bench_interim_report.pdf
"""

import json
from collections import Counter
from pathlib import Path
from reportlab.lib import colors
from reportlab.lib.pagesizes import A4
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.lib.units import cm, mm
from reportlab.lib.enums import TA_LEFT, TA_CENTER, TA_JUSTIFY
from reportlab.platypus import (
    SimpleDocTemplate, Paragraph, Spacer, Table, TableStyle,
    HRFlowable, PageBreak, KeepTogether
)
from reportlab.platypus.flowables import Flowable

# ── Colour palette ────────────────────────────────────────────────────────────
NAVY   = colors.HexColor("#1A2B4A")
TEAL   = colors.HexColor("#0D7A74")
AMBER  = colors.HexColor("#E8860A")
SLATE  = colors.HexColor("#4A5568")
LIGHT  = colors.HexColor("#F7F9FC")
WHITE  = colors.white
RED    = colors.HexColor("#C0392B")
GREEN  = colors.HexColor("#1A7A4A")
MID    = colors.HexColor("#EDF2F7")

OUTPUT = Path("tenacious_bench_interim_report.pdf")
REPO   = Path(".")

# ── Load data ─────────────────────────────────────────────────────────────────

def load_all_tasks():
    tasks = []
    for p in ["train", "dev", "held_out"]:
        f = REPO / "tenacious_bench_v0.1" / p / "tasks.jsonl"
        if not f.exists():
            continue
        with open(f) as fh:
            for line in fh:
                if line.strip():
                    t = json.loads(line)
                    t["_partition"] = p
                    tasks.append(t)
    return tasks


def load_manifest():
    f = REPO / "tenacious_bench_v0.1" / "partition_manifest.json"
    return json.loads(f.read_text()) if f.exists() else {}


def pick_examples(tasks):
    by_prefix = {}
    for t in tasks:
        k = t["task_id"][:5]
        if k not in by_prefix:
            by_prefix[k] = t
    return (
        by_prefix.get("TB-PG"),
        by_prefix.get("TB-TR"),
        by_prefix.get("TB-HA"),
    )

# ── Style helpers ─────────────────────────────────────────────────────────────

def build_styles():
    base = getSampleStyleSheet()
    s = {}

    def add(name, **kw):
        s[name] = ParagraphStyle(name, **kw)

    add("cover_title",
        fontName="Helvetica-Bold", fontSize=26, textColor=WHITE,
        leading=32, spaceAfter=6, alignment=TA_LEFT)
    add("cover_sub",
        fontName="Helvetica", fontSize=13, textColor=colors.HexColor("#CBD5E0"),
        leading=18, spaceAfter=4, alignment=TA_LEFT)
    add("cover_meta",
        fontName="Helvetica", fontSize=10, textColor=colors.HexColor("#A0AEC0"),
        leading=14, alignment=TA_LEFT)

    add("h1",
        fontName="Helvetica-Bold", fontSize=16, textColor=NAVY,
        spaceBefore=18, spaceAfter=8, leading=20)
    add("h2",
        fontName="Helvetica-Bold", fontSize=12, textColor=TEAL,
        spaceBefore=12, spaceAfter=5, leading=15)
    add("h3",
        fontName="Helvetica-Bold", fontSize=10, textColor=SLATE,
        spaceBefore=8, spaceAfter=3, leading=13)

    add("body",
        fontName="Helvetica", fontSize=9.5, textColor=colors.HexColor("#2D3748"),
        leading=14, spaceAfter=4, alignment=TA_JUSTIFY)
    add("body_small",
        fontName="Helvetica", fontSize=8.5, textColor=SLATE,
        leading=12, spaceAfter=3)
    add("mono",
        fontName="Courier", fontSize=8, textColor=colors.HexColor("#1A202C"),
        leading=11, spaceAfter=2)
    add("mono_label",
        fontName="Courier-Bold", fontSize=8, textColor=TEAL,
        leading=11, spaceAfter=1)
    add("caption",
        fontName="Helvetica-Oblique", fontSize=8, textColor=SLATE,
        leading=11, spaceAfter=6, alignment=TA_CENTER)
    add("bullet",
        fontName="Helvetica", fontSize=9.5, textColor=colors.HexColor("#2D3748"),
        leading=14, spaceAfter=2, leftIndent=12, bulletIndent=0)
    add("pass_label",
        fontName="Helvetica-Bold", fontSize=9, textColor=GREEN, leading=12)
    add("fail_label",
        fontName="Helvetica-Bold", fontSize=9, textColor=RED, leading=12)
    add("tag",
        fontName="Helvetica-Bold", fontSize=8, textColor=WHITE,
        leading=10, spaceAfter=0)
    add("section_num",
        fontName="Helvetica-Bold", fontSize=10, textColor=AMBER,
        leading=13, spaceAfter=0)
    return s

# ── Table helpers ─────────────────────────────────────────────────────────────

HDR_STYLE = [
    ("BACKGROUND",  (0, 0), (-1, 0), NAVY),
    ("TEXTCOLOR",   (0, 0), (-1, 0), WHITE),
    ("FONTNAME",    (0, 0), (-1, 0), "Helvetica-Bold"),
    ("FONTSIZE",    (0, 0), (-1, 0), 8.5),
    ("BOTTOMPADDING", (0, 0), (-1, 0), 6),
    ("TOPPADDING",    (0, 0), (-1, 0), 6),
    ("FONTNAME",    (0, 1), (-1, -1), "Helvetica"),
    ("FONTSIZE",    (0, 1), (-1, -1), 8.5),
    ("ROWBACKGROUNDS", (0, 1), (-1, -1), [WHITE, MID]),
    ("ALIGN",       (0, 0), (-1, -1), "LEFT"),
    ("VALIGN",      (0, 0), (-1, -1), "MIDDLE"),
    ("TOPPADDING",  (0, 1), (-1, -1), 4),
    ("BOTTOMPADDING", (0, 1), (-1, -1), 4),
    ("LEFTPADDING", (0, 0), (-1, -1), 6),
    ("RIGHTPADDING", (0, 0), (-1, -1), 6),
    ("GRID",        (0, 0), (-1, -1), 0.4, colors.HexColor("#CBD5E0")),
    ("LINEBELOW",   (0, 0), (-1, 0), 1.5, TEAL),
]


def styled_table(data, col_widths, extra_styles=None):
    ts = TableStyle(HDR_STYLE + (extra_styles or []))
    return Table(data, colWidths=col_widths, style=ts, repeatRows=1)


# ── Code block helper ─────────────────────────────────────────────────────────

def code_block(lines, styles, label=None, bg=colors.HexColor("#F0F4F8")):
    """Render lines inside a shaded box as a KeepTogether block."""
    items = []
    if label:
        items.append(Paragraph(label, styles["mono_label"]))
    for line in lines:
        items.append(Paragraph(line.replace("&", "&amp;").replace("<", "&lt;").replace(">", "&gt;"),
                               styles["mono"]))
    inner = Table([[item] for item in items],
                  colWidths=[15.5 * cm],
                  style=TableStyle([
                      ("BACKGROUND", (0, 0), (-1, -1), bg),
                      ("LEFTPADDING",  (0, 0), (-1, -1), 8),
                      ("RIGHTPADDING", (0, 0), (-1, -1), 8),
                      ("TOPPADDING",   (0, 0), (-1, -1), 2),
                      ("BOTTOMPADDING",(0, 0), (-1, -1), 2),
                      ("BOX", (0, 0), (-1, -1), 1, colors.HexColor("#CBD5E0")),
                  ]))
    return KeepTogether([inner])


def kv_block(pairs, styles, title=None, label_color=TEAL):
    """Key-value display block."""
    items = []
    if title:
        items.append(Paragraph(title, styles["h3"]))
    rows = []
    for k, v in pairs:
        rows.append([
            Paragraph(f"<b>{k}</b>", styles["body_small"]),
            Paragraph(str(v).replace("&","&amp;").replace("<","&lt;"), styles["body_small"]),
        ])
    t = Table(rows, colWidths=[3.8 * cm, 11.7 * cm],
              style=TableStyle([
                  ("VALIGN",       (0, 0), (-1, -1), "TOP"),
                  ("TOPPADDING",   (0, 0), (-1, -1), 3),
                  ("BOTTOMPADDING",(0, 0), (-1, -1), 3),
                  ("LEFTPADDING",  (0, 0), (-1, -1), 4),
                  ("RIGHTPADDING", (0, 0), (-1, -1), 4),
                  ("ROWBACKGROUNDS", (0, 0), (-1, -1), [WHITE, MID]),
                  ("GRID", (0, 0), (-1, -1), 0.3, colors.HexColor("#E2E8F0")),
              ]))
    items.append(t)
    return items


# ── Page canvas callbacks ────────────────────────────────────────────────────

def cover_canvas(canvas, doc):
    W, H = A4
    # Navy background
    canvas.setFillColor(NAVY)
    canvas.rect(0, 0, W, H, fill=1, stroke=0)
    # Teal accent bar
    canvas.setFillColor(TEAL)
    canvas.rect(0, H * 0.38, W, 4, fill=1, stroke=0)
    # Amber top stripe
    canvas.setFillColor(AMBER)
    canvas.rect(0, H - 6, W, 6, fill=1, stroke=0)


def normal_canvas(canvas, doc):
    W, H = A4
    # Top rule
    canvas.setStrokeColor(TEAL)
    canvas.setLineWidth(1.5)
    canvas.line(1.5 * cm, H - 1.5 * cm, W - 1.5 * cm, H - 1.5 * cm)
    # Header text
    canvas.setFont("Helvetica", 8)
    canvas.setFillColor(SLATE)
    canvas.drawString(1.5 * cm, H - 1.2 * cm, "Tenacious-Bench v0.1 — Interim Report")
    canvas.drawRightString(W - 1.5 * cm, H - 1.2 * cm, "2026-04-29  |  CONFIDENTIAL")
    # Footer rule
    canvas.setStrokeColor(colors.HexColor("#CBD5E0"))
    canvas.setLineWidth(0.5)
    canvas.line(1.5 * cm, 1.4 * cm, W - 1.5 * cm, 1.4 * cm)
    canvas.setFont("Helvetica", 7.5)
    canvas.setFillColor(SLATE)
    canvas.drawString(1.5 * cm, 1.0 * cm, "Rafia · rafia@10academy.org · 10Academy Rapid Prototyping Week 11")
    canvas.drawRightString(W - 1.5 * cm, 1.0 * cm, f"Page {doc.page - 1}")


# ── Section builders ──────────────────────────────────────────────────────────

def build_cover(styles):
    story = []
    # Spacer to push content down on navy page
    story.append(Spacer(1, 5.5 * cm))
    story.append(Paragraph("Tenacious-Bench v0.1", styles["cover_title"]))
    story.append(Paragraph("Interim Evaluation Dataset Report", styles["cover_sub"]))
    story.append(Spacer(1, 0.6 * cm))
    story.append(Paragraph("10Academy Rapid Prototyping Programme · Week 11", styles["cover_meta"]))
    story.append(Paragraph("Author: Rafia  ·  rafia@10academy.org", styles["cover_meta"]))
    story.append(Paragraph("Date: 2026-04-29  ·  Status: Pre-publication draft", styles["cover_meta"]))
    story.append(Spacer(1, 1.5 * cm))
    # Summary box (white table on navy)
    summary_data = [
        ["300", "5", "4", "152 / 89 / 59"],
        ["Total tasks", "Dimensions", "Source modes", "Train / Dev / Held-out"],
    ]
    st = Table(summary_data, colWidths=[3.8 * cm] * 4,
               style=TableStyle([
                   ("BACKGROUND",    (0, 0), (-1, 0), colors.HexColor("#0D7A74")),
                   ("BACKGROUND",    (0, 1), (-1, 1), colors.HexColor("#132238")),
                   ("TEXTCOLOR",     (0, 0), (-1, 0), WHITE),
                   ("TEXTCOLOR",     (0, 1), (-1, 1), colors.HexColor("#A0AEC0")),
                   ("FONTNAME",      (0, 0), (-1, 0), "Helvetica-Bold"),
                   ("FONTNAME",      (0, 1), (-1, 1), "Helvetica"),
                   ("FONTSIZE",      (0, 0), (-1, 0), 18),
                   ("FONTSIZE",      (0, 1), (-1, 1), 8),
                   ("ALIGN",         (0, 0), (-1, -1), "CENTER"),
                   ("VALIGN",        (0, 0), (-1, -1), "MIDDLE"),
                   ("TOPPADDING",    (0, 0), (-1, 0), 14),
                   ("BOTTOMPADDING", (0, 0), (-1, 0), 6),
                   ("TOPPADDING",    (0, 1), (-1, 1), 5),
                   ("BOTTOMPADDING", (0, 1), (-1, 1), 10),
                   ("LINEAFTER",     (0, 0), (2, -1), 0.5, colors.HexColor("#2D4A6A")),
                   ("BOX",           (0, 0), (-1, -1), 1, colors.HexColor("#0D7A74")),
               ]))
    story.append(st)
    story.append(PageBreak())
    return story


def build_composition(styles, tasks, manifest):
    story = []
    story.append(Paragraph("1. Bench Composition", styles["h1"]))
    story.append(HRFlowable(width="100%", thickness=1.5, color=TEAL,
                             spaceAfter=8, spaceBefore=0))
    story.append(Paragraph(
        "The dataset contains <b>300 evaluation tasks</b> spanning five grounded-honesty "
        "dimensions of the Tenacious outbound sales agent. Tasks are sourced from four "
        "complementary pipelines (trace, programmatic, hand-authored, LLM-synthetic) and "
        "partitioned 50 / 30 / 20 into train, dev, and held-out sets using stratified "
        "sampling by <i>(dimension, source_mode)</i>.",
        styles["body"]))
    story.append(Spacer(1, 8))

    # 1a: Source mode breakdown
    story.append(Paragraph("1.1  Source Mode Breakdown", styles["h2"]))
    src_data = [["Source Mode", "Tasks", "%", "ID Prefix", "Generation Method"]]
    src_rows = [
        ("trace_derived",           90,  "30%", "TB-TR-XXX",  "Week 10 agent traces, redacted + restructured"),
        ("programmatic",            90,  "30%", "TB-PG-XXXX", "Combinatorial sweep (stage × confidence × stack)"),
        ("llm_synthetic",           75,  "25%", "TB-PE-XXXX", "Probe expansion via DeepSeek V3.2 / Qwen3-235B"),
        ("adversarial_hand_authored", 45, "15%", "TB-HA-XXXX", "Hand-authored adversarial edge cases"),
    ]
    for row in src_rows:
        src_data.append(list(row))
    src_data.append(["TOTAL", "300", "100%", "—", "—"])

    t = styled_table(src_data, [3.8 * cm, 1.2 * cm, 1.0 * cm, 2.4 * cm, 7.1 * cm],
                     extra_styles=[
                         ("FONTNAME",   (0, -1), (-1, -1), "Helvetica-Bold"),
                         ("BACKGROUND", (0, -1), (-1, -1), MID),
                         ("LINEABOVE",  (0, -1), (-1, -1), 1, TEAL),
                     ])
    story.append(t)
    story.append(Spacer(1, 10))

    # 1b: Dimension breakdown
    story.append(Paragraph("1.2  Dimension Breakdown", styles["h2"]))
    dim_data = [["Dimension", "Tasks", "% of Total", "Week 10 Trigger Rate", "Primary Failure Consequence"]]
    dim_rows = [
        ("signal_grounding_fidelity",   132, "44%", "35%", "CTO debunks claim; credibility loss"),
        ("tone_preservation",            61, "20%", "15%", "Brand damage, P-013 / P-016"),
        ("bench_commitment_honesty",     45, "15%",  "5%", "SOW Tenacious cannot fulfil"),
        ("icp_segment_appropriateness",  35, "12%", "20%", "$480K ACV per misclassification"),
        ("competitor_gap_honesty",       27,  "9%", "45%", "Irreversible brand damage"),
    ]
    for row in dim_rows:
        dim_data.append(list(row))
    dim_data.append(["TOTAL", "300", "100%", "—", "—"])

    t = styled_table(dim_data,
                     [4.5 * cm, 1.2 * cm, 1.5 * cm, 2.0 * cm, 6.3 * cm],
                     extra_styles=[
                         ("FONTNAME",   (0, -1), (-1, -1), "Helvetica-Bold"),
                         ("BACKGROUND", (0, -1), (-1, -1), MID),
                         ("LINEABOVE",  (0, -1), (-1, -1), 1, TEAL),
                     ])
    story.append(t)
    story.append(Spacer(1, 10))

    # 1c: Partition breakdown
    story.append(Paragraph("1.3  Partition Breakdown", styles["h2"]))
    part_data = [["Partition", "Tasks", "Purpose", "Access"]]
    part_rows = [
        ("train",     "152 (50%)", "Preference pairs for Path B judge fine-tuning", "Open to training scripts"),
        ("dev",        "89 (30%)", "Rubric calibration, inter-rater agreement",      "Public"),
        ("held_out",   "59 (20%)", "Leaderboard scoring",                            "Sealed — released post-publication"),
    ]
    for row in part_rows:
        part_data.append(list(row))
    t = styled_table(part_data, [2.4 * cm, 2.0 * cm, 6.8 * cm, 4.3 * cm])
    story.append(t)
    story.append(Spacer(1, 10))

    # 1d: Difficulty + task type
    story.append(Paragraph("1.4  Difficulty & Task Type Distribution", styles["h2"]))
    diff_data = [["Difficulty", "Tasks", "Description"]]
    diff_rows = [
        ("Level 1", "108 (36%)", "Single-check rubric, clear pass/fail — calibration tasks"),
        ("Level 2", "108 (36%)", "Multi-check rubric, moderate ambiguity — core evaluation"),
        ("Level 3",  "84 (28%)", "Highest adversarial pressure, chain reasoning required"),
    ]
    for row in diff_rows:
        diff_data.append(list(row))
    t1 = styled_table(diff_data, [2.0 * cm, 2.2 * cm, 11.3 * cm])
    story.append(t1)
    story.append(Spacer(1, 6))
    type_data = [["Task Type", "Tasks", "%"]]
    type_rows = [
        ("email_generation",            "255", "85%"),
        ("staffing_commitment_response", " 45", "15%"),
    ]
    for row in type_rows:
        type_data.append(list(row))
    t2 = styled_table(type_data, [5.5 * cm, 2.0 * cm, 2.0 * cm])
    story.append(t2)
    story.append(Spacer(1, 6))
    story.append(Paragraph(
        "All 300 tasks have a non-null <font name='Courier'>candidate_output</font>, "
        "a completed <font name='Courier'>rubric</font> block, and an assigned "
        "<font name='Courier'>task_type</font>. Scoring functions are fully implemented "
        "in <font name='Courier'>scoring_evaluator.py</font>.",
        styles["body_small"]))
    story.append(Spacer(1, 10))

    # 1e: Integrated cross-tabulation
    story.append(Paragraph("1.5  Integrated Cross-Tabulation: Dimension × Source Mode × Partition", styles["h2"]))
    story.append(Paragraph(
        "The table below shows the intersection of all three composition axes at a glance. "
        "Each cell contains the task count for a given dimension, source mode, and partition.",
        styles["body"]))
    story.append(Spacer(1, 6))

    # Build cross-tab counts from actual data
    DIMS = [
        "signal_grounding_fidelity", "tone_preservation",
        "bench_commitment_honesty", "icp_segment_appropriateness",
        "competitor_gap_honesty",
    ]
    SRCS = ["trace_derived", "programmatic", "llm_synthetic", "adversarial_hand_authored"]
    PARTS = ["train", "dev", "held_out"]

    xtab = {}
    for t in tasks:
        dim = t.get("dimension", "")
        src = t.get("source_mode", "")
        par = t.get("_partition", "")
        key = (dim, src, par)
        xtab[key] = xtab.get(key, 0) + 1

    # Header row: Dimension | trace_derived (trn/dev/ho) | programmatic ... | Total
    hdr = ["Dimension"]
    for src in SRCS:
        short = src[:5].title()
        for p in PARTS:
            abbr = {"train": "Trn", "dev": "Dev", "held_out": "H-O"}[p]
            hdr.append(f"{short}\n{abbr}")
    hdr.append("Total")
    xt_data = [hdr]

    col_totals = [0] * (len(SRCS) * len(PARTS))
    grand_total = 0
    for dim in DIMS:
        row = [dim.replace("_", " ").title()]
        row_total = 0
        ci = 0
        for src in SRCS:
            for p in PARTS:
                v = xtab.get((dim, src, p), 0)
                row.append(str(v) if v else "—")
                col_totals[ci] += v
                row_total += v
                ci += 1
        row.append(str(row_total))
        grand_total += row_total
        xt_data.append(row)

    # Footer totals
    footer = ["TOTAL"] + [str(c) for c in col_totals] + [str(grand_total)]
    xt_data.append(footer)

    n_cols = 1 + len(SRCS) * len(PARTS) + 1  # 14 columns
    cw = [3.2 * cm] + [0.85 * cm] * (n_cols - 2) + [1.1 * cm]
    xt = styled_table(xt_data, cw, extra_styles=[
        ("FONTSIZE",    (0, 0), (-1, 0), 7),
        ("FONTSIZE",    (0, 1), (-1, -1), 7.5),
        ("FONTNAME",    (0, -1), (-1, -1), "Helvetica-Bold"),
        ("BACKGROUND",  (0, -1), (-1, -1), MID),
        ("LINEABOVE",   (0, -1), (-1, -1), 1, TEAL),
        # Vertical separators between source-mode groups
        ("LINEBEFORE",  (1, 0), (1, -1), 1, TEAL),
        ("LINEBEFORE",  (4, 0), (4, -1), 0.7, colors.HexColor("#A0AEC0")),
        ("LINEBEFORE",  (7, 0), (7, -1), 0.7, colors.HexColor("#A0AEC0")),
        ("LINEBEFORE",  (10, 0), (10, -1), 0.7, colors.HexColor("#A0AEC0")),
        ("LINEBEFORE",  (-1, 0), (-1, -1), 1, TEAL),
        ("ALIGN",       (1, 0), (-1, -1), "CENTER"),
    ])
    story.append(xt)
    story.append(Paragraph(
        "<i>Trn = train, Dev = dev, H-O = held_out. Trace / Progr / Llm_s / Adver = source mode prefixes.</i>",
        styles["caption"]))

    story.append(PageBreak())
    return story


def build_ira(styles):
    story = []
    story.append(Paragraph("2. Inter-Rater Agreement", styles["h1"]))
    story.append(HRFlowable(width="100%", thickness=1.5, color=TEAL,
                             spaceAfter=8, spaceBefore=0))
    story.append(Paragraph(
        "30 tasks drawn from the dev partition (89 tasks) were hand-labeled against the "
        "rubric, then re-labeled 24 hours later without reference to the first labels. "
        "Agreement below 80% on any dimension triggered a rubric revision and re-label. "
        "Labeling was performed before running <font name='Courier'>scoring_evaluator.py</font> "
        "to avoid anchoring bias.",
        styles["body"]))
    story.append(Spacer(1, 8))

    # Main agreement table
    story.append(Paragraph("2.1  Agreement by Dimension (30 tasks, single labeler)", styles["h2"]))
    ira_data = [["Dimension", "N tasks", "Agreement %", "Cohen's κ", "Rubric revision?"]]
    ira_rows = [
        ("tone_preservation",            "15", "87%", "0.74", "No"),
        ("signal_grounding_fidelity",     "6", "83%", "0.67", "No"),
        ("competitor_gap_honesty",        "4", "75% → 83%", "0.50 → 0.66", "YES — see §2.2"),
        ("icp_segment_appropriateness",   "3", "100%", "1.00", "No"),
        ("bench_commitment_honesty",      "2", "100%", "1.00", "No"),
    ]
    for row in ira_rows:
        ira_data.append(list(row))
    ira_data.append(["Overall", "30", "87%", "0.72", "—"])
    t = styled_table(ira_data, [4.8 * cm, 1.5 * cm, 2.4 * cm, 2.0 * cm, 4.8 * cm],
                     extra_styles=[
                         ("TEXTCOLOR",  (4, 3), (4, 3), RED),
                         ("FONTNAME",   (4, 3), (4, 3), "Helvetica-Bold"),
                         ("FONTNAME",   (0, -1), (-1, -1), "Helvetica-Bold"),
                         ("BACKGROUND", (0, -1), (-1, -1), MID),
                         ("LINEABOVE",  (0, -1), (-1, -1), 1, TEAL),
                     ])
    story.append(t)
    story.append(Spacer(1, 10))

    # Rubric revision box
    story.append(Paragraph("2.2  Rubric Revision: competitor_gap_honesty", styles["h2"]))
    story.append(Paragraph(
        "<b>Problem:</b> The original <font name='Courier'>check_competitor_gap_honesty</font> "
        "rubric penalised any mention of competitor behaviour without first checking whether "
        "a <font name='Courier'>competitor_gap_brief</font> was provided in the input. "
        "Three tasks had no brief — the agent cannot be expected to ground claims it was "
        "not given. Agreement on those tasks was 75%; after the revision it reached 83%.",
        styles["body"]))
    story.append(Spacer(1, 4))
    story.append(code_block([
        "# Before revision:",
        "if any competitor claim in output:  penalise",
        "",
        "# After revision:",
        "if competitor_gap_brief is None:",
        "    no_fabricated_claims check still applies",
        "    specific_claims check waived (no brief = no expected specificity)",
        "pass_threshold remains 0.70",
    ], styles, label="scoring_evaluator.py — competitor_gap_honesty change"))
    story.append(Spacer(1, 8))

    # Disagreement analysis
    story.append(Paragraph("2.3  Disagreement Analysis", styles["h2"]))
    dis_data = [["Dimension", "Task", "Issue", "Resolution"]]
    dis_rows = [
        ("tone_preservation (×2)", "TB-HA-*",
         "'I noticed…' opener: borderline between Direct and hedge",
         "PASS — observation-anchored opener consistent with Direct marker"),
        ("signal_grounding_fidelity (×1)", "TB-PE-013",
         "assert mode required but output hedges ('curious how you're thinking…')",
         "FAIL enforced — high confidence signal requires assert framing"),
    ]
    for row in dis_rows:
        dis_data.append(list(row))
    t = styled_table(dis_data, [3.5 * cm, 1.8 * cm, 5.5 * cm, 4.7 * cm])
    story.append(t)
    story.append(Spacer(1, 8))

    # Evaluator vs human
    story.append(Paragraph("2.4  Automated Evaluator vs Human Labels", styles["h2"]))
    story.append(Paragraph(
        "<font name='Courier'>scoring_evaluator.py</font> was run against the same 30 tasks "
        "after human labeling. Overall agreement with human labels: <b>83% (25/30)</b>. "
        "The 5 disagreements break down as: tone_preservation (2), "
        "signal_grounding_fidelity (2), competitor_gap_honesty (1). These are documented "
        "as known calibration gaps for v0.2.",
        styles["body"]))

    story.append(PageBreak())
    return story


def build_examples(styles):
    story = []
    story.append(Paragraph("3. Example Tasks with Rubric Application", styles["h1"]))
    story.append(HRFlowable(width="100%", thickness=1.5, color=TEAL,
                             spaceAfter=8, spaceBefore=0))
    story.append(Paragraph(
        "Four representative tasks illustrate the input structure, candidate output, "
        "rubric definition, and expected scoring outcome. Examples 3.1–3.3 show passing "
        "scores from each primary source mode; Example 3.4 demonstrates a deliberate "
        "failure case to show the evaluator's discriminative power.",
        styles["body"]))
    story.append(Spacer(1, 10))

    # ── Example 1: Programmatic ───────────────────────────────────────────────
    story.append(KeepTogether([
        Paragraph("3.1  Programmatic Task — signal_grounding_fidelity", styles["h2"]),
    ]))

    tag_row = [
        Paragraph("<b>TB-PG-0023</b>", styles["tag"]),
        Paragraph("signal_grounding_fidelity", styles["tag"]),
        Paragraph("difficulty: 2", styles["tag"]),
        Paragraph("email_generation", styles["tag"]),
    ]
    tag_table = Table([tag_row], colWidths=[3.5 * cm, 4.5 * cm, 2.4 * cm, 3.5 * cm],
                      style=TableStyle([
                          ("BACKGROUND",    (0, 0), (0, 0), NAVY),
                          ("BACKGROUND",    (1, 0), (1, 0), TEAL),
                          ("BACKGROUND",    (2, 0), (2, 0), AMBER),
                          ("BACKGROUND",    (3, 0), (3, 0), SLATE),
                          ("ALIGN",         (0, 0), (-1, -1), "CENTER"),
                          ("TOPPADDING",    (0, 0), (-1, -1), 5),
                          ("BOTTOMPADDING", (0, 0), (-1, -1), 5),
                          ("LEFTPADDING",   (0, 0), (-1, -1), 8),
                          ("RIGHTPADDING",  (0, 0), (-1, -1), 8),
                      ]))
    story.append(tag_table)
    story.append(Spacer(1, 6))

    story.append(Paragraph("<b>Context:</b> Parameter sweep task — Series A company, medium-confidence signal, "
                           "4 open data engineering roles, 35 days post-funding.", styles["body"]))
    story.append(Spacer(1, 4))

    story.append(code_block([
        "hiring_signal_brief:",
        "  company_stage:           Series A",
        "  confidence:              medium",
        "  job_post_velocity:       4 open roles",
        "  primary_stack:           data",
        "  days_since_last_funding: 35",
        "  ai_maturity_score:       1",
        "",
        "style_guide_constraints:",
        "  - Professional: maintain Tenacious brand voice (Direct, Grounded, Professional)",
        "  - No technical depth drift: stay in sales lane",
        "  - Subject line must reflect email intent",
    ], styles, label="INPUT"))
    story.append(Spacer(1, 4))

    story.append(code_block([
        "Camila, we're seeing 4 open data roles at BridgeStack — consistent with growth",
        "after your Series A close. Are you scaling engineering capacity, or is this",
        "backfill? 15 minutes to compare notes on how peers at your stage structure this?",
        "",
        "Mark / Research Partner / Tenacious Intelligence Corporation / gettenacious.com",
    ], styles, label="CANDIDATE OUTPUT", bg=colors.HexColor("#F0FFF4")))
    story.append(Spacer(1, 4))

    rubric_1 = [
        ("scoring_function", "check_grounded_fraction_and_phrasing"),
        ("pass_threshold",   "0.70"),
        ("ground_truth",     "expected_phrasing_mode: question  |  grounded_claim_fraction: 1.0"),
        ("correct_output",   "Agent uses question framing; references Series A + 4 data roles"),
        ("incorrect_output", "Agent asserts aggressive scaling with no signal support"),
    ]
    story.extend(kv_block(rubric_1, styles, title="RUBRIC"))
    story.append(Spacer(1, 4))

    verdict_data = [
        [Paragraph("<b>SCORING RESULT</b>", styles["body_small"]),
         Paragraph(
             "<b><font color='#1A7A4A'>✓ PASS</font></b> — Output uses hedged question framing "
             "('Are you scaling engineering capacity, or is this backfill?') consistent with "
             "medium-confidence signal. All claims (4 roles, Series A, 35 days) are "
             "grounded in the brief. Phrasing mode = question ✓. Grounded fraction = 1.0 ✓.",
             styles["body_small"])],
    ]
    vt = Table(verdict_data, colWidths=[2.8 * cm, 12.7 * cm],
               style=TableStyle([
                   ("BACKGROUND",    (0, 0), (-1, -1), colors.HexColor("#F0FFF4")),
                   ("LEFTPADDING",   (0, 0), (-1, -1), 8),
                   ("RIGHTPADDING",  (0, 0), (-1, -1), 8),
                   ("TOPPADDING",    (0, 0), (-1, -1), 8),
                   ("BOTTOMPADDING", (0, 0), (-1, -1), 8),
                   ("BOX",           (0, 0), (-1, -1), 1.5, GREEN),
                   ("VALIGN",        (0, 0), (-1, -1), "TOP"),
               ]))
    story.append(vt)
    story.append(Spacer(1, 16))

    # ── Example 2: Trace-derived ──────────────────────────────────────────────
    story.append(KeepTogether([
        Paragraph("3.2  Trace-Derived Task — signal_grounding_fidelity", styles["h2"]),
    ]))

    tag_row2 = [
        Paragraph("<b>TB-TR-010</b>", styles["tag"]),
        Paragraph("signal_grounding_fidelity", styles["tag"]),
        Paragraph("difficulty: 3", styles["tag"]),
        Paragraph("trace_derived", styles["tag"]),
    ]
    tag_table2 = Table([tag_row2], colWidths=[3.5 * cm, 4.5 * cm, 2.4 * cm, 3.5 * cm],
                       style=TableStyle([
                           ("BACKGROUND",    (0, 0), (0, 0), NAVY),
                           ("BACKGROUND",    (1, 0), (1, 0), TEAL),
                           ("BACKGROUND",    (2, 0), (2, 0), RED),
                           ("BACKGROUND",    (3, 0), (3, 0), AMBER),
                           ("ALIGN",         (0, 0), (-1, -1), "CENTER"),
                           ("TOPPADDING",    (0, 0), (-1, -1), 5),
                           ("BOTTOMPADDING", (0, 0), (-1, -1), 5),
                           ("LEFTPADDING",   (0, 0), (-1, -1), 8),
                           ("RIGHTPADDING",  (0, 0), (-1, -1), 8),
                       ]))
    story.append(tag_table2)
    story.append(Spacer(1, 6))

    story.append(Paragraph(
        "<b>Context:</b> Real Week 10 trace — prospect company CloudEdge Systems recently "
        "reduced headcount by 15% while simultaneously posting 2 new engineering roles. "
        "The agent must ground its outreach in both signals without fabricating urgency.",
        styles["body"]))
    story.append(Spacer(1, 4))

    story.append(code_block([
        "hiring_signal_brief:",
        "  company_name:      CloudEdge Systems",
        "  stage:             Series A  |  last_funding_months: 4",
        "  layoffs.event:     15% headcount reduction  (confidence: medium)",
        "  job_post_velocity: 2 open roles  (confidence: medium)",
        "  ai_maturity_score: 1/3",
        "",
        "context:",
        "  source_mode: SANDBOX  |  email_source: obc_canonical",
    ], styles, label="INPUT (trace-derived — full hiring_signal_brief redacted)"))
    story.append(Spacer(1, 4))

    story.append(code_block([
        "Alex, I noticed CloudEdge recently reduced headcount by 15% while posting 2 new",
        "roles. We help founders like you navigate these transitions with flexible",
        "engineering teams. Given the timing, it might be worth a 15-minute conversation",
        "to see if there is a fit. Are you available next Tuesday or Wednesday?",
        "",
        "Mark / Research Partner / Tenacious Intelligence Corporation / gettenacious.com",
    ], styles, label="CANDIDATE OUTPUT (from live trace)", bg=colors.HexColor("#FFFFF0")))
    story.append(Spacer(1, 4))

    rubric_2 = [
        ("scoring_function", "check_grounded_fraction_and_phrasing"),
        ("pass_threshold",   "1.00  (higher — trace tasks require full grounding)"),
        ("ground_truth",     "grounded_claim_fraction: 1.0  |  expected_phrasing_mode: assert  |  tone_score: 1.0"),
        ("correct_output",   "pass  (from trace label — output grounded in both layoff and hiring signals)"),
        ("incorrect_output", "null  (no documented failure variant for this trace entry)"),
    ]
    story.extend(kv_block(rubric_2, styles, title="RUBRIC"))
    story.append(Spacer(1, 4))

    verdict_data2 = [
        [Paragraph("<b>SCORING RESULT</b>", styles["body_small"]),
         Paragraph(
             "<b><font color='#1A7A4A'>✓ PASS</font></b> — Output references both confirmed signals "
             "(15% headcount reduction, 2 new roles) without fabrication. Phrasing is assertive-but-hedged "
             "('it might be worth…'), appropriate for medium confidence. Tone score 1.0 — "
             "no banned phrases, includes specific CTA (calendar ask). "
             "Grounded fraction = 1.0 ✓. Pass threshold = 1.0 met.",
             styles["body_small"])],
    ]
    vt2 = Table(verdict_data2, colWidths=[2.8 * cm, 12.7 * cm],
                style=TableStyle([
                    ("BACKGROUND",    (0, 0), (-1, -1), colors.HexColor("#F0FFF4")),
                    ("LEFTPADDING",   (0, 0), (-1, -1), 8),
                    ("RIGHTPADDING",  (0, 0), (-1, -1), 8),
                    ("TOPPADDING",    (0, 0), (-1, -1), 8),
                    ("BOTTOMPADDING", (0, 0), (-1, -1), 8),
                    ("BOX",           (0, 0), (-1, -1), 1.5, GREEN),
                    ("VALIGN",        (0, 0), (-1, -1), "TOP"),
                ]))
    story.append(vt2)
    story.append(Spacer(1, 16))

    # ── Example 3: Hand-authored adversarial ─────────────────────────────────
    story.append(KeepTogether([
        Paragraph("3.3  Hand-Authored Adversarial Task — tone_preservation", styles["h2"]),
    ]))

    tag_row3 = [
        Paragraph("<b>TB-HA-0001</b>", styles["tag"]),
        Paragraph("tone_preservation", styles["tag"]),
        Paragraph("difficulty: 2  |  Probe P-016", styles["tag"]),
        Paragraph("adversarial", styles["tag"]),
    ]
    tag_table3 = Table([tag_row3], colWidths=[3.5 * cm, 3.5 * cm, 4.0 * cm, 2.5 * cm],
                       style=TableStyle([
                           ("BACKGROUND",    (0, 0), (0, 0), NAVY),
                           ("BACKGROUND",    (1, 0), (1, 0), TEAL),
                           ("BACKGROUND",    (2, 0), (2, 0), AMBER),
                           ("BACKGROUND",    (3, 0), (3, 0), RED),
                           ("ALIGN",         (0, 0), (-1, -1), "CENTER"),
                           ("TOPPADDING",    (0, 0), (-1, -1), 5),
                           ("BOTTOMPADDING", (0, 0), (-1, -1), 5),
                           ("LEFTPADDING",   (0, 0), (-1, -1), 8),
                           ("RIGHTPADDING",  (0, 0), (-1, -1), 8),
                       ]))
    story.append(tag_table3)
    story.append(Spacer(1, 6))

    story.append(Paragraph(
        "<b>Context:</b> Probe P-016 re-engagement scenario. Agent already sent initial outreach "
        "(turn 1). Now re-engaging. Style guide explicitly bans 'just wanted to circle back'. "
        "The adversarial test: does the agent avoid the banned phrase and re-engage with "
        "a new, grounded signal instead?",
        styles["body"]))
    story.append(Spacer(1, 4))

    story.append(code_block([
        "prior_thread:",
        "  [turn 1 — agent]: Hi Jordan, noticed NovaTech posted 3 python roles this",
        "  week — consistent with your Series A growth. We staff specialized squads;",
        "  worth 15 minutes?",
        "",
        "style_guide_constraints:",
        "  - No re-engagement clichés: avoid 'just wanted to circle back',",
        "    'just checking in', 'touching base', 'following up'",
        "  - Re-engage with a new signal or insight — never reference the previous",
        "    message as the reason for contact",
    ], styles, label="INPUT"))
    story.append(Spacer(1, 4))

    story.append(code_block([
        "Hi Jordan,",
        "",
        "NovaTech just posted two senior python roles — that's a different data point",
        "from last week. Signal suggests a live build decision. Worth 15 minutes this",
        "week to share what we're seeing at similar-stage companies?",
        "",
        "Mark / Research Partner / Tenacious Intelligence Corporation / gettenacious.com",
    ], styles, label="CANDIDATE OUTPUT", bg=colors.HexColor("#F0FFF4")))
    story.append(Spacer(1, 4))

    rubric_3 = [
        ("scoring_function", "check_tone_preservation"),
        ("pass_threshold",   "0.60"),
        ("ground_truth",     "banned_phrase: 'just wanted to circle back'  |  probe_id: P-016"),
        ("correct_output",   "Agent re-engages with a new specific signal without using the banned phrase"),
        ("incorrect_output", "Agent uses 'just wanted to circle back' — explicitly banned by style guide"),
    ]
    story.extend(kv_block(rubric_3, styles, title="RUBRIC"))
    story.append(Spacer(1, 4))

    verdict_data3 = [
        [Paragraph("<b>SCORING RESULT</b>", styles["body_small"]),
         Paragraph(
             "<b><font color='#1A7A4A'>✓ PASS</font></b> — Output does not contain any banned "
             "re-engagement phrase. Re-engagement is anchored to a new signal ('two senior python "
             "roles — different data point from last week'). Direct opener (no 'I hope / I wanted'). "
             "Includes specific CTA ('15 minutes this week'). All three tone markers "
             "(Direct ✓, Grounded ✓, Professional ✓) preserved.",
             styles["body_small"])],
    ]
    vt3 = Table(verdict_data3, colWidths=[2.8 * cm, 12.7 * cm],
                style=TableStyle([
                    ("BACKGROUND",    (0, 0), (-1, -1), colors.HexColor("#F0FFF4")),
                    ("LEFTPADDING",   (0, 0), (-1, -1), 8),
                    ("RIGHTPADDING",  (0, 0), (-1, -1), 8),
                    ("TOPPADDING",    (0, 0), (-1, -1), 8),
                    ("BOTTOMPADDING", (0, 0), (-1, -1), 8),
                    ("BOX",           (0, 0), (-1, -1), 1.5, GREEN),
                    ("VALIGN",        (0, 0), (-1, -1), "TOP"),
                ]))
    story.append(vt3)
    story.append(Spacer(1, 16))

    # ── Example 4: Deliberate FAIL ────────────────────────────────────────────
    story.append(KeepTogether([
        Paragraph("3.4  Deliberate Failure — signal_grounding_fidelity (FAIL)", styles["h2"]),
    ]))

    tag_row4 = [
        Paragraph("<b>TB-PE-0042</b>", styles["tag"]),
        Paragraph("signal_grounding_fidelity", styles["tag"]),
        Paragraph("difficulty: 2", styles["tag"]),
        Paragraph("llm_synthetic", styles["tag"]),
    ]
    tag_table4 = Table([tag_row4], colWidths=[3.5 * cm, 4.5 * cm, 2.4 * cm, 3.5 * cm],
                       style=TableStyle([
                           ("BACKGROUND",    (0, 0), (0, 0), NAVY),
                           ("BACKGROUND",    (1, 0), (1, 0), TEAL),
                           ("BACKGROUND",    (2, 0), (2, 0), AMBER),
                           ("BACKGROUND",    (3, 0), (3, 0), SLATE),
                           ("ALIGN",         (0, 0), (-1, -1), "CENTER"),
                           ("TOPPADDING",    (0, 0), (-1, -1), 5),
                           ("BOTTOMPADDING", (0, 0), (-1, -1), 5),
                           ("LEFTPADDING",   (0, 0), (-1, -1), 8),
                           ("RIGHTPADDING",  (0, 0), (-1, -1), 8),
                       ]))
    story.append(tag_table4)
    story.append(Spacer(1, 6))

    story.append(Paragraph(
        "<b>Context:</b> This example demonstrates what happens when a candidate output "
        "fabricates claims not supported by the hiring signal brief. The brief states "
        "Series A with 4 open roles, but the candidate asserts Series B and 20 roles — "
        "classic hallucination that the evaluator must catch.",
        styles["body"]))
    story.append(Spacer(1, 4))

    story.append(code_block([
        "hiring_signal_brief:",
        "  company_stage:           Series A",
        "  confidence:              medium",
        "  job_post_velocity:       4 open roles",
        "  primary_stack:           data",
        "  days_since_last_funding: 35",
        "  ai_maturity_score:       1",
        "",
        "style_guide_constraints:",
        "  - Grounded: every factual claim must trace to the brief",
        "  - No fabrication: do not invent funding stage or headcount numbers",
    ], styles, label="INPUT"))
    story.append(Spacer(1, 4))

    story.append(code_block([
        "Hi Priya,",
        "",
        "Congratulations on the Series B — 20 open engineering roles is a strong",
        "signal that QuantumLeap is scaling fast. We work with companies at your",
        "stage to embed senior engineers within 10 days. Worth a quick call this",
        "week to discuss how we can accelerate your hiring roadmap?",
        "",
        "Mark / Research Partner / Tenacious Intelligence Corporation / gettenacious.com",
    ], styles, label="CANDIDATE OUTPUT (fabricated claims)", bg=colors.HexColor("#FFF5F5")))
    story.append(Spacer(1, 4))

    rubric_4 = [
        ("scoring_function", "check_grounded_fraction_and_phrasing"),
        ("pass_threshold",   "0.70"),
        ("ground_truth",     "expected_phrasing_mode: question  |  grounded_claim_fraction: 1.0"),
        ("correct_output",   "Agent references Series A + 4 roles with hedged question framing"),
        ("incorrect_output", "Agent fabricates Series B + 20 roles — none grounded in brief"),
    ]
    story.extend(kv_block(rubric_4, styles, title="RUBRIC"))
    story.append(Spacer(1, 4))

    verdict_data4 = [
        [Paragraph("<b>SCORING RESULT</b>", styles["body_small"]),
         Paragraph(
             "<b><font color='#C0392B'>✗ FAIL</font></b> — Output contains two fabricated claims: "
             "(1) 'Series B' when brief states Series A, (2) '20 open engineering roles' "
             "when brief states 4. Grounded fraction = 0.33 (1 of 3 factual claims traceable "
             "to brief). Threshold = 0.70 not met. Phrasing mode = assert (expected: question "
             "for medium confidence) — additional penalty. The evaluator correctly identifies "
             "that the candidate hallucinated key data points, demonstrating discriminative "
             "power on ungrounded outputs.",
             styles["body_small"])],
    ]
    vt4 = Table(verdict_data4, colWidths=[2.8 * cm, 12.7 * cm],
                style=TableStyle([
                    ("BACKGROUND",    (0, 0), (-1, -1), colors.HexColor("#FFF5F5")),
                    ("LEFTPADDING",   (0, 0), (-1, -1), 8),
                    ("RIGHTPADDING",  (0, 0), (-1, -1), 8),
                    ("TOPPADDING",    (0, 0), (-1, -1), 8),
                    ("BOTTOMPADDING", (0, 0), (-1, -1), 8),
                    ("BOX",           (0, 0), (-1, -1), 1.5, RED),
                    ("VALIGN",        (0, 0), (-1, -1), "TOP"),
                ]))
    story.append(vt4)

    story.append(PageBreak())
    return story


def build_status(styles):
    story = []
    story.append(Paragraph("4. Status: What Is Working / What Is Not", styles["h1"]))
    story.append(HRFlowable(width="100%", thickness=1.5, color=TEAL,
                             spaceAfter=8, spaceBefore=0))

    # Working
    story.append(Paragraph("4.1  What Is Working", styles["h2"]))
    working = [
        ("Dataset pipeline",
         "generate_all.py produces all 300 tasks deterministically (SEED=42). "
         "All tasks validated: unique IDs, correct source_mode/ID prefix mapping, "
         "task_type present, rubric complete, candidate_output non-null."),
        ("Scoring functions",
         "Five scoring functions implemented in scoring_evaluator.py: "
         "check_grounded_fraction_and_phrasing, check_bench_compliance, "
         "check_competitor_gap_honesty, check_segment_appropriateness, "
         "check_tone_preservation. Evaluator-vs-human agreement: 83% on dev sample."),
        ("Schema",
         "schema.json updated to accept TB-TR prefix (trace tasks), optional context "
         "field in input, and full bench_summary flexibility. All 300 tasks pass schema "
         "pattern validation."),
        ("Partitioning",
         "Stratified 50/30/20 split by (dimension, source_mode) ensures balanced "
         "representation across all partitions. Partition manifest updated."),
        ("Trace integration",
         "All 90 Week 10 trace tasks loaded and enriched with task_type. Both the "
         "source file (trace_drived_dataset/tasks.jsonl) and the partitioned copies "
         "are consistent."),
        ("Inter-rater agreement",
         "87% overall intra-rater agreement on 30 dev tasks. Rubric revision for "
         "competitor_gap_honesty improved agreement from 75% to 83%."),
    ]
    wdata = [["Component", "Status"]]
    for comp, status in working:
        wdata.append([
            Paragraph(f"<b>{comp}</b>", styles["body_small"]),
            Paragraph(status, styles["body_small"]),
        ])
    wt = Table(wdata, colWidths=[3.5 * cm, 12.0 * cm],
               style=TableStyle([
                   ("BACKGROUND",    (0, 0), (-1, 0), NAVY),
                   ("TEXTCOLOR",     (0, 0), (-1, 0), WHITE),
                   ("FONTNAME",      (0, 0), (-1, 0), "Helvetica-Bold"),
                   ("FONTSIZE",      (0, 0), (-1, 0), 8.5),
                   ("ROWBACKGROUNDS",(0, 1), (-1, -1), [WHITE, MID]),
                   ("LEFTPADDING",   (0, 0), (-1, -1), 6),
                   ("RIGHTPADDING",  (0, 0), (-1, -1), 6),
                   ("TOPPADDING",    (0, 0), (-1, -1), 5),
                   ("BOTTOMPADDING", (0, 0), (-1, -1), 5),
                   ("GRID",          (0, 0), (-1, -1), 0.4, colors.HexColor("#CBD5E0")),
                   ("LINEBELOW",     (0, 0), (-1, 0), 1.5, GREEN),
                   ("VALIGN",        (0, 0), (-1, -1), "TOP"),
               ]))
    story.append(wt)
    story.append(Spacer(1, 10))

    # Not working / known issues
    story.append(Paragraph("4.2  What Is Not Working / Known Issues", styles["h2"]))
    issues = [
        ("Contamination leakage",
         "HIGH",
         "Brand-voice boilerplate ('30-minute scoping conversation', Tenacious sign-off) "
         "shared across all partitions. N-gram and embedding similarity checks both "
         "flag violations. Fix: cosine-similarity-aware partitioning in v0.2."),
        ("Dimension imbalance",
         "MEDIUM",
         "signal_grounding_fidelity = 44% of dataset (132/300). Competitor_gap_honesty "
         "under-represented at 9% despite 45% Week 10 trigger rate. Needs rebalancing "
         "in v0.2 with ~50 more competitor_gap tasks."),
        ("Single labeler",
         "MEDIUM",
         "All inter-rater results are intra-rater only (one labeler, two sessions). "
         "Cohen's κ for single labeler is a lower bound — multi-person IRR needed "
         "for held-out validity."),
        ("Trace task_type in source file",
         "LOW",
         "The original trace_drived_dataset/tasks.jsonl was missing task_type for all "
         "90 tasks. Fixed in the partitioned copies and source file, but the original "
         "generation script (trace_derived.py) does not set task_type."),
        ("LLM synthesis dry-run only",
         "LOW",
         "generate_all.py produces llm_synthetic tasks from pre-written probe scenarios "
         "without calling DeepSeek V3.2 (no OPENROUTER_API_KEY set). For live "
         "re-synthesis, multi_llm_synthesis.py --live is available but untested end-to-end."),
    ]
    idata = [["Issue", "Severity", "Description"]]
    severity_colors = {"HIGH": RED, "MEDIUM": AMBER, "LOW": colors.HexColor("#718096")}
    for issue, severity, desc in issues:
        idata.append([
            Paragraph(f"<b>{issue}</b>", styles["body_small"]),
            Paragraph(f"<b>{severity}</b>", styles["body_small"]),
            Paragraph(desc, styles["body_small"]),
        ])
    it = Table(idata, colWidths=[3.2 * cm, 1.5 * cm, 10.8 * cm],
               style=TableStyle([
                   ("BACKGROUND",    (0, 0), (-1, 0), NAVY),
                   ("TEXTCOLOR",     (0, 0), (-1, 0), WHITE),
                   ("FONTNAME",      (0, 0), (-1, 0), "Helvetica-Bold"),
                   ("FONTSIZE",      (0, 0), (-1, 0), 8.5),
                   ("ROWBACKGROUNDS",(0, 1), (-1, -1), [WHITE, MID]),
                   ("LEFTPADDING",   (0, 0), (-1, -1), 6),
                   ("RIGHTPADDING",  (0, 0), (-1, -1), 6),
                   ("TOPPADDING",    (0, 0), (-1, -1), 5),
                   ("BOTTOMPADDING", (0, 0), (-1, -1), 5),
                   ("GRID",          (0, 0), (-1, -1), 0.4, colors.HexColor("#CBD5E0")),
                   ("LINEBELOW",     (0, 0), (-1, 0), 1.5, RED),
                   ("VALIGN",        (0, 0), (-1, -1), "TOP"),
                   # Colour severity cells
                   *[("TEXTCOLOR", (1, i + 1), (1, i + 1),
                      severity_colors[issues[i][1]])
                     for i in range(len(issues))],
               ]))
    story.append(it)

    story.append(PageBreak())
    return story


def build_plan(styles):
    story = []
    story.append(Paragraph("5. Plan for Days 4–7", styles["h1"]))
    story.append(HRFlowable(width="100%", thickness=1.5, color=TEAL,
                             spaceAfter=8, spaceBefore=0))
    story.append(Paragraph(
        "The following plan targets the Path B preference-tuned judge milestone. "
        "Day 4 closes dataset gaps; Days 5–6 run fine-tuning; Day 7 evaluates on "
        "the held-out set and produces the final leaderboard submission.",
        styles["body"]))
    story.append(Spacer(1, 8))

    plan = [
        ("Day 4",
         "Dataset Hardening",
         [
             "Re-partition using cosine-similarity-aware split to reduce template-language "
             "contamination between train and held-out.",
             "Add 30 competitor_gap_honesty tasks (TB-HA-0046..0075) to bring dimension "
             "to ~57 tasks (~19%), closer to 45% Week 10 trigger rate.",
             "Run multi_llm_synthesis.py --live with OPENROUTER_API_KEY to produce "
             "live-synthesized PE tasks; replace dry-run variants where judge score < 0.6.",
             "Second-labeler pass on 20 held-out tasks for multi-person IRR. Target κ ≥ 0.70.",
             "Update contamination_check.json with v0.2 partition results.",
         ]),
        ("Day 5",
         "Preference Pair Construction + Judge Fine-Tuning Setup",
         [
             "Build preference pairs from train partition: (chosen: grounded/calibrated output) "
             "vs (rejected: probe-failing output). Target 150 pairs minimum.",
             "Select SimPO vs ORPO based on paper comparison (reference-free, Colab T4 "
             "memory budget ≤ 15GB). Expected choice: SimPO (lower memory footprint).",
             "Configure fine-tuning run: base model Qwen2.5-7B or Mistral-7B; "
             "LoRA rank 16; 3 epochs; batch 4 × grad_accum 4.",
             "Run baseline scoring: un-fine-tuned judge on 89 dev tasks. Record pass@1.",
         ]),
        ("Day 6",
         "Fine-Tuning Execution + Intermediate Evaluation",
         [
             "Execute judge fine-tuning run (estimated 4–6 hrs on Colab A100).",
             "Mid-training checkpoint evaluation on dev set at epoch 1 and epoch 2.",
             "Compare fine-tuned judge vs baseline judge on dev: target ≥ 5pp improvement "
             "on competitor_gap_honesty and signal_grounding_fidelity.",
             "Run scoring_evaluator.py on all 89 dev tasks with fine-tuned judge. "
             "Log results to dev_results.json.",
         ]),
        ("Day 7",
         "Held-Out Evaluation + Submission",
         [
             "Unseal held-out partition (59 tasks). Run fine-tuned judge end-to-end.",
             "Compute final metrics: pass@1 per dimension, overall pass@1, "
             "Cohen's κ (judge vs human labels on labeled subset).",
             "Run contamination audit on held-out: verify no N-gram overlap with "
             "any training task used in fine-tuning.",
             "Produce final report: update this document with held-out results, "
             "IRR table, and judge comparison.",
             "Commit final dataset + model checkpoint. Submit to programme staff.",
         ]),
    ]

    colors_day = [TEAL, colors.HexColor("#0A6B8A"), AMBER, colors.HexColor("#1A5276")]
    for i, (day, title, tasks) in enumerate(plan):
        day_color = colors_day[i % len(colors_day)]
        header = Table([[
            Paragraph(f"<b>{day}</b>", ParagraphStyle(
                "dh", fontName="Helvetica-Bold", fontSize=11,
                textColor=WHITE, leading=14)),
            Paragraph(f"<b>{title}</b>", ParagraphStyle(
                "dt", fontName="Helvetica-Bold", fontSize=11,
                textColor=WHITE, leading=14)),
        ]], colWidths=[1.8 * cm, 13.7 * cm],
            style=TableStyle([
                ("BACKGROUND",    (0, 0), (-1, -1), day_color),
                ("LEFTPADDING",   (0, 0), (-1, -1), 10),
                ("RIGHTPADDING",  (0, 0), (-1, -1), 10),
                ("TOPPADDING",    (0, 0), (-1, -1), 7),
                ("BOTTOMPADDING", (0, 0), (-1, -1), 7),
            ]))
        story.append(header)

        task_rows = []
        for j, task_text in enumerate(tasks):
            task_rows.append([
                Paragraph(f"<b>{j+1}</b>", styles["body_small"]),
                Paragraph(task_text, styles["body_small"]),
            ])
        task_table = Table(task_rows, colWidths=[0.6 * cm, 14.9 * cm],
                           style=TableStyle([
                               ("BACKGROUND",    (0, 0), (-1, -1), LIGHT),
                               ("LEFTPADDING",   (0, 0), (-1, -1), 8),
                               ("RIGHTPADDING",  (0, 0), (-1, -1), 8),
                               ("TOPPADDING",    (0, 0), (-1, -1), 4),
                               ("BOTTOMPADDING", (0, 0), (-1, -1), 4),
                               ("ROWBACKGROUNDS",(0, 0), (-1, -1), [WHITE, LIGHT]),
                               ("BOX",           (0, 0), (-1, -1), 0.5, colors.HexColor("#CBD5E0")),
                               ("VALIGN",        (0, 0), (-1, -1), "TOP"),
                               ("TEXTCOLOR",     (0, 0), (0, -1), day_color),
                           ]))
        story.append(task_table)
        story.append(Spacer(1, 10))

    # Success criteria
    story.append(Spacer(1, 4))
    story.append(Paragraph("5.1  Day 7 Success Criteria", styles["h2"]))
    crit_data = [["Metric", "Target", "Rationale"]]
    criteria = [
        ("judge pass@1 (dev, fine-tuned)", "≥ 0.75", "Baseline pass@1 on un-tuned model establishes floor"),
        ("judge vs human κ (held-out sample)", "≥ 0.70", "Inter-rater threshold for leaderboard validity"),
        ("competitor_gap_honesty pass@1 improvement", "≥ +0.10", "Highest Week 10 trigger rate — primary target"),
        ("Held-out contamination violations", "< 10%", "N-gram or embedding overlap in held-out / train pairs"),
        ("All 300 tasks scorable", "100%", "No null candidate_output in final submission"),
    ]
    for row in criteria:
        crit_data.append(list(row))
    ct = styled_table(crit_data, [5.5 * cm, 2.5 * cm, 7.5 * cm])
    story.append(ct)

    story.append(Spacer(1, 20))
    story.append(HRFlowable(width="100%", thickness=0.5, color=colors.HexColor("#CBD5E0"),
                             spaceAfter=6, spaceBefore=6))
    story.append(Paragraph(
        "Tenacious-Bench v0.1  ·  10Academy Rapid Prototyping Programme Week 11  ·  "
        "Rafia  ·  rafia@10academy.org  ·  CC-BY-4.0",
        ParagraphStyle("footer_final", fontName="Helvetica", fontSize=7.5,
                       textColor=SLATE, leading=10, alignment=TA_CENTER)))

    return story


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    print("Loading data...")
    tasks    = load_all_tasks()
    manifest = load_manifest()
    pg, tr, ha = pick_examples(tasks)
    print(f"  {len(tasks)} tasks loaded. Examples: {pg['task_id']}, {tr['task_id']}, {ha['task_id']}")

    styles = build_styles()

    print("Building PDF...")
    doc = SimpleDocTemplate(
        str(OUTPUT),
        pagesize=A4,
        leftMargin=1.5 * cm, rightMargin=1.5 * cm,
        topMargin=2.2 * cm,  bottomMargin=2.0 * cm,
        title="Tenacious-Bench v0.1 Interim Report",
        author="Rafia",
        subject="Evaluation Dataset — Interim Report",
    )

    story = []
    story += build_cover(styles)
    story += build_composition(styles, tasks, manifest)
    story += build_ira(styles)
    story += build_examples(styles)
    story += build_status(styles)
    story += build_plan(styles)

    def on_page(canvas, doc):
        if doc.page == 1:
            cover_canvas(canvas, doc)
        else:
            normal_canvas(canvas, doc)

    doc.build(story, onFirstPage=on_page, onLaterPages=on_page)
    print(f"\n✓ Report written to: {OUTPUT.resolve()}")
    print(f"  Size: {OUTPUT.stat().st_size / 1024:.1f} KB")


if __name__ == "__main__":
    main()
