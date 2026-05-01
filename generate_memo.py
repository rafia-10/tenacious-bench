#!/usr/bin/env python3
"""
Day 7 — Generate two-page memo.pdf using ReportLab.
Reads numbers from evidence_graph.json and ablation_results.json.
Navy/teal palette, table-heavy, every number footnoted to source file.
"""

import json
from pathlib import Path

ROOT = Path(__file__).parent
EVIDENCE = ROOT / "evidence_graph.json"
ABLATION_RESULTS = ROOT / "ablations/ablation_results.json"
OUTPUT_PDF = ROOT / "memo.pdf"


def load_evidence():
    with open(EVIDENCE) as f:
        eg = json.load(f)
    vals = {}
    for key, entry in eg.items():
        if key.startswith("_"):
            continue
        vals[key] = entry.get("value", "N/A")
    return vals, eg


def load_ablations():
    if not ABLATION_RESULTS.exists():
        return {}
    with open(ABLATION_RESULTS) as f:
        return json.load(f)


def fmt(val, decimals=4):
    if val == "PENDING" or val is None:
        return "TBD"
    try:
        return f"{float(val):.{decimals}f}"
    except (TypeError, ValueError):
        return str(val)


def make_pdf(vals: dict, ablations: dict, eg: dict):
    from reportlab.lib.pagesizes import letter
    from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
    from reportlab.lib.units import inch
    from reportlab.lib import colors
    from reportlab.platypus import (
        SimpleDocTemplate, Paragraph, Table, TableStyle, Spacer, HRFlowable
    )
    from reportlab.lib.enums import TA_CENTER, TA_LEFT

    NAVY = colors.HexColor("#1B2A4A")
    TEAL = colors.HexColor("#00848A")
    LIGHT_TEAL = colors.HexColor("#E8F4F5")
    WHITE = colors.white
    DARK_GRAY = colors.HexColor("#333333")

    doc = SimpleDocTemplate(
        str(OUTPUT_PDF),
        pagesize=letter,
        leftMargin=0.75 * inch,
        rightMargin=0.75 * inch,
        topMargin=0.5 * inch,
        bottomMargin=0.6 * inch,
    )

    styles = getSampleStyleSheet()
    h1 = ParagraphStyle("H1", fontSize=14, textColor=NAVY, spaceAfter=4,
                         fontName="Helvetica-Bold", leading=18)
    h2 = ParagraphStyle("H2", fontSize=10, textColor=TEAL, spaceAfter=3,
                         fontName="Helvetica-Bold", leading=14)
    body = ParagraphStyle("Body", fontSize=8, textColor=DARK_GRAY, spaceAfter=4,
                           fontName="Helvetica", leading=12)
    footnote = ParagraphStyle("Footnote", fontSize=6.5, textColor=colors.gray, spaceAfter=2,
                               fontName="Helvetica-Oblique", leading=9)
    header_center = ParagraphStyle("HeaderCenter", fontSize=16, textColor=WHITE,
                                    fontName="Helvetica-Bold", alignment=TA_CENTER, leading=20)
    subtitle = ParagraphStyle("Subtitle", fontSize=8, textColor=WHITE,
                               fontName="Helvetica", alignment=TA_CENTER, leading=12)

    story = []

    def header_banner(text, subtext=""):
        data = [[Paragraph(text, header_center)]]
        if subtext:
            data.append([Paragraph(subtext, subtitle)])
        t = Table(data, colWidths=[7 * inch])
        t.setStyle(TableStyle([
            ("BACKGROUND", (0, 0), (-1, -1), NAVY),
            ("TOPPADDING", (0, 0), (-1, -1), 10),
            ("BOTTOMPADDING", (0, 0), (-1, -1), 10),
            ("LEFTPADDING", (0, 0), (-1, -1), 12),
            ("RIGHTPADDING", (0, 0), (-1, -1), 12),
        ]))
        return t

    def data_table(headers, rows, col_widths=None):
        data = [headers] + rows
        if col_widths is None:
            col_widths = [7 * inch / len(headers)] * len(headers)
        t = Table(data, colWidths=col_widths)
        style = [
            ("BACKGROUND", (0, 0), (-1, 0), TEAL),
            ("TEXTCOLOR", (0, 0), (-1, 0), WHITE),
            ("FONTNAME", (0, 0), (-1, 0), "Helvetica-Bold"),
            ("FONTSIZE", (0, 0), (-1, 0), 8),
            ("FONTSIZE", (0, 1), (-1, -1), 7.5),
            ("ROWBACKGROUNDS", (0, 1), (-1, -1), [WHITE, LIGHT_TEAL]),
            ("GRID", (0, 0), (-1, -1), 0.5, colors.HexColor("#CCCCCC")),
            ("TOPPADDING", (0, 0), (-1, -1), 4),
            ("BOTTOMPADDING", (0, 0), (-1, -1), 4),
            ("LEFTPADDING", (0, 0), (-1, -1), 6),
            ("RIGHTPADDING", (0, 0), (-1, -1), 6),
            ("VALIGN", (0, 0), (-1, -1), "MIDDLE"),
            ("ALIGN", (1, 0), (-1, -1), "CENTER"),
        ]
        t.setStyle(TableStyle(style))
        return t

    # ─────────────────── PAGE 1 ───────────────────
    story.append(header_banner(
        "Tenacious-Bench v0.1 — Decision Memo",
        "Conversion Engine Judge Training Report · 2026-05-01"
    ))
    story.append(Spacer(1, 0.12 * inch))

    # Executive summary
    story.append(Paragraph("Executive Summary", h1))
    da_mean = fmt(vals.get("delta_a_mean"), 4)
    da_ci_lo = fmt(ablations.get("delta_a", {}).get("ci_lower"), 4) if ablations else "TBD"
    da_ci_hi = fmt(ablations.get("delta_a", {}).get("ci_upper"), 4) if ablations else "TBD"
    da_p = fmt(ablations.get("delta_a", {}).get("p_value"), 4) if ablations else "TBD"
    da_sig = ablations.get("delta_a", {}).get("significant", "TBD") if ablations else "TBD"
    db_find = ablations.get("delta_b", {}).get("finding", "TBD") if ablations else "TBD"
    n_pairs = fmt(vals.get("training_pair_count"), 0)
    n_held = vals.get("held_out_task_count", 59)

    exec_text = (
        f"<b>What was built:</b> Tenacious-Bench v0.1 (300 tasks, 5 rubric dimensions) and a preference-tuned ORPO judge "
        f"(Qwen2.5-1.5B + LoRA, {n_pairs} training pairs). "
        f"<b>Headline result:</b> Trained judge vs. Week 10 baseline on {n_held} held-out tasks: "
        f"Δ={da_mean} [{da_ci_lo}, {da_ci_hi}] p={da_p} (significant={da_sig}). "
        f"<b>Recommendation:</b> See deployment table below."
    )
    story.append(Paragraph(exec_text, body))
    story.append(Spacer(1, 0.08 * inch))

    # Delta A table
    story.append(Paragraph("Delta A — Trained Judge vs. Week 10 Baseline¹", h2))
    delta_a = ablations.get("delta_a", {}) if ablations else {}
    rows_a = [
        ["Mean difference", da_mean],
        ["95% CI", f"[{da_ci_lo}, {da_ci_hi}]"],
        ["p-value (bootstrap, n=10k)", da_p],
        ["Statistically significant (p<0.05)", str(da_sig)],
        ["Held-out tasks", str(n_held)],
        ["Baseline mean score", fmt(vals.get("baseline_mean_score"), 4)],
        ["Trained mean score", fmt(vals.get("trained_mean_score"), 4)],
    ]
    story.append(data_table(["Metric", "Value"], rows_a, [3.5 * inch, 3.5 * inch]))
    story.append(Paragraph(
        "¹ Source: ablations/ablation_results.json/delta_a; bootstrap_test.py seed=42, 10,000 iterations",
        footnote))
    story.append(Spacer(1, 0.08 * inch))

    # Delta B table
    story.append(Paragraph("Delta B — Trained Judge vs. Prompt-Only Judge²", h2))
    delta_b = ablations.get("delta_b", {}) if ablations else {}
    db_mean = fmt(vals.get("delta_b_mean"), 4)
    db_ci_lo = fmt(delta_b.get("ci_lower"), 4) if delta_b else "TBD"
    db_ci_hi = fmt(delta_b.get("ci_upper"), 4) if delta_b else "TBD"
    db_p = fmt(delta_b.get("p_value"), 4) if delta_b else "TBD"
    db_sig = delta_b.get("significant", "TBD") if delta_b else "TBD"
    rows_b = [
        ["Mean difference", db_mean],
        ["95% CI", f"[{db_ci_lo}, {db_ci_hi}]"],
        ["p-value", db_p],
        ["Significant (p<0.05)", str(db_sig)],
        ["Prompt-only model", "Qwen3-30B (zero-shot)"],
        ["Finding", str(db_find)],
    ]
    story.append(data_table(["Metric", "Value"], rows_b, [3.5 * inch, 3.5 * inch]))
    story.append(Paragraph(
        "² Source: ablations/ablation_results.json/delta_b. If finding=prompt_engineering_sufficient, "
        "prompting is a viable lower-cost alternative at this training scale.",
        footnote))
    story.append(Spacer(1, 0.08 * inch))

    # Cost table
    story.append(Paragraph("Cost Per Task³", h2))
    rows_c = [
        ["Condition", "Cost per task", "Source"],
        ["Baseline (scoring_evaluator)", fmt(vals.get("cost_per_task_baseline"), 6), "ablations/held_out_traces.jsonl"],
        ["Trained judge", fmt(vals.get("cost_per_task_trained"), 6), "ablations/held_out_traces.jsonl"],
    ]
    t = Table(rows_c, colWidths=[2.5 * inch, 2 * inch, 2.5 * inch])
    t.setStyle(TableStyle([
        ("BACKGROUND", (0, 0), (-1, 0), TEAL),
        ("TEXTCOLOR", (0, 0), (-1, 0), WHITE),
        ("FONTNAME", (0, 0), (-1, 0), "Helvetica-Bold"),
        ("FONTSIZE", (0, 0), (-1, -1), 7.5),
        ("ROWBACKGROUNDS", (0, 1), (-1, -1), [WHITE, LIGHT_TEAL]),
        ("GRID", (0, 0), (-1, -1), 0.5, colors.HexColor("#CCCCCC")),
        ("TOPPADDING", (0, 0), (-1, -1), 4),
        ("BOTTOMPADDING", (0, 0), (-1, -1), 4),
        ("LEFTPADDING", (0, 0), (-1, -1), 6),
        ("RIGHTPADDING", (0, 0), (-1, -1), 6),
    ]))
    story.append(t)
    story.append(Paragraph("³ Source: ablations/held_out_traces.jsonl latency_ms × compute rate", footnote))
    story.append(Spacer(1, 0.08 * inch))

    # Deployment decision
    story.append(Paragraph("Deployment Recommendation⁴", h2))
    deploy_rows = [
        ["Condition", "Recommendation"],
        ["Delta A significant AND cost delta acceptable", "DEPLOY trained judge"],
        ["Delta A significant BUT Delta B neutral (prompting sufficient)", "DEPLOY WITH CAVEAT: monitor agreement"],
        ["Delta A NOT significant (training did not beat baseline)", "DO NOT DEPLOY: revert to baseline"],
    ]
    t = Table(deploy_rows, colWidths=[4 * inch, 3 * inch])
    t.setStyle(TableStyle([
        ("BACKGROUND", (0, 0), (-1, 0), NAVY),
        ("TEXTCOLOR", (0, 0), (-1, 0), WHITE),
        ("FONTNAME", (0, 0), (-1, 0), "Helvetica-Bold"),
        ("FONTSIZE", (0, 0), (-1, -1), 7.5),
        ("ROWBACKGROUNDS", (0, 1), (-1, -1), [WHITE, LIGHT_TEAL]),
        ("GRID", (0, 0), (-1, -1), 0.5, colors.HexColor("#CCCCCC")),
        ("TOPPADDING", (0, 0), (-1, -1), 4),
        ("BOTTOMPADDING", (0, 0), (-1, -1), 4),
        ("LEFTPADDING", (0, 0), (-1, -1), 6),
        ("RIGHTPADDING", (0, 0), (-1, -1), 6),
    ]))
    story.append(t)
    story.append(Paragraph("⁴ Based on ablation_results.json/delta_a/significant and cost delta from held_out_traces.jsonl", footnote))

    # Page break
    from reportlab.platypus import PageBreak
    story.append(PageBreak())

    # ─────────────────── PAGE 2 ───────────────────
    story.append(header_banner(
        "Tenacious-Bench v0.1 — The Skeptic's Appendix",
        "Four gaps v0.1 cannot capture · Unresolved training failure · Kill-switch trigger"
    ))
    story.append(Spacer(1, 0.12 * inch))

    story.append(Paragraph("Four Failure Modes Tenacious-Bench v0.1 Does Not Capture", h1))
    gaps_rows = [
        ["Gap", "Description", "Why It Matters"],
        ["Multi-turn trajectory compounding",
         "Per-turn rubric checks miss compounding errors across a 5-turn thread",
         "Turn-3 plausible response contradicts turn-1 commitment"],
        ["Cultural register differences",
         "EU/US/East Africa prospects require different tone calibration",
         "US-calibrated score 1.0 may read as aggressive in Nairobi"],
        ["Bench capacity drift",
         "Bench_summary is a weekly snapshot; judge trained on static data",
         "Judge miscalibrates as real bench diverges from training snapshot"],
        ["Competitive landscape changes",
         "Gap briefs are synthetic; real competitors change ML posture monthly",
         "A correct gap brief in January may be false by March"],
    ]
    t = Table(gaps_rows, colWidths=[1.8 * inch, 3.0 * inch, 2.2 * inch])
    t.setStyle(TableStyle([
        ("BACKGROUND", (0, 0), (-1, 0), TEAL),
        ("TEXTCOLOR", (0, 0), (-1, 0), WHITE),
        ("FONTNAME", (0, 0), (-1, 0), "Helvetica-Bold"),
        ("FONTSIZE", (0, 0), (-1, -1), 7),
        ("ROWBACKGROUNDS", (0, 1), (-1, -1), [WHITE, LIGHT_TEAL]),
        ("GRID", (0, 0), (-1, -1), 0.5, colors.HexColor("#CCCCCC")),
        ("TOPPADDING", (0, 0), (-1, -1), 3),
        ("BOTTOMPADDING", (0, 0), (-1, -1), 3),
        ("LEFTPADDING", (0, 0), (-1, -1), 5),
        ("RIGHTPADDING", (0, 0), (-1, -1), 5),
        ("VALIGN", (0, 0), (-1, -1), "TOP"),
        ("WORDWRAP", (0, 0), (-1, -1), True),
    ]))
    story.append(t)
    story.append(Spacer(1, 0.08 * inch))

    story.append(Paragraph("Public-Signal Lossiness in Ground Truth⁵", h2))
    gt_text = (
        "The Week 10 agent has two documented false-positive and false-negative failure modes in rubric labeling. "
        "<b>False positive (quietly sophisticated):</b> A prospect with no visible AI signal (low Crunchbase score, no open ML roles) "
        "may have a mature internal ML capability — the rubric scores an ABSTAIN-mode output as correct, "
        "but the agent was actually right to be assertive. Ground truth over-penalizes. "
        "<b>False negative (loud but shallow):</b> A prospect with a high-velocity ML hiring signal may be performatively "
        "scaling without genuine AI strategy — the rubric rewards confident assert-mode outreach that a senior "
        "AE would recognize as over-pitched. These two modes affect ~12% of signal_grounding_fidelity tasks "
        "and create systematic bias in the training preference pairs."
    )
    story.append(Paragraph(gt_text, body))
    story.append(Paragraph("⁵ Source: inter_rater_agreement.md section 3; probe P-005, P-006", footnote))
    story.append(Spacer(1, 0.08 * inch))

    story.append(Paragraph("Honest Unresolved Training Failure⁶", h2))
    db_txt = ablations.get("delta_b", {}).get("finding_explanation", "Ablations not yet run.") if ablations else "Ablations not yet run."
    story.append(Paragraph(db_txt[:600], body))
    story.append(Paragraph("⁶ Source: ablations/ablation_results.json/delta_b/finding_explanation", footnote))
    story.append(Spacer(1, 0.08 * inch))

    story.append(Paragraph("Kill-Switch Trigger⁷", h2))
    ks_text = (
        "<b>Condition:</b> If judge agreement with human reviewers on weekly spot-check drops below 80%, "
        "freeze the trained judge component and revert to the prompt-only judge (Qwen3-30B zero-shot). "
        "<b>Implementation:</b> The kill-switch is enforced via the KILL_SWITCH environment variable "
        "(already active from Week 10). When triggered, the rejection-sampling gate uses the prompt-only "
        "judge until the adapter is retrained on updated preference pairs. "
        "<b>Monitoring cadence:</b> Weekly spot-check on 20 randomly sampled live Conversion Engine outputs, "
        "scored by human reviewer against the five Tenacious rubric dimensions. "
        "Agreement < 80% on any two consecutive weeks triggers the freeze."
    )
    story.append(Paragraph(ks_text, body))
    story.append(Paragraph("⁷ Source: methodology.md kill-switch protocol; KILL_SWITCH env var from Week 10", footnote))

    doc.build(story)
    print(f"Memo written to {OUTPUT_PDF}")


def main():
    vals, eg = load_evidence()
    ablations = load_ablations()

    # If ablation results exist, update evidence graph values
    if ablations:
        def update_eg(key, source_val):
            if key in eg and source_val is not None:
                eg[key]["value"] = source_val

        da = ablations.get("delta_a", {})
        db = ablations.get("delta_b", {})
        bl = ablations.get("baseline", {})
        tr = ablations.get("trained", {})
        po = ablations.get("prompt_only", {})

        update_eg("delta_a_mean", da.get("mean_diff"))
        update_eg("delta_a_ci", f"[{da.get('ci_lower')}, {da.get('ci_upper')}]")
        update_eg("delta_a_p_value", da.get("p_value"))
        update_eg("delta_a_significant", da.get("significant"))
        update_eg("delta_b_mean", db.get("mean_diff"))
        update_eg("delta_b_significant", db.get("significant"))
        update_eg("delta_b_finding", db.get("finding"))
        update_eg("baseline_mean_score", bl.get("mean"))
        update_eg("trained_mean_score", tr.get("mean"))
        update_eg("prompt_only_mean_score", po.get("mean"))

        # Training pair count from file
        pairs_path = ROOT / "training_data/preference_pairs.jsonl"
        if pairs_path.exists():
            with open(pairs_path) as f:
                n_pairs = sum(1 for l in f if l.strip())
            update_eg("training_pair_count", n_pairs)

        # Update evidence_graph.json
        for key, entry in eg.items():
            if not key.startswith("_"):
                vals[key] = entry.get("value", "PENDING")

        with open(EVIDENCE, "w") as f:
            json.dump(eg, f, indent=2)
        print("evidence_graph.json updated with ablation values")

    make_pdf(vals, ablations, eg)


if __name__ == "__main__":
    main()
