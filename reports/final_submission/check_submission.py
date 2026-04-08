#!/usr/bin/env python3
"""Lightweight verification for the final submission package."""

from __future__ import annotations

from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[2]
REPORT_DIR = Path(__file__).resolve().parent
OUTPUT_STEM = "Deloitte_Quantum_Sustainability_Challenge_2026_Submission"

required_inputs = [
    REPO_ROOT / "reports" / "tables" / "model_comparison.csv",
    REPO_ROOT / "reports" / "tables" / "quantum_resources.csv",
    REPO_ROOT / "insurance-model" / "results" / "metrics" / "all_experiments_combined.csv",
    REPO_ROOT / "reports" / "figures" / "quantum_vs_classical.png",
    REPO_ROOT / "insurance-model" / "results" / "figures" / "experiment_comparison.png",
]

required_outputs = [
    REPORT_DIR / f"{OUTPUT_STEM}.md",
    REPORT_DIR / f"{OUTPUT_STEM}.pdf",
]


def main() -> None:
    missing = [path for path in [*required_inputs, *required_outputs] if not path.exists()]
    if missing:
        print("Missing files:")
        for path in missing:
            print(f"  - {path}")
        raise SystemExit(1)

    markdown = required_outputs[0].read_text(encoding="utf-8")
    abstract_section = markdown.split("## One-Page Summary / Abstract", 1)[1].split(
        "## Detailed Description of the Participant's Algorithm", 1
    )[0]
    abstract_words = len([token for token in abstract_section.split() if token.strip()])
    github_links = markdown.count("https://github.com/")
    pdf_size = required_outputs[1].stat().st_size

    required_headings = [
        "## Overview of the Individual or Team and Backgrounds",
        "## One-Page Summary / Abstract",
        "## Detailed Description of the Participant's Algorithm",
        "## Description of Results",
        "## Description of the Envisioned Algorithm",
        "## Clickable Repo Links",
    ]
    required_markers = [
        "task 1a",
        "task 1b",
        "task 2",
        "additional data used",
    ]

    print("Submission package looks complete.")
    print(f"Abstract word count: {abstract_words}")
    print(f"GitHub links found in markdown: {github_links}")
    print(f"PDF size: {pdf_size} bytes")

    if abstract_words > 400:
        raise SystemExit("Abstract exceeds the 400-word competition limit.")
    if github_links < 5:
        raise SystemExit("Expected at least five clickable GitHub links in the report.")
    if pdf_size <= 0:
        raise SystemExit("Generated PDF is empty.")
    for heading in required_headings:
        if heading not in markdown:
            raise SystemExit(f"Missing required report heading: {heading}")
    markdown_lower = markdown.lower()
    for marker in required_markers:
        if marker not in markdown_lower:
            raise SystemExit(f"Missing required task/compliance marker: {marker}")


if __name__ == "__main__":
    main()
