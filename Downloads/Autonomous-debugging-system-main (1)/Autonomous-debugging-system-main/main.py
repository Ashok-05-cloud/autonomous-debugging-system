#!/usr/bin/env python3
"""
AI Multi-Agent Debugger - CLI Entry Point

Usage:
  python main.py --bug-report mock_repo/bug_report.json --logs mock_repo/logs/production.log --repo mock_repo
  python main.py --bug-report mock_repo/bug_report.json --logs mock_repo/logs/production.log
  python main.py --help

All inputs are optional with graceful fallbacks.
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time
from pathlib import Path

# Ensure project root is on PYTHONPATH
sys.path.insert(0, str(Path(__file__).parent))

from orchestrator.graph import run_pipeline
from utils.output_formatter import build_output, save_output
from utils.logger import get_logger

logger = get_logger("main")


def load_bug_report(path: str) -> dict:
    """Load bug report from JSON or Markdown file."""
    p = Path(path)
    if not p.exists():
        logger.error(f"Bug report not found: {path}")
        return {}

    content = p.read_text(encoding="utf-8", errors="replace")

    if p.suffix.lower() == ".json":
        try:
            return json.loads(content)
        except json.JSONDecodeError as e:
            logger.error(f"Invalid JSON in bug report: {e}")
            return {"title": "Unparseable bug report", "description": content[:500]}

    # Markdown / plain text
    return {
        "title": p.stem.replace("_", " ").replace("-", " ").title(),
        "description": content[:2000],
    }


def load_logs(path: str) -> str:
    """Load logs from file."""
    p = Path(path)
    if not p.exists():
        logger.warning(f"Log file not found: {path}")
        return ""
    try:
        return p.read_text(encoding="utf-8", errors="replace")
    except Exception as e:
        logger.warning(f"Cannot read logs: {e}")
        return ""


def print_summary(output: dict) -> None:
    """Print a human-readable summary to stdout."""
    sep = "=" * 70

    print(f"\n{sep}")
    print("  AI MULTI-AGENT DEBUGGER — ANALYSIS COMPLETE")
    print(sep)

    bs = output.get("bug_summary", {})
    rc = output.get("root_cause", {})
    repro = output.get("repro", {})
    pp = output.get("patch_plan", {})
    meta = output.get("metadata", {})

    print(f"\n📋 BUG: {bs.get('title', 'Unknown')}")
    print(f"   Severity: {bs.get('severity', 'unknown').upper()}")
    print(f"   Component: {bs.get('scope', 'unknown')}")

    print(f"\n🔍 ROOT CAUSE ({rc.get('confidence', 0):.0%} confidence):")
    print(f"   {rc.get('hypothesis', 'Not determined')}")

    print(f"\n🧪 REPRODUCTION:")
    status_icon = "✅" if repro.get("bug_confirmed") else "❌"
    print(f"   {status_icon} Status: {repro.get('status', 'unknown')}")
    if repro.get("file"):
        print(f"   File: {repro.get('file')}")
    if repro.get("output"):
        # Print last few lines of repro output
        lines = repro["output"].strip().splitlines()
        preview = "\n      ".join(lines[-5:])
        print(f"   Output (last 5 lines):\n      {preview}")

    print(f"\n🔧 PATCH PLAN:")
    print(f"   {pp.get('summary', 'No patch plan')}")
    for f in pp.get("files_to_change", [])[:3]:
        print(f"   → {f.get('file')} ({f.get('lines')}): {f.get('change', '')[:80]}")

    print(f"\n📊 PIPELINE METRICS:")
    raw_conf = meta.get("pipeline_confidence", 0)
    try:
        conf_display = f"{float(raw_conf):.0%}"
    except (TypeError, ValueError):
        conf_display = "N/A"
    print(f"   Overall Confidence: {conf_display}")
    print(f"   LLM Used: {'Yes' if meta.get('llm_used') else 'No (fallback mode)'}")
    print(f"   Recommendation: {meta.get('review_recommendation', 'N/A')}")

    errors = meta.get("pipeline_errors", [])
    if errors:
        print(f"\n⚠️  PIPELINE WARNINGS ({len(errors)}):")
        for e in errors[:5]:
            print(f"   • {e[:100]}")

    oq = output.get("open_questions", [])
    if oq:
        print(f"\n❓ OPEN QUESTIONS:")
        for q in oq[:4]:
            print(f"   • {q}")

    print(f"\n{sep}\n")


def main():
    parser = argparse.ArgumentParser(
        description="AI Multi-Agent Debugger",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Full run with repo
  python main.py --bug-report mock_repo/bug_report.json \\
                 --logs mock_repo/logs/production.log \\
                 --repo mock_repo

  # No repo (standalone repro generation)
  python main.py --bug-report mock_repo/bug_report.json \\
                 --logs mock_repo/logs/production.log

  # Just a log file (minimal mode)
  python main.py --logs mock_repo/logs/production.log
        """,
    )
    parser.add_argument("--bug-report", "-b", help="Path to bug report (.json or .md)")
    parser.add_argument("--logs", "-l", help="Path to log file (.log or .txt)")
    parser.add_argument("--repo", "-r", help="Path to repository root directory")
    parser.add_argument("--output", "-o", help="Output JSON file path (default: auto-named in outputs/reports/)")
    parser.add_argument("--no-summary", action="store_true", help="Suppress human-readable summary")

    args = parser.parse_args()

    # ── Load inputs ───────────────────────────────────────────────────────
    bug_report = load_bug_report(args.bug_report) if args.bug_report else {}
    logs = load_logs(args.logs) if args.logs else ""
    repo_path = args.repo or ""

    if not bug_report and not logs:
        print("ERROR: Provide at least --bug-report or --logs", file=sys.stderr)
        print("Run with --help for usage.", file=sys.stderr)
        sys.exit(1)

    if not bug_report:
        bug_report = {
            "title": "Log-based investigation (no bug report)",
            "description": "No bug report provided — analyzing from logs only",
        }

    # ── Run pipeline ──────────────────────────────────────────────────────
    start = time.time()
    logger.info("Starting pipeline...")

    final_state = run_pipeline(bug_report, logs, repo_path)

    elapsed = time.time() - start
    logger.info(f"Pipeline completed in {elapsed:.1f}s")

    # ── Format output ─────────────────────────────────────────────────────
    output = build_output(final_state)

    # ── Save output ───────────────────────────────────────────────────────
    output_path = save_output(output, filename=args.output)
    print(f"\n📄 Report saved: {output_path}")

    # ── Print summary ─────────────────────────────────────────────────────
    if not args.no_summary:
        print_summary(output)

    # Exit with non-zero if confidence is very low
    confidence = output.get("metadata", {}).get("pipeline_confidence", 0)
    sys.exit(0 if confidence > 0.1 else 1)


if __name__ == "__main__":
    main()
