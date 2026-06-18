"""Command-line entrypoint for the ICRAS master pipeline.

Run the full pipeline end-to-end on a single bundle with one command:

    python -m app.main --bundle data/bundles/scenario_03_net_90_payment_terms

Exits with a non-zero status if the pipeline fails.
"""

from __future__ import annotations

import argparse
import sys
from collections.abc import Sequence

from app.orchestrator import PipelineResult, run_pipeline


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="app.main",
        description="Run the ICRAS contract review & risk analysis pipeline.",
    )
    parser.add_argument(
        "--bundle",
        required=True,
        help="Path to the contract bundle directory to process.",
    )
    parser.add_argument(
        "--runs-root",
        default="runs",
        help="Base directory under which the run directory is created "
        "(default: runs).",
    )
    return parser


def _print_summary(result: PipelineResult) -> None:
    if result.succeeded:
        print("ICRAS pipeline completed successfully.")
    else:
        print(f"ICRAS pipeline FAILED: {result.status}")
        if result.error_message:
            print(f"  error: {result.error_message}")

    print(f"  run id        : {result.run_id or 'n/a'}")
    print(f"  run directory : {result.run_directory or 'n/a'}")
    if result.agents_completed:
        print(f"  agents        : {', '.join(result.agents_completed)}")
    print(f"  final decision: {result.final_decision or 'n/a'}")
    print(f"  overall risk  : {result.overall_risk or 'n/a'}")


def main(argv: Sequence[str] | None = None) -> int:
    args = _build_parser().parse_args(list(argv) if argv is not None else None)

    print("ICRAS pipeline started.")
    print(f"  bundle: {args.bundle}")

    result = run_pipeline(args.bundle, runs_root=args.runs_root)

    # Per-agent completion summary.
    for agent_name in result.agents_completed:
        print(f"  [done] agent: {agent_name}")

    _print_summary(result)

    return 0 if result.succeeded else 1


if __name__ == "__main__":
    raise SystemExit(main())
