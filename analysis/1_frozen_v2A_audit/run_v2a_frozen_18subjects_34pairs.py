"""Run frozen v2A retrieval on the local 18-subject/34-pair ds004998 cohort.

This BYJoWR-inclusive expansion keeps the frozen v2A scorer unchanged while
rebuilding the local Hold/Move cohort after the additional
sub-BYJoWR HoldR MedOff/MedOn run-2 files were downloaded. The original
17-subject frozen outputs are not modified.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from scripts import run_v2a_frozen_17subjects_33pairs as frozen17


DATA_ROOT = frozen17.DATA_ROOT
AUDIT_DIR = Path("outputs/ds004998_full_cohort_completeness_audit_18subjects_34pairs")
OUTPUT_DIR = Path("outputs/v2A_frozen_18subjects_34pairs")
TABLES_DIR = OUTPUT_DIR / "tables"
EXPECTED_COMPLETE_PAIRS = 34
EXPECTED_SUBJECTS_WITH_COMPLETE_PAIRS = 18
EXPECTED_COMPLETE_GROUPS = set(frozen17.EXPECTED_COMPLETE_GROUPS) | {("sub-BYJoWR", "HoldR")}


def parse_args() -> argparse.Namespace:
    """Parse command-line arguments."""

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-root", default=str(DATA_ROOT))
    parser.add_argument("--quality-table", default="outputs/tables/real_recording_quality.csv")
    parser.add_argument("--subspace-definitions", default="outputs/tables/subspace_definitions.csv")
    parser.add_argument("--expected-pairs", type=int, default=EXPECTED_COMPLETE_PAIRS)
    parser.add_argument("--expected-subjects", type=int, default=EXPECTED_SUBJECTS_WITH_COMPLETE_PAIRS)
    parser.add_argument("--allow-count-mismatch", action="store_true")
    parser.add_argument("--max-duration-sec", type=float, default=120.0)
    parser.add_argument("--max-windows", type=int, default=12)
    parser.add_argument("--n-jobs", type=int, default=4)
    parser.add_argument("--parallel-backend", choices=["process", "thread", "sequential"], default="thread")
    parser.add_argument("--recording-batch-size", type=int, default=None)
    return parser.parse_args()


def main() -> None:
    """Entry point."""

    frozen17.AUDIT_DIR = AUDIT_DIR
    frozen17.OUTPUT_DIR = OUTPUT_DIR
    frozen17.TABLES_DIR = TABLES_DIR
    frozen17.EXPECTED_COMPLETE_PAIRS = EXPECTED_COMPLETE_PAIRS
    frozen17.EXPECTED_SUBJECTS_WITH_COMPLETE_PAIRS = EXPECTED_SUBJECTS_WITH_COMPLETE_PAIRS
    frozen17.EXPECTED_COMPLETE_GROUPS = EXPECTED_COMPLETE_GROUPS
    frozen17.run(parse_args())


if __name__ == "__main__":
    main()
