"""
src.utils.safeguard  ·  Hardware safety guards.
================================================
Prevents accidental heavy workloads on a local laptop
(RTX 4050 6 GB).  Allows unrestricted runs only on Kaggle.

Usage:
    from src.utils.safeguard import check_environment
    check_environment(num_samples=50_000)  # raises on laptop
"""

from __future__ import annotations

import os
import sys

# ── Constants ────────────────────────────────────────────────────
_LOCAL_SAMPLE_LIMIT = 200  # max samples allowed outside Kaggle


def is_kaggle() -> bool:
    """Return True if running inside a Kaggle kernel."""
    return "KAGGLE_KERNEL_RUN_TYPE" in os.environ


def check_environment(num_samples: int) -> None:
    """
    Refuse to proceed if running locally with too many samples.

    Parameters
    ----------
    num_samples : int
        Number of samples that would be processed.

    Raises
    ------
    RuntimeError
        If not on Kaggle and num_samples > _LOCAL_SAMPLE_LIMIT.
    """
    if is_kaggle():
        return  # no restrictions on Kaggle

    if num_samples > _LOCAL_SAMPLE_LIMIT:
        msg = (
            f"\n{'=' * 60}\n"
            f"  🛑  SAFEGUARD: {num_samples:,} samples detected but you are\n"
            f"      running LOCALLY (limit: {_LOCAL_SAMPLE_LIMIT}).\n"
            f"      Training / extraction with large datasets must run\n"
            f"      on Kaggle.  Use --debug for local smoke tests.\n"
            f"{'=' * 60}\n"
        )
        print(msg, file=sys.stderr)
        raise RuntimeError(
            f"Local run blocked: {num_samples} samples > "
            f"{_LOCAL_SAMPLE_LIMIT} limit.  Use Kaggle or --debug."
        )
    else:
        print(
            f"[SAFEGUARD] Local run OK - {num_samples} samples "
            f"(<= {_LOCAL_SAMPLE_LIMIT} limit)."
        )
