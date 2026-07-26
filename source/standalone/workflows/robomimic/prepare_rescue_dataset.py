# Copyright (c) 2026, Dr.Anmar Project Developers.
# SPDX-License-Identifier: BSD-3-Clause

"""Merge Autonomous Rescue OR recordings into an imitation dataset."""

from __future__ import annotations

import argparse
from pathlib import Path
import sys


REPOSITORY_ROOT = Path(__file__).resolve().parents[4]
sys.path.insert(0, str(REPOSITORY_ROOT / "scripts"))

from dr_anmar_rescue_dataset import (  # noqa: E402
    SCHEMA,
    merge_rescue_training_hdf5,
)


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Merge transition-aligned Autonomous Rescue OR HDF5 episodes. "
            "Train/validation masks are split by whole episode."
        )
    )
    parser.add_argument(
        "sources",
        nargs="+",
        type=Path,
        help="rescue training HDF5 files emitted beside workstation recordings",
    )
    parser.add_argument(
        "--output",
        type=Path,
        required=True,
        help="destination merged HDF5 dataset",
    )
    parser.add_argument(
        "--validation-fraction",
        type=float,
        default=0.15,
    )
    parser.add_argument("--seed", type=int, default=7777)
    parser.add_argument(
        "--allow-non-reference",
        action="store_true",
        help=(
            "include failed, interrupted, or operator runs; off by default "
            "so behavior cloning cannot silently ingest failed experts"
        ),
    )
    args = parser.parse_args()
    destination = args.output.expanduser().resolve()
    destination.parent.mkdir(parents=True, exist_ok=True)
    result = merge_rescue_training_hdf5(
        [path.expanduser().resolve() for path in args.sources],
        destination,
        validation_fraction=args.validation_fraction,
        seed=args.seed,
        require_reference_eligible=not args.allow_non_reference,
    )
    print(
        f"Wrote {SCHEMA} dataset to {result} "
        f"from {len(args.sources)} complete episodes"
    )


if __name__ == "__main__":
    main()
