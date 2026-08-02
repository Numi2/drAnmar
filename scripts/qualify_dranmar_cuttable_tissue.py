#!/usr/bin/env python3
"""Run the deterministic intact-tissue/scalpel-contact qualification."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from dr_anmar_cuttable_tissue_solver import (
    DEFAULT_PROFILE_PATH,
    load_profile,
    run_intact_scalpel_qualification,
)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--profile", type=Path, default=DEFAULT_PROFILE_PATH)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    profile = load_profile(args.profile)
    receipt = run_intact_scalpel_qualification(profile, profile_path=args.profile)
    payload = receipt.payload()
    encoded = json.dumps(payload, indent=2) + "\n"
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(encoded, encoding="utf-8")
    print(encoded, end="")
    return 0 if receipt.qualified else 1


if __name__ == "__main__":
    raise SystemExit(main())
