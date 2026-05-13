#!/usr/bin/env python3
"""
CLI: generate training trajectories and write a HuggingFace Dataset.

Examples
--------
# Small smoke test (10 programs, 2 trajectories each)
python scripts/generate_data.py \\
  --source mbpp --split "train[:10]" \\
  --trajectories-per-trace 2 \\
  --output /tmp/smoke_data

# Full MBPP training set
python scripts/generate_data.py \\
  --source mbpp --split train \\
  --trajectories-per-trace 10 \\
  --output data/mbpp_train
"""

import argparse
import sys
from pathlib import Path

# Make src importable from project root
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.data.generate import generate_dataset


def main():
    p = argparse.ArgumentParser(description="Generate neural-debugger training data.")
    p.add_argument("--source", default="mbpp", choices=["mbpp", "cruxeval"])
    p.add_argument("--split", default="train")
    p.add_argument("--trajectories-per-trace", type=int, default=10)
    p.add_argument("--max-programs", type=int, default=None)
    p.add_argument("--output", required=True, help="Path to save HuggingFace Dataset")
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--verbose", action="store_true")
    args = p.parse_args()

    ds = generate_dataset(
        source=args.source,
        split=args.split,
        trajectories_per_trace=args.trajectories_per_trace,
        max_programs=args.max_programs,
        seed=args.seed,
        verbose=args.verbose,
    )

    out_path = args.output
    ds.save_to_disk(out_path)
    print(f"Saved {len(ds)} examples to {out_path}")


if __name__ == "__main__":
    main()
