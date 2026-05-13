#!/usr/bin/env python3
"""
CLI: evaluate a fine-tuned neural debugger on CruxEval.

Examples
--------
# Quick test with 10 problems
python scripts/eval_cruxeval.py \\
  --model /tmp/smoke_model \\
  --limit 10 \\
  --verbose

# Full evaluation
python scripts/eval_cruxeval.py \\
  --model checkpoints/nd-1.3b \\
  --device cuda
"""

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.eval.cruxeval import evaluate


def main():
    p = argparse.ArgumentParser(description="Evaluate neural debugger on CruxEval.")
    p.add_argument("--model", required=True, help="Model checkpoint path or HuggingFace model ID")
    p.add_argument("--limit", type=int, default=None, help="Limit number of problems evaluated")
    p.add_argument("--max-new-tokens", type=int, default=256)
    p.add_argument("--device", default="cpu")
    p.add_argument("--verbose", action="store_true")
    args = p.parse_args()

    results = evaluate(
        model_path=args.model,
        limit=args.limit,
        max_new_tokens=args.max_new_tokens,
        device=args.device,
        verbose=args.verbose,
    )

    print(json.dumps(results, indent=2))
    print(f"\nOutput prediction pass@1: {results['output_pass_at_1']:.1%}")
    print(f"Input  prediction pass@1: {results['input_pass_at_1']:.1%}")
    print(f"Total problems evaluated: {results['n']}")


if __name__ == "__main__":
    main()
