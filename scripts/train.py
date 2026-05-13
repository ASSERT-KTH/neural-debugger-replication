#!/usr/bin/env python3
"""
CLI: fine-tune a causal LM on neural-debugger trajectories.

Examples
--------
# CPU smoke test with GPT-2 (no QLoRA needed)
python scripts/train.py \\
  --model gpt2 \\
  --dataset /tmp/smoke_data \\
  --max-steps 50 \\
  --no-lora \\
  --output /tmp/smoke_model

# Full run (GPU required)
python scripts/train.py \\
  --model deepseek-ai/deepseek-coder-1.3b-base \\
  --dataset data/mbpp_train \\
  --epochs 3 \\
  --output checkpoints/nd-1.3b
"""

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.model.train import TrainConfig, train


def main():
    p = argparse.ArgumentParser(description="Fine-tune neural debugger model.")
    p.add_argument("--model", default="deepseek-ai/deepseek-coder-1.3b-base")
    p.add_argument("--dataset", required=True, help="Path to HuggingFace Dataset on disk")
    p.add_argument("--output", required=True)
    p.add_argument("--epochs", type=int, default=3)
    p.add_argument("--max-steps", type=int, default=-1)
    p.add_argument("--batch-size", type=int, default=4)
    p.add_argument("--grad-accum", type=int, default=4)
    p.add_argument("--lr", type=float, default=1e-4)
    p.add_argument("--max-seq-len", type=int, default=4096)
    p.add_argument("--lora-r", type=int, default=16)
    p.add_argument("--no-lora", action="store_true")
    p.add_argument("--fp16", action="store_true")
    p.add_argument("--bf16", action="store_true")
    p.add_argument("--seed", type=int, default=42)
    args = p.parse_args()

    cfg = TrainConfig(
        model_name=args.model,
        dataset_path=args.dataset,
        output_dir=args.output,
        num_train_epochs=args.epochs,
        max_steps=args.max_steps,
        per_device_train_batch_size=args.batch_size,
        gradient_accumulation_steps=args.grad_accum,
        learning_rate=args.lr,
        max_seq_length=args.max_seq_len,
        use_lora=not args.no_lora,
        lora_r=args.lora_r,
        fp16=args.fp16,
        bf16=args.bf16,
        seed=args.seed,
    )

    train(cfg)


if __name__ == "__main__":
    main()
