"""
Fine-tuning script for the neural debugger.

Trains a causal language model on serialized debugger trajectories using
HuggingFace Trainer. Supports QLoRA (via peft) for GPU-efficient fine-tuning.

Key choices:
  - Causal LM loss over full sequence (no masking of the prompt portion)
  - Sequence length: 4096 tokens (reduced from paper's 16384 for small-scale runs)
  - QLoRA r=16 by default; set --no-lora for full fine-tuning (requires more VRAM)

Usage (via scripts/train.py):
  python scripts/train.py \
    --model deepseek-ai/deepseek-coder-1.3b-base \
    --dataset data/mbpp_train \
    --output checkpoints/nd-1.3b \
    --epochs 3
"""

from __future__ import annotations

from dataclasses import dataclass

from src.model.tokenizer import load_tokenizer


@dataclass
class TrainConfig:
    model_name: str = "deepseek-ai/deepseek-coder-1.3b-base"
    dataset_path: str = "data/mbpp_train"
    output_dir: str = "checkpoints/nd-1.3b"
    num_train_epochs: int = 3
    max_steps: int = -1               # -1 = use num_train_epochs
    per_device_train_batch_size: int = 4
    gradient_accumulation_steps: int = 4
    learning_rate: float = 1e-4
    max_seq_length: int = 4096
    use_lora: bool = True
    lora_r: int = 16
    lora_alpha: int = 32
    lora_dropout: float = 0.05
    fp16: bool = False
    bf16: bool = False
    logging_steps: int = 10
    save_steps: int = 200
    seed: int = 42


def train(cfg: TrainConfig) -> None:
    import datasets as hf_datasets
    import torch
    from transformers import (
        AutoModelForCausalLM,
        DataCollatorForLanguageModeling,
        Trainer,
        TrainingArguments,
    )

    # ---------------------------------------------------------------- load --
    tokenizer = load_tokenizer(cfg.model_name)

    model = AutoModelForCausalLM.from_pretrained(
        cfg.model_name,
        trust_remote_code=True,
        torch_dtype=torch.bfloat16 if cfg.bf16 else (torch.float16 if cfg.fp16 else torch.float32),
    )

    # Resize embeddings for added special tokens
    model.resize_token_embeddings(len(tokenizer))

    # ---------------------------------------------------------------- LoRA --
    if cfg.use_lora:
        from peft import LoraConfig, TaskType, get_peft_model

        lora_cfg = LoraConfig(
            r=cfg.lora_r,
            lora_alpha=cfg.lora_alpha,
            lora_dropout=cfg.lora_dropout,
            bias="none",
            task_type=TaskType.CAUSAL_LM,
            # Target the attention projection layers (works for most decoder models)
            target_modules=["q_proj", "v_proj", "k_proj", "o_proj"],
        )
        model = get_peft_model(model, lora_cfg)
        model.print_trainable_parameters()

    # ------------------------------------------------------------- dataset --
    ds = hf_datasets.load_from_disk(cfg.dataset_path)

    def tokenize(batch):
        return tokenizer(
            batch["text"],
            truncation=True,
            max_length=cfg.max_seq_length,
            padding=False,
        )

    tokenized = ds.map(tokenize, batched=True, remove_columns=ds.column_names)

    collator = DataCollatorForLanguageModeling(tokenizer=tokenizer, mlm=False)

    # ------------------------------------------------------------ training --
    training_args = TrainingArguments(
        output_dir=cfg.output_dir,
        num_train_epochs=cfg.num_train_epochs,
        max_steps=cfg.max_steps,
        per_device_train_batch_size=cfg.per_device_train_batch_size,
        gradient_accumulation_steps=cfg.gradient_accumulation_steps,
        learning_rate=cfg.learning_rate,
        fp16=cfg.fp16,
        bf16=cfg.bf16,
        logging_steps=cfg.logging_steps,
        save_steps=cfg.save_steps,
        save_total_limit=2,
        gradient_checkpointing=True,
        report_to="none",
        seed=cfg.seed,
        dataloader_num_workers=0,
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=tokenized,
        data_collator=collator,
        tokenizer=tokenizer,
    )

    trainer.train()

    # Save final model and tokenizer
    model.save_pretrained(cfg.output_dir)
    tokenizer.save_pretrained(cfg.output_dir)
    print(f"Model saved to {cfg.output_dir}")
