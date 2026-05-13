# Journal

## 2026-05-13 — First smoke-test training run

**Goal:** clone the repo and complete one successful training run end-to-end.

### Bugs fixed

Three bugs prevented any training from running; all were fixed in commit `746765f` (rebased to `b4d43cf`).

#### 1. `src/trace/collector.py` — all execution events silently dropped (critical)

`TraceCollector._is_stdlib()` contained:

```python
return filename.startswith(self._stdlib_prefixes) or filename.startswith("<")
```

The `or filename.startswith("<")` clause filtered out every frame whose filename starts with `<`, including `<string>` — the filename Python assigns to code compiled by `exec()`. Because MBPP functions are executed via `exec()`, every event was dropped, every program produced zero events, and data generation raised:

```
RuntimeError: No trajectories generated — check that functions executed successfully.
```

Fix: remove the over-broad clause. `<frozen` (importlib bootstrap) is already in `_stdlib_prefixes`, so frozen frames remain excluded.

#### 2. `src/data/generate.py` — deprecated `trust_remote_code` argument

```python
hf_datasets.load_dataset("google-research-datasets/mbpp", split=split, trust_remote_code=True)
```

Current `datasets` raises an error when `trust_remote_code=True` is passed for a dataset that is now Parquet-backed (MBPP was converted). Fix: drop the argument.

#### 3. `src/model/train.py` — two transformers 5.x API breaks

| Old (transformers 4.x) | New (transformers 5.x) |
|---|---|
| `AutoModelForCausalLM.from_pretrained(..., torch_dtype=...)` | `dtype=...` |
| `Trainer(..., tokenizer=tokenizer)` | `processing_class=tokenizer` |

#### 4. `.gitignore` — `data/` shadowing `src/data/` source package

The pattern `data/` matched any directory named `data/`, including `src/data/`, making `git add src/data/generate.py` fail without `-f`. Anchored to `/data/` and `/checkpoints/` so only the top-level generated-data directories are ignored.

### Training run

**Command:**

```bash
# Step 1 – generate dataset (10 MBPP programs, 2 trajectories each)
python3.11 scripts/generate_data.py \
  --source mbpp --split "train[:10]" \
  --trajectories-per-trace 2 \
  --output /tmp/smoke_data --verbose

# Step 2 – train (GPT-2, no LoRA, CPU, 10 steps)
python3.11 scripts/train.py \
  --model gpt2 \
  --dataset /tmp/smoke_data \
  --max-steps 10 \
  --no-lora \
  --output /tmp/smoke_model
```

**Data generation output:**

```
Generated 108 trajectory examples.
Saved 108 examples to /tmp/smoke_data
```

**Training output:**

```
{'loss': '3.403', 'grad_norm': '7.244', 'learning_rate': '1e-05', 'epoch': '1.444'}
{'train_runtime': '207.5', 'train_samples_per_second': '0.771',
 'train_steps_per_second': '0.048', 'train_loss': '3.403', 'epoch': '1.444'}
Model saved to /tmp/smoke_model
```

**Environment:** Python 3.11.13, transformers 5.8.1, torch 2.11.0, CPU only (CUDA driver too old for available GPU).
