"""
Tokenizer utilities for the neural debugger.

Loads a base HuggingFace tokenizer and extends it with the four special
tokens required by the formal debugger grammar.
"""

from __future__ import annotations

from src.trace.serializer import SPECIAL_TOKENS


def load_tokenizer(model_name_or_path: str):
    """
    Load tokenizer from `model_name_or_path` and add debugger special tokens.

    Returns the extended tokenizer. Call tokenizer.save_pretrained(path) to
    persist it alongside the model checkpoint.
    """
    from transformers import AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(model_name_or_path, trust_remote_code=True)

    # Add pad token if missing (common for decoder-only models)
    if tokenizer.pad_token is None:
        tokenizer.add_special_tokens({"pad_token": "<|pad|>"})

    # Add the debugger grammar tokens
    added = tokenizer.add_special_tokens({"additional_special_tokens": SPECIAL_TOKENS})
    if added > 0:
        print(f"Added {added} special token(s) to tokenizer.")

    return tokenizer
