"""
CruxEval evaluation harness for the neural debugger.

CruxEval (https://huggingface.co/datasets/cruxeval-org/cruxeval) contains
800 Python functions each with an input/output pair.

Two tasks are evaluated:
  output_prediction : given f + input, predict f(input)   (uses step_return)
  input_prediction  : given f + output, infer input        (uses inv_step_call)

For each task the model is prompted with a partial trajectory and asked
to generate the final state. The generated state is parsed and verified
by executing the function in a sandboxed subprocess.
"""

from __future__ import annotations

import ast
import contextlib
import io
import multiprocessing
import textwrap
from typing import Any

from src.trace.serializer import (
    ACTION_SEP,
    FRAME_SEP,
    ARG_SEP,
    SRC_SEP,
    deserialize_state,
    serialize_trajectory,
)
from src.eval.metrics import pass_at_1, field_accuracy


# ------------------------------------------------------------------ public --

def evaluate(
    model_path: str,
    limit: int | None = None,
    max_new_tokens: int = 256,
    device: str = "cpu",
    verbose: bool = False,
) -> dict[str, Any]:
    """
    Run full CruxEval evaluation and return a results dict.

    Parameters
    ----------
    model_path : str
        Path to a fine-tuned model checkpoint (or HuggingFace model ID).
    limit : int | None
        Evaluate only the first `limit` problems (for quick tests).
    max_new_tokens : int
        Maximum tokens to generate per prediction.
    device : str
        Torch device ('cpu', 'cuda', 'mps').
    verbose : bool
        Print per-problem results.
    """
    import datasets as hf_datasets
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
    model = AutoModelForCausalLM.from_pretrained(
        model_path, trust_remote_code=True, torch_dtype=torch.float32
    ).to(device)
    model.eval()

    ds = hf_datasets.load_dataset("cruxeval-org/cruxeval", split="test", trust_remote_code=True)
    if limit is not None:
        ds = ds.select(range(min(limit, len(ds))))

    output_results: list[bool] = []
    input_results: list[bool] = []

    for row in ds:
        func_code = row["code"]
        inp_str = row["input"]
        out_str = row["output"]

        # -- output prediction --
        prompt = build_output_prompt(func_code, inp_str)
        generated = _generate(model, tokenizer, prompt, max_new_tokens, device)
        pred_output = _parse_return_value(generated)
        ok_out = _verify_output(func_code, inp_str, out_str, pred_output)
        output_results.append(ok_out)

        # -- input prediction --
        prompt_inv = build_input_prompt(func_code, out_str)
        generated_inv = _generate(model, tokenizer, prompt_inv, max_new_tokens, device)
        pred_input = _parse_return_value(generated_inv)
        ok_in = _verify_input(func_code, pred_input, out_str)
        input_results.append(ok_in)

        if verbose:
            print(f"  out={'✓' if ok_out else '✗'}  in={'✓' if ok_in else '✗'}  {func_code[:40]!r}")

    return {
        "output_pass_at_1": pass_at_1(output_results),
        "input_pass_at_1": pass_at_1(input_results),
        "n": len(output_results),
    }


def build_output_prompt(func_code: str, input_str: str) -> str:
    """
    Build the output-prediction prompt.

    The model receives the function source + a step_return action and must
    predict the return state (i.e., the function's output).
    """
    return (
        f"{func_code.strip()}\n"
        f"# Input: {input_str}\n"
        f"{ACTION_SEP}step_return{ACTION_SEP}"
        f"{FRAME_SEP}return{SRC_SEP}"
    )


def build_input_prompt(func_code: str, output_str: str) -> str:
    """
    Build the input-prediction prompt.

    The model receives the function source + expected output and an
    inv_step_call action, and must predict the call arguments.
    """
    return (
        f"{func_code.strip()}\n"
        f"# Output: {output_str}\n"
        f"{ACTION_SEP}inv_step_call{ACTION_SEP}"
        f"{FRAME_SEP}call{SRC_SEP}"
    )


# ----------------------------------------------------------------- helpers --

def _generate(model, tokenizer, prompt: str, max_new_tokens: int, device: str) -> str:
    import torch

    inputs = tokenizer(prompt, return_tensors="pt").to(device)
    with torch.no_grad():
        out = model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            do_sample=False,
            pad_token_id=tokenizer.pad_token_id or tokenizer.eos_token_id,
        )
    # Decode only the newly generated tokens
    new_tokens = out[0][inputs["input_ids"].shape[1]:]
    return tokenizer.decode(new_tokens, skip_special_tokens=False)


def _parse_return_value(text: str) -> str:
    """
    Extract the args/return-value portion from a generated state string.

    Looks for the pattern: ...<|arg_sep|>{value}\n{locals}
    and returns just {value}.
    """
    if ARG_SEP in text:
        after = text.split(ARG_SEP, 1)[1]
        value = after.split("\n", 1)[0].strip()
        return value
    return text.strip()


def _verify_output(func_code: str, inp_str: str, expected_out: str, predicted: str) -> bool:
    """Check predicted == expected_out (string comparison after normalization)."""
    try:
        pred_val = ast.literal_eval(predicted)
        exp_val = ast.literal_eval(expected_out)
        return pred_val == exp_val
    except Exception:
        return predicted.strip() == expected_out.strip()


def _verify_input(func_code: str, pred_input_str: str, expected_out: str) -> bool:
    """
    Verify input prediction by executing f(pred_input) and comparing to expected_out.
    Runs in a subprocess with a 5-second timeout for safety.
    """
    def _run(q, code, inp_s, out_s):
        try:
            g: dict = {}
            exec(textwrap.dedent(code), g)  # noqa: S102
            fn = next(v for v in g.values() if callable(v) and not isinstance(v, type))
            args = ast.literal_eval(inp_s)
            if not isinstance(args, tuple):
                args = (args,)
            result = fn(*args)
            expected = ast.literal_eval(out_s)
            q.put(result == expected)
        except Exception:
            q.put(False)

    q: multiprocessing.Queue = multiprocessing.Queue()
    p = multiprocessing.Process(target=_run, args=(q, func_code, pred_input_str, expected_out))
    p.start()
    p.join(timeout=5)
    if p.is_alive():
        p.kill()
        return False
    if q.empty():
        return False
    return bool(q.get())
