"""
Dataset generation pipeline.

Loads Python functions from a source corpus (MBPP by default), executes them
under the trace collector, builds state trees, samples trajectories, serialises
them, and writes a HuggingFace Dataset to disk.

Usage (as a library)::

    from src.data.generate import generate_dataset
    ds = generate_dataset(source="mbpp", split="train", trajectories_per_trace=10)
    ds.save_to_disk("data/mbpp_train")

CLI usage: see scripts/generate_data.py
"""

from __future__ import annotations

import ast
import sys
import textwrap
import traceback
from typing import Any, Iterator

from src.data.trajectory_sampler import TrajectoryPolicy
from src.trace.collector import TraceCollector
from src.trace.serializer import serialize_trajectory
from src.trace.state_tree import build_tree


# ------------------------------------------------------------------ public --

def generate_dataset(
    source: str = "mbpp",
    split: str = "train",
    trajectories_per_trace: int = 10,
    max_programs: int | None = None,
    seed: int = 42,
    verbose: bool = False,
) -> "datasets.Dataset":  # type: ignore[name-defined]
    """
    Generate a HuggingFace Dataset of serialized neural-debugger trajectories.

    Parameters
    ----------
    source : str
        Dataset source. Currently supported: 'mbpp', 'cruxeval'.
    split : str
        Dataset split (e.g., 'train', 'test', 'train[:10]').
    trajectories_per_trace : int
        Number of trajectories (forward + inverse) to sample per execution trace.
    max_programs : int | None
        Limit number of programs processed (for smoke tests).
    seed : int
        Base RNG seed.
    verbose : bool
        Print progress information.
    """
    import datasets as hf_datasets

    collector = TraceCollector(max_events=5_000)
    policy = TrajectoryPolicy(seed=seed)

    records: list[dict[str, Any]] = []
    programs = list(_load_programs(source, split))

    if max_programs is not None:
        programs = programs[:max_programs]

    for i, prog in enumerate(programs):
        if verbose:
            print(f"[{i+1}/{len(programs)}] {prog['name']}", file=sys.stderr)

        for call_input in prog["inputs"][:3]:  # at most 3 inputs per function
            try:
                events = _execute_and_collect(collector, prog["code"], call_input)
            except Exception as exc:
                if verbose:
                    print(f"  skip ({exc})", file=sys.stderr)
                continue

            if len(events) < 3:
                continue

            try:
                tree = build_tree(events)
            except ValueError:
                continue

            func_src = prog["code"]

            # Forward trajectories
            for j in range(trajectories_per_trace):
                local_policy = TrajectoryPolicy(seed=seed + i * 1000 + j)
                steps = local_policy.sample_forward(tree)
                if len(steps) >= 2:
                    text = serialize_trajectory(func_src, steps)
                    records.append({"text": text, "direction": "forward", "source": source})

            # Inverse trajectories (same count)
            for j in range(trajectories_per_trace):
                local_policy = TrajectoryPolicy(seed=seed + i * 1000 + j + 500)
                steps = local_policy.sample_inverse(tree)
                if len(steps) >= 2:
                    text = serialize_trajectory(func_src, steps)
                    records.append({"text": text, "direction": "inverse", "source": source})

    if not records:
        raise RuntimeError("No trajectories generated — check that functions executed successfully.")

    ds = hf_datasets.Dataset.from_list(records)
    if verbose:
        print(f"Generated {len(ds)} trajectory examples.", file=sys.stderr)
    return ds


# ----------------------------------------------------------------- loaders --

def _load_programs(source: str, split: str) -> Iterator[dict]:
    """Yield dicts with keys: name, code, inputs."""
    if source == "mbpp":
        yield from _load_mbpp(split)
    elif source == "cruxeval":
        yield from _load_cruxeval(split)
    else:
        raise ValueError(f"Unknown source: {source!r}. Use 'mbpp' or 'cruxeval'.")


def _load_mbpp(split: str) -> Iterator[dict]:
    import datasets as hf_datasets

    ds = hf_datasets.load_dataset("google-research-datasets/mbpp", split=split, trust_remote_code=True)
    for row in ds:
        code = row.get("code", "")
        if not code.strip():
            continue
        # Extract test assertions to derive call inputs
        test_cases = row.get("test_list", []) or []
        inputs = _parse_mbpp_inputs(code, test_cases)
        yield {
            "name": f"mbpp_{row.get('task_id', '?')}",
            "code": code,
            "inputs": inputs,
        }


def _load_cruxeval(split: str) -> Iterator[dict]:
    import datasets as hf_datasets

    ds = hf_datasets.load_dataset("cruxeval-org/cruxeval", split=split, trust_remote_code=True)
    for row in ds:
        code = row.get("code", "")
        inp = row.get("input", None)
        yield {
            "name": f"cruxeval_{row.get('id', '?')}",
            "code": code,
            "inputs": [inp] if inp is not None else [],
        }


# ----------------------------------------------------------------- helpers --

def _parse_mbpp_inputs(code: str, test_cases: list[str]) -> list[Any]:
    """
    Heuristically extract call arguments from MBPP test assertions.

    MBPP tests look like: assert func_name(args) == expected
    We extract the args tuple for each test case.
    """
    import re

    # Find the first function name defined in the code
    func_name = None
    try:
        tree = ast.parse(code)
        for node in ast.walk(tree):
            if isinstance(node, ast.FunctionDef):
                func_name = node.name
                break
    except SyntaxError:
        return []

    if func_name is None:
        return []

    inputs: list[Any] = []
    for test in test_cases:
        # Match: assert func_name(...) == ...
        pattern = rf"assert\s+{re.escape(func_name)}\s*\((.+)\)\s*==\s*"
        m = re.search(pattern, test)
        if not m:
            continue
        args_str = m.group(1)
        try:
            val = ast.literal_eval(f"({args_str},)")
            inputs.append(val)
        except (ValueError, SyntaxError):
            continue
    return inputs


def _execute_and_collect(
    collector: TraceCollector,
    code: str,
    call_input: Any,
) -> list:
    """
    Execute the function defined in `code` with `call_input` and return events.

    `call_input` should be a tuple of positional arguments.
    """
    g: dict[str, Any] = {}
    exec(textwrap.dedent(code), g)  # noqa: S102

    # Find the first callable defined
    fn = None
    for v in g.values():
        if callable(v) and not isinstance(v, type):
            fn = v
            break

    if fn is None:
        raise ValueError("No callable found in code")

    args = call_input if isinstance(call_input, tuple) else (call_input,)
    collector.run(fn, *args)
    return collector.events
