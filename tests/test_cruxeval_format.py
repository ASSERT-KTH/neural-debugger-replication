"""Tests for CruxEval prompt construction and response parsing."""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.eval.cruxeval import (
    _parse_return_value,
    _verify_output,
    build_input_prompt,
    build_output_prompt,
)
from src.trace.serializer import ACTION_SEP, ARG_SEP, FRAME_SEP, SRC_SEP


SAMPLE_FUNC = """\
def add(a, b):
    return a + b
"""


def test_output_prompt_contains_step_return():
    prompt = build_output_prompt(SAMPLE_FUNC, "(2, 3)")
    assert "step_return" in prompt
    assert ACTION_SEP in prompt
    assert FRAME_SEP in prompt


def test_input_prompt_contains_inv_step_call():
    prompt = build_input_prompt(SAMPLE_FUNC, "5")
    assert "inv_step_call" in prompt
    assert ACTION_SEP in prompt


def test_output_prompt_ends_with_partial_state():
    prompt = build_output_prompt(SAMPLE_FUNC, "(2, 3)")
    assert prompt.endswith(f"{FRAME_SEP}return{SRC_SEP}")


def test_parse_return_value_with_arg_sep():
    generated = f"{ARG_SEP}42\n{{\"x\": 1}}"
    val = _parse_return_value(generated)
    assert val == "42"


def test_parse_return_value_plain():
    val = _parse_return_value("  5  ")
    assert val == "5"


def test_verify_output_correct():
    assert _verify_output(SAMPLE_FUNC, "(2, 3)", "5", "5") is True


def test_verify_output_wrong():
    assert _verify_output(SAMPLE_FUNC, "(2, 3)", "5", "99") is False


def test_verify_output_literal_eval():
    # Both sides are Python literals — should compare by value
    assert _verify_output(SAMPLE_FUNC, "([1,2],)", "[1, 2]", "[1,2]") is True


def test_trajectory_sampler_produces_output():
    """Smoke test: run the full pipeline on a trivial function."""
    from src.trace.collector import TraceCollector
    from src.trace.state_tree import build_tree
    from src.data.trajectory_sampler import TrajectoryPolicy
    from src.trace.serializer import serialize_trajectory

    def multiply(x, y):
        return x * y

    collector = TraceCollector()
    collector.run(multiply, 3, 4)
    tree = build_tree(collector.events)

    policy = TrajectoryPolicy(seed=0)
    steps = policy.sample_forward(tree)
    assert len(steps) >= 2

    traj = serialize_trajectory("def multiply(x, y): return x * y", steps)
    assert ACTION_SEP in traj
    assert FRAME_SEP in traj
