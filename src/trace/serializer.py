"""
Formal grammar serialization for neural debugger trajectories.

Special tokens (must be added to the model tokenizer):
  <|frame_sep|>   separates the state header from the rest
  <|action_sep|>  separates actions from states in a trajectory
  <|src_sep|>     separates EVT from SRC within a state
  <|arg_sep|>     separates SRC from ARGS/LOCALS within a state

State format:
  <|frame_sep|>{EVT}<|src_sep|>{src_line}<|arg_sep|>{args}\\n{locals_json}

Trajectory format:
  {function_source}\\n
  <|action_sep|>{action_1}<|action_sep|>{state_1}
  <|action_sep|>{action_2}<|action_sep|>{state_2}
  ...

Locals are delta-encoded: keys unchanged since the previous state are
replaced with the sentinel value UNCHANGED_SENTINEL in the JSON output.
"""

from __future__ import annotations

import json
from typing import Any

from src.trace.state_tree import TreeNode

# ------------------------------------------------------------------ tokens --

FRAME_SEP = "<|frame_sep|>"
ACTION_SEP = "<|action_sep|>"
SRC_SEP = "<|src_sep|>"
ARG_SEP = "<|arg_sep|>"

SPECIAL_TOKENS: list[str] = [FRAME_SEP, ACTION_SEP, SRC_SEP, ARG_SEP]

# Sentinel used in JSON to indicate an unchanged local variable.
UNCHANGED_SENTINEL = ".."


# ------------------------------------------------------------------ public --

def serialize_state(node: TreeNode, prev_locals: dict | None = None) -> str:
    """
    Serialize a single TreeNode into the formal state string.

    If `prev_locals` is provided, variables with identical values are
    replaced by UNCHANGED_SENTINEL in the locals JSON (delta encoding).
    """
    evt = node.event
    args_str = _format_args(evt.evt, evt.args)
    locals_str = _format_locals(evt.locals, prev_locals)
    return f"{FRAME_SEP}{evt.evt}{SRC_SEP}{evt.src}{ARG_SEP}{args_str}\n{locals_str}"


def serialize_trajectory(
    function_source: str,
    steps: list[tuple[str, TreeNode]],
) -> str:
    """
    Serialize a full trajectory to a training string.

    `steps` is a list of (action_name, tree_node) pairs representing the
    sequence of debugger actions taken and the resulting states.

    The function source is prepended so the model sees the full program.
    """
    parts: list[str] = [function_source]
    prev_locals: dict | None = None
    for action, node in steps:
        state_str = serialize_state(node, prev_locals)
        parts.append(f"{ACTION_SEP}{action}{ACTION_SEP}{state_str}")
        prev_locals = node.event.locals
    return "\n".join(parts)


def deserialize_state(text: str) -> dict[str, Any]:
    """
    Parse a serialized state string back into its components.
    Returns a dict with keys: evt, src, args, locals.
    """
    if not text.startswith(FRAME_SEP):
        raise ValueError(f"Expected state to start with {FRAME_SEP!r}")
    text = text[len(FRAME_SEP):]

    evt, rest = text.split(SRC_SEP, 1)
    src, rest2 = rest.split(ARG_SEP, 1)

    if "\n" in rest2:
        args_str, locals_str = rest2.split("\n", 1)
    else:
        args_str, locals_str = rest2, "{}"

    try:
        locals_dict = json.loads(locals_str)
    except json.JSONDecodeError:
        locals_dict = {}

    return {
        "evt": evt.strip(),
        "src": src.strip(),
        "args": args_str.strip(),
        "locals": locals_dict,
    }


# ----------------------------------------------------------------- private --

def _format_args(evt: str, args: Any) -> str:
    if args is None:
        return ""
    if evt == "return":
        return _safe_json(args)
    if evt == "exception":
        exc_type, exc_val = args if isinstance(args, tuple) and len(args) == 2 else (None, args)
        return f"{exc_type.__name__ if exc_type else 'Exception'}: {exc_val}"
    return ""


def _format_locals(
    current: dict,
    previous: dict | None,
) -> str:
    if previous is None:
        return _safe_json(current)

    delta: dict[str, Any] = {}
    for k, v in current.items():
        if k in previous and _values_equal(previous[k], v):
            delta[k] = UNCHANGED_SENTINEL
        else:
            delta[k] = v
    # Include keys that disappeared (deleted variables)
    return _safe_json(delta)


def _values_equal(a: Any, b: Any) -> bool:
    try:
        return a == b
    except Exception:
        return False


def _safe_json(obj: Any) -> str:
    try:
        return json.dumps(obj, default=repr, ensure_ascii=False)
    except Exception:
        return repr(obj)
