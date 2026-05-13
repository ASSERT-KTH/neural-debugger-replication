"""Tests for debugger action navigation."""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.debugger.actions import (
    apply_action,
    continue_run,
    inv_step_call,
    inv_step_into,
    inv_step_over,
    make_breakpoint,
    step_into,
    step_over,
    step_return,
)
from src.trace.collector import RawEvent, TraceCollector
from src.trace.state_tree import build_tree


def _evt(evt, lineno=1, depth=0, src=""):
    return RawEvent(evt=evt, filename="<test>", lineno=lineno, src=src,
                    locals={}, args=None, depth=depth, func_name="f")


def _build(events):
    return build_tree(events)


# ---------------------------------------------------------------- fixtures --

def _nested_tree():
    """
    Tree layout:
      root (call, line=1, depth=0)
        line (line=2, depth=0)
        inner (call, line=3, depth=1)
          inner_line (line=10, depth=1)
          inner_return (return, line=10, depth=1)
        line2 (line=4, depth=0)
        outer_return (return, line=4, depth=0)
    """
    events = [
        _evt("call", lineno=1, depth=0),
        _evt("line", lineno=2, depth=0),
        _evt("call", lineno=3, depth=1),
        _evt("line", lineno=10, depth=1),
        _evt("return", lineno=10, depth=1),
        _evt("line", lineno=4, depth=0),
        _evt("return", lineno=4, depth=0),
    ]
    return build_tree(events)


# ------------------------------------------------------------------ tests --

def test_step_into_descends():
    tree = _nested_tree()
    nxt = step_into(tree)
    assert nxt is not None
    assert nxt.event.lineno == 2


def test_step_over_skips_inner_call():
    tree = _nested_tree()
    inner_call = tree.children[1]
    assert inner_call.event.evt == "call"
    result = step_over(inner_call)
    assert result is not None
    assert result.event.lineno == 4  # jumped over inner call subtree


def test_step_return_finds_return():
    tree = _nested_tree()
    inner_call = tree.children[1]
    inner_line = inner_call.children[0]
    ret = step_return(inner_line)
    assert ret is not None
    assert ret.event.evt in ("return", "exception")


def test_breakpoint_finds_target_line():
    tree = _nested_tree()
    bp = make_breakpoint(10)
    found = bp(tree)
    assert found is not None
    assert found.event.lineno == 10


def test_breakpoint_returns_none_for_missing_line():
    tree = _nested_tree()
    bp = make_breakpoint(999)
    assert bp(tree) is None


def test_continue_reaches_final_return():
    tree = _nested_tree()
    end = continue_run(tree)
    assert end is not None
    assert end.event.evt in ("return", "exception")


def test_inv_step_into_goes_back():
    tree = _nested_tree()
    line2 = tree.children[0]  # lineno=2
    prev = inv_step_into(line2)
    assert prev is tree  # went back to root


def test_inv_step_over_previous_sibling():
    tree = _nested_tree()
    inner_call = tree.children[1]   # lineno=3
    ps = inv_step_over(inner_call)
    assert ps is not None
    assert ps.event.lineno == 2


def test_inv_step_call_finds_call_node():
    tree = _nested_tree()
    inner_call = tree.children[1]
    inner_line = inner_call.children[0]  # line inside inner call
    # call_node should find the inner_call (or outer call depending on depth)
    result = inv_step_call(inner_line)
    # Result should be the inner call or the outer call
    assert result is not None
    assert result.event.evt == "call"


def test_apply_action_dispatch():
    tree = _nested_tree()
    result = apply_action("step_into", tree)
    assert result is not None


def test_apply_action_breakpoint_syntax():
    tree = _nested_tree()
    result = apply_action("breakpoint(10)", tree)
    assert result is not None
    assert result.event.lineno == 10


def test_apply_action_unknown_raises():
    import pytest
    tree = _nested_tree()
    with pytest.raises(ValueError):
        apply_action("fly_to_moon", tree)


# ---------------------------------------------------------------- integration

def test_full_forward_walk():
    """Walk an entire real trace with step_into and reach the end."""
    def add(a, b):
        return a + b

    collector = TraceCollector()
    collector.run(add, 1, 2)
    tree = build_tree(collector.events)

    node = tree
    visited = [node]
    for _ in range(100):
        nxt = step_into(node)
        if nxt is None:
            break
        visited.append(nxt)
        node = nxt

    assert len(visited) == len(collector.events)
