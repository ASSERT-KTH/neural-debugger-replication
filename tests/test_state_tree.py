"""Tests for state tree construction and navigation."""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.trace.collector import RawEvent, TraceCollector
from src.trace.state_tree import build_tree


def _evt(evt, lineno=1, depth=0, src=""):
    return RawEvent(evt=evt, filename="<test>", lineno=lineno, src=src,
                    locals={}, args=None, depth=depth, func_name="f")


def test_build_tree_simple():
    events = [
        _evt("call", lineno=1, depth=0),
        _evt("line", lineno=2, depth=0),
        _evt("return", lineno=2, depth=0),
    ]
    root = build_tree(events)
    assert root.event.evt == "call"
    assert len(root.children) == 2  # line + return


def test_build_tree_nested_call():
    events = [
        _evt("call", lineno=1, depth=0),          # root
        _evt("line", lineno=2, depth=0),
        _evt("call", lineno=3, depth=1),           # inner call
        _evt("line", lineno=10, depth=1),
        _evt("return", lineno=10, depth=1),        # inner return
        _evt("line", lineno=4, depth=0),
        _evt("return", lineno=4, depth=0),         # outer return
    ]
    root = build_tree(events)
    inner_call = root.children[1]
    assert inner_call.event.evt == "call"
    assert len(inner_call.children) == 2  # line + return inside inner call


def test_step_over_skips_subtree():
    """step_over on a call node should go to its next sibling, not its children."""
    events = [
        _evt("call", lineno=1, depth=0),
        _evt("line", lineno=2, depth=0),
        _evt("call", lineno=3, depth=1),
        _evt("line", lineno=10, depth=1),
        _evt("return", lineno=10, depth=1),
        _evt("line", lineno=4, depth=0),
        _evt("return", lineno=4, depth=0),
    ]
    root = build_tree(events)
    inner_call = root.children[1]
    assert inner_call.event.evt == "call"

    from src.debugger.actions import step_over
    after = step_over(inner_call)
    # Should land on the line at lineno=4, not inside inner_call
    assert after is not None
    assert after.event.lineno == 4


def test_dfs_next_descends():
    events = [
        _evt("call", lineno=1, depth=0),
        _evt("call", lineno=2, depth=1),
        _evt("return", lineno=2, depth=1),
        _evt("return", lineno=1, depth=0),
    ]
    root = build_tree(events)
    nxt = root.dfs_next()
    assert nxt is not None
    assert nxt.event.depth == 1  # descended into inner call


def test_prev_sibling():
    events = [
        _evt("call", lineno=1, depth=0),
        _evt("line", lineno=2, depth=0),
        _evt("line", lineno=3, depth=0),
        _evt("return", lineno=3, depth=0),
    ]
    root = build_tree(events)
    line3 = root.children[1]
    assert line3.event.lineno == 3
    ps = line3.prev_sibling()
    assert ps is not None
    assert ps.event.lineno == 2


def test_find_line():
    events = [
        _evt("call", lineno=1, depth=0),
        _evt("line", lineno=5, depth=0),
        _evt("line", lineno=10, depth=0),
        _evt("return", lineno=10, depth=0),
    ]
    root = build_tree(events)
    found = root.find_line(10)
    assert found is not None
    assert found.event.lineno == 10


def test_all_nodes_count():
    def add(a, b):
        return a + b

    collector = TraceCollector()
    collector.run(add, 1, 2)
    tree = build_tree(collector.events)
    nodes = tree.all_nodes()
    assert len(nodes) == len(collector.events)


def test_build_tree_raises_on_empty():
    import pytest
    with pytest.raises(ValueError):
        build_tree([])
