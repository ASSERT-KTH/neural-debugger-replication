"""Tests for the serializer module."""

import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.trace.collector import RawEvent, TraceCollector
from src.trace.serializer import (
    ACTION_SEP,
    ARG_SEP,
    FRAME_SEP,
    SRC_SEP,
    UNCHANGED_SENTINEL,
    deserialize_state,
    serialize_state,
    serialize_trajectory,
)
from src.trace.state_tree import TreeNode, build_tree


def _make_node(evt="line", lineno=1, src="x = 1", locals_=None, args=None, depth=0):
    raw = RawEvent(
        evt=evt, filename="<test>", lineno=lineno, src=src,
        locals=locals_ or {}, args=args, depth=depth, func_name="f"
    )
    return TreeNode(event=raw)


def test_serialize_state_structure():
    node = _make_node(evt="line", src="x = 1", locals_={"x": 1})
    s = serialize_state(node)
    assert s.startswith(FRAME_SEP)
    assert SRC_SEP in s
    assert ARG_SEP in s
    assert "line" in s
    assert "x = 1" in s


def test_serialize_state_return():
    node = _make_node(evt="return", args=42)
    s = serialize_state(node)
    assert "42" in s


def test_serialize_state_delta_encoding():
    node2 = _make_node(locals_={"x": 1, "y": 3})
    s = serialize_state(node2, prev_locals={"x": 1, "y": 2})
    d = json.loads(s.split(ARG_SEP, 1)[1].split("\n", 1)[1])
    assert d["x"] == UNCHANGED_SENTINEL   # unchanged
    assert d["y"] == 3                    # changed


def test_deserialize_roundtrip():
    node = _make_node(evt="line", src="y = x + 1", locals_={"x": 5, "y": 6})
    s = serialize_state(node)
    parsed = deserialize_state(s)
    assert parsed["evt"] == "line"
    assert parsed["src"] == "y = x + 1"
    assert parsed["locals"]["x"] == 5


def test_serialize_trajectory_format():
    def add(a, b):
        return a + b

    collector = TraceCollector()
    collector.run(add, 2, 3)
    tree = build_tree(collector.events)
    nodes = tree.all_nodes()[:3]
    steps = [("start", nodes[0]), ("step_into", nodes[1]), ("step_into", nodes[2])]
    traj = serialize_trajectory("def add(a, b): return a + b", steps)
    assert "def add" in traj
    assert ACTION_SEP in traj
    assert FRAME_SEP in traj


def test_trajectory_has_correct_action_count():
    def f(x):
        return x * 2

    collector = TraceCollector()
    collector.run(f, 3)
    tree = build_tree(collector.events)
    nodes = tree.all_nodes()
    n_steps = min(3, len(nodes))
    steps = [("start", nodes[0])] + [("step_into", nodes[i]) for i in range(1, n_steps)]
    traj = serialize_trajectory("def f(x): return x * 2", steps)
    # Each step (after "start") contributes one ACTION_SEP pair
    sep_count = traj.count(ACTION_SEP)
    # start + n_steps-1 actions, each adds 2 seps (action + state)
    assert sep_count >= (n_steps - 1) * 2
