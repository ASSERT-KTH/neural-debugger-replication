"""Tests for the trace collector."""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.trace.collector import TraceCollector


def add(a, b):
    return a + b


def fib(n):
    if n <= 1:
        return n
    return fib(n - 1) + fib(n - 2)


def test_basic_events_captured():
    collector = TraceCollector()
    collector.run(add, 2, 3)
    evts = [e.evt for e in collector.events]
    assert "call" in evts
    assert "return" in evts


def test_return_value_captured():
    collector = TraceCollector()
    collector.run(add, 2, 3)
    returns = [e for e in collector.events if e.evt == "return"]
    assert len(returns) >= 1
    assert returns[-1].args == 5


def test_locals_snapshot():
    def assign_locals(x):
        y = x * 2
        return y

    collector = TraceCollector()
    collector.run(assign_locals, 7)
    # All events (line + return) after the assignment should have y=14
    events_with_y = [e for e in collector.events if isinstance(e.locals, dict) and "y" in e.locals]
    assert len(events_with_y) >= 1
    assert events_with_y[0].locals["y"] == 14


def test_depth_increments_on_call():
    collector = TraceCollector()
    collector.run(fib, 3)
    call_events = [e for e in collector.events if e.evt == "call"]
    depths = [e.depth for e in call_events]
    # Recursive calls go deeper
    assert max(depths) > 0


def test_max_events_truncation():
    collector = TraceCollector(max_events=5)
    collector.run(fib, 10)
    assert len(collector.events) <= 5
    assert collector._truncated


def test_run_code():
    collector = TraceCollector()
    g = collector.run_code("x = 1 + 1\n")
    assert g.get("x") == 2


def test_no_stdlib_frames():
    collector = TraceCollector(skip_stdlib=True)
    collector.run(add, 1, 2)
    filenames = {e.filename for e in collector.events}
    for fn in filenames:
        assert not fn.startswith("<frozen"), f"Stdlib frame leaked: {fn}"
