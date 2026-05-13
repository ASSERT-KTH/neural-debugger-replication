"""
Execution trace collector using sys.settrace().

Captures a flat list of RawEvent objects from running a Python function,
filtering out stdlib frames and taking deep-copied locals snapshots.
"""

from __future__ import annotations

import copy
import linecache
import sys
from dataclasses import dataclass
from typing import Any


@dataclass
class RawEvent:
    evt: str        # 'call' | 'line' | 'return' | 'exception'
    filename: str
    lineno: int
    src: str        # source text of the executing line (stripped)
    locals: dict    # deep-copied snapshot of frame.f_locals
    args: Any       # return value or (exc_type, exc_val, exc_tb) for exceptions; None otherwise
    depth: int      # call stack depth (0 = top-level call)
    func_name: str  # f_code.co_name


def _get_source_line(filename: str, lineno: int) -> str:
    line = linecache.getline(filename, lineno)
    return line.rstrip() if line else ""


class TraceCollector:
    """
    Instruments a callable with sys.settrace() and collects execution events.

    Usage::

        collector = TraceCollector()
        result = collector.run(my_function, arg1, arg2)
        events = collector.events  # list[RawEvent]
    """

    def __init__(self, *, max_events: int = 10_000, skip_stdlib: bool = True):
        self.max_events = max_events
        self.skip_stdlib = skip_stdlib
        self.events: list[RawEvent] = []
        self._depth = -1          # incremented on 'call', decremented after 'return'
        self._root_filename: str | None = None
        self._stdlib_prefixes: tuple[str, ...] = _stdlib_prefixes()
        self._truncated = False

    # ------------------------------------------------------------------
    # sys.settrace callback
    # ------------------------------------------------------------------

    def _trace(self, frame, event: str, arg: Any):
        filename = frame.f_code.co_filename

        if event == "call":
            self._depth += 1
            # Record the root filename on the very first call so we can
            # optionally filter out unrelated frames.
            if self._root_filename is None:
                self._root_filename = filename

        skip = self.skip_stdlib and self._is_stdlib(filename)

        if not skip:
            if len(self.events) >= self.max_events:
                self._truncated = True
                sys.settrace(None)
                return None

            processed_arg: Any = None
            if event == "return":
                processed_arg = _safe_copy(arg)
            elif event == "exception":
                # arg is (exc_type, exc_value, traceback); keep first two
                processed_arg = (arg[0], arg[1])

            self.events.append(
                RawEvent(
                    evt=event,
                    filename=filename,
                    lineno=frame.f_lineno,
                    src=_get_source_line(filename, frame.f_lineno),
                    locals=_safe_copy_locals(frame.f_locals),
                    args=processed_arg,
                    depth=self._depth,
                    func_name=frame.f_code.co_name,
                )
            )

        if event == "return":
            self._depth -= 1

        # Return self to keep tracing into called functions.
        return self._trace

    def _is_stdlib(self, filename: str) -> bool:
        return filename.startswith(self._stdlib_prefixes) or filename.startswith("<")

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def run(self, fn, *args, **kwargs) -> Any:
        """Execute fn(*args, **kwargs) under tracing and return its result."""
        self.events = []
        self._depth = -1
        self._root_filename = None
        self._truncated = False

        old_trace = sys.gettrace()
        sys.settrace(self._trace)
        try:
            result = fn(*args, **kwargs)
        except Exception:
            raise
        finally:
            sys.settrace(old_trace)

        return result

    def run_code(self, source: str, globals_: dict | None = None) -> Any:
        """
        Execute a source string under tracing.

        The source is compiled and exec'd. The *last expression* in the
        source (if it is an expression statement) is returned.
        """
        import textwrap

        source = textwrap.dedent(source)
        g = globals_ if globals_ is not None else {}
        code = compile(source, "<traced>", "exec")

        self.events = []
        self._depth = -1
        self._root_filename = None
        self._truncated = False

        old_trace = sys.gettrace()
        sys.settrace(self._trace)
        try:
            exec(code, g)  # noqa: S102
        finally:
            sys.settrace(old_trace)

        return g


# ------------------------------------------------------------------
# Helpers
# ------------------------------------------------------------------

def _stdlib_prefixes() -> tuple[str, ...]:
    """Return filesystem prefixes that indicate stdlib / site-packages."""
    prefixes = set()
    for mod_name in ("os", "sys", "collections", "typing"):
        mod = sys.modules.get(mod_name)
        if mod and hasattr(mod, "__file__") and mod.__file__:
            prefixes.add(mod.__file__.rsplit("/", 2)[0] + "/")
    # Also skip importlib bootstrap
    prefixes.add("<frozen")
    return tuple(prefixes)


def _safe_copy_locals(frame_locals: Any) -> dict:
    """
    Copy frame locals into a plain dict, deep-copying each value independently.

    Python 3.13 changed frame.f_locals to return a FrameLocalsProxy which
    may not be deepcopy-able as a whole. We iterate key-by-key instead.
    """
    result: dict = {}
    try:
        items = list(dict(frame_locals).items())
    except Exception:
        return {}
    for k, v in items:
        result[k] = _safe_copy(v)
    return result


def _safe_copy(obj: Any) -> Any:
    """Deep-copy obj, falling back to shallow copy then repr on failure."""
    try:
        return copy.deepcopy(obj)
    except Exception:
        try:
            return copy.copy(obj)
        except Exception:
            return repr(obj)
