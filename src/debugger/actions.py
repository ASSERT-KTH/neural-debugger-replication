"""
Debugger actions implemented as pure navigation functions on the state tree.

Each forward action takes a TreeNode and returns the next TreeNode
(or None if the action cannot be applied / reaches program end).

Inverse actions mirror the forward ones for inverse-execution modeling.

Forward actions
---------------
step_into    – next node in DFS order (descends into function calls)
step_over    – next sibling, skipping subtrees
step_return  – jump to the return/exception node of the current call level
breakpoint   – jump to the first future node whose lineno matches a target
continue_run – jump to the final return node of the entire program

Inverse actions
---------------
inv_step_into  – previous node in DFS order
inv_step_over  – previous sibling
inv_step_call  – jump to the call node that initiated the current level
                 (this is what the model uses to infer function inputs)
"""

from __future__ import annotations

from typing import Callable

from src.trace.state_tree import TreeNode

# ------------------------------------------------------------------ forward --

def step_into(node: TreeNode) -> TreeNode | None:
    """Next node in DFS traversal (descend into calls)."""
    return node.dfs_next()


def step_over(node: TreeNode) -> TreeNode | None:
    """
    Advance to the next node at the same level.
    If the current node is a call, skip its entire subtree.
    """
    if node.event.evt == "call":
        # Skip to the node *after* the call's closing return, i.e. the
        # next sibling of the call node from its parent's perspective.
        return node.next_sibling()
    return node.next_sibling()


def step_return(node: TreeNode) -> TreeNode | None:
    """Jump to the return/exception node that closes the current call."""
    ret = node.return_node()
    return ret


def make_breakpoint(lineno: int) -> Callable[[TreeNode], TreeNode | None]:
    """Return a breakpoint action targeting a specific source line."""
    def _breakpoint(node: TreeNode) -> TreeNode | None:
        return node.find_line(lineno)
    _breakpoint.__name__ = f"breakpoint({lineno})"
    return _breakpoint


def continue_run(node: TreeNode) -> TreeNode | None:
    """Jump to the final return of the entire program (tree root's return)."""
    root = _get_root(node)
    # The last node in the root's children that is a return/exception
    for child in reversed(root.children):
        if child.event.evt in ("return", "exception"):
            return child
    return None


# ------------------------------------------------------------------ inverse --

def inv_step_into(node: TreeNode) -> TreeNode | None:
    """Previous node in DFS traversal (inverse of step_into)."""
    return node.dfs_prev()


def inv_step_over(node: TreeNode) -> TreeNode | None:
    """Previous sibling (inverse of step_over)."""
    return node.prev_sibling()


def inv_step_call(node: TreeNode) -> TreeNode | None:
    """
    Jump to the call node that opened the current call level.
    Used to infer function inputs during inverse execution.
    """
    return node.call_node()


# ---------------------------------------------------------------- registry --

#: All forward action names understood by the trajectory sampler.
FORWARD_ACTIONS: list[str] = [
    "step_into",
    "step_over",
    "step_return",
    "continue",
]

#: All inverse action names understood by the trajectory sampler.
INVERSE_ACTIONS: list[str] = [
    "inv_step_into",
    "inv_step_over",
    "inv_step_call",
]

_ACTION_FN: dict[str, Callable[[TreeNode], TreeNode | None]] = {
    "step_into": step_into,
    "step_over": step_over,
    "step_return": step_return,
    "continue": continue_run,
    "inv_step_into": inv_step_into,
    "inv_step_over": inv_step_over,
    "inv_step_call": inv_step_call,
}


def apply_action(action: str, node: TreeNode) -> TreeNode | None:
    """
    Apply a named action to a node.

    `action` may be either a plain name ('step_into') or a
    parameterised breakpoint ('breakpoint(42)').
    """
    if action.startswith("breakpoint(") and action.endswith(")"):
        lineno = int(action[len("breakpoint("):-1])
        return make_breakpoint(lineno)(node)
    fn = _ACTION_FN.get(action)
    if fn is None:
        raise ValueError(f"Unknown action: {action!r}")
    return fn(node)


# ----------------------------------------------------------------- helpers --

def _get_root(node: TreeNode) -> TreeNode:
    while node.parent is not None:
        node = node.parent
    return node
