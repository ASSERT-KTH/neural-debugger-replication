"""
State tree construction from flat RawEvent lists.

The paper organises execution into a call-stack tree:
  - Each function call creates a sub-tree.
  - Children of a call-node are the events that occurred inside that call.
  - Tree depth mirrors call-stack depth.

Navigation methods on TreeNode implement the primitives required by
each debugger action (step_into, step_over, step_return, breakpoint, ...).
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Iterator

if TYPE_CHECKING:
    from src.trace.collector import RawEvent


@dataclass
class TreeNode:
    event: "RawEvent"
    parent: "TreeNode | None" = field(default=None, repr=False)
    children: list["TreeNode"] = field(default_factory=list, repr=False)

    # ------------------------------------------------------------------
    # Forward navigation
    # ------------------------------------------------------------------

    def first_child(self) -> "TreeNode | None":
        """First event inside this call (step_into when on a call node)."""
        return self.children[0] if self.children else None

    def next_sibling(self) -> "TreeNode | None":
        """Next node at the same level (used by step_over)."""
        if self.parent is None:
            return None
        siblings = self.parent.children
        idx = siblings.index(self)
        if idx + 1 < len(siblings):
            return siblings[idx + 1]
        return None

    def return_node(self) -> "TreeNode | None":
        """
        The return/exception node that closes *this* call level.

        If this node is itself a call-node, its last child should be the
        return.  Otherwise walk up to the enclosing call and return its
        closing node.
        """
        if self.event.evt == "call" and self.children:
            last = self.children[-1]
            if last.event.evt in ("return", "exception"):
                return last
        # Walk up to find the enclosing call's return
        target_depth = self.event.depth
        node: TreeNode | None = self
        while node is not None:
            if node.event.evt in ("return", "exception") and node.event.depth == target_depth:
                return node
            node = node.next_sibling() or (node.parent and node.parent.next_sibling())
        return None

    def find_line(self, lineno: int) -> "TreeNode | None":
        """
        First future node (DFS) whose lineno matches `lineno`.
        Searches the full subtree rooted at this node plus right siblings.
        """
        for node in self._dfs_forward():
            if node is self:
                continue
            if node.event.lineno == lineno:
                return node
        return None

    def dfs_next(self) -> "TreeNode | None":
        """Next node in full DFS traversal (step_into semantics)."""
        if self.children:
            return self.children[0]
        return self._next_in_dfs()

    def _next_in_dfs(self) -> "TreeNode | None":
        node: TreeNode | None = self
        while node is not None:
            sib = node.next_sibling()
            if sib is not None:
                return sib
            node = node.parent
        return None

    # ------------------------------------------------------------------
    # Inverse navigation
    # ------------------------------------------------------------------

    def prev_sibling(self) -> "TreeNode | None":
        """Previous sibling (inv_step_over)."""
        if self.parent is None:
            return None
        siblings = self.parent.children
        idx = siblings.index(self)
        if idx > 0:
            return siblings[idx - 1]
        return None

    def dfs_prev(self) -> "TreeNode | None":
        """Previous node in full DFS traversal (inv_step_into)."""
        ps = self.prev_sibling()
        if ps is not None:
            # Descend to its deepest rightmost leaf
            return ps._rightmost_leaf()
        return self.parent

    def call_node(self) -> "TreeNode | None":
        """
        The call-node that opened the current call level (inv_step_call).
        Walks up until we find a 'call' node at depth == self.depth.
        """
        target_depth = self.event.depth
        node: TreeNode | None = self.parent
        while node is not None:
            if node.event.evt == "call" and node.event.depth == target_depth:
                return node
            node = node.parent
        return None

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    def _rightmost_leaf(self) -> "TreeNode":
        node = self
        while node.children:
            node = node.children[-1]
        return node

    def _dfs_forward(self) -> Iterator["TreeNode"]:
        """DFS iterator starting from this node."""
        stack = [self]
        while stack:
            node = stack.pop()
            yield node
            for child in reversed(node.children):
                stack.append(child)

    def all_nodes(self) -> list["TreeNode"]:
        """Flat list of all nodes in DFS order."""
        return list(self._dfs_forward())

    def __repr__(self) -> str:
        return (
            f"TreeNode(evt={self.event.evt!r}, "
            f"lineno={self.event.lineno}, "
            f"src={self.event.src!r:.30}, "
            f"depth={self.event.depth}, "
            f"children={len(self.children)})"
        )


def build_tree(events: "list[RawEvent]") -> TreeNode:
    """
    Build a call-stack tree from a flat list of RawEvent objects.

    Returns the root TreeNode (the outermost 'call' event).
    Raises ValueError if events is empty.
    """
    if not events:
        raise ValueError("Cannot build tree from empty event list")

    root = TreeNode(event=events[0])
    current_parent: TreeNode = root

    for raw in events[1:]:
        node = TreeNode(event=raw, parent=current_parent)
        current_parent.children.append(node)

        if raw.evt == "call":
            # Descend: new function call starts a sub-tree
            current_parent = node
        elif raw.evt in ("return", "exception"):
            # Ascend: this call level is closing
            if current_parent.parent is not None:
                current_parent = current_parent.parent

    return root
