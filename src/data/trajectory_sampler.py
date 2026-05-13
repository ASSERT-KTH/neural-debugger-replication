"""
Stochastic trajectory sampler implementing the paper's mixed action policy.

Policy (§3.3 of arXiv 2603.09951):
  With probability p_uniform=0.5: sample uniformly from ALL available actions.
  With probability 1-p_uniform=0.5: sample uniformly from {step_into, step_over}.

The sampler walks the state tree by repeatedly:
  1. Choosing an action according to the policy.
  2. Applying the action to get the next node.
  3. Recording the (action, node) pair.

The trajectory ends when 'continue' is chosen, no further nodes exist,
or max_steps is reached.

For inverse trajectories the same structure is used with inverse actions.
"""

from __future__ import annotations

import random
from typing import TYPE_CHECKING

from src.debugger.actions import (
    FORWARD_ACTIONS,
    INVERSE_ACTIONS,
    apply_action,
)

if TYPE_CHECKING:
    from src.trace.state_tree import TreeNode

# Actions that are always available regardless of position.
_STEP_BASIC = ["step_into", "step_over"]
_STEP_BASIC_INV = ["inv_step_into", "inv_step_over"]


class TrajectoryPolicy:
    """
    Generates a single debugger trajectory from a starting tree node.

    Parameters
    ----------
    p_uniform : float
        Probability of sampling uniformly from *all* available forward
        actions (vs. sampling only from {step_into, step_over}).
    max_steps : int
        Hard cap on trajectory length to avoid infinite loops.
    seed : int | None
        Optional RNG seed for reproducibility.
    """

    def __init__(
        self,
        p_uniform: float = 0.5,
        max_steps: int = 200,
        seed: int | None = None,
    ):
        self.p_uniform = p_uniform
        self.max_steps = max_steps
        self._rng = random.Random(seed)

    # ------------------------------------------------------------------
    # Forward trajectory
    # ------------------------------------------------------------------

    def sample_forward(self, root: "TreeNode") -> list[tuple[str, "TreeNode"]]:
        """
        Sample a forward trajectory starting from `root`.

        Returns a list of (action_name, resulting_node) pairs.
        The first entry is always ("start", root) to provide the initial state.
        """
        steps: list[tuple[str, "TreeNode"]] = [("start", root)]
        node = root

        for _ in range(self.max_steps):
            action = self._choose_forward_action(node)
            next_node = apply_action(action, node)

            if next_node is None:
                # Dead end: append a continue to close the trajectory
                steps.append(("continue", node))
                break

            steps.append((action, next_node))
            node = next_node

            if action == "continue" or node.event.evt == "return" and node.parent is None:
                break

        return steps

    # ------------------------------------------------------------------
    # Inverse trajectory
    # ------------------------------------------------------------------

    def sample_inverse(self, root: "TreeNode") -> list[tuple[str, "TreeNode"]]:
        """
        Sample an inverse trajectory starting from the last node in DFS order.

        Returns (action_name, resulting_node) pairs walking *backward*.
        """
        # Start from the deepest rightmost leaf (program end)
        all_nodes = root.all_nodes()
        if not all_nodes:
            return []

        node = all_nodes[-1]
        steps: list[tuple[str, "TreeNode"]] = [("start", node)]

        for _ in range(self.max_steps):
            action = self._choose_inverse_action(node)
            next_node = apply_action(action, node)

            if next_node is None:
                steps.append(("inv_continue", node))
                break

            steps.append((action, next_node))
            node = next_node

            if next_node.parent is None:
                break

        return steps

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _choose_forward_action(self, node: "TreeNode") -> str:
        available = self._available_forward(node)
        if self._rng.random() < self.p_uniform:
            return self._rng.choice(available)
        else:
            # Restrict to basic stepping actions if they apply
            basic = [a for a in _STEP_BASIC if a in available]
            pool = basic if basic else available
            return self._rng.choice(pool)

    def _choose_inverse_action(self, node: "TreeNode") -> str:
        available = self._available_inverse(node)
        if self._rng.random() < self.p_uniform:
            return self._rng.choice(available)
        else:
            basic = [a for a in _STEP_BASIC_INV if a in available]
            pool = basic if basic else available
            return self._rng.choice(pool)

    @staticmethod
    def _available_forward(node: "TreeNode") -> list[str]:
        """Which forward actions are non-null from this node."""
        from src.debugger.actions import (
            step_into, step_over, step_return, continue_run
        )
        available: list[str] = []
        if step_into(node) is not None:
            available.append("step_into")
        if step_over(node) is not None:
            available.append("step_over")
        if step_return(node) is not None:
            available.append("step_return")
        available.append("continue")  # always available
        return available

    @staticmethod
    def _available_inverse(node: "TreeNode") -> list[str]:
        """Which inverse actions are non-null from this node."""
        from src.debugger.actions import inv_step_into, inv_step_over, inv_step_call
        available: list[str] = []
        if inv_step_into(node) is not None:
            available.append("inv_step_into")
        if inv_step_over(node) is not None:
            available.append("inv_step_over")
        if inv_step_call(node) is not None:
            available.append("inv_step_call")
        if not available:
            available.append("inv_continue")
        return available
