"""
Rubric for trajectory-level scoring of the Meta Ad Optimizer.

Uses exponential discounting to assign temporal credit across
steps within an episode, with the final score coming from the
task-specific grader in tasks.py.
"""

from __future__ import annotations

from typing import Any, List, Tuple

try:
    from openenv.core.rubrics.trajectory import ExponentialDiscountingTrajectoryRubric
except ModuleNotFoundError:

    class ExponentialDiscountingTrajectoryRubric:  # type: ignore[no-redef]
        """Compatibility fallback when rubrics module is absent."""

        def __init__(self, gamma: float = 0.99, intermediate_reward: float = 0.0):
            self.gamma = gamma
            self.intermediate_reward = intermediate_reward
            self._trajectory: List[Tuple[Any, Any]] = []

        def __call__(self, action: Any, observation: Any) -> float:
            self._trajectory.append((action, observation))
            if getattr(observation, "done", False):
                return self.score_trajectory(self._trajectory)
            return self.intermediate_reward

        def reset(self) -> None:
            self._trajectory = []

        def compute_step_rewards(self) -> List[float]:
            if not self._trajectory:
                return []
            final_score = self.score_trajectory(self._trajectory)
            n = len(self._trajectory)
            return [
                self.gamma ** (n - 1 - i) * final_score
                for i in range(n)
            ]


class AdOptimizerRubric(ExponentialDiscountingTrajectoryRubric):
    """Score an ad-optimizer episode using the task grader.

    The per-step discounted reward is:
        r_t = gamma^(T-1-t) * final_score

    where *final_score* is the 0.0–1.0 value produced by
    ``tasks.grade_episode(state)`` at the terminal observation.
    """

    def score_trajectory(self, trajectory: List[Tuple[Any, Any]]) -> float:
        if not trajectory:
            return 0.0
        _, final_obs = trajectory[-1]
        metrics = getattr(final_obs, "last_action_metrics", {})
        if isinstance(metrics, dict):
            return metrics.get("episode_score", 0.0)
        return 0.0
