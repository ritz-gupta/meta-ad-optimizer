"""
Composable rubrics for AdMarket Arena (Round 2).

Three rubrics for the *advertiser* training loop are stubbed here for
Plan 1 to fill in (PerStepEngagementRubric, DailyPacingRubric,
WeeklyROASRubric); the fourth — OversightF1Rubric — is fully
implemented because Plan 2 owns the entire oversight stack.

Each rubric implements `score(state, **kwargs) -> float` and can be
composed by `arena_env.py` as the env reward layer. This is the
"composable rubrics > monolithic scoring" pattern called out by the
hackathon judging criteria.
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional, Protocol

try:
    from ..models import GroundTruthViolation, ViolationFlag
    from ..oversight import score_flags
except ImportError:
    from models import GroundTruthViolation, ViolationFlag  # type: ignore
    from oversight import score_flags  # type: ignore


# ---------------------------------------------------------------------------
# Rubric protocol
# ---------------------------------------------------------------------------

class ArenaRubric(Protocol):
    """All arena rubrics expose `score(...) -> float` and a `name` attr.

    Each rubric is independent, testable in isolation, and can be
    enabled/disabled per `reset(enabled_rubrics=...)` for ablations.
    """

    name: str

    def score(self, *args: Any, **kwargs: Any) -> float:
        ...


# ---------------------------------------------------------------------------
# Plan 1 stubs — to be filled by P1 owner
# ---------------------------------------------------------------------------

class PerStepEngagementRubric:
    """STUB (Plan 1 fills): per-step auction outcome reward.

    Plan 1 implementation should score the auction outcome:
      - won + clicked: +1.0 * (revenue_value - clearing_price)
      - won + no click: -clearing_price * 0.1
      - skipped: +0.02
      - invalid bid (over budget): -0.5
    """

    name = "per_step_engagement"

    def score(self, *args: Any, **kwargs: Any) -> float:
        raise NotImplementedError("Plan 1 owns the implementation.")


class DailyPacingRubric:
    """STUB (Plan 1 fills): daily ROAS + pacing alignment bonus."""

    name = "daily_pacing"

    def score(self, *args: Any, **kwargs: Any) -> float:
        raise NotImplementedError("Plan 1 owns the implementation.")


class WeeklyROASRubric:
    """STUB (Plan 1 fills): sparse weekly ROAS bonus + over/underspend penalties."""

    name = "weekly_roas"

    def score(self, *args: Any, **kwargs: Any) -> float:
        raise NotImplementedError("Plan 1 owns the implementation.")


# ---------------------------------------------------------------------------
# Plan 2 — Oversight F1 reward (full implementation)
# ---------------------------------------------------------------------------

class OversightF1Rubric:
    """Reward signal for OversightAgent training (Fleet AI bonus).

    Fires at the end of each day with the daily F1 component, and at
    the end of the week with a weekly F1 bonus. False positives carry
    a small fixed penalty so the rubric cannot be gamed by flagging
    everyone (precision matters as much as recall).

    Design (master Section 12.4):

        daily F1     -> +1.0 * f1_today
        weekly F1    -> +3.0 * f1_week  (sparse, fires at episode end)
        FP penalty   -> -0.5 per false positive

    Why these coefficients:
      - 3.0 weekly bonus dominates the ~7 daily F1 signals so that
        end-of-week credit assignment is non-trivial (the agent must
        reason across the week, not just maximize day 7).
      - 0.5 FP penalty is roughly equal to the marginal value of one
        true positive at f1=0.5, which means the rubric is indifferent
        between flagging a borderline case and skipping it — neutral,
        not biased toward false caution.
    """

    name = "oversight_f1"

    def __init__(
        self,
        daily_weight: float = 1.0,
        weekly_weight: float = 3.0,
        fp_penalty: float = 0.5,
    ) -> None:
        self.daily_weight = daily_weight
        self.weekly_weight = weekly_weight
        self.fp_penalty = fp_penalty

    def score_day(
        self,
        predicted: List[ViolationFlag],
        ground_truth: List[GroundTruthViolation],
    ) -> Dict[str, float]:
        f1 = score_flags(predicted, ground_truth)
        reward = self.daily_weight * f1.f1 - self.fp_penalty * f1.false_positives
        return {
            "reward": reward,
            "f1": f1.f1,
            "precision": f1.precision,
            "recall": f1.recall,
            "true_positives": f1.true_positives,
            "false_positives": f1.false_positives,
            "false_negatives": f1.false_negatives,
        }

    def score_week(
        self,
        predicted_all: List[ViolationFlag],
        ground_truth_all: List[GroundTruthViolation],
    ) -> Dict[str, float]:
        f1 = score_flags(predicted_all, ground_truth_all)
        reward = self.weekly_weight * f1.f1 - self.fp_penalty * f1.false_positives
        return {
            "reward": reward,
            "f1": f1.f1,
            "precision": f1.precision,
            "recall": f1.recall,
            "true_positives": f1.true_positives,
            "false_positives": f1.false_positives,
            "false_negatives": f1.false_negatives,
        }

    # Generic Protocol-compatible entry point. Decides between day or
    # week scoring based on whether `kind="week"` is passed.
    def score(
        self,
        predicted: List[ViolationFlag],
        ground_truth: List[GroundTruthViolation],
        kind: str = "day",
    ) -> float:
        if kind == "week":
            return self.score_week(predicted, ground_truth)["reward"]
        return self.score_day(predicted, ground_truth)["reward"]


# ---------------------------------------------------------------------------
# Composer
# ---------------------------------------------------------------------------

def build_arena_rubrics(
    enabled: Optional[List[str]] = None,
) -> Dict[str, ArenaRubric]:
    """Factory used by `arena_env.py`. Returns a dict keyed by rubric
    name. If `enabled` is None, returns all four; otherwise filters to
    the named subset (used for ablation experiments).
    """
    all_rubrics: Dict[str, ArenaRubric] = {
        "per_step_engagement": PerStepEngagementRubric(),
        "daily_pacing": DailyPacingRubric(),
        "weekly_roas": WeeklyROASRubric(),
        "oversight_f1": OversightF1Rubric(),
    }
    if enabled is None:
        return all_rubrics
    return {name: rubric for name, rubric in all_rubrics.items() if name in enabled}
