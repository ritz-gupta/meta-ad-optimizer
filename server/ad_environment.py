"""
Meta Ad Optimizer Environment — core OpenEnv implementation.

Provides reset(), step(), and state() for RL training of an agent
that optimises ad delivery across Instagram and Facebook.
"""

from __future__ import annotations

import random
from typing import Any, Optional
from uuid import uuid4

from openenv.core.env_server.interfaces import Environment
from openenv.core.env_server.types import State

try:
    from ..models import AdAction, AdObservation, AdState
    from ..simulation import (
        compute_engagement,
        generate_master_catalog,
        generate_user,
        sample_creatives,
        transition_surface,
        update_fatigue,
        UserProfile,
    )
    from ..tasks import DEFAULT_TASK, TASKS, TaskConfig, grade_episode
except ImportError:
    from models import AdAction, AdObservation, AdState
    from simulation import (
        compute_engagement,
        generate_master_catalog,
        generate_user,
        sample_creatives,
        transition_surface,
        update_fatigue,
        UserProfile,
    )
    from tasks import DEFAULT_TASK, TASKS, TaskConfig, grade_episode


class AdOptimizerEnvironment(Environment):
    """RL environment for Meta ad optimization.

    Supports three task tiers via ``reset(task=...)``:
      - ``creative_matcher`` (easy)
      - ``placement_optimizer`` (medium)
      - ``campaign_optimizer`` (hard)
    """

    SUPPORTS_CONCURRENT_SESSIONS: bool = True

    def __init__(self, catalog_size: int = 80, catalog_seed: int = 0):
        super().__init__()
        self._catalog = generate_master_catalog(n_creatives=catalog_size, seed=catalog_seed)
        self._state: AdState = AdState(episode_id=str(uuid4()), step_count=0)
        self._task_cfg: TaskConfig = TASKS[DEFAULT_TASK]
        self._user: Optional[UserProfile] = None
        self._creative_pool: list[dict[str, Any]] = []
        self._current_surface: str = "feed"
        self._rng = random.Random()

    # ------------------------------------------------------------------
    # reset
    # ------------------------------------------------------------------

    def reset(
        self,
        seed: Optional[int] = None,
        task: Optional[str] = None,
        **kwargs: Any,
    ) -> AdObservation:
        task_name = task or DEFAULT_TASK
        self._task_cfg = TASKS.get(task_name, TASKS[DEFAULT_TASK])

        self._rng = random.Random(seed)

        self._user = generate_user(self._rng)

        if self._task_cfg.fixed_platform:
            self._user = UserProfile(
                segment=self._user.segment,
                interests=self._user.interests,
                device=self._user.device,
                platform=self._task_cfg.fixed_platform,
                starting_surface=self._task_cfg.fixed_surface or self._user.starting_surface,
            )

        self._creative_pool = sample_creatives(
            self._catalog,
            self._task_cfg.creatives_per_episode,
            self._rng,
        )

        self._current_surface = self._user.starting_surface

        self._state = AdState(
            episode_id=str(uuid4()),
            step_count=0,
            total_impressions_shown=0,
            total_clicks=0,
            total_view_time=0.0,
            cumulative_satisfaction=0.0,
            fatigue_level=0.0,
            task=self._task_cfg.name,
            valid_actions=0,
            invalid_actions=0,
        )

        return self._build_observation(done=False, reward=0.0, last_metrics={})

    # ------------------------------------------------------------------
    # step
    # ------------------------------------------------------------------

    def step(self, action: AdAction) -> AdObservation:  # type: ignore[override]
        if self._user is None:
            raise RuntimeError("Call reset() before step().")

        self._state.step_count += 1
        cfg = self._task_cfg

        done = self._state.step_count >= cfg.steps_per_episode

        # ----- resolve action fields (apply task-level overrides) -----
        show_ad = action.show_ad if cfg.allow_skip else True
        platform = cfg.fixed_platform or action.platform
        placement = cfg.fixed_surface or action.placement
        ad_format = cfg.fixed_format or action.ad_format
        creative_id = max(0, min(action.creative_id, len(self._creative_pool) - 1))

        last_metrics: dict[str, Any] = {}

        if show_ad:
            creative = self._creative_pool[creative_id]

            engagement = compute_engagement(
                user=self._user,
                creative=creative,
                platform=platform,
                placement=placement,
                ad_format=ad_format,
                fatigue_level=self._state.fatigue_level,
                rng=self._rng,
            )

            if engagement["valid_action"]:
                self._state.valid_actions += 1
            else:
                self._state.invalid_actions += 1

            self._state.total_impressions_shown += 1
            if engagement["clicked"]:
                self._state.total_clicks += 1
            self._state.total_view_time += engagement["view_time"]
            self._state.cumulative_satisfaction += max(0.0, engagement["satisfaction"])

            reward = self._compute_reward(engagement)

            last_metrics = {
                "clicked": engagement["clicked"],
                "view_time": engagement["view_time"],
                "effective_ctr": engagement["effective_ctr"],
                "satisfaction": engagement["satisfaction"],
                "valid_action": engagement["valid_action"],
                "creative_id": creative_id,
                "platform": platform,
                "placement": placement,
                "ad_format": ad_format,
            }
        else:
            reward = self._skip_reward()
            last_metrics = {"skipped": True}

        # ----- fatigue update -----
        if cfg.fatigue_enabled:
            self._state.fatigue_level = update_fatigue(
                self._state.fatigue_level,
                show_ad,
                self._state.total_impressions_shown,
                fatigue_increment=cfg.fatigue_increment,
                fatigue_recovery=cfg.fatigue_recovery,
            )

        # ----- surface transition -----
        if cfg.surface_transitions:
            self._current_surface = transition_surface(
                self._current_surface, self._user.platform, self._rng,
            )

        if done:
            last_metrics["episode_score"] = grade_episode(self._state)

        return self._build_observation(done=done, reward=reward, last_metrics=last_metrics)

    # ------------------------------------------------------------------
    # state property
    # ------------------------------------------------------------------

    @property
    def state(self) -> AdState:
        return self._state

    # ------------------------------------------------------------------
    # internals
    # ------------------------------------------------------------------

    def _compute_reward(self, engagement: dict[str, Any]) -> float:
        if not engagement["valid_action"]:
            return -0.2

        click_r = 1.0 if engagement["clicked"] else 0.0
        max_view = 15.0
        view_r = min(1.0, engagement["view_time"] / max_view)
        sat_r = max(0.0, engagement["satisfaction"])
        fatigue_pen = self._state.fatigue_level

        return round(
            0.35 * click_r
            + 0.25 * view_r
            + 0.25 * sat_r
            + 0.15 * (-fatigue_pen),
            5,
        )

    def _skip_reward(self) -> float:
        recovery_bonus = 0.05
        missed_revenue = -0.02
        return round(recovery_bonus + missed_revenue, 5)

    def _build_observation(
        self,
        done: bool,
        reward: float,
        last_metrics: dict[str, Any],
    ) -> AdObservation:
        assert self._user is not None

        impr = self._state.total_impressions_shown
        session_ctr = (
            self._state.total_clicks / impr if impr > 0 else 0.0
        )

        pool_dicts = [
            {
                "pool_index": c["pool_index"],
                "category": c["category"],
                "tone": c["tone"],
                "target_segment": c["target_segment"],
                "base_ctr": c["base_ctr"],
                "base_view_time": c["base_view_time"],
            }
            for c in self._creative_pool
        ]

        return AdObservation(
            task=self._task_cfg.name,
            user_segment=self._user.segment,
            user_interests=self._user.interests,
            user_device=self._user.device,
            current_platform=self._user.platform,
            current_surface=self._current_surface,
            available_creatives=pool_dicts,
            impression_count=self._state.total_impressions_shown,
            fatigue_level=self._state.fatigue_level,
            step=self._state.step_count,
            total_steps=self._task_cfg.steps_per_episode,
            last_action_metrics=last_metrics,
            session_metrics={
                "ctr": round(session_ctr, 5),
                "total_clicks": self._state.total_clicks,
                "total_view_time": round(self._state.total_view_time, 3),
                "cumulative_satisfaction": round(self._state.cumulative_satisfaction, 4),
                "impressions_shown": self._state.total_impressions_shown,
            },
            done=done,
            reward=reward,
        )
