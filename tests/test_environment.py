"""
pytest test suite for the Meta Ad Optimizer environment.

Covers:
  - Environment basics (reset, step)
  - Reward signals (invalid action, skip)
  - All 3 tasks (full episode run)
  - Grader (score in [0, 1])
  - Fatigue mechanics (increase, decrease, disabled)
  - Episode termination (correct step count)
  - Gymnasium wrapper (shapes, types, roundtrip)
  - Agent comparison (rule-based > random)

Run:
    cd "c:/Projects/meta hackathon/meta-ad-optimizer"
    python -m pytest tests/ -v
"""

from __future__ import annotations

import random
import statistics

import numpy as np
import pytest

from meta_ad_optimizer.server.ad_environment import AdOptimizerEnvironment
from meta_ad_optimizer.models import AdAction, AdObservation
from meta_ad_optimizer.tasks import grade_episode
from meta_ad_optimizer.simulation import SEGMENT_NAMES
from meta_ad_optimizer.baseline import RandomAgent, RuleBasedAgent, run_evaluation
from meta_ad_optimizer.gym_wrapper import (
    MetaAdEnv,
    obs_to_vector,
    decode_action,
    OBS_DIM,
    PLATFORMS,
    SURFACES,
    FORMATS,
    MAX_CREATIVES,
)

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

ALL_SURFACES = ["feed", "reels", "stories", "explore", "search", "marketplace", "right_column"]
ALL_DEVICES  = ["mobile", "desktop", "tablet"]


def _show_action(platform="instagram", placement="feed", ad_format="image", creative_id=0):
    return AdAction(
        show_ad=True,
        creative_id=creative_id,
        platform=platform,
        placement=placement,
        ad_format=ad_format,
    )


def _skip_action():
    return AdAction(show_ad=False, creative_id=0, platform="instagram", placement="feed", ad_format="image")


def _run_full_episode(task: str, seed: int = 42) -> AdObservation:
    """Run one full episode with the show action and return the terminal observation."""
    env = AdOptimizerEnvironment()
    obs = env.reset(seed=seed, task=task)
    while not obs.done:
        obs = env.step(_show_action(
            platform=obs.current_platform,
            placement=obs.current_surface,
            ad_format="image",
        ))
    return obs


# ===========================================================================
# Group 1: Environment Basics
# ===========================================================================

class TestEnvironmentBasics:
    def test_reset_returns_valid_observation(self):
        env = AdOptimizerEnvironment()
        obs = env.reset(seed=0, task="campaign_optimizer")

        assert isinstance(obs, AdObservation)
        assert obs.step == 0
        assert obs.done is False
        assert obs.reward == 0.0
        assert 0.0 <= obs.fatigue_level <= 1.0
        assert obs.user_segment in SEGMENT_NAMES
        assert obs.user_device in ALL_DEVICES
        assert obs.current_platform in ["instagram", "facebook"]
        assert obs.current_surface in ALL_SURFACES
        assert isinstance(obs.available_creatives, list)
        assert len(obs.available_creatives) == 12  # campaign_optimizer

    def test_step_returns_observation_and_reward(self):
        env = AdOptimizerEnvironment()
        obs = env.reset(seed=1, task="campaign_optimizer")
        action = _show_action(
            platform=obs.current_platform,
            placement=obs.current_surface,
        )
        obs2 = env.step(action)

        assert isinstance(obs2, AdObservation)
        assert obs2.reward is not None
        assert isinstance(obs2.reward, float)
        assert obs2.step == 1

    def test_step_before_reset_raises(self):
        env = AdOptimizerEnvironment()
        with pytest.raises(RuntimeError, match="reset"):
            env.step(_show_action())


# ===========================================================================
# Group 2: Reward Signals
# ===========================================================================

class TestRewardSignals:
    def test_invalid_format_surface_gives_negative_reward(self):
        """instagram + right_column is invalid — right_column is FB only."""
        env = AdOptimizerEnvironment()
        env.reset(seed=0, task="campaign_optimizer")
        action = AdAction(
            show_ad=True,
            creative_id=0,
            platform="instagram",
            placement="right_column",
            ad_format="image",
        )
        obs = env.step(action)
        assert obs.reward == pytest.approx(-0.2)
        assert obs.last_action_metrics.get("valid_action") is False

    def test_skip_action_gives_positive_reward(self):
        env = AdOptimizerEnvironment()
        env.reset(seed=0, task="campaign_optimizer")
        obs = env.step(_skip_action())
        assert obs.reward == pytest.approx(0.03)
        assert obs.last_action_metrics.get("skipped") is True

    def test_valid_action_reward_in_range(self):
        env = AdOptimizerEnvironment()
        obs = env.reset(seed=0, task="campaign_optimizer")
        obs2 = env.step(_show_action(
            platform=obs.current_platform,
            placement=obs.current_surface,
        ))
        # Valid action: reward is in [-0.15, 1.0] (fatigue penalty can make it slightly negative)
        assert -0.2 < obs2.reward <= 1.0


# ===========================================================================
# Group 3: All 3 Tasks — Full Episode
# ===========================================================================

@pytest.mark.parametrize("task,expected_steps,expected_creatives", [
    ("creative_matcher",    10, 4),
    ("placement_optimizer", 15, 8),
    ("campaign_optimizer",  20, 12),
])
class TestAllTasksFullEpisode:
    def test_task_config(self, task, expected_steps, expected_creatives):
        env = AdOptimizerEnvironment()
        obs = env.reset(seed=42, task=task)
        assert obs.total_steps == expected_steps
        assert len(obs.available_creatives) == expected_creatives
        assert obs.step == 0
        assert obs.done is False

    def test_full_episode_terminates(self, task, expected_steps, expected_creatives):
        obs = _run_full_episode(task)
        assert obs.done is True
        assert obs.step == expected_steps


# ===========================================================================
# Group 4: Grader
# ===========================================================================

class TestGrader:
    @pytest.mark.parametrize("task", [
        "creative_matcher", "placement_optimizer", "campaign_optimizer"
    ])
    def test_grader_score_in_0_1_range(self, task):
        obs = _run_full_episode(task)
        env = AdOptimizerEnvironment()
        # Re-run to get state accessible
        obs2 = env.reset(seed=42, task=task)
        while not obs2.done:
            obs2 = env.step(_show_action(
                platform=obs2.current_platform,
                placement=obs2.current_surface,
            ))
        score = grade_episode(env.state)
        assert 0.0 <= score <= 1.0, f"{task} score {score} out of range"

    def test_grader_score_included_in_terminal_obs(self):
        env = AdOptimizerEnvironment()
        obs = env.reset(seed=42, task="campaign_optimizer")
        while not obs.done:
            obs = env.step(_show_action(
                platform=obs.current_platform,
                placement=obs.current_surface,
            ))
        assert "episode_score" in obs.last_action_metrics
        score = obs.last_action_metrics["episode_score"]
        assert 0.0 <= score <= 1.0


# ===========================================================================
# Group 5: Fatigue Mechanics
# ===========================================================================

class TestFatigueMechanics:
    def test_fatigue_increases_on_show_ad(self):
        env = AdOptimizerEnvironment()
        obs = env.reset(seed=42, task="campaign_optimizer")
        initial = obs.fatigue_level
        obs2 = env.step(_show_action(
            platform=obs.current_platform,
            placement=obs.current_surface,
        ))
        assert obs2.fatigue_level > initial

    def test_fatigue_decreases_on_skip(self):
        env = AdOptimizerEnvironment()
        obs = env.reset(seed=42, task="campaign_optimizer")
        # Build up fatigue with 5 show steps
        for _ in range(5):
            obs = env.step(_show_action(
                platform=obs.current_platform,
                placement=obs.current_surface,
            ))
        fatigue_before = obs.fatigue_level
        assert fatigue_before > 0.0
        obs2 = env.step(_skip_action())
        assert obs2.fatigue_level < fatigue_before

    def test_fatigue_stays_zero_in_creative_matcher(self):
        """creative_matcher has fatigue_enabled=False."""
        env = AdOptimizerEnvironment()
        obs = env.reset(seed=42, task="creative_matcher")
        obs2 = env.step(_show_action(platform="instagram", placement="feed"))
        assert obs2.fatigue_level == 0.0


# ===========================================================================
# Group 6: Episode Termination
# ===========================================================================

@pytest.mark.parametrize("task,expected_steps", [
    ("creative_matcher",    10),
    ("placement_optimizer", 15),
    ("campaign_optimizer",  20),
])
def test_episode_terminates_at_correct_step_count(task, expected_steps):
    env = AdOptimizerEnvironment()
    obs = env.reset(seed=0, task=task)
    step_count = 0
    while not obs.done:
        obs = env.step(_skip_action())
        step_count += 1
    assert step_count == expected_steps
    assert obs.step == expected_steps
    assert obs.done is True


# ===========================================================================
# Group 7: Gymnasium Wrapper
# ===========================================================================

class TestGymnasiumWrapper:
    def test_reset_returns_correct_shape_and_dtype(self):
        env = MetaAdEnv(task="campaign_optimizer", seed=0)
        obs_vec, info = env.reset()
        assert obs_vec.shape == (OBS_DIM,)
        assert obs_vec.dtype == np.float32

    def test_obs_in_observation_space(self):
        env = MetaAdEnv(task="campaign_optimizer", seed=0)
        obs_vec, _ = env.reset()
        assert env.observation_space.contains(obs_vec)

    def test_obs_values_in_unit_range(self):
        env = MetaAdEnv(task="campaign_optimizer", seed=0)
        obs_vec, _ = env.reset()
        assert float(obs_vec.min()) >= 0.0
        assert float(obs_vec.max()) <= 1.0

    def test_step_returns_correct_5_tuple(self):
        env = MetaAdEnv(task="campaign_optimizer", seed=0)
        env.reset()
        action = env.action_space.sample()
        result = env.step(action)
        assert len(result) == 5
        obs_vec, reward, terminated, truncated, info = result
        assert obs_vec.shape == (OBS_DIM,)
        assert obs_vec.dtype == np.float32
        assert isinstance(reward, float)
        assert isinstance(terminated, bool)
        assert isinstance(truncated, bool)
        assert truncated is False

    def test_decode_action_roundtrip(self):
        env = MetaAdEnv(task="campaign_optimizer", seed=0)
        for _ in range(20):
            action = env.action_space.sample()
            ad_action = decode_action(action)
            assert ad_action.platform in PLATFORMS
            assert ad_action.placement in SURFACES
            assert ad_action.ad_format in FORMATS
            assert 0 <= ad_action.creative_id < MAX_CREATIVES

    def test_full_episode_terminates(self):
        env = MetaAdEnv(task="creative_matcher", seed=42)
        obs, _ = env.reset()
        done = False
        steps = 0
        while not done:
            obs, reward, terminated, truncated, info = env.step(env.action_space.sample())
            done = terminated
            steps += 1
        assert steps == 10

    def test_ad_env_property_accessible(self):
        env = MetaAdEnv(task="campaign_optimizer", seed=0)
        env.reset()
        assert env.ad_env is not None

    def test_all_tasks_work_in_wrapper(self):
        for task, expected_steps in [
            ("creative_matcher",    10),
            ("placement_optimizer", 15),
            ("campaign_optimizer",  20),
        ]:
            env = MetaAdEnv(task=task, seed=0)
            obs, _ = env.reset()
            assert obs.shape == (OBS_DIM,)
            done = False
            count = 0
            while not done:
                _, _, done, _, _ = env.step(env.action_space.sample())
                count += 1
            assert count == expected_steps


# ===========================================================================
# Group 8: Agent Comparison
# ===========================================================================

class TestAgentComparison:
    def test_rule_based_outscores_random_over_20_episodes(self):
        random_scores = run_evaluation(
            "campaign_optimizer", RandomAgent(random.Random(42)), 20, 42
        )
        rule_scores = run_evaluation(
            "campaign_optimizer", RuleBasedAgent(), 20, 42
        )
        assert statistics.mean(rule_scores) > statistics.mean(random_scores)
