"""
Meta Ad Optimizer Environment Client.

WebSocket-based client for interacting with the AdOptimizer
environment server using the standard EnvClient API.
"""

from __future__ import annotations

from typing import Dict

from openenv.core import EnvClient
from openenv.core.client_types import StepResult

from .models import AdAction, AdObservation, AdState


class AdEnv(EnvClient[AdAction, AdObservation, AdState]):
    """Client for the Meta Ad Optimizer Environment.

    Example (sync):
        >>> with AdEnv(base_url="http://localhost:8000").sync() as env:
        ...     result = env.reset(task="creative_matcher")
        ...     print(result.observation.user_segment)
        ...     result = env.step(AdAction(show_ad=True, creative_id=0))
        ...     print(result.observation.last_action_metrics)

    Example (async):
        >>> async with AdEnv(base_url="http://localhost:8000") as env:
        ...     result = await env.reset(task="campaign_optimizer")
        ...     result = await env.step(AdAction(
        ...         show_ad=True, creative_id=2,
        ...         platform="instagram", placement="reels", ad_format="reel",
        ...     ))
    """

    def _step_payload(self, action: AdAction) -> Dict:
        return {
            "show_ad": action.show_ad,
            "creative_id": action.creative_id,
            "platform": action.platform,
            "placement": action.placement,
            "ad_format": action.ad_format,
        }

    def _parse_result(self, payload: Dict) -> StepResult[AdObservation]:
        obs_data = payload.get("observation", {})
        observation = AdObservation(
            task=obs_data.get("task", ""),
            user_segment=obs_data.get("user_segment", ""),
            user_interests=obs_data.get("user_interests", []),
            user_device=obs_data.get("user_device", "mobile"),
            current_platform=obs_data.get("current_platform", "instagram"),
            current_surface=obs_data.get("current_surface", "feed"),
            available_creatives=obs_data.get("available_creatives", []),
            impression_count=obs_data.get("impression_count", 0),
            fatigue_level=obs_data.get("fatigue_level", 0.0),
            step=obs_data.get("step", 0),
            total_steps=obs_data.get("total_steps", 20),
            last_action_metrics=obs_data.get("last_action_metrics", {}),
            session_metrics=obs_data.get("session_metrics", {}),
            done=payload.get("done", False),
            reward=payload.get("reward"),
            metadata=obs_data.get("metadata", {}),
        )
        return StepResult(
            observation=observation,
            reward=payload.get("reward"),
            done=payload.get("done", False),
        )

    def _parse_state(self, payload: Dict) -> AdState:
        return AdState(
            episode_id=payload.get("episode_id", ""),
            step_count=payload.get("step_count", 0),
            total_impressions_shown=payload.get("total_impressions_shown", 0),
            total_clicks=payload.get("total_clicks", 0),
            total_view_time=payload.get("total_view_time", 0.0),
            cumulative_satisfaction=payload.get("cumulative_satisfaction", 0.0),
            fatigue_level=payload.get("fatigue_level", 0.0),
            task=payload.get("task", ""),
            valid_actions=payload.get("valid_actions", 0),
            invalid_actions=payload.get("invalid_actions", 0),
        )
