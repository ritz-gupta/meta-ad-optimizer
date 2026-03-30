"""
Data models for the Meta Ad Optimizer Environment.

Defines typed Action, Observation, and State for ad optimization
across Instagram and Facebook surfaces.
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional

from openenv.core.env_server.types import Action, Observation, State
from pydantic import Field


VALID_PLATFORMS = ("instagram", "facebook")
VALID_PLACEMENTS_IG = ("feed", "reels", "stories", "explore", "search")
VALID_PLACEMENTS_FB = ("feed", "reels", "stories", "marketplace", "search", "right_column")
VALID_FORMATS = ("image", "video", "carousel", "reel", "collection")


class AdAction(Action):
    """Agent decision for a single impression opportunity.

    Attributes:
        show_ad: False to skip this slot and let fatigue recover.
        creative_id: Index into the per-episode creative pool.
        platform: Target platform for the ad.
        placement: Target surface within the platform.
        ad_format: Creative format to render.
    """

    show_ad: bool = Field(..., description="Whether to show an ad or skip")
    creative_id: int = Field(default=0, description="Index into available_creatives pool (0 to N-1)")
    platform: str = Field(default="instagram", description="'instagram' | 'facebook'")
    placement: str = Field(default="feed", description="Surface to place the ad on")
    ad_format: str = Field(default="image", description="'image' | 'video' | 'carousel' | 'reel' | 'collection'")


class AdObservation(Observation):
    """What the agent sees at each step.

    Includes user context, available creatives, fatigue state,
    and cumulative session metrics.
    """

    task: str = Field(default="campaign_optimizer", description="Active task name")
    user_segment: str = Field(default="", description="User segment archetype")
    user_interests: List[str] = Field(default_factory=list)
    user_device: str = Field(default="mobile")
    current_platform: str = Field(default="instagram")
    current_surface: str = Field(default="feed")
    available_creatives: List[Dict[str, Any]] = Field(default_factory=list)
    impression_count: int = Field(default=0, description="Ads shown so far")
    fatigue_level: float = Field(default=0.0, ge=0.0, le=1.0)
    step: int = Field(default=0)
    total_steps: int = Field(default=20)
    last_action_metrics: Dict[str, Any] = Field(default_factory=dict)
    session_metrics: Dict[str, Any] = Field(default_factory=dict)


class AdState(State):
    """Internal episode state tracked by the environment."""

    total_impressions_shown: int = 0
    total_clicks: int = 0
    total_view_time: float = 0.0
    cumulative_satisfaction: float = 0.0
    fatigue_level: float = 0.0
    task: str = "campaign_optimizer"
    valid_actions: int = 0
    invalid_actions: int = 0
