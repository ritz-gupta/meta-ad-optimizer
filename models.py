"""
Data models for the Meta Ad Optimizer Environment.

Defines typed Action, Observation, and State for both:
  - The single-agent ad optimizer (Round 1: AdAction / AdObservation / AdState)
  - The multi-agent AdMarket Arena (Round 2: AuctionAction / AuctionObservation,
    plus OversightAction / OversightObservation for the oversight agent).
"""

from __future__ import annotations

from typing import Any, Dict, List, Literal, Optional

from openenv.core.env_server.types import Action, Observation, State
from pydantic import BaseModel, Field


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


# ---------------------------------------------------------------------------
# AdMarket Arena (Round 2) — multi-agent auction schemas
# ---------------------------------------------------------------------------

VIOLATION_TYPES = ("frequency_cap", "budget_overspend", "shill_bidding")
ViolationType = Literal["frequency_cap", "budget_overspend", "shill_bidding"]


class AuctionAction(Action):
    """Advertiser bid for a single impression opportunity in the arena."""

    skip: bool = Field(default=False, description="True = no bid, save budget")
    bid_amount: float = Field(default=0.0, ge=0.0, description="CPM-like bid value")
    creative_id: int = Field(default=0, description="Index into available_creatives if won")


class AuctionObservation(Observation):
    """What an advertiser sees at each auction step."""

    user_segment: str = Field(default="")
    user_interests: List[str] = Field(default_factory=list)
    current_surface: str = Field(default="feed")
    day_of_week: int = Field(default=0, ge=0, le=6)
    step_in_day: int = Field(default=0)
    weekly_budget_remaining: float = Field(default=0.0)
    daily_budget_remaining: float = Field(default=0.0)
    spent_so_far_today: float = Field(default=0.0)
    spent_so_far_week: float = Field(default=0.0)
    clicks_today: int = Field(default=0)
    clicks_week: int = Field(default=0)
    per_segment_fatigue: Dict[str, float] = Field(default_factory=dict)
    available_creatives: List[Dict[str, Any]] = Field(default_factory=list)
    recent_clearing_prices: List[float] = Field(default_factory=list)
    yesterday_recap: str = Field(default="")
    target_weekly_roas: float = Field(default=2.0)
    floor_price: float = Field(default=0.0)
    frequency_cap_per_user: int = Field(default=999)


class AuctionResult(BaseModel):
    """Outcome of a single second-price auction."""

    winner_id: Optional[int] = None
    clearing_price: float = 0.0
    no_contest: bool = False
    rejected_below_floor: List[int] = Field(default_factory=list)
    rejected_freq_cap: List[int] = Field(default_factory=list)


# ---------------------------------------------------------------------------
# Oversight Agent (Fleet AI bonus) — schemas
# ---------------------------------------------------------------------------

class AuctionRecord(BaseModel):
    """Single auction outcome as exposed to the OversightAgent.

    Redacted view: enough info to detect violations from auction-log analysis,
    no privileged ground-truth fields.
    """

    step: int
    day: int
    step_in_day: int
    user_id: str = Field(description="Synthetic per-user ID for tracking impressions")
    user_segment: str
    advertiser_id: int = Field(description="Bidder identity")
    bid: float
    won: bool
    clearing_price: float = Field(description="Price the winner paid (0 if no winner)")
    floor_price: float
    no_contest: bool = Field(description="True if all bids were below floor")
    creative_id: Optional[int] = None


class CampaignStateSummary(BaseModel):
    """Per-advertiser snapshot at a day boundary, exposed to OversightAgent."""

    advertiser_id: int
    advertiser_name: str = ""
    spent_today: float
    daily_budget_cap: float
    spent_total: float
    weekly_budget_cap: float
    impressions_today: int
    clicks_today: int


class ViolationFlag(BaseModel):
    """One predicted violation emitted by the OversightAgent."""

    advertiser_id: int
    violation_type: ViolationType
    confidence: float = Field(default=0.5, ge=0.01, le=0.99)
    evidence_step_ids: List[int] = Field(default_factory=list)


class OversightObservation(Observation):
    """What the OversightAgent sees at each day boundary."""

    day: int = Field(default=0)
    auction_log: List[AuctionRecord] = Field(default_factory=list)
    campaign_states: List[CampaignStateSummary] = Field(default_factory=list)
    floor_price: float = Field(default=0.0)
    frequency_cap_per_user: int = Field(default=999)
    advertiser_names: Dict[int, str] = Field(default_factory=dict)


class OversightAction(Action):
    """OversightAgent emits a list of violation flags per day boundary."""

    flags: List[ViolationFlag] = Field(default_factory=list)


class GroundTruthViolation(BaseModel):
    """Ground-truth violation tracked in env state for F1 scoring.

    Never exposed in any observation. Only the env's reward/scoring layer
    sees this.
    """

    advertiser_id: int
    violation_type: ViolationType
    day: int
    step_ids: List[int] = Field(default_factory=list)
    note: str = Field(default="", description="Optional human-readable detail")
