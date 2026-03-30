"""
Simulation engine for the Meta Ad Optimizer Environment.

Contains user segment definitions, creative catalog generation,
CTR/view-time computation, fatigue modelling, and surface transitions
for Instagram and Facebook ad surfaces.
"""

from __future__ import annotations

import math
import random
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple

# ---------------------------------------------------------------------------
# Platform / surface / format validity tables
# ---------------------------------------------------------------------------

VALID_SURFACES: Dict[str, List[str]] = {
    "instagram": ["feed", "reels", "stories", "explore", "search"],
    "facebook": ["feed", "reels", "stories", "marketplace", "search", "right_column"],
}

VALID_FORMATS: Dict[str, List[str]] = {
    "feed": ["image", "video", "carousel", "reel"],
    "reels": ["reel"],
    "stories": ["image", "video", "reel", "collection"],
    "explore": ["image", "video", "carousel", "reel"],
    "search": ["image", "video"],
    "marketplace": ["image", "carousel", "collection"],
    "right_column": ["image"],
}

ALL_FORMATS = ["image", "video", "carousel", "reel", "collection"]

# ---------------------------------------------------------------------------
# Placement CTR factors (surface-level base engagement)
# ---------------------------------------------------------------------------

PLACEMENT_CTR_FACTORS: Dict[str, Dict[str, float]] = {
    "instagram": {
        "reels": 1.4,
        "stories": 1.3,   # mobile; desktop handled via device modifier
        "feed": 1.0,
        "explore": 1.2,
        "search": 0.9,
    },
    "facebook": {
        "reels": 1.3,
        "stories": 1.1,
        "feed": 1.0,
        "marketplace": 1.5,
        "search": 1.1,
        "right_column": 0.4,
    },
}

# Stories / right_column device modifiers
DEVICE_SURFACE_MODIFIERS: Dict[str, Dict[str, float]] = {
    "stories": {"mobile": 1.0, "desktop": 0.38, "tablet": 0.75},
    "right_column": {"mobile": 0.0, "desktop": 1.0, "tablet": 0.5},
}

# ---------------------------------------------------------------------------
# Format engagement factors
# ---------------------------------------------------------------------------

FORMAT_CTR_FACTOR: Dict[str, float] = {
    "reel": 1.4,
    "video": 1.2,
    "carousel": 1.15,
    "collection": 1.3,
    "image": 0.9,
}

FORMAT_VIEW_TIME_FACTOR: Dict[str, float] = {
    "reel": 3.0,
    "video": 2.5,
    "carousel": 2.0,
    "collection": 2.5,
    "image": 1.0,
}

# ---------------------------------------------------------------------------
# Format–surface synergy bonuses (multiplicative on top of base)
# ---------------------------------------------------------------------------

FORMAT_SURFACE_SYNERGY: Dict[Tuple[str, str], float] = {
    ("reel", "reels"): 1.35,
    ("reel", "stories"): 1.25,
    ("carousel", "feed"): 1.20,
    ("carousel", "explore"): 1.20,
    ("collection", "marketplace"): 1.40,
    ("collection", "stories"): 1.15,
    ("image", "right_column"): 1.10,
}

# ---------------------------------------------------------------------------
# User segments
# ---------------------------------------------------------------------------

USER_SEGMENTS: Dict[str, Dict[str, Any]] = {
    "gen_z_creator": {
        "interests": ["fashion", "music", "memes", "beauty"],
        "device_weights": {"mobile": 0.90, "desktop": 0.05, "tablet": 0.05},
        "platform_bias": {"instagram": 0.80, "facebook": 0.20},
        "preferred_surfaces": {
            "instagram": ["reels", "explore", "stories"],
            "facebook": ["reels", "feed"],
        },
        "base_ctr_modifier": 1.1,
        "base_view_time_modifier": 0.7,
    },
    "millennial_parent": {
        "interests": ["kids", "home_decor", "deals", "family"],
        "device_weights": {"mobile": 0.55, "desktop": 0.35, "tablet": 0.10},
        "platform_bias": {"instagram": 0.40, "facebook": 0.60},
        "preferred_surfaces": {
            "instagram": ["feed", "stories"],
            "facebook": ["feed", "marketplace", "stories"],
        },
        "base_ctr_modifier": 0.9,
        "base_view_time_modifier": 1.3,
    },
    "business_pro": {
        "interests": ["saas", "finance", "productivity", "b2b"],
        "device_weights": {"mobile": 0.35, "desktop": 0.55, "tablet": 0.10},
        "platform_bias": {"instagram": 0.20, "facebook": 0.80},
        "preferred_surfaces": {
            "instagram": ["feed", "search"],
            "facebook": ["feed", "search", "right_column"],
        },
        "base_ctr_modifier": 0.7,
        "base_view_time_modifier": 1.0,
    },
    "casual_scroller": {
        "interests": ["entertainment", "food", "travel", "memes"],
        "device_weights": {"mobile": 0.80, "desktop": 0.10, "tablet": 0.10},
        "platform_bias": {"instagram": 0.60, "facebook": 0.40},
        "preferred_surfaces": {
            "instagram": ["feed", "reels", "explore"],
            "facebook": ["feed", "reels"],
        },
        "base_ctr_modifier": 1.0,
        "base_view_time_modifier": 0.8,
    },
    "fitness_enthusiast": {
        "interests": ["health", "sports", "outdoors", "supplements"],
        "device_weights": {"mobile": 0.85, "desktop": 0.10, "tablet": 0.05},
        "platform_bias": {"instagram": 0.70, "facebook": 0.30},
        "preferred_surfaces": {
            "instagram": ["reels", "stories", "explore"],
            "facebook": ["feed", "reels"],
        },
        "base_ctr_modifier": 1.15,
        "base_view_time_modifier": 1.2,
    },
    "bargain_hunter": {
        "interests": ["deals", "coupons", "electronics", "secondhand"],
        "device_weights": {"mobile": 0.55, "desktop": 0.35, "tablet": 0.10},
        "platform_bias": {"instagram": 0.30, "facebook": 0.70},
        "preferred_surfaces": {
            "instagram": ["feed", "search"],
            "facebook": ["marketplace", "feed", "search"],
        },
        "base_ctr_modifier": 1.05,
        "base_view_time_modifier": 1.1,
    },
}

SEGMENT_NAMES = list(USER_SEGMENTS.keys())

# ---------------------------------------------------------------------------
# Creative catalog
# ---------------------------------------------------------------------------

CREATIVE_CATEGORIES = [
    "fashion", "music", "beauty", "kids", "home_decor", "deals",
    "saas", "finance", "productivity", "entertainment", "food",
    "travel", "health", "sports", "outdoors", "supplements",
    "electronics", "secondhand", "memes", "family", "coupons", "b2b",
]

CREATIVE_TONES = ["humorous", "informational", "emotional", "action_oriented"]

# Mapping from segment to the categories that resonate
SEGMENT_CATEGORY_AFFINITY: Dict[str, List[str]] = {
    "gen_z_creator": ["fashion", "music", "memes", "beauty"],
    "millennial_parent": ["kids", "home_decor", "deals", "family"],
    "business_pro": ["saas", "finance", "productivity", "b2b"],
    "casual_scroller": ["entertainment", "food", "travel", "memes"],
    "fitness_enthusiast": ["health", "sports", "outdoors", "supplements"],
    "bargain_hunter": ["deals", "coupons", "electronics", "secondhand"],
}


def generate_master_catalog(n_creatives: int = 80, seed: int = 0) -> List[Dict[str, Any]]:
    """Build a deterministic master catalog of *n_creatives* ad creatives.

    The catalog is balanced so that every user segment has ~10-15
    well-matched creatives.
    """
    rng = random.Random(seed)
    catalog: List[Dict[str, Any]] = []

    segments = SEGMENT_NAMES
    per_segment = n_creatives // len(segments)
    remainder = n_creatives % len(segments)

    cid = 0
    for idx, seg in enumerate(segments):
        count = per_segment + (1 if idx < remainder else 0)
        affinity_cats = SEGMENT_CATEGORY_AFFINITY[seg]
        for _ in range(count):
            category = rng.choice(affinity_cats)
            tone = rng.choice(CREATIVE_TONES)
            base_ctr = round(rng.uniform(0.10, 0.45), 4)
            base_view_time = round(rng.uniform(2.0, 8.0), 2)
            catalog.append({
                "id": cid,
                "category": category,
                "tone": tone,
                "target_segment": seg,
                "base_ctr": base_ctr,
                "base_view_time": base_view_time,
            })
            cid += 1

    rng.shuffle(catalog)
    for i, c in enumerate(catalog):
        c["id"] = i
    return catalog


def sample_creatives(
    catalog: List[Dict[str, Any]],
    n: int,
    rng: random.Random,
) -> List[Dict[str, Any]]:
    """Sample *n* creatives from the master catalog for one episode."""
    selected = rng.sample(catalog, min(n, len(catalog)))
    pool = []
    for i, c in enumerate(selected):
        entry = dict(c)
        entry["pool_index"] = i
        pool.append(entry)
    return pool


# ---------------------------------------------------------------------------
# User generation
# ---------------------------------------------------------------------------

@dataclass
class UserProfile:
    segment: str
    interests: List[str]
    device: str
    platform: str
    starting_surface: str

    def to_dict(self) -> Dict[str, Any]:
        return {
            "segment": self.segment,
            "interests": self.interests,
            "device": self.device,
            "platform": self.platform,
            "starting_surface": self.starting_surface,
        }


def generate_user(rng: random.Random, segment: str | None = None) -> UserProfile:
    """Create a random user profile, optionally forcing a segment."""
    if segment is None:
        segment = rng.choice(SEGMENT_NAMES)
    seg_data = USER_SEGMENTS[segment]

    devices = list(seg_data["device_weights"].keys())
    weights = list(seg_data["device_weights"].values())
    device = rng.choices(devices, weights=weights, k=1)[0]

    platforms = list(seg_data["platform_bias"].keys())
    p_weights = list(seg_data["platform_bias"].values())
    platform = rng.choices(platforms, weights=p_weights, k=1)[0]

    preferred = seg_data["preferred_surfaces"].get(platform, ["feed"])
    starting_surface = rng.choice(preferred)

    return UserProfile(
        segment=segment,
        interests=list(seg_data["interests"]),
        device=device,
        platform=platform,
        starting_surface=starting_surface,
    )


# ---------------------------------------------------------------------------
# Surface transitions
# ---------------------------------------------------------------------------

# Transition weights: from_surface -> {to_surface: weight}
_SURFACE_TRANSITION_WEIGHTS: Dict[str, Dict[str, float]] = {
    "feed": {"feed": 0.50, "reels": 0.20, "stories": 0.15, "explore": 0.10, "search": 0.05},
    "reels": {"reels": 0.55, "feed": 0.20, "stories": 0.10, "explore": 0.10, "search": 0.05},
    "stories": {"stories": 0.40, "feed": 0.25, "reels": 0.20, "explore": 0.10, "search": 0.05},
    "explore": {"explore": 0.45, "reels": 0.25, "feed": 0.15, "stories": 0.10, "search": 0.05},
    "search": {"search": 0.40, "feed": 0.25, "explore": 0.15, "reels": 0.10, "stories": 0.10},
    "marketplace": {"marketplace": 0.50, "feed": 0.25, "search": 0.15, "stories": 0.10},
    "right_column": {"right_column": 0.30, "feed": 0.40, "search": 0.20, "marketplace": 0.10},
}


def transition_surface(
    current_surface: str,
    platform: str,
    rng: random.Random,
) -> str:
    """Simulate user drifting to another surface."""
    weights_map = _SURFACE_TRANSITION_WEIGHTS.get(current_surface, {"feed": 1.0})
    valid = VALID_SURFACES[platform]

    surfaces: List[str] = []
    weights: List[float] = []
    for s in valid:
        surfaces.append(s)
        weights.append(weights_map.get(s, 0.02))

    return rng.choices(surfaces, weights=weights, k=1)[0]


# ---------------------------------------------------------------------------
# Engagement computation
# ---------------------------------------------------------------------------

def is_valid_action(platform: str, placement: str, ad_format: str) -> bool:
    """Check if the platform/placement/format combination is valid."""
    valid_surfs = VALID_SURFACES.get(platform)
    if valid_surfs is None or placement not in valid_surfs:
        return False
    valid_fmts = VALID_FORMATS.get(placement)
    if valid_fmts is None or ad_format not in valid_fmts:
        return False
    return True


def compute_segment_affinity(
    user_segment: str,
    user_interests: List[str],
    creative: Dict[str, Any],
) -> float:
    """Compute affinity multiplier between a creative and a user."""
    if creative.get("target_segment") == user_segment:
        return 2.0
    if creative.get("category") in user_interests:
        return 1.2
    return 0.3


def compute_engagement(
    user: UserProfile,
    creative: Dict[str, Any],
    platform: str,
    placement: str,
    ad_format: str,
    fatigue_level: float,
    rng: random.Random,
) -> Dict[str, Any]:
    """Simulate user engagement: click, view_time, satisfaction.

    Returns a dict with keys: clicked, view_time, effective_ctr,
    satisfaction, valid_action.
    """
    valid = is_valid_action(platform, placement, ad_format)

    if not valid:
        return {
            "clicked": False,
            "view_time": 0.0,
            "effective_ctr": 0.0,
            "satisfaction": -0.3,
            "valid_action": False,
        }

    base_ctr = creative["base_ctr"]
    base_view = creative["base_view_time"]

    seg_affinity = compute_segment_affinity(user.segment, user.interests, creative)

    plat_factors = PLACEMENT_CTR_FACTORS.get(platform, {})
    placement_factor = plat_factors.get(placement, 1.0)

    device_mod = DEVICE_SURFACE_MODIFIERS.get(placement, {}).get(user.device, 1.0)
    placement_factor *= device_mod

    # Context bonus: ad placed on the surface the user is browsing
    context_bonus = 1.2 if placement == user.starting_surface else 1.0
    # Platform mismatch penalty
    platform_match = 1.0 if platform == user.platform else 0.5

    format_ctr = FORMAT_CTR_FACTOR.get(ad_format, 1.0)
    format_view = FORMAT_VIEW_TIME_FACTOR.get(ad_format, 1.0)

    synergy = FORMAT_SURFACE_SYNERGY.get((ad_format, placement), 1.0)

    seg_data = USER_SEGMENTS.get(user.segment, {})
    seg_ctr_mod = seg_data.get("base_ctr_modifier", 1.0)
    seg_view_mod = seg_data.get("base_view_time_modifier", 1.0)

    fatigue_ctr_decay = max(0.0, 1.0 - fatigue_level)
    fatigue_view_decay = max(0.0, 1.0 - fatigue_level ** 0.5)

    effective_ctr = (
        base_ctr
        * seg_affinity
        * placement_factor
        * context_bonus
        * platform_match
        * format_ctr
        * synergy
        * seg_ctr_mod
        * fatigue_ctr_decay
    )
    effective_ctr = min(effective_ctr, 0.95)

    effective_view = (
        base_view
        * format_view
        * synergy
        * seg_view_mod
        * fatigue_view_decay
    )

    clicked = rng.random() < effective_ctr
    view_time = max(0.0, rng.gauss(effective_view, 0.3 * effective_view))

    relevance = seg_affinity / 1.5  # normalised to 0–1
    intrusiveness = 0.2 if placement == user.starting_surface else 0.5
    satisfaction = relevance * (1.0 - intrusiveness)

    return {
        "clicked": clicked,
        "view_time": round(view_time, 3),
        "effective_ctr": round(effective_ctr, 5),
        "satisfaction": round(satisfaction, 4),
        "valid_action": True,
    }


# ---------------------------------------------------------------------------
# Fatigue model
# ---------------------------------------------------------------------------

def update_fatigue(
    fatigue: float,
    show_ad: bool,
    impression_count: int,
    fatigue_increment: float = 0.06,
    fatigue_recovery: float = 0.04,
) -> float:
    """Update fatigue level after an impression opportunity."""
    if show_ad:
        delta = fatigue_increment * (1.0 + 0.1 * impression_count)
        fatigue = min(1.0, fatigue + delta)
    else:
        fatigue = max(0.0, fatigue - fatigue_recovery)
    return round(fatigue, 5)
