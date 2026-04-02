"""
Gymnasium-compatible wrapper for the Meta Ad Optimizer environment.

Converts AdOptimizerEnvironment into a standard gymnasium.Env with:
  - Flat float32 observation vector (452 dims)
  - MultiDiscrete action space [show_ad, creative_id, platform, placement, ad_format]

Compatible with Stable Baselines3, RLlib, and CleanRL.

Usage:
    from meta_ad_optimizer.gym_wrapper import MetaAdEnv
    env = MetaAdEnv(task="campaign_optimizer")
    obs, info = env.reset()
    obs, reward, terminated, truncated, info = env.step(env.action_space.sample())
"""

from __future__ import annotations

from typing import Any, Dict, Optional, Tuple

import numpy as np
import gymnasium
from gymnasium import spaces

# ---------------------------------------------------------------------------
# Vocabulary constants (exported for use in tests and external code)
# ---------------------------------------------------------------------------

SEGMENTS = [
    "gen_z_creator",
    "millennial_parent",
    "business_pro",
    "casual_scroller",
    "fitness_enthusiast",
    "bargain_hunter",
]

DEVICES = ["mobile", "desktop", "tablet"]

PLATFORMS = ["instagram", "facebook"]

SURFACES = [
    "feed",
    "reels",
    "stories",
    "explore",
    "search",
    "marketplace",
    "right_column",
]

FORMATS = ["image", "video", "carousel", "reel", "collection"]

TONES = ["humorous", "informational", "emotional", "action_oriented"]

# 22 creative categories — CREATIVE_CATEGORIES insertion order from simulation.py
CATEGORIES = [
    "fashion", "music", "beauty", "kids", "home_decor", "deals",
    "saas", "finance", "productivity", "entertainment", "food",
    "travel", "health", "sports", "outdoors", "supplements",
    "electronics", "secondhand", "memes", "family", "coupons", "b2b",
]

# 22 interest categories sorted alphabetically (for multi-hot encoding)
INTERESTS = sorted(CATEGORIES)

OBS_DIM = 452       # 4 + 6 + 3 + 2 + 7 + 22 + 12 * 34
MAX_CREATIVES = 12  # campaign_optimizer pool size

# Pre-build index lookups for speed
_SEG_IDX  = {s: i for i, s in enumerate(SEGMENTS)}
_DEV_IDX  = {d: i for i, d in enumerate(DEVICES)}
_PLT_IDX  = {p: i for i, p in enumerate(PLATFORMS)}
_SRF_IDX  = {s: i for i, s in enumerate(SURFACES)}
_INT_IDX  = {v: i for i, v in enumerate(INTERESTS)}
_TON_IDX  = {t: i for i, t in enumerate(TONES)}
_CAT_IDX  = {c: i for i, c in enumerate(CATEGORIES)}


# ---------------------------------------------------------------------------
# Observation encoder
# ---------------------------------------------------------------------------

def obs_to_vector(obs: Any) -> np.ndarray:
    """Convert an AdObservation to a flat float32 numpy array of shape (452,).

    Layout:
      [0:4]   scalars: fatigue, impression_count/total_steps, step/total_steps, total_steps/20
      [4:10]  user_segment one-hot (6)
      [10:13] user_device one-hot (3)
      [13:15] current_platform one-hot (2)
      [15:22] current_surface one-hot (7)
      [22:44] user_interests multi-hot (22, alphabetical)
      [44:452] 12 creative slots × 34 values each
    """
    vec = np.zeros(OBS_DIM, dtype=np.float32)

    # Block A — scalars
    total = max(float(obs.total_steps), 1.0)
    vec[0] = float(obs.fatigue_level)
    vec[1] = float(obs.impression_count) / total
    vec[2] = float(obs.step) / total
    vec[3] = float(obs.total_steps) / 20.0

    # Block B — user context one-hots
    offset = 4
    _set_onehot(vec, offset, _SEG_IDX, obs.user_segment);      offset += len(SEGMENTS)
    _set_onehot(vec, offset, _DEV_IDX, obs.user_device);        offset += len(DEVICES)
    _set_onehot(vec, offset, _PLT_IDX, obs.current_platform);   offset += len(PLATFORMS)
    _set_onehot(vec, offset, _SRF_IDX, obs.current_surface);    offset += len(SURFACES)

    # Block C — user interests multi-hot
    for interest in obs.user_interests:
        idx = _INT_IDX.get(interest)
        if idx is not None:
            vec[offset + idx] = 1.0
    offset += len(INTERESTS)

    # Block D — creative pool (12 slots × 34 values)
    for i, creative in enumerate(obs.available_creatives[:MAX_CREATIVES]):
        base = offset + i * 34
        vec[base + 0] = float(creative.get("base_ctr", 0.0))
        vec[base + 1] = min(1.0, float(creative.get("base_view_time", 0.0)) / 8.0)
        _set_onehot(vec, base + 2,  _TON_IDX, creative.get("tone", ""))
        _set_onehot(vec, base + 6,  _CAT_IDX, creative.get("category", ""))
        _set_onehot(vec, base + 28, _SEG_IDX, creative.get("target_segment", ""))

    return vec


def _set_onehot(vec: np.ndarray, offset: int, index: dict, value: str) -> None:
    idx = index.get(value)
    if idx is not None:
        vec[offset + idx] = 1.0


# ---------------------------------------------------------------------------
# Action decoder
# ---------------------------------------------------------------------------

def decode_action(action: np.ndarray) -> Any:
    """Map a MultiDiscrete action array [show_ad, creative_id, platform, placement, ad_format]
    back to an AdAction instance.

    Args:
        action: integer array of shape (5,) from MultiDiscrete([2, 12, 2, 7, 5])

    Returns:
        AdAction with decoded fields
    """
    from meta_ad_optimizer.models import AdAction

    return AdAction(
        show_ad=bool(int(action[0])),
        creative_id=int(action[1]),
        platform=PLATFORMS[int(action[2])],
        placement=SURFACES[int(action[3])],
        ad_format=FORMATS[int(action[4])],
    )


# ---------------------------------------------------------------------------
# Gymnasium environment
# ---------------------------------------------------------------------------

class MetaAdEnv(gymnasium.Env):
    """Gymnasium wrapper around AdOptimizerEnvironment.

    Supports all three tasks:
      - "creative_matcher"      (easy,   10 steps, 4 creatives)
      - "placement_optimizer"   (medium, 15 steps, 8 creatives)
      - "campaign_optimizer"    (hard,   20 steps, 12 creatives)

    Observation: flat float32 vector of shape (452,), all values in [0, 1].
    Action:      MultiDiscrete([2, 12, 2, 7, 5])
                   dim 0 — show_ad:    0=skip, 1=show
                   dim 1 — creative_id: 0..11
                   dim 2 — platform:   0=instagram, 1=facebook
                   dim 3 — placement:  0=feed, 1=reels, 2=stories, 3=explore,
                                       4=search, 5=marketplace, 6=right_column
                   dim 4 — ad_format:  0=image, 1=video, 2=carousel, 3=reel, 4=collection

    The action space is always campaign_optimizer-sized regardless of task.
    The environment clamps creative_id to the active pool size internally.

    Compatible with Stable Baselines3, RLlib, and CleanRL.

    Example:
        env = MetaAdEnv(task="campaign_optimizer")
        obs, info = env.reset(seed=42)
        obs, reward, terminated, truncated, info = env.step(env.action_space.sample())
    """

    metadata = {"render_modes": []}

    def __init__(
        self,
        task: str = "campaign_optimizer",
        seed: Optional[int] = None,
    ):
        super().__init__()
        self._task = task
        self._ep_seed = seed if seed is not None else 0

        from meta_ad_optimizer.server.ad_environment import AdOptimizerEnvironment
        self._env = AdOptimizerEnvironment()

        self.observation_space = spaces.Box(
            low=0.0, high=1.0, shape=(OBS_DIM,), dtype=np.float32
        )
        # [show_ad(2), creative_id(12), platform(2), placement(7), ad_format(5)]
        self.action_space = spaces.MultiDiscrete([2, MAX_CREATIVES, 2, len(SURFACES), len(FORMATS)])

    # ------------------------------------------------------------------
    # gymnasium.Env interface
    # ------------------------------------------------------------------

    def reset(
        self,
        *,
        seed: Optional[int] = None,
        options: Optional[Dict[str, Any]] = None,
    ) -> Tuple[np.ndarray, Dict[str, Any]]:
        super().reset(seed=seed)
        ep_seed = seed if seed is not None else self._ep_seed
        self._ep_seed = ep_seed + 1  # auto-increment for sequential episodes
        obs = self._env.reset(seed=ep_seed, task=self._task)
        return obs_to_vector(obs), {}

    def step(
        self, action: np.ndarray
    ) -> Tuple[np.ndarray, float, bool, bool, Dict[str, Any]]:
        ad_action = decode_action(action)
        obs = self._env.step(ad_action)
        terminated = bool(obs.done)
        truncated = False
        info = dict(obs.last_action_metrics) if obs.last_action_metrics else {}
        return obs_to_vector(obs), float(obs.reward or 0.0), terminated, truncated, info

    def render(self) -> None:
        pass

    # ------------------------------------------------------------------
    # Extras
    # ------------------------------------------------------------------

    @property
    def ad_env(self):
        """Direct access to the underlying AdOptimizerEnvironment.

        Useful for calling grade_episode(env.ad_env.state) after an episode.
        """
        return self._env
