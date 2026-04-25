"""
Persona-shell competitors for AdMarket Arena.

Plan 1 was supposed to ship the ``PersonaBot`` + ``LLMPolicyBot`` stubs;
Plan 3 fills both implementations because the same module is consumed by:

  - ``server/arena_env.py`` (when Plan 1 lands) for in-env opponent bidding.
  - ``scripts/advertiser_eval.py`` (Plan 3) for self-play evaluation.
  - ``train_grpo.ipynb`` (Plan 3) for opponent rollouts during training.

Two classes share the same ``bid(observation, state) -> AuctionAction``
contract so the auction loop never knows whether an opponent is a
scripted persona or a frozen LLM checkpoint:

  - ``PersonaBot`` — five named archetypes (PremiumBrand, BargainReseller,
    PerformanceMarketer, SpamFlooder, CautiousChallenger), each with a
    5-trait vector that is jittered per-episode and clipped to
    [0.01, 0.995] to prevent degenerate zero-bid streams while
    preserving recognizable behavior.
  - ``LLMPolicyBot`` — wraps a frozen Unsloth checkpoint behind the
    same interface; used for self-play eval (Section 7.2) and the
    optional Phase 3 self-play training stretch.

A sixth ``OpportunisticArbitrageur`` persona is exposed via
``HELD_OUT_PERSONA`` for the Section 7.2 edge-case eval mode (held-out
generalization test).
"""

from __future__ import annotations

import json
import math
import random
import re
from dataclasses import dataclass, field, replace
from typing import Any, Callable, Dict, List, Optional, Tuple

try:
    from .models import AuctionAction, AuctionObservation
except ImportError:
    from models import AuctionAction, AuctionObservation  # type: ignore


# ---------------------------------------------------------------------------
# Trait vector and named personas
# ---------------------------------------------------------------------------

TRAIT_NAMES: Tuple[str, ...] = (
    "aggression",
    "pacing_strength",
    "segment_focus",
    "fatigue_awareness",
    "price_elasticity",
)

TRAIT_FLOOR = 0.01
TRAIT_CEILING = 0.995


@dataclass(frozen=True)
class PersonaSpec:
    """Immutable definition of a named persona archetype.

    Per-episode jitter is applied symmetrically (uniform in
    [-jitter, +jitter]) and the result is clipped to
    [TRAIT_FLOOR, TRAIT_CEILING].
    """

    name: str
    aggression: float
    pacing_strength: float
    segment_focus: float
    fatigue_awareness: float
    price_elasticity: float
    jitter: float

    def base_vector(self) -> Dict[str, float]:
        return {
            "aggression": self.aggression,
            "pacing_strength": self.pacing_strength,
            "segment_focus": self.segment_focus,
            "fatigue_awareness": self.fatigue_awareness,
            "price_elasticity": self.price_elasticity,
        }


PERSONAS: Dict[str, PersonaSpec] = {
    "PremiumBrand": PersonaSpec(
        name="PremiumBrand",
        aggression=0.85, pacing_strength=0.40, segment_focus=0.80,
        fatigue_awareness=0.60, price_elasticity=0.30, jitter=0.15,
    ),
    "BargainReseller": PersonaSpec(
        name="BargainReseller",
        aggression=0.40, pacing_strength=0.85, segment_focus=0.30,
        fatigue_awareness=0.50, price_elasticity=0.85, jitter=0.15,
    ),
    "PerformanceMarketer": PersonaSpec(
        name="PerformanceMarketer",
        aggression=0.65, pacing_strength=0.65, segment_focus=0.70,
        fatigue_awareness=0.85, price_elasticity=0.55, jitter=0.10,
    ),
    "SpamFlooder": PersonaSpec(
        name="SpamFlooder",
        aggression=0.95, pacing_strength=0.10, segment_focus=0.10,
        fatigue_awareness=0.05, price_elasticity=0.20, jitter=0.10,
    ),
    "CautiousChallenger": PersonaSpec(
        name="CautiousChallenger",
        aggression=0.35, pacing_strength=0.70, segment_focus=0.65,
        fatigue_awareness=0.70, price_elasticity=0.75, jitter=0.20,
    ),
}

# Held-out persona for --eval-mode edge "held-out persona" sub-condition
# (master Section 7.2). Never used during training so we can attribute
# trained-agent performance against it to genuine generalization rather
# than intra-distribution jitter robustness.
HELD_OUT_PERSONA = PersonaSpec(
    name="OpportunisticArbitrageur",
    aggression=0.55, pacing_strength=0.45, segment_focus=0.80,
    fatigue_awareness=0.60, price_elasticity=0.90, jitter=0.10,
)


def _clip_trait(value: float) -> float:
    return max(TRAIT_FLOOR, min(TRAIT_CEILING, value))


def jitter_persona(spec: PersonaSpec, rng: random.Random, jitter_scale: float = 1.0) -> Dict[str, float]:
    """Sample a per-episode trait vector by jittering the base spec.

    `jitter_scale` doubles in --eval-mode edge "extreme jitter"
    sub-condition; default 1.0 uses the spec's declared range.
    Always clipped to [TRAIT_FLOOR, TRAIT_CEILING] post-jitter.
    """
    width = spec.jitter * jitter_scale
    return {
        trait: _clip_trait(getattr(spec, trait) + rng.uniform(-width, width))
        for trait in TRAIT_NAMES
    }


def maxed_persona(spec: PersonaSpec) -> Dict[str, float]:
    """Pin the persona's dominant trait at TRAIT_CEILING and the
    opposing trait at TRAIT_FLOOR. Used by --eval-mode edge "maxed
    personas" sub-condition.

    Mapping (dominant -> opposing):
      - aggression -> fatigue_awareness   (SpamFlooder shape)
      - pacing_strength -> aggression     (BargainReseller shape)
      - segment_focus -> aggression       (cautious targeted shape)
      - fatigue_awareness -> aggression   (Performance shape)
      - price_elasticity -> aggression    (BargainReseller shape)
    """
    base = spec.base_vector()
    dominant_trait = max(base, key=base.get)
    opposing = {
        "aggression": "fatigue_awareness",
        "pacing_strength": "aggression",
        "segment_focus": "aggression",
        "fatigue_awareness": "aggression",
        "price_elasticity": "aggression",
    }[dominant_trait]
    out = dict(base)
    out[dominant_trait] = TRAIT_CEILING
    out[opposing] = TRAIT_FLOOR
    return out


# ---------------------------------------------------------------------------
# PersonaBot — bid formula
# ---------------------------------------------------------------------------

@dataclass
class PersonaBot:
    """Scripted persona-shell competitor.

    Bid formula (deterministic given user, state, trait vector):

        bid = aggression
              * valuation_estimate
              * pacing_factor(spent_today, daily_target, pacing_strength)
              * segment_factor(user, segment_focus)
              * fatigue_factor(per_segment_fatigue, fatigue_awareness)
              * price_factor(recent_clearing_prices, price_elasticity)

    Each factor returns ~1.0 in the 'neutral' case and pulls the bid
    toward 0 (skip) or upward (boost) as the trait/state combination
    suggests. The trained advertiser must learn to recognize each
    persona's bid distribution and exploit it without memorizing
    point values (because of per-episode jitter).
    """

    spec: PersonaSpec
    traits: Dict[str, float] = field(default_factory=dict)
    valuation_anchor: float = 1.0  # scales raw bid magnitude

    def __post_init__(self) -> None:
        if not self.traits:
            self.traits = self.spec.base_vector()

    @classmethod
    def from_persona_name(
        cls,
        persona_name: str,
        rng: Optional[random.Random] = None,
        jitter_enabled: bool = True,
        jitter_scale: float = 1.0,
        valuation_anchor: float = 1.0,
    ) -> "PersonaBot":
        """Convenience: instantiate by name with optional per-episode jitter."""
        if persona_name == HELD_OUT_PERSONA.name:
            spec = HELD_OUT_PERSONA
        else:
            spec = PERSONAS[persona_name]
        if jitter_enabled and rng is not None:
            traits = jitter_persona(spec, rng, jitter_scale=jitter_scale)
        else:
            traits = spec.base_vector()
        return cls(spec=spec, traits=traits, valuation_anchor=valuation_anchor)

    # --- factor helpers ---
    def _pacing_factor(self, spent_today: float, daily_target: float) -> float:
        if daily_target <= 0.0:
            return 1.0
        spent_ratio = spent_today / daily_target
        # Strong pacing trait => steep tapering once over half of the
        # day's nominal budget; weak pacing => roughly flat.
        s = self.traits["pacing_strength"]
        # tapered = max(0, 1 - s * (spent_ratio - 0.5))
        tapered = max(0.0, 1.0 - s * max(0.0, spent_ratio - 0.5))
        return float(tapered)

    def _segment_factor(self, user_segment: str, target_segment: Optional[str]) -> float:
        # If we don't know the persona's "preferred" segment, use the
        # observation-supplied target_segment as a proxy. With strong
        # focus, mismatched segments collapse the bid; with weak focus,
        # nearly indifferent.
        if not target_segment:
            return 1.0
        f = self.traits["segment_focus"]
        match = (user_segment == target_segment)
        if match:
            return 1.0
        # Mismatch: 1 - f maps focus=1 -> 0, focus=0 -> 1
        return float(1.0 - f * 0.85)

    def _fatigue_factor(self, per_segment_fatigue: Dict[str, float], user_segment: str) -> float:
        a = self.traits["fatigue_awareness"]
        fatigue = float(per_segment_fatigue.get(user_segment, 0.0))
        # awareness=1 + fatigue=1 -> bid factor 0.15 (strong avoidance)
        # awareness=0 -> always 1.0 (ignores fatigue)
        return float(max(0.05, 1.0 - a * fatigue * 0.85))

    def _price_factor(self, recent_clearing_prices: List[float]) -> float:
        e = self.traits["price_elasticity"]
        if not recent_clearing_prices:
            return 1.0
        recent = [p for p in recent_clearing_prices if p > 0]
        if not recent:
            return 1.0
        mean_price = sum(recent) / len(recent)
        # If recent mean > our anchor, elastic personas back off; if
        # recent mean is low, they pile in. e=0 means inelastic (always 1).
        ratio = mean_price / max(self.valuation_anchor, 1e-6)
        return float(max(0.10, 1.0 - e * (ratio - 1.0)))

    def _valuation_estimate(self, observation: AuctionObservation) -> float:
        # Simple proxy: bigger when recent clearing prices are higher
        # (signals competitive interest), capped by valuation_anchor.
        recent = observation.recent_clearing_prices or []
        if recent:
            mean_recent = sum(recent) / len(recent)
            return min(self.valuation_anchor * 1.5, max(self.valuation_anchor * 0.5, mean_recent))
        return self.valuation_anchor

    def bid(
        self,
        observation: AuctionObservation,
        state: Optional[Dict[str, Any]] = None,
    ) -> AuctionAction:
        state = state or {}
        spent_today = float(state.get("spent_today", observation.spent_so_far_today))
        daily_target = float(
            state.get("daily_target", max(observation.daily_budget_remaining + spent_today, 1e-6))
        )
        target_segment = state.get("target_segment")  # caller may pass persona's preferred segment

        v = self._valuation_estimate(observation)
        a = self.traits["aggression"]
        pacing = self._pacing_factor(spent_today, daily_target)
        segment = self._segment_factor(observation.user_segment, target_segment)
        fatigue = self._fatigue_factor(observation.per_segment_fatigue, observation.user_segment)
        price = self._price_factor(observation.recent_clearing_prices)

        bid_amount = max(0.0, a * v * pacing * segment * fatigue * price)
        skip = bid_amount < observation.floor_price * 0.5

        creative_id = 0
        if observation.available_creatives:
            # Heuristic: pick the creative whose target_segment matches the user
            # or fall back to index 0.
            for idx, c in enumerate(observation.available_creatives):
                if c.get("target_segment") == observation.user_segment:
                    creative_id = idx
                    break

        return AuctionAction(
            skip=skip,
            bid_amount=round(bid_amount, 4),
            creative_id=creative_id,
        )


# ---------------------------------------------------------------------------
# LLMPolicyBot — wraps a frozen LLM checkpoint behind the same interface
# ---------------------------------------------------------------------------

ADVERTISER_SYSTEM_PROMPT = """\
You are an LLM advertiser bidding in a multi-day second-price auction.
Each step you receive a user, your remaining budget, segment fatigue,
recent clearing prices, and a target weekly ROAS.

Decide whether to bid; if you bid, choose the bid amount and creative.
Reply with ONLY a JSON object (no markdown, no commentary):
{"skip": bool, "bid_amount": float, "creative_id": int}

Rules:
- skip=true means do not bid (saves budget, no impression served).
- bid_amount in [0.0, 5.0]. Pay second-highest price if you win.
- creative_id is the 0-based index into available_creatives.
- Pace spend across days; an underspend penalty applies at week end.
"""


def _format_observation_for_advertiser(obs: AuctionObservation) -> str:
    creatives_lines: List[str] = []
    for idx, c in enumerate(obs.available_creatives):
        creatives_lines.append(
            f"  idx={idx}: target={c.get('target_segment','-')}, "
            f"category={c.get('category','-')}, "
            f"base_ctr={float(c.get('base_ctr', 0.0)):.3f}"
        )

    fatigue_lines = ", ".join(f"{k}={v:.2f}" for k, v in obs.per_segment_fatigue.items()) or "<none>"
    recent_prices = ", ".join(f"{p:.3f}" for p in obs.recent_clearing_prices[-5:]) or "<none>"

    return (
        f"DAY {obs.day_of_week} step_in_day={obs.step_in_day} "
        f"floor_price={obs.floor_price:.3f} freq_cap={obs.frequency_cap_per_user}\n"
        f"User: segment={obs.user_segment} interests={obs.user_interests} surface={obs.current_surface}\n"
        f"Budget: weekly_remaining={obs.weekly_budget_remaining:.2f} "
        f"daily_remaining={obs.daily_budget_remaining:.2f} "
        f"spent_today={obs.spent_so_far_today:.2f} spent_week={obs.spent_so_far_week:.2f}\n"
        f"Clicks: today={obs.clicks_today} week={obs.clicks_week} target_roas={obs.target_weekly_roas:.2f}\n"
        f"Per-segment fatigue: {fatigue_lines}\n"
        f"Recent clearing prices: {recent_prices}\n"
        f"Yesterday recap: {obs.yesterday_recap or '<n/a>'}\n"
        f"Available creatives:\n" + ("\n".join(creatives_lines) or "  <none>")
    )


_JSON_BLOCK_RE = re.compile(r"\{[\s\S]*\}")


def parse_llm_advertiser_action(raw_text: str, n_creatives: int) -> AuctionAction:
    """Robust JSON -> AuctionAction. Falls back to ``skip=True`` on parse failure."""
    candidates: List[str] = [raw_text]
    match = _JSON_BLOCK_RE.search(raw_text)
    if match:
        candidates.append(match.group(0))

    parsed: Optional[Dict[str, Any]] = None
    for candidate in candidates:
        try:
            parsed = json.loads(candidate)
            break
        except json.JSONDecodeError:
            continue

    if not isinstance(parsed, dict):
        return AuctionAction(skip=True, bid_amount=0.0, creative_id=0)

    try:
        skip = bool(parsed.get("skip", False))
        bid_amount = float(parsed.get("bid_amount", 0.0))
        creative_id = int(parsed.get("creative_id", 0))
    except (TypeError, ValueError):
        return AuctionAction(skip=True, bid_amount=0.0, creative_id=0)

    bid_amount = max(0.0, min(5.0, bid_amount))
    creative_id = max(0, min(n_creatives - 1, creative_id)) if n_creatives > 0 else 0
    if bid_amount == 0.0 and not skip:
        skip = True
    return AuctionAction(skip=skip, bid_amount=round(bid_amount, 4), creative_id=creative_id)


@dataclass
class LLMPolicyBot:
    """Adapter around a frozen LLM checkpoint that emits ``AuctionAction``s.

    The ``completion_fn`` is ``(system_prompt, user_prompt) -> str``.
    This decouples the bot from any specific runtime:

      - ``unsloth_completion_fn(checkpoint)`` for self-play eval and
        in-env rollouts (used by ``scripts/advertiser_eval.py``).
      - An OpenAI-style client for the legacy hosted-LLM baseline.

    The bot is *frozen* by contract — it neither updates weights nor
    accumulates conversation history. Each step is independent so the
    env can call ``bid`` from many parallel auction loops.
    """

    completion_fn: Callable[[str, str], str]
    name: str = "LLMPolicyBot"
    system_prompt: str = ADVERTISER_SYSTEM_PROMPT
    fallback: Optional[PersonaBot] = None

    def bid(
        self,
        observation: AuctionObservation,
        state: Optional[Dict[str, Any]] = None,
    ) -> AuctionAction:
        prompt = _format_observation_for_advertiser(observation)
        try:
            raw = self.completion_fn(self.system_prompt, prompt)
        except Exception:
            return self._fallback_action(observation, state)
        if not isinstance(raw, str):
            return self._fallback_action(observation, state)
        return parse_llm_advertiser_action(raw, n_creatives=len(observation.available_creatives))

    def _fallback_action(
        self,
        observation: AuctionObservation,
        state: Optional[Dict[str, Any]],
    ) -> AuctionAction:
        if self.fallback is not None:
            return self.fallback.bid(observation, state)
        return AuctionAction(skip=True, bid_amount=0.0, creative_id=0)


# ---------------------------------------------------------------------------
# Unsloth inference helper (lazy import)
# ---------------------------------------------------------------------------

def make_unsloth_advertiser_completion_fn(
    checkpoint_path: str,
    max_new_tokens: int = 64,
    max_seq_length: int = 4096,
) -> Callable[[str, str], str]:
    """Returns a ``(system, user) -> str`` callable backed by a frozen
    Unsloth LoRA checkpoint with ``FastLanguageModel.for_inference()``
    enabled. Lazy import keeps the module usable without unsloth installed.

    Used by:
      - ``scripts/advertiser_eval.py`` for selfplay mode (load the
        early-step checkpoint as the v1 opponent).
      - The post-training demo rollout in ``train_grpo.ipynb``.
    """
    try:
        from unsloth import FastLanguageModel  # type: ignore
    except ImportError as e:
        raise ImportError(
            "unsloth is required for make_unsloth_advertiser_completion_fn. "
            "Install with `pip install unsloth`."
        ) from e

    model, tokenizer = FastLanguageModel.from_pretrained(
        model_name=checkpoint_path,
        max_seq_length=max_seq_length,
        dtype=None,
        load_in_4bit=True,
    )
    FastLanguageModel.for_inference(model)

    def completion(system_prompt: str, user_prompt: str) -> str:
        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt},
        ]
        inputs = tokenizer.apply_chat_template(
            messages,
            return_tensors="pt",
            add_generation_prompt=True,
        ).to(model.device)
        output = model.generate(
            input_ids=inputs,
            max_new_tokens=max_new_tokens,
            do_sample=False,
            temperature=0.0,
            pad_token_id=tokenizer.pad_token_id or tokenizer.eos_token_id,
        )
        text = tokenizer.decode(output[0][inputs.shape[-1]:], skip_special_tokens=True)
        return text

    return completion


# ---------------------------------------------------------------------------
# Convenience constructor for an opponent slate (used by eval + tests)
# ---------------------------------------------------------------------------

def build_opponent_slate(
    persona_names: List[str],
    rng: random.Random,
    jitter_enabled: bool = True,
    jitter_scale: float = 1.0,
    valuation_anchor: float = 1.0,
) -> List[PersonaBot]:
    """Construct one ``PersonaBot`` per name with per-episode jitter.

    Used in the env (when Plan 1 lands) and in ``scripts/advertiser_eval.py``
    where it powers the standard / edge / selfplay opponent populations.
    """
    return [
        PersonaBot.from_persona_name(
            persona_name=name,
            rng=rng,
            jitter_enabled=jitter_enabled,
            jitter_scale=jitter_scale,
            valuation_anchor=valuation_anchor,
        )
        for name in persona_names
    ]
