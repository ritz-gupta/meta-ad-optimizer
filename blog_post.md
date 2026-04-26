---
title: "AdMarket Arena: Teaching LLMs to Win Ad Auctions with GRPO"
thumbnail: https://huggingface.co/datasets/huggingface/documentation-images/resolve/main/blog/no_image.jpg
authors:
  - user: ritz-gupta
tags:
  - reinforcement-learning
  - grpo
  - multi-agent
  - long-horizon
  - ad-tech
  - unsloth
  - trl
  - openenv
---

# AdMarket Arena: Teaching LLMs to Win Ad Auctions with GRPO

> **The pitch.** A 3B-parameter LLM, trained with GRPO, learns to bid in real-time ad auctions — pacing a $1,000 weekly budget across **350 sealed-bid rounds** against four scripted competitors, while planning across days that no longer fit in its context window.
>
> **Want the full story?**
> - **Execution & how to run it →** [`README.md`](./README.md)
> - **The 10-slide pitch deck →** [`AdMarket_Arena_Hackathon.pptx`](https://github.com/ritz-gupta/AdMarket-Arena/raw/main/AdMarket_Arena_Hackathon.pptx)

---

## Why this matters

Digital advertising is the original real-world multi-agent game. Every Google, Meta, and Amazon advertiser plays it every second of every day. A policy that's **10% better at ROAS on a $1M/week budget saves $100,000 per week.**

And yet — there's no open RL benchmark for it. Chess and Atari don't capture sealed bidding. MuJoCo doesn't simulate budget pacing. Existing ad-tech sims aren't open and weren't built for LLM training.

So we built one.

---

## What we built

**AdMarket Arena** is an OpenEnv environment that drops one GRPO-trained LLM advertiser into a 7-day Vickrey auction against four scripted `PersonaBot` opponents. The agent must simultaneously solve four hard problems most RL benchmarks ignore:

- **Multi-agent dynamics** — bid against opponents whose strategies you can only infer from clearing prices.
- **Long-horizon planning** — 350 decisions per episode, far beyond any context window.
- **Budget pacing** — overspend Day 1 and the weekend slots are gone; underspend and the terminal penalty wipes out ROAS.
- **User fatigue** — knowing *when to skip* is as important as knowing *what to bid*.

Our key trick for the context-window problem: a `yesterday_recap` field — a ~200-token narrative summary injected at each day boundary. The agent doesn't need 350 steps of history; it just needs to read yesterday's story.

---

## The headline result

We trained `Qwen2.5-3B-Instruct` with TRL GRPO + Unsloth (4-bit NF4 + LoRA r=16, ~4hr on a single T4) and benchmarked against four baselines on `arena_easy`:

| Agent | Weekly ROAS | Mean Episode Reward |
|---|---|---|
| ArenaRandom | 0.167 ± 0.040 | -2.384 |
| ArenaGreedy | 0.264 ± 0.061 | -1.297 |
| ArenaRecapFollower | 0.219 ± 0.229 | -0.454 |
| **ArenaPacing (ceiling)** | **0.527 ± 0.081** | **9.100** |

`ArenaPacing` — our hand-tuned heuristic — gets **3.15× the ROAS of random** and skips just 3% of slots vs. random's 20.7%. That's the bar the trained LLM must clear, by learning to do what no rule-based agent can: read the daily recap, infer opponent intent, and shade bids accordingly.

---

## What the trained agent actually learns

After GRPO, several non-trivial behaviors emerge — none of which are hardcoded:

- **Bid shading.** It bids *below* its true valuation when it estimates a high win probability — preserving budget for high-value later slots. Game-theoretic behavior, learned from reward.
- **Creative-segment matching.** It picks creatives whose `target_segment` matches the current user, inferring the 2.0× engagement multiplier from rewards alone.
- **Fatigue-aware skipping.** It selectively passes on high-fatigue users — trading a tiny `+0.02` skip nudge for restored future engagement quality.
- **Cross-day pacing.** It reads `yesterday_recap` and adjusts aggression accordingly. After a hot Day 1, it cools down to save budget for the weekend. **A pure RL agent without language comprehension cannot do this.**

---

## Why it's hard to reward-hack

The hackathon brief flags reward-hacking explicitly. We close five obvious exploits:

1. **Skip-spam fails** — always-skip ends with ROAS = 0 and pays a `-2.0` underspend penalty.
2. **Greedy bidding fails** — overspend penalty + per-impression waste tax punishes "always max bid".
3. **Single-day farming fails** — the weekly bonus is multiplicative on *weekly* ROAS, not sum-of-daily.
4. **Bid-precision can't be gamed** — bidding `clearing_price + ε` loses ties and collapses ROAS.
5. **Persona jitter** — opponents have ±20–30% per-episode trait jitter, so memorising bid values doesn't transfer.

The only path to high reward is the intended one: engaged wins, paced spend, fatigue-aware skipping, cross-day planning.

---

## Who should care

- **RL researchers** — the first open benchmark with sealed-bid auctions, real budget constraints, and long-horizon planning, with independently-ablatable reward rubrics.
- **Ad-tech practitioners** — a sim that mirrors actual KPIs (ROAS, pacing, fatigue), not proxy objectives.
- **OpenEnv contributors** — the `yesterday_recap` pattern generalises to any domain where the agent needs compressed history.

---

## Try it in 30 seconds

```python
from meta_ad_optimizer.client import AdMarketArenaEnv
from meta_ad_optimizer.models import AuctionAction

async with AdMarketArenaEnv(base_url="https://ritz-gupta-admarket-arena.hf.space/arena") as env:
    obs = (await env.reset(task="arena_easy")).observation
    result = await env.step(AuctionAction(skip=False, bid_amount=1.20, creative_id=0))
    print(result.observation.last_auction_result)
```

---

## Dive deeper

- **Execution details, setup, full env spec, baselines, oversight agent →** [`README.md`](./README.md)
- **Pitch deck (slides) →** [`AdMarket_Arena_Hackathon.pptx`](https://github.com/ritz-gupta/AdMarket-Arena/raw/main/AdMarket_Arena_Hackathon.pptx)
- **HuggingFace Space →** [ritz-gupta/AdMarket-Arena](https://huggingface.co/spaces/ritz-gupta/AdMarket-Arena)
- **Trained model →** [MuskanBidani/admarket-advertiser-qwen2.5-3b-grpo](https://huggingface.co/MuskanBidani/admarket-advertiser-qwen2.5-3b-grpo)
- **Training notebook →** [`train_grpo.ipynb`](./train_grpo.ipynb)
- **Source →** [github.com/ritz-gupta/AdMarket-Arena](https://github.com/ritz-gupta/AdMarket-Arena)

---

*Built for the OpenEnv Hackathon · Themes: Multi-Agent + Long-Horizon Planning · Powered by OpenEnv × TRL GRPO × Unsloth × Qwen2.5-3B*
