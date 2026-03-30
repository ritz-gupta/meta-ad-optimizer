# Meta Ad Optimizer — Comprehensive Documentation

## Table of Contents

1. [Project Overview](#1-project-overview)
2. [Background & Motivation](#2-background--motivation)
3. [Architecture](#3-architecture)
4. [OpenEnv Integration](#4-openenv-integration)
5. [Simulation Engine](#5-simulation-engine)
   - 5.1 [User Segments](#51-user-segments)
   - 5.2 [Creative Catalog](#52-creative-catalog)
   - 5.3 [Platform, Surface & Format Validity](#53-platform-surface--format-validity)
   - 5.4 [Engagement Model](#54-engagement-model)
   - 5.5 [Fatigue Model](#55-fatigue-model)
   - 5.6 [Surface Transitions](#56-surface-transitions)
6. [RL Environment](#6-rl-environment)
   - 6.1 [Action Space](#61-action-space)
   - 6.2 [Observation Space](#62-observation-space)
   - 6.3 [State](#63-state)
   - 6.4 [Reward Function](#64-reward-function)
7. [Tasks & Difficulty Tiers](#7-tasks--difficulty-tiers)
   - 7.1 [Creative Matcher (Easy)](#71-creative-matcher-easy)
   - 7.2 [Placement Optimizer (Medium)](#72-placement-optimizer-medium)
   - 7.3 [Campaign Optimizer (Hard)](#73-campaign-optimizer-hard)
8. [Grading & Rubrics](#8-grading--rubrics)
9. [End-to-End Flow](#9-end-to-end-flow)
10. [Client API](#10-client-api)
11. [Baseline Agents](#11-baseline-agents)
12. [Server & Deployment](#12-server--deployment)
13. [Project Structure & File Reference](#13-project-structure--file-reference)
14. [Dependencies](#14-dependencies)
15. [Configuration Reference](#15-configuration-reference)

---

## 1. Project Overview

Meta Ad Optimizer is an OpenEnv reinforcement-learning environment that simulates ad delivery across **Instagram** and **Facebook**. An RL agent learns to optimise ad placement, creative selection, format choice, and impression frequency to maximise user engagement while managing user fatigue.

The environment is fully self-contained: it includes a simulation engine that models user behaviour, a creative catalog, engagement mechanics, fatigue dynamics, and a multi-objective reward function — all exposed through the standard OpenEnv `reset()` / `step()` / `state()` protocol.

**Key characteristics:**

- Stochastic, episodic environment with 3 difficulty tiers
- Multi-dimensional action space (show/skip, creative, platform, surface, format)
- Rich observation with user context, creative pool, fatigue, and session metrics
- Multi-objective reward balancing CTR, view time, satisfaction, and fatigue
- Supports concurrent sessions (up to 50 by default)
- Compatible with heuristic agents, LLM agents, and standard RL training pipelines

---

## 2. Background & Motivation

Ad delivery is Meta's core business. Every day, billions of impression opportunities arise across Feed, Reels, Stories, Explore, Marketplace, and Search on both Instagram and Facebook. A real ad delivery system must continuously make decisions that balance:

- **Relevance**: Which creative resonates with which user segment?
- **Placement**: Which surface and format produce the highest CTR and view time?
- **Frequency**: When to skip an impression to avoid fatigue and maintain long-term engagement?
- **Multi-objective trade-offs**: How to balance competing goals — CTR vs. view duration vs. user satisfaction vs. revenue?

This environment distils those real-world trade-offs into a tractable RL problem. It serves as:

1. **A training ground** for RL agents learning ad delivery policies
2. **A benchmark** with reproducible baseline scores (Random, Greedy, Rule-Based, LLM)
3. **A testbed** for LLM-based RL pipelines that treat the environment as a tool-use problem

---

## 3. Architecture

The system follows a client-server architecture built on top of the OpenEnv framework:

```
┌─────────────────────────────────────────────────────────────────────┐
│                          CLIENT LAYER                               │
│                                                                     │
│  ┌─────────────┐  ┌──────────────┐  ┌───────────────┐              │
│  │ AdEnv        │  │ BaselineAgent│  │ LLM Agent     │              │
│  │ (client.py)  │  │ (baseline.py)│  │ (baseline.py) │              │
│  │              │  │              │  │               │              │
│  │ EnvClient    │  │ Direct Python│  │ OpenAI API    │              │
│  │ WebSocket    │  │ (no server)  │  │ + env         │              │
│  └──────┬───────┘  └──────┬───────┘  └───────┬───────┘              │
│         │                 │                  │                      │
└─────────┼─────────────────┼──────────────────┼──────────────────────┘
          │ WebSocket       │ Direct call      │ Direct call
          ▼                 ▼                  ▼
┌─────────────────────────────────────────────────────────────────────┐
│                          SERVER LAYER                               │
│                                                                     │
│  ┌──────────────────────────────────────────────────────────────┐   │
│  │ FastAPI App  (server/app.py)                                 │   │
│  │   created via openenv.create_app()                           │   │
│  │   HTTP + WebSocket endpoints                                 │   │
│  │   max_concurrent_envs = 50                                   │   │
│  └──────────────────────┬───────────────────────────────────────┘   │
│                         │                                           │
│  ┌──────────────────────▼───────────────────────────────────────┐   │
│  │ AdOptimizerEnvironment  (server/ad_environment.py)           │   │
│  │   implements Environment interface                           │   │
│  │   reset() / step() / state                                   │   │
│  └──────────────────────┬───────────────────────────────────────┘   │
│                         │                                           │
│  ┌──────────────────────▼───────────────────────────────────────┐   │
│  │ Simulation Engine  (simulation.py)                           │   │
│  │   User segments, creative catalog, engagement computation,   │   │
│  │   fatigue model, surface transitions                         │   │
│  └──────────────────────────────────────────────────────────────┘   │
│                                                                     │
│  ┌──────────────────────────────────────────────────────────────┐   │
│  │ Tasks & Graders  (tasks.py)                                  │   │
│  │   3 difficulty tiers, grader functions (0.0–1.0)             │   │
│  └──────────────────────────────────────────────────────────────┘   │
│                                                                     │
│  ┌──────────────────────────────────────────────────────────────┐   │
│  │ Rubrics  (server/rubrics.py)                                 │   │
│  │   Trajectory-level scoring with exponential discounting      │   │
│  └──────────────────────────────────────────────────────────────┘   │
│                                                                     │
│  ┌──────────────────────────────────────────────────────────────┐   │
│  │ Models  (models.py)                                          │   │
│  │   AdAction, AdObservation, AdState (Pydantic + OpenEnv)      │   │
│  └──────────────────────────────────────────────────────────────┘   │
│                                                                     │
└─────────────────────────────────────────────────────────────────────┘
```

---

## 4. OpenEnv Integration

The project integrates with OpenEnv at three layers:

### 4.1 Type System — Action, Observation, State

All data contracts extend OpenEnv base types from `openenv.core.env_server.types`:

| Class            | Extends              | Purpose                                      |
|------------------|----------------------|----------------------------------------------|
| `AdAction`       | `Action`             | Agent's decision for one impression slot      |
| `AdObservation`  | `Observation`        | What the agent sees after each step           |
| `AdState`        | `State`              | Internal episode state (counters, fatigue)    |

These are Pydantic models, so they get automatic validation, serialization, and schema generation.

### 4.2 Server — Environment Interface

`AdOptimizerEnvironment` extends `openenv.core.env_server.interfaces.Environment` and implements:

- `reset(seed, task, **kwargs) → AdObservation` — initialise a new episode
- `step(action: AdAction) → AdObservation` — process one impression opportunity
- `state → AdState` — property exposing internal state

The `SUPPORTS_CONCURRENT_SESSIONS = True` flag tells OpenEnv this environment can handle multiple parallel sessions.

### 4.3 App Factory

`openenv.core.env_server.http_server.create_app()` takes the environment class, action type, and observation type and produces a ready-to-run FastAPI application with HTTP and WebSocket endpoints.

### 4.4 Client

`AdEnv` extends `openenv.core.EnvClient[AdAction, AdObservation, AdState]`, providing a typed WebSocket client with `reset()`, `step()`, and `state()` methods in both sync and async variants.

### 4.5 Rubric

`AdOptimizerRubric` extends `openenv.core.rubrics.trajectory.ExponentialDiscountingTrajectoryRubric` for trajectory-level RL credit assignment.

### 4.6 Manifest

The `openenv.yaml` file declares the environment for OpenEnv's discovery and deployment tooling:

```yaml
spec_version: 1
name: meta_ad_optimizer
type: space
runtime: fastapi
app: server.app:app
port: 8000
```

---

## 5. Simulation Engine

All simulation logic lives in `simulation.py`. It models the ad delivery ecosystem with the following components.

### 5.1 User Segments

Six user archetypes represent distinct audience demographics:

| Segment              | Interests                            | Device Bias                    | Platform Bias          | CTR Modifier | View Time Modifier |
|----------------------|--------------------------------------|--------------------------------|------------------------|:------------:|:------------------:|
| `gen_z_creator`      | fashion, music, memes, beauty        | 90% mobile, 5% desktop/tablet | 80% Instagram          | 1.10         | 0.70               |
| `millennial_parent`  | kids, home_decor, deals, family      | 55% mobile, 35% desktop       | 60% Facebook           | 0.90         | 1.30               |
| `business_pro`       | saas, finance, productivity, b2b     | 35% mobile, 55% desktop       | 80% Facebook           | 0.70         | 1.00               |
| `casual_scroller`    | entertainment, food, travel, memes   | 80% mobile                    | 60% Instagram          | 1.00         | 0.80               |
| `fitness_enthusiast` | health, sports, outdoors, supplements| 85% mobile                    | 70% Instagram          | 1.15         | 1.20               |
| `bargain_hunter`     | deals, coupons, electronics, secondhand | 55% mobile, 35% desktop   | 70% Facebook           | 1.05         | 1.10               |

Each segment also has **preferred surfaces** per platform. For example, `gen_z_creator` prefers Reels, Explore, and Stories on Instagram but Reels and Feed on Facebook.

**User generation**: At episode start, `generate_user()` samples a segment uniformly (or uses a forced segment), then draws a device, platform, and starting surface from the segment's weighted distributions.

### 5.2 Creative Catalog

A **master catalog** of 80 ad creatives is generated deterministically at server startup via `generate_master_catalog(n_creatives=80, seed=0)`.

Each creative has:

| Property         | Description                                            | Range            |
|------------------|--------------------------------------------------------|------------------|
| `id`             | Unique identifier within the catalog                   | 0–79             |
| `category`       | Product category (from 22 categories)                  | e.g., "fashion"  |
| `tone`           | Creative tone                                          | humorous, informational, emotional, action_oriented |
| `target_segment` | Which user segment this was designed for               | One of 6 segments|
| `base_ctr`       | Baseline click-through probability                     | 0.10–0.45        |
| `base_view_time` | Baseline expected view duration (seconds)              | 2.0–8.0          |

The catalog is **balanced**: each of the 6 segments gets approximately 13 creatives whose categories are drawn from that segment's affinity set.

**Per-episode sampling**: At episode start, `sample_creatives()` randomly draws N creatives from the master catalog (N = 4, 8, or 12 depending on the task). Each sampled creative gets a `pool_index` (0 to N-1) that the agent uses as `creative_id`.

**Creative categories** (22 total): fashion, music, beauty, kids, home_decor, deals, saas, finance, productivity, entertainment, food, travel, health, sports, outdoors, supplements, electronics, secondhand, memes, family, coupons, b2b.

### 5.3 Platform, Surface & Format Validity

The environment enforces strict validity rules for platform/surface/format combinations.

**Valid surfaces per platform:**

| Platform  | Surfaces                                                 |
|-----------|----------------------------------------------------------|
| Instagram | feed, reels, stories, explore, search                    |
| Facebook  | feed, reels, stories, marketplace, search, right_column  |

**Valid formats per surface:**

| Surface      | Allowed Formats                      |
|--------------|--------------------------------------|
| feed         | image, video, carousel, reel         |
| reels        | reel                                 |
| stories      | image, video, reel, collection       |
| explore      | image, video, carousel, reel         |
| search       | image, video                         |
| marketplace  | image, carousel, collection          |
| right_column | image                                |

An invalid combination (e.g., carousel on Reels, or video on right_column) results in:
- `valid_action = False`
- `satisfaction = -0.3`
- A **-0.2 penalty reward**
- Zero engagement (no click, no view time)

### 5.4 Engagement Model

When a valid ad is shown, `compute_engagement()` calculates engagement through a multiplicative factor chain:

```
effective_ctr = base_ctr
              × segment_affinity
              × placement_factor
              × context_bonus
              × platform_match
              × format_ctr
              × synergy
              × segment_ctr_modifier
              × fatigue_ctr_decay

effective_ctr = min(effective_ctr, 0.95)   # capped
```

**Factor breakdown:**

| Factor                     | Value / Logic                                                                 |
|----------------------------|-------------------------------------------------------------------------------|
| **Segment affinity**       | 2.0× if creative's `target_segment` matches user's segment; 1.2× if creative's `category` is in user's interests; 0.3× otherwise |
| **Placement factor**       | Surface-level base engagement. Examples: IG Reels = 1.4×, FB Marketplace = 1.5×, FB right_column = 0.4×. Further multiplied by device modifier for certain surfaces (Stories on desktop = 0.38×, right_column on mobile = 0.0×) |
| **Context bonus**          | 1.2× if the ad's placement matches the surface the user is currently browsing; 1.0× otherwise |
| **Platform match**         | 1.0× if ad targets the user's current platform; 0.5× penalty otherwise       |
| **Format CTR factor**      | reel = 1.4×, video = 1.2×, carousel = 1.15×, collection = 1.3×, image = 0.9× |
| **Format-surface synergy** | Multiplicative bonus for optimal pairings (reel on Reels = 1.35×, collection on Marketplace = 1.40×, carousel on Feed = 1.20×, etc.) |
| **Segment CTR modifier**   | Per-segment base modifier (e.g., fitness_enthusiast = 1.15×, business_pro = 0.7×) |
| **Fatigue CTR decay**      | `max(0, 1 - fatigue_level)` — at 50% fatigue, CTR is halved                  |

**View time** is computed similarly:

```
effective_view = base_view_time
               × format_view_time_factor
               × synergy
               × segment_view_time_modifier
               × fatigue_view_decay

fatigue_view_decay = max(0, 1 - fatigue^0.5)
```

Format view time factors: reel = 3.0×, video = 2.5×, collection = 2.5×, carousel = 2.0×, image = 1.0×.

**Stochastic outcomes:**

- **Click**: sampled as `Bernoulli(effective_ctr)` — `rng.random() < effective_ctr`
- **View time**: sampled as `Gaussian(effective_view, 0.3 × effective_view)`, floored at 0

**Satisfaction** is computed as:

```
relevance = segment_affinity / 1.5     # normalised to ~0–1.33
intrusiveness = 0.2 if on user's current surface, 0.5 otherwise
satisfaction = relevance × (1 - intrusiveness)
```

### 5.5 Fatigue Model

Fatigue tracks user annoyance from seeing too many ads. It is a float in [0.0, 1.0].

**When an ad is shown:**
```
fatigue += fatigue_increment × (1 + 0.1 × impressions_shown)
fatigue = min(1.0, fatigue)
```

The multiplicative factor `(1 + 0.1 × impressions_shown)` means fatigue accelerates — later impressions cause more fatigue per ad.

**When an ad is skipped:**
```
fatigue -= fatigue_recovery
fatigue = max(0.0, fatigue)
```

**Fatigue parameters by task:**

| Task                 | Enabled | Increment | Recovery |
|----------------------|:-------:|:---------:|:--------:|
| creative_matcher     | No      | 0.00      | 0.00     |
| placement_optimizer  | Yes     | 0.03      | 0.05     |
| campaign_optimizer   | Yes     | 0.06      | 0.04     |

The hard task has aggressive fatigue (double the increment, less recovery) making fatigue management a critical skill.

### 5.6 Surface Transitions

In the **campaign_optimizer** (hard) task only, the user drifts between surfaces each step according to a Markov transition matrix.

Transition probabilities (row = current surface, values = probability of moving to each target):

| From \ To      | Same   | feed  | reels | stories | explore | search | marketplace |
|----------------|:------:|:-----:|:-----:|:-------:|:-------:|:------:|:-----------:|
| feed           | 0.50   | —     | 0.20  | 0.15    | 0.10    | 0.05   | —           |
| reels          | 0.55   | 0.20  | —     | 0.10    | 0.10    | 0.05   | —           |
| stories        | 0.40   | 0.25  | 0.20  | —       | 0.10    | 0.05   | —           |
| explore        | 0.45   | 0.15  | 0.25  | 0.10    | —       | 0.05   | —           |
| search         | 0.40   | 0.25  | 0.10  | 0.10    | 0.15    | —      | —           |
| marketplace    | 0.50   | 0.25  | —     | 0.10    | —       | 0.15   | —           |
| right_column   | 0.30   | 0.40  | —     | —       | —       | 0.20   | 0.10        |

The transition is filtered to only include surfaces valid for the user's current platform. Surfaces not in the transition map get a default weight of 0.02.

---

## 6. RL Environment

### 6.1 Action Space

The agent submits an `AdAction` each step:

| Field         | Type   | Description                                      | Constraints                         |
|---------------|--------|--------------------------------------------------|-------------------------------------|
| `show_ad`     | `bool` | Whether to show an ad or skip this slot           | `False` only if task allows skipping|
| `creative_id` | `int`  | Index into the per-episode creative pool          | 0 to N-1 (clamped by environment)   |
| `platform`    | `str`  | Target platform                                   | "instagram" or "facebook"           |
| `placement`   | `str`  | Target surface                                    | Must be valid for the platform      |
| `ad_format`   | `str`  | Creative format                                   | Must be valid for the surface       |

Task-level overrides can lock down fields. For example, in the easy task, `platform`, `placement`, and `ad_format` are all fixed — the agent only chooses `creative_id`.

### 6.2 Observation Space

The agent receives an `AdObservation` after each step:

| Field                | Type             | Description                                                |
|----------------------|------------------|------------------------------------------------------------|
| `task`               | `str`            | Active task name                                           |
| `user_segment`       | `str`            | User archetype (e.g., "gen_z_creator")                     |
| `user_interests`     | `list[str]`      | Interest categories for this user                          |
| `user_device`        | `str`            | "mobile", "desktop", or "tablet"                           |
| `current_platform`   | `str`            | Platform the user is browsing                              |
| `current_surface`    | `str`            | Surface the user is currently on                           |
| `available_creatives`| `list[dict]`     | Pool of ad creatives with properties (see below)           |
| `impression_count`   | `int`            | Total ads shown so far this session                        |
| `fatigue_level`      | `float`          | 0.0–1.0 fatigue meter                                     |
| `step`               | `int`            | Current step number                                        |
| `total_steps`        | `int`            | Episode length                                             |
| `last_action_metrics`| `dict`           | Engagement metrics from the previous step                  |
| `session_metrics`    | `dict`           | Cumulative CTR, total clicks, view time, satisfaction      |
| `done`               | `bool`           | Whether the episode has ended                              |
| `reward`             | `float`          | Reward from the last step                                  |

**Each creative in `available_creatives`:**

| Property         | Description                                  |
|------------------|----------------------------------------------|
| `pool_index`     | Index to use as `creative_id` in the action  |
| `category`       | Product category                             |
| `tone`           | humorous / informational / emotional / action_oriented |
| `target_segment` | Which user segment it was designed for       |
| `base_ctr`       | Baseline click-through probability           |
| `base_view_time` | Baseline expected view duration (seconds)    |

**`last_action_metrics`** (when an ad was shown):

| Key              | Description                                  |
|------------------|----------------------------------------------|
| `clicked`        | Whether the user clicked                     |
| `view_time`      | Actual view duration (seconds)               |
| `effective_ctr`  | Computed click probability                   |
| `satisfaction`   | User satisfaction score                      |
| `valid_action`   | Whether the format/surface combo was valid   |
| `creative_id`    | Which creative was shown                     |
| `platform`       | Platform used                                |
| `placement`      | Surface used                                 |
| `ad_format`      | Format used                                  |
| `episode_score`  | (Final step only) 0.0–1.0 grader score      |

**`session_metrics`:**

| Key                       | Description                           |
|---------------------------|---------------------------------------|
| `ctr`                     | Session click-through rate            |
| `total_clicks`            | Total clicks this episode             |
| `total_view_time`         | Cumulative view time (seconds)        |
| `cumulative_satisfaction` | Sum of satisfaction scores            |
| `impressions_shown`       | Total ads shown                       |

### 6.3 State

`AdState` tracks internal episode state (not directly exposed to the agent through the observation, but accessible via the `state` property):

| Field                     | Type    | Description                        |
|---------------------------|---------|------------------------------------|
| `episode_id`              | `str`   | Unique episode identifier (UUID)   |
| `step_count`              | `int`   | Steps taken so far                 |
| `total_impressions_shown` | `int`   | Ads shown (excludes skips)         |
| `total_clicks`            | `int`   | Total clicks                       |
| `total_view_time`         | `float` | Cumulative view time               |
| `cumulative_satisfaction` | `float` | Sum of positive satisfaction scores|
| `fatigue_level`           | `float` | Current fatigue (0.0–1.0)          |
| `task`                    | `str`   | Active task name                   |
| `valid_actions`           | `int`   | Count of valid format/surface combos|
| `invalid_actions`         | `int`   | Count of invalid combos            |

### 6.4 Reward Function

**When an ad is shown (valid action):**
```
reward = 0.35 × click       (1.0 if clicked, 0.0 otherwise)
       + 0.25 × view_time   (normalised: min(1.0, view_time / 15.0))
       + 0.25 × satisfaction (clamped ≥ 0.0)
       + 0.15 × (−fatigue)  (penalises high fatigue)
```

**When an ad is shown (invalid format/surface combination):**
```
reward = −0.2
```

**When the agent skips (show_ad = False):**
```
reward = +0.05 (fatigue recovery bonus) − 0.02 (missed revenue) = +0.03
```

The reward is rounded to 5 decimal places.

---

## 7. Tasks & Difficulty Tiers

### 7.1 Creative Matcher (Easy)

| Property        | Value              |
|-----------------|--------------------|
| Steps           | 10                 |
| Creatives       | 4 per episode      |
| Platform        | Instagram (fixed)  |
| Surface         | Feed (fixed)       |
| Format          | Image (fixed)      |
| Fatigue         | Disabled           |
| Skip allowed    | No                 |
| Transitions     | No                 |

**Agent only picks `creative_id`.** The challenge is to learn which creative best matches the user segment from a small pool of 4.

**Grader**: Pure session CTR.
```
score = total_clicks / total_impressions_shown
```

### 7.2 Placement Optimizer (Medium)

| Property        | Value              |
|-----------------|--------------------|
| Steps           | 15                 |
| Creatives       | 8 per episode      |
| Platform        | Instagram (fixed)  |
| Surfaces        | All 5 IG surfaces  |
| Formats         | All 5 formats      |
| Fatigue         | Light (incr=0.03, recovery=0.05) |
| Skip allowed    | Yes                |
| Transitions     | No                 |

**Agent picks creative + surface + format + skip.** Must learn valid format/surface combinations, surface engagement differences, and basic fatigue management.

**Grader**: 30% validity + 70% engagement.
```
validity   = valid_actions / total_actions
engagement = 0.5 × min(1.0, ctr / 0.5) + 0.5 × normalised_view_time
score      = 0.3 × validity + 0.7 × engagement
```

### 7.3 Campaign Optimizer (Hard)

| Property        | Value                  |
|-----------------|------------------------|
| Steps           | 20                     |
| Creatives       | 12 per episode         |
| Platforms       | Instagram + Facebook   |
| Surfaces        | All surfaces           |
| Formats         | All formats            |
| Fatigue         | Aggressive (incr=0.06, recovery=0.04) |
| Skip allowed    | Yes                    |
| Transitions     | Yes (Markov drift)     |

**Full action space.** The agent must handle cross-platform decisions, aggressive fatigue, surface drift, and a larger creative pool.

**Grader**: Multi-objective.
```
validity       = valid_actions / total_actions
ctr_score      = min(1.0, ctr / 0.5)
view_score     = min(1.0, total_view_time / (15.0 × step_count))
sat_score      = min(1.0, cumulative_satisfaction / step_count)
fatigue_score  = 1.0 − fatigue_level

score = 0.15 × validity
      + 0.25 × ctr_score
      + 0.20 × view_score
      + 0.25 × sat_score
      + 0.15 × fatigue_score
```

---

## 8. Grading & Rubrics

### Episode Grading

Each task has a dedicated grader function in `tasks.py` that maps the terminal `AdState` to a normalised 0.0–1.0 score. The grader is called at the end of every episode and the score is included in the final observation's `last_action_metrics["episode_score"]`.

### Trajectory Rubric

`AdOptimizerRubric` (in `server/rubrics.py`) provides trajectory-level credit assignment for RL training. It extends `ExponentialDiscountingTrajectoryRubric` with gamma=0.99 (default).

The per-step discounted reward is:
```
r_t = gamma^(T − 1 − t) × final_score
```

where `T` is the episode length and `final_score` is the 0.0–1.0 grader output. Later steps receive more credit than earlier ones.

---

## 9. End-to-End Flow

### Phase 1: Startup

1. `openenv.yaml` declares the environment
2. `create_app()` in `server/app.py` creates a FastAPI application
3. `AdOptimizerEnvironment.__init__()` generates the 80-creative master catalog
4. Server starts listening on port 8000

### Phase 2: Episode Reset

1. Client sends `reset(task="...", seed=N)`
2. Task config is loaded (steps, fatigue params, constraints)
3. RNG is seeded for reproducibility
4. A random `UserProfile` is generated (segment, device, platform, surface)
5. N creatives are sampled from the master catalog
6. Internal state (`AdState`) is zeroed
7. Initial `AdObservation` is returned to the client

### Phase 3: Step Loop

For each step until `done`:

1. Client sends an `AdAction` (show/skip, creative_id, platform, placement, format)
2. Environment applies task-level overrides (fixed platform/surface/format)
3. If `show_ad = True`:
   - Validate the format/surface combination
   - If valid: compute engagement (effective_ctr, click, view_time, satisfaction)
   - If invalid: assign penalty
   - Update state counters (impressions, clicks, view_time, satisfaction)
   - Compute multi-objective reward
4. If `show_ad = False`:
   - Assign skip reward (+0.03)
5. Update fatigue (if enabled for this task)
6. Transition surface (if enabled for this task — Markov drift)
7. Check termination (`step_count >= steps_per_episode`)
8. If terminal: compute `episode_score` via the task grader
9. Return `AdObservation` with reward, metrics, updated state, done flag

### Phase 4: Episode End

The final observation includes `episode_score` in `last_action_metrics`. The client can read this score or call `grade_episode(env.state)` directly.

---

## 10. Client API

### WebSocket Client (AdEnv)

```python
from meta_ad_optimizer import AdEnv, AdAction

# Async usage
async with AdEnv(base_url="http://localhost:8000") as env:
    result = await env.reset(task="campaign_optimizer")
    print(result.observation.user_segment)

    result = await env.step(AdAction(
        show_ad=True,
        creative_id=2,
        platform="instagram",
        placement="reels",
        ad_format="reel",
    ))
    print(result.reward, result.observation.last_action_metrics)

# Sync usage
with AdEnv(base_url="http://localhost:8000").sync() as env:
    result = env.reset(task="creative_matcher")
    result = env.step(AdAction(show_ad=True, creative_id=0))
```

### Raw WebSocket

```python
import asyncio, json, websockets

async def main():
    async with websockets.connect("ws://localhost:8000/ws") as ws:
        await ws.send(json.dumps({"type": "reset", "data": {"task": "creative_matcher"}}))
        obs = json.loads(await ws.recv())["data"]["observation"]

        await ws.send(json.dumps({
            "type": "step",
            "data": {
                "show_ad": True,
                "creative_id": 0,
                "platform": "instagram",
                "placement": "feed",
                "ad_format": "image",
            }
        }))
        result = json.loads(await ws.recv())["data"]

asyncio.run(main())
```

### Direct Python (No Server)

```python
from meta_ad_optimizer.server.ad_environment import AdOptimizerEnvironment
from meta_ad_optimizer.models import AdAction
from meta_ad_optimizer.tasks import grade_episode

env = AdOptimizerEnvironment()
obs = env.reset(seed=42, task="campaign_optimizer")

while not obs.done:
    action = AdAction(show_ad=True, creative_id=0, platform="instagram",
                      placement="feed", ad_format="image")
    obs = env.step(action)

final_score = grade_episode(env.state)
```

---

## 11. Baseline Agents

Four baseline agents are provided in `baseline.py`:

### Random Agent
- Randomly picks platform, surface (valid for that platform), format (valid for that surface)
- Skips 20% of the time
- Serves as a lower bound

### Greedy Agent
- Always picks the creative with the highest `base_ctr`
- Always targets Reels with reel format (highest engagement factors)
- Never skips — no fatigue awareness
- Demonstrates that raw CTR maximisation without context is suboptimal

### Rule-Based Agent
- Matches creative `target_segment` to user segment (1.5× score bonus) or category to interests (1.0×)
- Selects the format with the highest synergy for the current surface
- Skips every 4th impression or when fatigue > 0.5 for fatigue management
- Strongest heuristic baseline

### LLM Agent (Optional)
- Sends the full observation as a structured text prompt to an OpenAI-compatible model
- Parses the model's JSON response into an `AdAction`
- Falls back to a safe default action on parse failure
- Demonstrates LLM-as-agent compatibility

### Baseline Scores (100 episodes, seed=42)

| Task                  | Random        | Greedy        | Rule-Based    |
|-----------------------|:-------------:|:-------------:|:-------------:|
| creative_matcher      | 0.1720 ± 0.14| 0.2430 ± 0.27| 0.3730 ± 0.30|
| placement_optimizer   | 0.4859 ± 0.12| 0.7141 ± 0.13| 0.7333 ± 0.15|
| campaign_optimizer    | 0.2800 ± 0.05| 0.3529 ± 0.13| 0.5899 ± 0.09|

---

## 12. Server & Deployment

### Local Development

```bash
cd meta_ad_optimizer
uv sync                        # install dependencies
uvicorn meta_ad_optimizer.server.app:app --host 0.0.0.0 --port 8000
```

Or via the registered script entry point:

```bash
server   # runs meta_ad_optimizer.server.app:main
```

### Docker

The `server/Dockerfile` uses a multi-stage build:

1. **Builder stage**: installs `uv`, syncs dependencies from `uv.lock` or `pyproject.toml`
2. **Runtime stage**: copies the venv and app code, sets `PATH` and `PYTHONPATH`
3. **Healthcheck**: pings `http://localhost:8000/health` every 30 seconds

```bash
docker build -t meta-ad-optimizer -f server/Dockerfile .
docker run -p 8000:8000 meta-ad-optimizer
```

### Hugging Face Spaces

```bash
openenv push
```

### Running Baselines

```bash
# Heuristic baselines (no server required)
python -m meta_ad_optimizer.baseline --episodes 100 --seed 42

# With LLM agent
OPENAI_API_KEY=sk-... python -m meta_ad_optimizer.baseline --episodes 5 --seed 42 --llm --model gpt-4o-mini
```

---

## 13. Project Structure & File Reference

```
meta_ad_optimizer/
│
├── __init__.py              Package exports (AdAction, AdObservation, AdState, AdEnv)
├── models.py                Pydantic data models extending OpenEnv types
├── simulation.py            Simulation engine (users, creatives, engagement, fatigue, transitions)
├── tasks.py                 Task configs (3 tiers) + grader functions
├── client.py                WebSocket client (AdEnv) wrapping OpenEnv EnvClient
├── baseline.py              Baseline agents (Random, Greedy, Rule-Based, LLM) + evaluation loop
├── openenv.yaml             OpenEnv deployment manifest
├── pyproject.toml           Package metadata + dependencies
├── README.md                Quick-start README
├── DOCUMENTATION.md         This file
│
└── server/
    ├── app.py               FastAPI app created via openenv.create_app()
    ├── ad_environment.py    AdOptimizerEnvironment — core reset/step/state implementation
    ├── rubrics.py           AdOptimizerRubric — trajectory-level exponential discounting
    ├── requirements.txt     Pinned server dependencies
    └── Dockerfile           Multi-stage Docker build
```

### File-by-file summary

| File                        | Lines | Purpose                                                              |
|-----------------------------|:-----:|----------------------------------------------------------------------|
| `__init__.py`               | 12    | Package-level exports                                                |
| `models.py`                 | 73    | `AdAction`, `AdObservation`, `AdState` Pydantic models               |
| `simulation.py`             | 469   | All simulation logic: segments, catalog, engagement, fatigue, transitions |
| `tasks.py`                  | 149   | `TaskConfig` dataclass, 3 task definitions, 3 grader functions       |
| `client.py`                 | 85    | `AdEnv` WebSocket client with typed step/reset/state                 |
| `baseline.py`               | 362   | 4 baseline agents + CLI evaluation harness                           |
| `server/app.py`             | 42    | FastAPI app factory wiring                                           |
| `server/ad_environment.py`  | 280   | Core `AdOptimizerEnvironment` (reset, step, reward, observation)     |
| `server/rubrics.py`         | 63    | `AdOptimizerRubric` for trajectory-level RL scoring                  |
| `openenv.yaml`              | 6     | OpenEnv manifest (name, runtime, port)                               |
| `pyproject.toml`            | 34    | Build system, dependencies, entry points                             |

---

## 14. Dependencies

### Core

| Package          | Version    | Purpose                                    |
|------------------|------------|--------------------------------------------|
| openenv-core     | >= 0.2.1   | RL environment framework (interfaces, server, client) |
| numpy            | >= 1.24.0  | Numerical operations                       |
| pydantic         | >= 2.0.0   | Data validation and serialization          |
| fastapi          | >= 0.115.0 | HTTP/WebSocket server                      |
| uvicorn[standard]| >= 0.24.0  | ASGI server                                |

### Optional

| Package  | Version  | Purpose                     | Install group |
|----------|----------|-----------------------------|---------------|
| openai   | >= 1.0.0 | LLM baseline agent          | `llm`         |
| pytest   | >= 8.0.0 | Testing                     | `dev`         |
| pytest-cov| >= 4.0.0| Coverage reporting          | `dev`         |

### Python Version

Requires Python **3.10+**.

---

## 15. Configuration Reference

### Environment Constructor

| Parameter       | Default | Description                                 |
|-----------------|---------|---------------------------------------------|
| `catalog_size`  | 80      | Number of creatives in the master catalog   |
| `catalog_seed`  | 0       | Seed for deterministic catalog generation   |

### reset() Parameters

| Parameter | Default               | Description                              |
|-----------|-----------------------|------------------------------------------|
| `seed`    | `None`                | RNG seed for reproducibility             |
| `task`    | `"campaign_optimizer"`| Task tier to use                         |

### TaskConfig Fields

| Field                  | Type           | Description                                        |
|------------------------|----------------|----------------------------------------------------|
| `name`                 | `str`          | Task identifier                                    |
| `steps_per_episode`    | `int`          | Number of steps before episode terminates           |
| `creatives_per_episode`| `int`          | Number of creatives sampled for the pool            |
| `platforms`            | `list[str]`    | Available platforms                                 |
| `allow_skip`           | `bool`         | Whether `show_ad=False` is permitted                |
| `fatigue_enabled`      | `bool`         | Whether fatigue mechanics are active                |
| `fatigue_increment`    | `float`        | Base fatigue increase per shown ad                  |
| `fatigue_recovery`     | `float`        | Fatigue decrease per skipped impression             |
| `surface_transitions`  | `bool`         | Whether user drifts between surfaces each step      |
| `fixed_platform`       | `str \| None`  | Lock platform (agent cannot choose)                 |
| `fixed_surface`        | `str \| None`  | Lock surface                                        |
| `fixed_format`         | `str \| None`  | Lock format                                         |

### Placement CTR Factors

| Platform  | Surface      | Factor |
|-----------|--------------|:------:|
| Instagram | reels        | 1.4    |
| Instagram | stories      | 1.3    |
| Instagram | feed         | 1.0    |
| Instagram | explore      | 1.2    |
| Instagram | search       | 0.9    |
| Facebook  | marketplace  | 1.5    |
| Facebook  | reels        | 1.3    |
| Facebook  | search       | 1.1    |
| Facebook  | stories      | 1.1    |
| Facebook  | feed         | 1.0    |
| Facebook  | right_column | 0.4    |

### Device-Surface Modifiers

| Surface      | Mobile | Desktop | Tablet |
|--------------|:------:|:-------:|:------:|
| stories      | 1.0    | 0.38    | 0.75   |
| right_column | 0.0    | 1.0     | 0.5    |

### Format-Surface Synergy

| Format     | Surface     | Synergy |
|------------|-------------|:-------:|
| reel       | reels       | 1.35    |
| reel       | stories     | 1.25    |
| carousel   | feed        | 1.20    |
| carousel   | explore     | 1.20    |
| collection | marketplace | 1.40    |
| collection | stories     | 1.15    |
| image      | right_column| 1.10    |

### create_app() Configuration

| Parameter            | Value                      |
|----------------------|----------------------------|
| Environment class    | `AdOptimizerEnvironment`   |
| Action type          | `AdAction`                 |
| Observation type     | `AdObservation`            |
| Environment name     | `"meta_ad_optimizer"`      |
| Max concurrent envs  | 50                         |
