# CityNav / AgentNav — System Architecture

> Technical documentation for the CityNav autonomous visual navigation platform.
> See the [paper](https://arxiv.org/abs/2512.15933) and [project page](https://dwipddalal.github.io/AgentNav/) for results and methodology.

---

## Table of Contents

1. [What This System Actually Is](#1-what-this-system-actually-is)
2. [High-Level System Architecture](#2-high-level-system-architecture)
3. [Core Simulation Engine](#3-core-simulation-engine)
4. [Navigation Strategies (The AI Brain)](#4-navigation-strategies-the-ai-brain)
5. [Self-Positioning Agent (Zero-Shot Geo-Localization)](#5-self-positioning-agent-zero-shot-geo-localization)
6. [Infrastructure Layer](#6-infrastructure-layer)
7. [CLI & Experiment Orchestration](#7-cli--experiment-orchestration)
8. [Analysis & Observability Platform](#8-analysis--observability-platform)
9. [Path Generator Tool (Full-Stack Web App)](#9-path-generator-tool-full-stack-web-app)
10. [Ablation Study Framework](#10-ablation-study-framework)
11. [Key Engineering Decisions](#11-key-engineering-decisions)
12. [System Architecture Diagram (ASCII)](#12-system-architecture-diagram-ascii)
13. [Data Flow Diagrams](#13-data-flow-diagrams)

---

## 1. System Overview

CityNav is an end-to-end platform for autonomous visual navigation in real-world cities. The system:

- **Navigates real cities** (New York, Tokyo, Vienna, São Paulo) in Google Street View
- **Makes 50–150 sequential decisions** per run, each requiring vision + reasoning
- **Self-localizes without GPS** using only visual observations
- **Handles real-world failures** (dead ends, missing Street View coverage, API errors)
- **Runs experiments in parallel** with checkpointing and resumption
- **Evaluates decisions** against ground truth walking distances
- **Visualizes results** through web-based dashboards
- **Generates evaluation datasets** through a full-stack path generation tool

The platform spans: simulation engine, agent architecture, multi-provider LLM integration, experiment orchestration, persistent caching, analysis dashboards, and a dataset creation tool.

---

## 2. High-Level System Architecture

The system is organized into **7 layers**, each independently designed and testable:

| Layer | Directory | Lines of Code | Purpose |
|-------|-----------|--------------|---------|
| **Simulation Engine** | `core/` | ~2,340 | Environment, agent, simulation loop, utilities |
| **AI Strategies** | `strategies/` | ~2,400 | Vision-based decision-making with MLLMs |
| **Agent Modules** | `agents/` | ~680 | Self-positioning, panorama tiling math |
| **Infrastructure** | `infrastructure/` | ~990 | LLM abstraction, caching (LMDB), scoring, auth |
| **CLI / Orchestration** | `cli/` | ~1,440 | Single run, batch parallel, resume, multi-model |
| **Analysis Platform** | `analysis/` | ~3,470 | Error analysis, log crawling, stats, web viewer |
| **Path Generator** | `tools/path_generator/` | ~8,800 | Full-stack web app for dataset creation |

---

## 3. Core Simulation Engine

### 3.1 Environment (`core/environment.py` — 854 lines)

The environment wraps Google Street View as a **Gymnasium-style MDP**:

**What it does:**
- Maintains the agent's current position as a Street View panorama ID
- Provides observations: available directional links with compass headings
- Applies actions: moves the agent to the chosen panorama
- Determines termination: polygon-based or distance-based arrival detection

**Key engineering:**
- **Playwright browser automation** — programmatic access to Google Maps' internal panorama API (not just the static images API). Fetches the full panorama graph (linked panos, coordinates, headings) by executing JavaScript in a headless Chromium browser
- **LMDB-backed panorama cache** — every fetched pano is cached permanently. Subsequent runs skip the API entirely. Multi-process safe via LMDB (not SQLite, which has writer-lock contention)
- **Dead-end pruning** — detects dead-end links (panos that only link back to where you came from) and marks them in a persistent cache. Shared across ALL runs, so the system learns the Street View graph over time
- **Backlink sanitization** — Street View panoramas sometimes include inconsistent reverse links. The environment sanitizes these so the agent doesn't get stuck in 2-node loops
- **Stable heading calculation** — at intersections preceded by long corridors, raw heading can be noisy. The environment "walks forward" through linear (non-branching) corridors to compute bearing from start-to-end, yielding a more reliable heading
- **Polygon-based destination detection** — uses Shapely's geometric inclusion test against manually-annotated destination polygons (not just "within X meters")
- **Walking distance evaluation** — calls Google Directions API to compute real walking distance (not haversine) for decision quality evaluation. Results are also cached in LMDB

### 3.2 Agent (`core/agent.py` — 100 lines)

The agent follows a **Strategy pattern** — it holds state but delegates decision-making entirely to its strategy:

```
Agent
├── observation       # Current available links
├── destination       # Target destination name
├── decision_history  # Full sequence of past decisions
├── memory            # Markovian memory state (LLM-generated summary)
├── pos_agent         # Self-positioning sub-component
├── env_events        # Queue (dead-end detections from environment)
└── strategy          # Pluggable decision-making module
```

The agent is a **pure state container** — this clean separation means strategies can be swapped without touching agent code, enabling controlled ablation studies.

### 3.3 Simulation Loop (`core/simulation.py` — 624 lines)

The simulation loop is the **main orchestrator** with several production-grade features:

- **Per-decision checkpointing** — after every single decision (not periodic), the entire state (agent, environment, strategy, config) is serialized to disk. This means any run can be resumed from any decision point
- **Atomic checkpoint writes** — writes to a temp file, then atomic `os.rename()`. No corrupted checkpoints even on crash/kill
- **Non-pickleable object handling** — Playwright browsers, LMDB handles, and API clients can't be pickled. The checkpoint system detaches these before serialization and re-attaches them on resume
- **Signal handling** — Ctrl+C triggers a graceful shutdown: saves final checkpoint, dumps coordinates, writes evaluation log, then exits cleanly
- **Decision evaluation tracking** — after each decision, the system tracks whether walking distance to destination decreased (RIGHT) or increased (WRONG). This is evaluated asynchronously because the "true" result isn't known until the agent reaches the next intersection
- **Coordinate trajectory logging** — every position visited is logged with timestamps, creating a full GPS-like trace of the navigation

### 3.4 Utilities (`core/utils.py` — 728 lines)

Shared infrastructure used throughout the system:

- **Haversine distance** — coordinate distance calculations
- **URL signing** — HMAC-SHA1 signing for Google Street View Static API (required for high-volume usage)
- **Checkpoint save/load** — the serialization/deserialization logic for full system state
- **Tee logging** — custom `Tee` class that duplicates stdout/stderr to both console and a log file in real-time
- **Experiment folder generation** — creates timestamped, deterministic folder names for reproducible experiments
- **File-based locking** — `fcntl.flock()` for coordinating rate limits across parallel processes
- **Run status extraction** — parses log folders to extract success/failure, decision counts, distance metrics

---

## 4. Navigation Strategies

### 4.1 Baseline Strategy (`strategies/baseline.py` — 1,200 lines)

The base MLLM strategy that makes navigation decisions using vision:

**Decision flow at each intersection:**
1. **Observation** — receive list of available directional links with headings
2. **Image fetching** — download a Street View static image for each direction (looking down that street)
3. **Intersection detection** — SHA-256 hash the set of images to detect if we've returned to a previously-visited intersection
4. **Dead-end detection** — if we returned to the same intersection we just left, flag it as a dead end
5. **Prompt construction** — build a structured prompt with:
   - Current images (one per direction)
   - Cardinal directions for each choice
   - Decision history (previous N choices)
   - Memory state (LLM-generated summary of progress)
   - Self-positioning result (estimated lat/lon)
   - Previous visit warnings
   - Intersection summary
6. **LLM call** — send to MLLM (GPT-4, Gemini, Claude, etc.) with structured JSON schema
7. **Response parsing** — extract chosen direction, reasoning, and updated memory
8. **Recording** — log the decision to intersection memory for future reference

**The LLM response schema** is carefully designed:
```json
{
  "analysis": "free-form reasoning about what the agent sees and where it thinks it is",
  "decision": "the chosen direction alias (e.g., 'A', 'B', 'C')",
  "memory": "updated summary of progress, position, and plan for the next step"
}
```

**Prompt component toggles** — each piece of context (memory, history, self-positioning, etc.) can be individually enabled/disabled via config. This is what powers the ablation study in the paper.

**Retry logic** — up to 4 LLM call attempts with exponential backoff (2s, 30s, 60s, 180s). Handles rate limits, timeouts, and malformed responses gracefully using `json_repair` for automatic JSON fixing.

### 4.2 AgentNav Strategy (`strategies/agent_nav.py` — 1,272 lines)

The full AgentNav strategy adds **loop-breaking** on top of the baseline:

- **Escalating per-direction warnings** — tracks how many times the agent has chosen each direction at each intersection. At thresholds (3, 5, 10), injects increasingly urgent warnings into the prompt:
  - Level 1 (>3): "You have chosen direction X multiple times"
  - Level 2 (>5): "WARNING: You appear to be stuck in a loop"
  - Level 3 (>10): "CRITICAL: You MUST choose a different direction"
- **Global exploration advice** — based on the worst-case direction count across the current intersection, adjusts the overall tone of the prompt to encourage exploration
- **Configurable variant parameters** — grid parameters, self-positioning interval, signed URL support. These are passed through to the self-positioning agent

---

## 5. Self-Positioning Agent (Zero-Shot Geo-Localization)

### `agents/self_positioning.py` — 450 lines

The agent **estimates its own GPS coordinates** from a panoramic image — with no GPS, no landmarks, no map.

**How it works:**
1. **Panorama tiling** — the full 360-degree spherical panorama is divided into a grid of tiles (default 4x3 or 8x3). Each tile is fetched as a separate Street View Static API image
2. **Polar cap trimming** — the top and bottom of the sphere (sky and ground) are cropped out as they contain no useful location information
3. **Grid selection** — configurable which tiles to use: "all", "middle_row" (horizon band), or custom patterns. The middle row is most informative for geo-localization
4. **LLM geo-localization** — the tiles are sent to the MLLM with a prompt asking it to estimate the lat/lon coordinates based on visual cues (street signs, architecture, vegetation, sun position, etc.)
5. **Markovian memory** — the last N position estimates are tracked. The prompt includes this history to prevent drastic "jumps" in estimated location (e.g., suddenly thinking you're in a different city)

### `agents/pano_grid.py` — 227 lines

The mathematical backbone for panorama tiling:

- **Equal-angle spacing** — divides the sphere into tiles of equal angular width/height
- **Band limit calculation** — accounts for trimmed polar caps to determine the remaining latitude range
- **FOV computation** — calculates the correct `fov` parameter for Google's Street View Static API so tiles seamlessly tile the sphere
- **URL generation** — produces signed or unsigned Street View Static API URLs with correct heading, pitch, and FOV for each tile

---

## 6. Infrastructure Layer

### 6.1 LLM Wrapper (`infrastructure/llm_wrapper.py` — 524 lines)

A **unified, provider-agnostic LLM interface** supporting:

| Provider | Models |
|----------|--------|
| OpenAI | GPT-4o, GPT-4.1, GPT-5, etc. |
| Google | Gemini 2.0/2.5 Flash, Gemini Pro |
| Anthropic | Claude 3.5, Claude 4 |
| Azure OpenAI | Via TRAPI proxy |
| Ollama | Any local model |

**Built on LiteLLM** for provider abstraction, with custom additions:
- **TRAPI (Microsoft proxy) support** — Azure Active Directory token acquisition for corporate environments
- **File-based rate limiting** — uses `fcntl.flock()` to coordinate API calls across parallel processes (e.g., 10 processes sharing a rate limit). Each process acquires a file lock before making a call, enforcing a minimum delay
- **Detailed request/response logging** — every LLM call is saved to disk as a JSON file with full prompt, response, token counts, and timing. This creates an auditable trail of every decision
- **Automatic credential detection** — reads from environment variables, supports multiple key formats

### 6.2 Caching System (`infrastructure/cache.py` — 141 lines)

**LMDB-based persistent caching** with two databases:

**PanoCache** — caches Google Street View panorama data:
- Stores panorama metadata (ID, coordinates, links, headings)
- Dead-edge tracking (marks specific parent→child links as dead ends)
- Nearest-pano lookup by coordinates (haversine-based)
- Multi-process safe via LMDB (lock-free readers, single writer)
- 8GB default map size, configurable

**DistanceCache** — caches Google Directions API results:
- Stores walking distance between any two coordinate pairs
- Avoids redundant API calls during evaluation
- Same LMDB backend for consistency

**Why LMDB over SQLite/Redis:**
- SQLite has writer-lock contention in multi-process scenarios
- Redis requires a separate server process
- LMDB gives zero-copy reads, multi-process safety, and persistent storage with no daemon

### 6.3 Scoring System (`infrastructure/scoring.py` — 235 lines)

A **custom navigation quality scoring algorithm** designed for this paper:

- **Geometric streak scoring** — consecutive correct decisions earn exponentially growing rewards (base 2.4). Consecutive incorrect decisions earn exponentially growing penalties
- **Asymmetric weighting** — failures are weighted more heavily than successes (beta=0.8), reflecting that one wrong turn can undo many correct ones
- **Flip penalty** — alternating right-wrong-right-wrong patterns are penalized (lambda=2.0), discouraging random exploration
- **Dynamic caps** — reward/penalty caps scale with the total number of decisions, preventing score explosion on long runs
- **Multiple kernels** — implements geometric, polynomial, and logistic kernels for comparison
- **Normalization** — supports mean, z-score, and max-scaled normalization for cross-run comparison

### 6.4 TRAPI Integration (`infrastructure/trapi.py` — 92 lines)

Azure Active Directory authentication for Microsoft's TRAPI (Trusted Research API) proxy. Handles OAuth2 token acquisition and refresh for enterprise environments.

---

## 7. CLI & Experiment Orchestration

### 7.1 Single Run (`cli/run.py` — 282 lines)

Entry point for a single navigation experiment:
- Loads YAML config
- Initializes environment (Playwright browser, caches)
- Instantiates strategy and agent
- Creates experiment folder with timestamped name
- Runs simulation loop
- Saves final results (coordinates, evaluations, checkpoints)

### 7.2 Batch Parallel Execution (`cli/run_batch.py` — 400 lines)

**Distributed experiment orchestration** for running 100+ paths in parallel:
- Loads all paths from dataset JSON
- Distributes across a process pool
- Each process gets its own experiment folder
- Progress tracking across all runs
- Handles failures gracefully (one crashed run doesn't kill the batch)

### 7.3 Resume System (`cli/resume.py` — 299 lines)

**Checkpoint-based experiment resumption:**
- Scans a session folder for incomplete runs
- Finds the latest checkpoint for each run
- Re-initializes all non-pickleable objects (browser, caches, API clients)
- Continues simulation from the exact decision point where it stopped
- Supports "branching" — resume from decision N into a new folder for A/B testing

### 7.4 Multi-Model Runner (`cli/run_by_model.py` — 457 lines)

Runs the same path across **multiple LLM models** for comparison:
- Iterates over a list of model names
- Creates separate experiment folders per model
- Enables controlled model comparison (same path, different brains)

---

## 8. Analysis & Observability Platform

### 8.1 Error Analysis (`analysis/errors.py` — 373 lines)

Comprehensive post-run analysis:
- **Success detection** — checks if agent reached destination polygon
- **Distance metrics** — haversine to destination, distance to polygon boundary
- **Ray-casting polygon inclusion** — custom implementation of point-in-polygon test
- **403 error tracking** — counts Google API rate-limit rejections
- **Navigation status classification** — SUCCESS, COMPLETED_NO_ERROR, FAILED_403, FAILED_OTHER

### 8.2 Log Crawler (`analysis/stats/crawler.py` — 1,068 lines)

**Automated experiment discovery and aggregation:**
- Recursively traverses log directories to find experiment folders
- Identifies experiments by presence of coordinate files, API call logs
- Classifies success/failure by scanning terminal output
- Categorizes errors: LLM errors, JSON parsing errors, network errors, Playwright errors, checkpoint errors
- Detects random-choice fallbacks
- Aggregates statistics across runs: success rate, mean distance, error breakdown
- Outputs summary tables for paper-ready results

### 8.3 Advanced Metrics (`analysis/stats/advanced.py` — 332 lines)

Computes **SPL (Success weighted by Path Length)** and other advanced metrics:
- **L\* calculation** — optimal walking distance via Google Directions API
- **Path length** — total haversine distance of agent's actual trajectory
- **SPL** — `Success * (L* / max(L*, path_length))`
- **Min distance to polygon** — closest the agent ever got to the destination
- **Decision count** — total decisions made during the run
- **Polygon-aware distance** — computes distance to polygon boundary, not just centroid

### 8.4 Evaluation Recalculation (`analysis/recalculate/` — 992 lines)

**Post-hoc re-evaluation of decision quality:**
- Recomputes RIGHT/WRONG labels from saved coordinate data
- Uses cached walking distances to avoid redundant API calls
- Batch processing across all runs in a dataset
- Generates comparison charts (original vs. recalculated accuracy)
- Useful when evaluation criteria change after experiments are run

### 8.5 Web-Based Log Viewer (`analysis/viewer/` — 3,058 lines)

A **full web application** for interactive experiment visualization:

**Backend** (`server.py` — 445 lines):
- Python HTTP server serving the viewer UI
- API endpoints for listing experiments, fetching coordinates, loading decision logs
- Deep analysis integration (triggers advanced metric computation on demand)
- Recursive experiment folder discovery

**Frontend** (`index.html` — 2,613 lines):
- **Satellite map visualization** using Google Maps JS API
- **Path rendering** — polylines showing the agent's actual trajectory
- **Decision point markers** — gold markers for LLM decisions, blue for intermediate steps
- **Destination polygon overlay** — shows the target area on the map
- **Experiment browser** — hierarchical folder navigation with success/failure indicators
- **Resizable sidebar** — drag-to-resize experiment list
- **Lazy loading** — fetches data on demand as experiments are selected

---

## 9. Path Generator Tool (Full-Stack Web App)

A **complete web application** (~8,800 lines) for creating the CityNav evaluation dataset.

### 9.1 Backend Server (`tools/path_generator/src/server.py` — 2,417 lines)

A **Flask REST API** with sophisticated pathfinding:

**Pathfinding Algorithm:**
- **Depth-First Search** through the Street View panorama graph
- **Softmax temperature annealing** for heading-based direction selection (starts focused, becomes exploratory)
- **Direction bias** — can bias paths toward a compass heading for the first N decisions
- **Regression window** — detects if the path doubles back on itself and backtracks
- **Visit penalty** — exponentially penalizes revisiting the same panorama
- **Minimum final degree** — ensures the endpoint has multiple directions (is a real intersection)
- **Distance target** — generates paths of a specified length (e.g., 2km)

**Concurrency:**
- **Multi-job system** — runs multiple path generation jobs concurrently
- **Batch mode** — generates dozens of paths from a single seed with repulsion (paths avoid overlapping)
- **Thread-safe** — uses locks for shared state, per-thread environment instances

**API Endpoints:**
- `POST /api/walk/start` — begin a new path generation job
- `GET /api/walk/status/<id>` — poll job progress
- `POST /api/walk/stop/<id>` — cancel a running job
- `POST /api/walk/batch` — generate a batch of paths
- `GET /api/pano/<id>` — fetch panorama data
- `GET /api/paths/polygons` — load/save destination polygons
- `POST /api/paths/save` — save finalized paths to JSON
- `POST /api/paths/bulk_save` — batch-save multiple paths

### 9.2 Panorama Proxy (`tools/path_generator/src/proxy.py` — 524 lines)

A **FastAPI server** managing a pool of Playwright browsers for high-throughput panorama fetching:

- **Browser pool** — multiple headless Chromium instances with request queuing
- **Worker threads** — each browser runs in its own thread with a request queue
- **Health endpoint** — monitors browser pool status
- **Concurrent request handling** — distributes fetch requests across available browsers

### 9.3 Cache System (`tools/path_generator/src/cache.py` — 181 lines)

- **PanoCache** — LMDB cache for panorama data (same design as main system)
- **PathRunDB** — SQLite database for saving generated path run history

### 9.4 Frontend (`tools/path_generator/web/` — ~5,600 lines)

**A rich single-page application:**

**Main App** (`app.js` — 1,884 lines):
- Google Maps with Street View integration
- Click-to-navigate in Street View
- Real-time path visualization on the map
- Path start/end selection
- Polygon drawing and editing
- Auto-save to localStorage

**Selector** (`selector.js` — 1,525 lines):
- Batch path generation UI
- Distance/parameters configuration
- Progress monitoring for running jobs
- Path review and approval workflow

**Landmarks** (`landmarks.js` — 1,688 lines):
- Predefined landmarks for all 4 cities (New York, Tokyo, Vienna, Sao Paulo)
- Quick-jump to known locations
- City switching

**Styles** (`styles.css` — 336 lines):
- Full responsive UI styling

---

## 10. Ablation Study Framework

### `scripts/generate_ablation_configs.py` — 226 lines

An **automated ablation study configuration generator** that produces all experiment variants for the paper:

**Variant Sets:**
| Set | Description | Variants |
|-----|-------------|----------|
| **Base** | No enhancements | 1 |
| **Base+X** | Base + one component at a time | 6 (one per component) |
| **Ladder** | Incremental addition (L1→Full) | 6 |
| **Drop-one** | Full minus one component | 6 |
| **Full** | All components enabled | 1 |

**Components toggled:**
1. Self-positioning
2. Arrival heading
3. Decision history
4. Previous visit tracking
5. Intersection summary
6. Markovian memory

This generates **20 YAML config files** from a single base config, each with a different combination of features enabled/disabled. This is what produced the ablation table in the paper.

---

## 11. Key Engineering Decisions

### 11.1 Strategy Pattern for Decision-Making
The agent delegates ALL decision logic to a pluggable strategy. This clean separation enabled:
- Controlled A/B testing between baseline and AgentNav
- Ablation studies by toggling prompt components
- Easy addition of new strategies without touching core code

### 11.2 LMDB Over SQLite/Redis for Caching
- SQLite: writer-lock contention kills parallel execution
- Redis: requires a separate daemon process
- LMDB: zero-copy reads, multi-process safe, persistent, no daemon needed

### 11.3 Playwright for Street View API Access
Google doesn't expose a public API for panorama graph traversal. The system uses Playwright to:
- Load Google Maps in a headless browser
- Execute JavaScript to call internal panorama APIs
- Extract the full pano graph (IDs, coordinates, linked panos, headings)
This is significantly more robust than scraping HTML.

### 11.4 Per-Decision Atomic Checkpointing
Navigation experiments take 30-120 minutes. A single API failure, crash, or rate limit shouldn't lose an hour of work. The system checkpoints after EVERY decision with atomic writes, enabling seamless resumption.

### 11.5 File-Based Rate Limiting
When 10+ processes share a single API key, in-process rate limiting doesn't work. The system uses `fcntl.flock()` file locks to coordinate across all processes on the machine.

### 11.6 Markovian Memory for Self-Positioning
Without memory, the LLM might estimate the agent is in Tokyo one step and New York the next. Tracking the last N position estimates and including them in the prompt creates temporal consistency.

### 11.7 Dead-End Learning Across Runs
Dead ends detected in run #1 are cached persistently and shared with run #2, #3, etc. The system collectively learns the Street View graph's topology over time.

---

## 12. System Architecture Diagram (ASCII)

```
                            ┌──────────────────────────────────────────────────┐
                            │              CLI / ORCHESTRATION                 │
                            │                                                  │
                            │  run.py ─── run_batch.py ─── resume.py           │
                            │              run_by_model.py                      │
                            └──────────┬────────────────────┬──────────────────┘
                                       │                    │
                    ┌──────────────────▼──────────┐         │
                    │      SIMULATION ENGINE       │         │
                    │                              │         │
                    │  ┌────────────────────────┐  │         │
                    │  │     Simulation Loop     │  │         │
                    │  │  (checkpoints, signals, │  │         │
                    │  │   evaluation tracking)  │  │         │
                    │  └────────┬───────┬────────┘  │         │
                    │           │       │           │         │
                    │  ┌────────▼──┐ ┌──▼────────┐  │         │
                    │  │   Agent   │ │Environment │  │         │
                    │  │  (state,  │ │(Street View│  │         │
                    │  │  memory,  │ │ panoramas, │  │         │
                    │  │  history) │ │ dead-end   │  │         │
                    │  │     │     │ │ detection) │  │         │
                    │  │     │     │ │     │      │  │         │
                    │  └─────┼─────┘ └─────┼──────┘  │         │
                    │        │             │         │         │
                    └────────┼─────────────┼─────────┘         │
                             │             │                   │
              ┌──────────────▼──┐   ┌──────▼────────────┐      │
              │    STRATEGIES    │   │  INFRASTRUCTURE    │      │
              │                  │   │                    │      │
              │ ┌──────────────┐ │   │ ┌──────────────┐  │      │
              │ │   Baseline   │ │   │ │ LLM Wrapper  │  │      │
              │ │   Strategy   │ │   │ │ (LiteLLM,    │  │      │
              │ └──────┬───────┘ │   │ │  multi-model) │  │      │
              │        │         │   │ └──────┬───────┘  │      │
              │ ┌──────▼───────┐ │   │        │          │      │
              │ │  AgentNav    │ │   │ ┌──────▼───────┐  │      │
              │ │  (VoP +      │ │   │ │   LMDB       │  │      │
              │ │  loop-break) │ │   │ │   Caching     │  │      │
              │ └──────┬───────┘ │   │ │ (pano+dist)  │  │      │
              │        │         │   │ └──────────────┘  │      │
              └────────┼─────────┘   │                    │      │
                       │             │ ┌──────────────┐  │      │
              ┌────────▼─────────┐   │ │   Scoring    │  │      │
              │  SELF-POSITIONING │   │ │  (nav-score, │  │      │
              │      AGENT       │   │ │  streaks,    │  │      │
              │                  │   │ │  SPL)        │  │      │
              │ ┌──────────────┐ │   │ └──────────────┘  │      │
              │ │  Pano Grid   │ │   │                    │      │
              │ │  (spherical  │ │   │ ┌──────────────┐  │      │
              │ │   tiling)    │ │   │ │  TRAPI Auth  │  │      │
              │ └──────────────┘ │   │ │ (Azure AD)   │  │      │
              │ ┌──────────────┐ │   │ └──────────────┘  │      │
              │ │  Markovian   │ │   └────────────────────┘      │
              │ │  Memory      │ │                               │
              │ └──────────────┘ │                               │
              └──────────────────┘                               │
                                                                 │
       ┌─────────────────────────────────────────────────────────▼──┐
       │                     ANALYSIS PLATFORM                      │
       │                                                            │
       │  ┌──────────┐  ┌──────────┐  ┌──────────┐  ┌───────────┐  │
       │  │  Error    │  │   Log    │  │ Advanced │  │  Eval     │  │
       │  │ Analysis  │  │ Crawler  │  │ Metrics  │  │ Recalc   │  │
       │  └──────────┘  └──────────┘  │ (SPL)    │  └───────────┘  │
       │                              └──────────┘                  │
       │  ┌─────────────────────────────────────────────────────┐   │
       │  │           Web Log Viewer (Maps + Paths)             │   │
       │  │    Python HTTP Server + Google Maps JS Frontend     │   │
       │  └─────────────────────────────────────────────────────┘   │
       └────────────────────────────────────────────────────────────┘

       ┌────────────────────────────────────────────────────────────┐
       │              PATH GENERATOR (Full-Stack Web App)           │
       │                                                            │
       │  ┌─────────────────────┐  ┌─────────────────────────────┐  │
       │  │   Flask REST API    │  │  Frontend (Google Maps +     │  │
       │  │   (DFS pathfinding, │  │  Street View + polygon      │  │
       │  │    batch gen,       │  │  drawing + batch selector)   │  │
       │  │    job management)  │  │                              │  │
       │  └─────────┬───────────┘  └──────────────────────────────┘  │
       │            │                                                │
       │  ┌─────────▼───────────┐  ┌─────────────────────────────┐  │
       │  │  Panorama Proxy     │  │  PanoCache + PathRunDB      │  │
       │  │  (FastAPI +         │  │  (LMDB + SQLite)            │  │
       │  │   Browser Pool)     │  │                              │  │
       │  └─────────────────────┘  └──────────────────────────────┘  │
       └────────────────────────────────────────────────────────────┘

                         ┌──────────────────────┐
                         │   EXTERNAL SERVICES   │
                         │                       │
                         │  Google Street View   │
                         │  Google Directions    │
                         │  OpenAI / Gemini /    │
                         │  Anthropic / Azure /  │
                         │  Ollama               │
                         └──────────────────────┘
```

---

## 13. Data Flow Diagrams

### 13.1 Single Navigation Decision

```
Street View Pano (current position)
        │
        ▼
┌─────────────────────┐
│ Environment:         │
│ get_observation()    │──── Returns: [Link A (heading 45°), Link B (heading 180°), ...]
│ fetch_pano_data()    │
│ (Playwright → LMDB) │
└─────────┬───────────┘
          │
          ▼
┌─────────────────────────────────┐
│ Strategy: decide_with_vision()  │
│                                 │
│ 1. Fetch Street View images    │◄── Google Street View Static API
│    for each direction           │
│                                 │
│ 2. Hash images → intersection  │
│    detection                    │
│                                 │
│ 3. Check dead-end memory       │◄── LMDB PanoCache (dead edges)
│                                 │
│ 4. Build prompt:                │
│    - Images + directions        │
│    - Memory (LLM summary)      │
│    - Decision history           │
│    - Self-position estimate     │◄── Self-Positioning Agent
│    - Previous visit warnings    │
│                                 │
│ 5. Call LLM                    │◄── LiteLLM → GPT-4 / Gemini / Claude
│                                 │
│ 6. Parse JSON response          │
│    {analysis, decision, memory} │
└─────────┬───────────────────────┘
          │
          ▼
┌─────────────────────┐
│ Simulation:          │
│ apply_action()       │──── Move to chosen pano
│ save_checkpoint()    │──── Atomic write to disk
│ evaluate_decision()  │──── Walking distance Δ → RIGHT/WRONG
│ log_coordinates()    │──── Append to trajectory JSON
└──────────────────────┘
```

### 13.2 Experiment Lifecycle

```
config.yml + paths.json
        │
        ▼
   ┌─────────┐     ┌──────────────────────┐
   │ CLI:     │────►│ Experiment folder:    │
   │ run.py   │     │ logs/2025-01-15_.../ │
   └─────────┘     └──────────┬───────────┘
                              │
            ┌─────────────────┼─────────────────┐
            ▼                 ▼                 ▼
    ┌──────────────┐  ┌──────────────┐  ┌──────────────┐
    │ terminal.log │  │ coordinates  │  │ checkpoints/ │
    │ (full stdout)│  │ .json        │  │ decision_N   │
    └──────────────┘  └──────────────┘  └──────────────┘
            │                 │                 │
            ▼                 ▼                 ▼
    ┌──────────────┐  ┌──────────────┐  ┌──────────────┐
    │ openai_calls/│  │ evaluations  │  │ points.json  │
    │ call_001.json│  │ .json        │  │ (start, end, │
    │ call_002.json│  │ (RIGHT/WRONG │  │  polygon)    │
    │ ...          │  │  per decision│  │              │
    └──────────────┘  └──────────────┘  └──────────────┘
            │
            ▼
    ┌────────────────────┐
    │ Analysis Platform   │
    │ - Crawler (auto)    │
    │ - Advanced metrics  │
    │ - Web viewer        │
    └────────────────────┘
```

---


