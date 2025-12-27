# CityNav: City Navigation in the Wild

[![Python 3.10+](https://img.shields.io/badge/python-3.10+-blue.svg)](https://www.python.org/downloads/)
[![Platform](https://img.shields.io/badge/platform-macOS%20%7C%20Linux-lightgrey.svg)](https://github.com)

Official implementation for **"City Navigation in the Wild: Exploring Emergent Navigation from Web-Scale Knowledge in MLLMs"**.

**[Project Page](https://dwipddalal.github.io/AgentNav/)** | **[Paper (arXiv)]()** | **[Dataset]()** 

<!-- See project page for figures: https://dwipddalal.github.io/AgentNav/ -->

---

## Overview

**CityNav** introduces the task of *Sparsely Grounded Visual Navigation*: navigating real-world cities over long distances (~2km, 50+ decision points) using only intersection images—without any landmark annotations, natural language instructions, or GPS.

We propose **AgentNav**, a zero-shot framework that leverages *Verbalization of Path (VoP)* to elicit structured world knowledge from MLLMs, achieving **92% success rate** in New York City with GPT-4.1.

### Key Features

- **Real-world evaluation** on Google Street View across 4 diverse cities
- **Long-range navigation**: ~2km paths with 50+ sequential decisions
- **Zero-shot**: No fine-tuning required—uses MLLM's internal world knowledge
- **Robust platform**: Handles dead ends, missing links, and Street View inconsistencies

---

## Task Definition

> **Sparsely Grounded Visual Navigation**: An agent must navigate from a start location to a destination using only visual observations at intersections. The agent receives no map, GPS, landmark labels, or navigation instructions—it must leverage its internal world knowledge to self-localize and plan routes.

At each intersection, the agent:
1. Observes images of available street directions
2. Reasons about its current location and route to the destination
3. Selects which direction to proceed

<img width="1186" height="608" alt="image" src="https://github.com/user-attachments/assets/a9400e62-432d-4148-967f-e0c9eae58284" />


---

## AgentNav Method

AgentNav enhances MLLM-based navigation through two key components:

### Verbalization of Path (VoP)

VoP explicitly extracts the agent's latent world knowledge by prompting it to verbalize:

1. **Destination location**: The exact address/location of the goal
2. **Current estimated location**: The agent's best estimate of where it is now
3. **Walking directions**: Step-by-step route from current position to destination

This structured verbalization grounds the agent's reasoning in real-world spatial relationships.

### Memory Architecture

For efficient long-horizon reasoning (50+ decisions), AgentNav uses three memory components:

| Component | Description |
|-----------|-------------|
| **Markovian Memory** | Agent produces a memory state at each step, summarizing relevant past observations |
| **Decision History** | Ordered sequence of past actions for trajectory awareness |
| **Previous Visit Tracking** | Detects loops by tracking visit counts at each intersection |

---

## Dataset: CityNav Benchmark

| City | Region | Characteristics | Avg. Distance | Avg. Decisions |
|------|--------|-----------------|---------------|----------------|
| **New York** | USA | Grid-based, rich signage | 1.8 km | 44 |
| **São Paulo** | Brazil | Non-block structure, Portuguese | 2.0 km | 55 |
| **Tokyo** | Japan | Narrow alleys, Japanese script | 1.9 km | 80 |
| **Vienna** | Austria | Rail crossings, German signage | 2.1 km | 60 |

Each city contains **100 origin-destination pairs** with manually annotated destination polygons.

---

## Results

### Main Results (Success Rate %)

| MLLM | Config | New York | Tokyo | Vienna | São Paulo |
|------|--------|----------|-------|--------|-----------|
| GPT-5 | AgentNav | **94** | **30** | **56** | **29** |
| GPT-5 | Base | 54 | 10 | 11 | 7 |
| GPT-4.1 | AgentNav | **92** | 17 | 32 | 22 |
| GPT-4.1 | Base | 15 | 5 | 2 | 5 |
| Gemini 2.5 Flash | AgentNav | **73** | 17 | 17 | 12 |
| Gemini 2.5 Flash | Base | 12 | 8 | 1 | 5 |

### Ablation Study (New York, GPT-4.1)

| Method | Success | SPL |
|--------|---------|-----|
| Base GPT-4.1 | 15% | 0.097 |
| + Markovian Memory | 23% | 0.162 |
| + Decision History | 29% | 0.228 |
| + Previous Visit | 35% | 0.298 |
| + Partial VoP | 66% | 0.469 |
| **Full AgentNav** | **92%** | **0.557** |

---

## Installation

### Requirements

- Python 3.10+
- macOS or Linux (uses Unix file locking)

### Setup

```bash
# Clone the repository
git clone https://github.com/your-org/citynav.git
cd citynav

# Install dependencies
pip install -r requirements.txt

# Install Playwright browser
playwright install chromium

# Set up API keys
export GOOGLE_MAPS_API_KEY="your-google-maps-key"
export GOOGLE_GEMINI_API_KEY="your-gemini-key"  # or OPENAI_API_KEY, etc.

# Create config from template
cp config.example.yml config.yml
```

---

## Quick Start

### Run a Single Navigation

```bash
python cli/run.py config.yml
```

### Run Batch Experiments

```bash
# Parallel execution across multiple paths
python cli/run_batch.py config.yml

# Resume incomplete runs
python cli/resume.py logs/session_folder/
```

### View Results

```bash
# Launch the log viewer web UI
python -m analysis.viewer.server logs/ --port 8000
```

Then open `http://localhost:8000` in your browser.

---

## Configuration

Edit `config.yml` to configure experiments:

```yaml
# Dataset
paths_file: "data/paths/New_York.json"
prompts_file: "data/prompts/Prompt.txt"

# Model (supports Gemini, OpenAI, Anthropic, Ollama)
model_name: "gemini/gemini-2.5-flash"
# model_name: "openai/gpt-4o"

# Strategy
strategy_module: "strategies.memory_strategy"  # AgentNav
# strategy_mode: baseline                       # Base MLLM

# Simulation limits
max_steps: 2000
max_decision_points: 150

# Termination: "distance", "polygon", or "both"
termination_criteria: "polygon"
```

### Environment Variables

```bash
# Required
export GOOGLE_MAPS_API_KEY="..."

# LLM Provider (set one)
export GOOGLE_GEMINI_API_KEY="..."
export OPENAI_API_KEY="..."
export ANTHROPIC_API_KEY="..."
```

---

## Project Structure

```
.
├── core/                   # Simulation engine
│   ├── simulation.py       # Main simulation loop
│   ├── environment.py      # Google Street View environment
│   └── agent.py            # Navigation agent
├── strategies/             # Navigation strategies
│   ├── memory_strategy.py  # AgentNav (VoP + Memory)
│   ├── cot_strategy.py     # Chain-of-Thought variant
│   └── baseline.py         # Base MLLM comparison
├── infrastructure/         # LLM wrapper, caching, scoring
├── cli/                    # Command-line entry points
├── analysis/               # Log analysis and visualization
├── data/
│   ├── paths/              # Route definitions (4 cities)
│   └── prompts/            # System prompts
└── docs/                   # Documentation
```

See [`docs/STRUCTURE.md`](docs/STRUCTURE.md) for detailed documentation.

---

## Evaluation Metrics

| Metric | Description |
|--------|-------------|
| **Success** | Binary: 1 if agent reaches destination polygon, 0 otherwise |
| **SPL** | Success weighted by Path Length: `Success × (optimal_dist / actual_dist)` |
| **Decision Accuracy** | % of decisions that reduce walking distance to destination |

---

## Citation

```bibtex
@article{dalal2025city,
  title={City Navigation in the Wild: Exploring Emergent Navigation from Web-Scale Knowledge in MLLMs},
  author={Dalal, Dwip and Mishra, Utkarsh and Ahuja, Narendra and Jojic, Nebojsa},
  journal={arXiv preprint arXiv:2512.15933},
  year={2025}
}
```
