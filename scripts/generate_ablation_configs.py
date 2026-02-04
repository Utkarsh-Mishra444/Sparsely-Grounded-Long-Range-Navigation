#!/usr/bin/env python3
import argparse
import os
import sys
from typing import Dict, List, Tuple

try:
    import yaml  # type: ignore
except Exception:
    print("This script requires PyYAML. Install with: pip install pyyaml", file=sys.stderr)
    sys.exit(1)


# ------------------------------
# Ablation variant definitions
# ------------------------------

def _variant_dict(name: str,
                  include_self_positioning: bool = False,
                  include_arrival_heading: bool = False,
                  include_decision_history: bool = False,
                  include_previous_visits: bool = False,
                  include_intersection_summary: bool = False,
                  include_memory: bool = False) -> Dict:
    return {
        "name": name,
        "prompt_components": {
            "include_self_positioning": include_self_positioning,
            "include_arrival_heading": include_arrival_heading,
            "include_decision_history": include_decision_history,
            "include_previous_visits": include_previous_visits,
            "include_intersection_summary": include_intersection_summary,
            "include_memory": include_memory,
        },
        # Strategy flag also needs to match for self-positioning to actually run
        "use_self_positioning": include_self_positioning,
    }


def list_variants(which: str = "all") -> List[Dict]:
    """Return the ablation variants to generate based on a set selector.

    Selectors:
      - all: Base + Single-component + Ladder + Drop-one + Full
      - base: only Base
      - base_plus: Base+X set
      - ladder: L1..L5 + Full
      - drop: Full - X set
    """
    base = [
        _variant_dict("Base")
    ]

    base_plus = [
        _variant_dict("Base+Arrival", include_arrival_heading=True),
        _variant_dict("Base+DecHistory", include_decision_history=True),
        _variant_dict("Base+PrevVisits", include_previous_visits=True),
        _variant_dict("Base+InterSummary", include_intersection_summary=True),
        _variant_dict("Base+Memory", include_memory=True),
        _variant_dict("Base+SelfPos", include_self_positioning=True),
    ]

    ladder = [
        _variant_dict("L1+Arrival", include_arrival_heading=True),
        _variant_dict("L2+DecisionHistory", include_arrival_heading=True, include_decision_history=True),
        _variant_dict("L3+PrevVisits", include_arrival_heading=True, include_decision_history=True, include_previous_visits=True),
        _variant_dict("L4+InterSummary", include_arrival_heading=True, include_decision_history=True, include_previous_visits=True, include_intersection_summary=True),
        _variant_dict("L5+Memory", include_arrival_heading=True, include_decision_history=True, include_previous_visits=True, include_intersection_summary=True, include_memory=True),
        _variant_dict("Full", include_arrival_heading=True, include_decision_history=True, include_previous_visits=True, include_intersection_summary=True, include_memory=True, include_self_positioning=True),
    ]

    drop = [
        _variant_dict("Full-Arrival", include_decision_history=True, include_previous_visits=True, include_intersection_summary=True, include_memory=True, include_self_positioning=True),
        _variant_dict("Full-DecHistory", include_arrival_heading=True, include_previous_visits=True, include_intersection_summary=True, include_memory=True, include_self_positioning=True),
        _variant_dict("Full-PrevVisits", include_arrival_heading=True, include_decision_history=True, include_intersection_summary=True, include_memory=True, include_self_positioning=True),
        _variant_dict("Full-InterSummary", include_arrival_heading=True, include_decision_history=True, include_previous_visits=True, include_memory=True, include_self_positioning=True),
        _variant_dict("Full-Memory", include_arrival_heading=True, include_decision_history=True, include_previous_visits=True, include_intersection_summary=True, include_self_positioning=True),
        _variant_dict("Full-SelfPos", include_arrival_heading=True, include_decision_history=True, include_previous_visits=True, include_intersection_summary=True, include_memory=True),
    ]

    if which == "all":
        return base + base_plus + ladder + drop
    if which == "base":
        return base
    if which == "base_plus":
        return base_plus
    if which == "ladder":
        return ladder
    if which == "drop":
        return drop

    # Comma-separated names subset
    names = {n.strip() for n in which.split(",") if n.strip()}
    selected = []
    for v in base + base_plus + ladder + drop:
        if v["name"] in names:
            selected.append(v)
    if not selected:
        raise SystemExit(f"No matching variants for selector '{which}'.")
    return selected


# ------------------------------
# Helpers
# ------------------------------

def _read_text_lines(path: str) -> List[str]:
    try:
        with open(path, "r", encoding="utf-8") as f:
            return [ln.strip() for ln in f.readlines() if ln.strip()]
    except FileNotFoundError:
        return []


def _load_base_config(path: str) -> Dict:
    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def _safe_name(value: str) -> str:
    return value.replace(" ", "_").replace("/", "_")


def _paths_label(paths_file: str) -> str:
    base = os.path.splitext(os.path.basename(paths_file))[0]
    return _safe_name(base)


def _model_label(model_name: str) -> str:
    return _safe_name(model_name)


def _resolve_prompts(repo_root: str) -> Tuple[str, str]:
    basic = os.path.join(repo_root, "prompts", "basic.txt")
    # Prefer Final_Prompt_Plus.txt if it exists, otherwise fall back to Final_Prompt.txt
    full_plus = os.path.join(repo_root, "prompts", "Final_Prompt_Plus.txt")
    full = os.path.join(repo_root, "prompts", "Final_Prompt.txt")
    if os.path.exists(full_plus):
        full = full_plus
    return os.path.abspath(basic), os.path.abspath(full)


def _decide_prompt_file(variant_name: str, basic_path: str, full_path: str) -> str:
    # Use minimal prompt for Base and Base+X, full prompt otherwise
    if variant_name == "Base" or variant_name.startswith("Base+"):
        return basic_path
    return full_path


def _write_yaml(path: str, data: Dict) -> None:
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        yaml.safe_dump(data, f, sort_keys=False)


# ------------------------------
# Main
# ------------------------------

def main() -> int:
    parser = argparse.ArgumentParser(description="Generate ablation config YAMLs for run_multi.py")
    parser.add_argument("--base-config", default=os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "config.yml"), help="Path to base config.yml")
    parser.add_argument("--outdir", default=os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "bulk_run_configs"), help="Output folder for generated YAMLs")
    parser.add_argument("--sets", default="all", help="Which ablation set(s) to generate: all|base|base_plus|ladder|drop or comma-separated variant names")
    parser.add_argument("--models", default=None, help="Comma-separated model names (overrides --models-file if provided)")
    parser.add_argument("--models-file", default=os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "bulk_run_configs", "default_models.txt"), help="File with one model per line")
    parser.add_argument("--paths-file", default=None, help="Override paths_file from base config")
    args = parser.parse_args()

    repo_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

    # Load base config
    base_cfg = _load_base_config(args.base_config)
    if not isinstance(base_cfg, dict):
        raise SystemExit("Base config is not a YAML mapping")

    # Resolve dataset label
    paths_file = args.paths_file or base_cfg.get("paths_file")
    if not paths_file:
        raise SystemExit("Base config must include 'paths_file' or pass --paths-file")
    dataset_label = _paths_label(str(paths_file))

    # Resolve prompts
    basic_prompt, full_prompt = _resolve_prompts(repo_root)

    # Build variants
    variants = list_variants(args.sets)

    # Collect models
    if args.models:
        models = [m.strip() for m in args.models.split(",") if m.strip()]
    else:
        models = _read_text_lines(args.models_file) or [str(base_cfg.get("model_name", "trapi/gpt-4o"))]

    # Generate
    created = 0
    for model_name in models:
        model_dir = os.path.join(args.outdir, dataset_label, _model_label(model_name))
        for v in variants:
            out_cfg = dict(base_cfg)  # shallow copy is fine
            # Ensure the selected dataset file is embedded in each generated config
            out_cfg["paths_file"] = str(paths_file)
            out_cfg["model_name"] = model_name
            out_cfg["prompts_file"] = _decide_prompt_file(v["name"], basic_prompt, full_prompt)
            out_cfg["prompt_components"] = v["prompt_components"]
            out_cfg["use_self_positioning"] = bool(v.get("use_self_positioning", False))
            # Keep other base fields intact (maps_api_key, max_steps, etc.)

            dest_path = os.path.join(model_dir, f"{v['name']}.yml")
            _write_yaml(dest_path, out_cfg)
            created += 1

    print(f"Generated {created} config files under: {os.path.abspath(args.outdir)}")
    print(f"Dataset: {dataset_label}")
    print("Models:")
    for m in models:
        print(f" - {m}")
    print("Variants:")
    for v in variants:
        print(f" - {v['name']}")

    return 0


if __name__ == "__main__":
    sys.exit(main())
