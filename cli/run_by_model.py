#!/usr/bin/env python3
"""
Directory-driven runner for navigation experiments with a strict one-per-model concurrency cap.

Why this exists
---------------
- You generate many config YAMLs (e.g., under bulk_run_configs/...). Each file corresponds to a
  run_multi.py orchestration (fan-out over paths).
- You want to run them all, BUT never run two configs of the same model at the same time.
- This script discovers configs by directory, groups them by model_name (read from each YAML),
  and schedules them so at most one run_multi.py process per model is active concurrently.

Key properties
--------------
- Model-agnostic: no hardcoded model names. The script reads model_name from each YAML.
- Safety: continues if a config fails (non-zero exit). Failures are reported at the end.
- Control: two layers of concurrency
  1) --max-models-parallel  → how many different models to run at once
  2) --max-parallel         → passed to run_multi.py to control per-path parallelism within a run
- Isolation: each run is a separate Python process (no threading), minimizing shared-state risks.

Typical usage
-------------
    python run_dir_by_model.py bulk_run_configs --max-models-parallel 2 --max-parallel 4

This starts at most two run_multi.py processes (for two different models) at once, and each
run_multi.py will run up to four paths concurrently.

Behavior on failures
--------------------
- If an individual run_multi.py returns a non-zero exit code, this script records the failure,
  prints it at the end, and moves on to the next config. It does NOT stop the whole sequence.

Notes
-----
- Keep running this from the project root so relative paths resolve correctly.
- Logs are written per model to files named like: phase1_<model_label>.log
  (<model_label> is model_name with '/' replaced by '_').
 - Status shows: done/total, active count, queued count, active models with current cfg,
   and per-model queue sizes with the next pending cfg.
"""

from __future__ import annotations

import argparse
import collections
import glob
import os
import signal
import subprocess
import sys
import time
from typing import Dict, List, Tuple

import yaml  # PyYAML

MODEL_ORDER: List[str] = [
    "trapi/gpt-5",
    "trapi/gpt-4o",
    "trapi/gpt-4.1",
    "trapi/o3",
]

VARIANT_ORDER: List[str] = [
    "Full",
    "Base",
]

DATASET_ORDER: List[str] = [
    "Tokyo_2",
    "Vienna_2",
    "Sao_Paulo_2",
    "New_York",
    "Tokyo",
    "Vienna",
    "Sao_Paulo",
]

_MODEL_PRIORITY = {name: idx for idx, name in enumerate(MODEL_ORDER)}
_VARIANT_PRIORITY = {name: idx for idx, name in enumerate(VARIANT_ORDER)}
_DATASET_PRIORITY = {name: idx for idx, name in enumerate(DATASET_ORDER)}


def _priority(value: str, table: Dict[str, int]) -> int:
    return table.get(value, len(table))


def _model_sort_key(model_name: str) -> Tuple[int, str]:
    return (_priority(model_name, _MODEL_PRIORITY), model_name)


def _cfg_sort_key(cfg_path: str, root_root: str) -> Tuple[int, int, str]:
    try:
        rel = os.path.relpath(cfg_path, root_root)
    except Exception:
        rel = cfg_path
    parts = rel.split(os.sep)
    dataset = parts[0] if parts else ""
    variant = os.path.splitext(os.path.basename(cfg_path))[0]
    dataset_rank = _priority(dataset, _DATASET_PRIORITY)
    variant_rank = _priority(variant, _VARIANT_PRIORITY)
    return (dataset_rank, variant_rank, cfg_path)


def _find_config_paths(root_dir: str) -> List[str]:
    """Find all config files under root_dir.

    - Includes .yml and .yaml
    - Returns a stable, lexicographically sorted list
    """
    patterns = [os.path.join(root_dir, "**", "*.yml"), os.path.join(root_dir, "**", "*.yaml")]
    paths: List[str] = []
    for pat in patterns:
        paths.extend(glob.glob(pat, recursive=True))
    # Deduplicate and sort for determinism
    return sorted(dict.fromkeys(paths))


def _read_model_name(yaml_path: str) -> str:
    """Read model_name from a YAML file.

    Raises ValueError if the key is missing or invalid.
    """
    with open(yaml_path, "r", encoding="utf-8") as f:
        data = yaml.safe_load(f) or {}
    model_name = data.get("model_name")
    if not isinstance(model_name, str) or not model_name.strip():
        raise ValueError(f"model_name missing or invalid in {yaml_path}")
    return model_name


def _safe_label(value: str) -> str:
    """Make a value safe for filenames (e.g., replace '/')."""
    return str(value).replace("/", "_")


def _run_multi_py_path() -> str:
    """Return an absolute path to run_batch.py (assumed to be alongside this script)."""
    return os.path.join(os.path.dirname(os.path.abspath(__file__)), "run_batch.py")


def _print(msg: str) -> None:
    """Print a timestamped, flushed line to stdout."""
    now = time.strftime("%FT%T")
    print(f"[{now}] {msg}", flush=True)


def _status_string(
    active: Dict[str, Tuple[subprocess.Popen, "os.PathLike[str] | int | None", str]],
    queues: Dict[str, List[str]],
    done_count: int,
    total_count: int,
    max_models_parallel: int,
) -> str:
    """Build a concise status line with overall and per-model info.

    - done_count: incremented when a run_multi.py process exits (success or failure)
    - total_count: fixed total number of configs discovered at start
    - active: model -> (proc, log_file_handle, current_cfg)
    - queues: model -> list of remaining cfg paths
    """
    active_models = sorted(active.keys(), key=_model_sort_key)
    queued_counts = {m: len(q) for m, q in queues.items() if len(q) > 0}
    queued_total = sum(queued_counts.values())

    # Per-model details: active model current cfg, queued count and next cfg
    per_model_parts: List[str] = []
    for m in sorted(set(active_models) | set(queued_counts.keys()), key=_model_sort_key):
        label = _safe_label(m)
        if m in active:
            _, _, cfg = active[m]
            cur = os.path.basename(cfg)
        else:
            cur = "-"
        qn = len(queues.get(m, []))
        nxt = os.path.basename(queues[m][0]) if qn > 0 else "-"
        per_model_parts.append(f"{label}:cur={cur},q={qn},next={nxt}")

    parts = [
        f"done={done_count}/{total_count}",
        f"active={len(active_models)}/{max_models_parallel}",
        f"queued={queued_total}",
    ]
    if per_model_parts:
        parts.append("[" + "; ".join(per_model_parts) + "]")
    return "STATUS " + " | ".join(parts)


def _format_cfg_label(cfg_path: str, root_root: str) -> str:
    """Human-readable label: <DATASET> <MODEL> <VARIANT>, all uppercase.

    - DATASET derived from first path segment under root_root
    - MODEL derived from second path segment (strip 'trapi_' prefix, replace '_' and '-' with spaces)
    - VARIANT is the filename stem (e.g., Base, Full)
    """
    try:
        rel = os.path.relpath(cfg_path, root_root)
    except Exception:
        rel = cfg_path
    parts = rel.split(os.sep)
    dataset = parts[0] if len(parts) > 0 else ""
    model_dir = parts[1] if len(parts) > 1 else os.path.basename(os.path.dirname(cfg_path))
    variant = os.path.splitext(os.path.basename(cfg_path))[0]

    def up(s: str) -> str:
        return s.replace("_", " ").upper()

    model_disp = model_dir
    if model_disp.startswith("trapi_"):
        model_disp = model_disp[len("trapi_"):]
    model_disp = model_disp.replace("_", " ").replace("-", " ")

    return f"{up(dataset)} {model_disp.upper()} {variant.upper()}"


def _print_queue_lines(
    active: Dict[str, Tuple[subprocess.Popen, "os.PathLike[str] | int | None", str]],
    queues: Dict[str, List[str]],
    root_root: str,
) -> None:
    """Print the full ACTIVE and QUEUE with one line per config (human-readable)."""
    # ACTIVE
    _print("ACTIVE:")
    if not active:
        _print("(none)")
    else:
        for model in sorted(active.keys(), key=_model_sort_key):
            _, _, cfg = active[model]
            _print(_format_cfg_label(cfg, root_root))

    # QUEUE
    _print("QUEUE:")
    any_queued = False
    for model in sorted(queues.keys(), key=_model_sort_key):
        for cfg in queues[model]:
            any_queued = True
            _print(_format_cfg_label(cfg, root_root))
    if not any_queued:
        _print("(empty)")


def main(argv: List[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description=(
            "Run all config YAMLs in a directory, enforcing at most one concurrent run per model.\n"
            "Each config is executed via run_multi.py, with --max-parallel forwarded."
        )
    )
    parser.add_argument(
        "configs_root",
        help="Directory containing generated config YAMLs (searched recursively)",
    )
    parser.add_argument(
        "--max-models-parallel",
        type=int,
        default=2,
        help=(
            "Maximum number of different models to run at once. "
            "One run_multi.py process per model, never two of the same model concurrently."
        ),
    )
    parser.add_argument(
        "--max-parallel",
        type=int,
        default=4,
        help=(
            "run_multi.py --max-parallel (paths per config). "
            "This controls intra-run concurrency, independent of model-level concurrency."
        ),
    )
    parser.add_argument(
        "--poll-interval",
        type=float,
        default=0.5,
        help="Seconds between polling active processes.",
    )
    parser.add_argument(
        "--status-interval",
        type=float,
        default=30.0,
        help="How often (sec) to print a status line with active models and queue sizes.",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Print the planned run order and exit without executing anything.",
    )

    args = parser.parse_args(argv)

    # Discover configs and group them by model
    cfg_paths = _find_config_paths(args.configs_root)
    if not cfg_paths:
        _print(f"No YAML configs found under: {args.configs_root}")
        return 0

    model_to_configs: Dict[str, List[str]] = collections.defaultdict(list)
    for cfg in cfg_paths:
        try:
            mname = _read_model_name(cfg)
            model_to_configs[mname].append(cfg)
        except Exception as exc:  # Invalid/missing model_name → skip
            _print(f"[skip] {cfg}: {exc}")

    # Deterministic order per model, with preference: "Full" before "Base" when both exist.
    # We sort by a tuple: (priority, path) where priority=0 if filename contains '/Full.yml', else 1.
    for m in model_to_configs:
        model_to_configs[m].sort(key=lambda p: _cfg_sort_key(p, args.configs_root))

    # Optional dry run: just show the plan
    if args.dry_run:
        _print("Dry run (no execution):")
        for m in sorted(model_to_configs, key=_model_sort_key):
            _print(f"- Model: {m}  ({len(model_to_configs[m])} configs)")
            for c in sorted(model_to_configs[m], key=lambda p: _cfg_sort_key(p, args.configs_root)):
                _print(f"  {c}")
        return 0

    # Prepare scheduling structures
    queues: Dict[str, List[str]] = {}
    for m, paths in model_to_configs.items():
        queues[m] = list(sorted(paths, key=lambda p: _cfg_sort_key(p, args.configs_root)))
    active: Dict[str, Tuple[subprocess.Popen, "os.PathLike[str] | int | None", str]] = {}
    failures: List[Tuple[str, int]] = []
    terminated = False
    total_configs = sum(len(v) for v in model_to_configs.values())
    done_count = 0  # Counts finished processes regardless of success/failure
    configs_root_abs = os.path.abspath(args.configs_root)

    # Resolve run_multi.py absolute path once
    run_multi = _run_multi_py_path()

    def launch_next(model_name: str) -> bool:
        """Launch the next config for a model (if any). Returns True if launched."""
        if not queues.get(model_name):
            return False
        cfg = queues[model_name].pop(0)
        log_path = f"phase1_{_safe_label(model_name)}.log"
        try:
            log_file = open(log_path, "a", encoding="utf-8")
        except Exception:
            # Fallback to stdout if log file can't be opened
            log_file = None  # type: ignore[assignment]

        cmd = [sys.executable, run_multi, "--config", cfg, "--max-parallel", str(args.max_parallel)]
        _print(f"START {cfg}")
        try:
            proc = subprocess.Popen(
                cmd,
                stdout=log_file if log_file is not None else None,
                stderr=subprocess.STDOUT if log_file is not None else None,
            )
        except Exception as exc:
            _print(f"[error] Failed to launch {cfg}: {exc}")
            if log_file is not None:
                try:
                    log_file.close()
                except Exception:
                    pass
            failures.append((cfg, -1))
            return False

        active[model_name] = (proc, log_file, cfg)
        return True

    def handle_signal(signum, frame):  # noqa: ANN001 - handler signature fixed by signal
        nonlocal terminated
        if terminated:
            return
        terminated = True
        _print(f"Received signal {signum}. Terminating active child processes...")
        for m in list(active.keys()):
            proc, log_f, _ = active[m]
            try:
                proc.terminate()
            except Exception:
                pass
        time.sleep(2.0)
        for m in list(active.keys()):
            proc, log_f, _ = active[m]
            if proc.poll() is None:
                try:
                    proc.kill()
                except Exception:
                    pass

    # Register signal handlers to clean up children on Ctrl+C or SIGTERM
    signal.signal(signal.SIGINT, handle_signal)
    signal.signal(signal.SIGTERM, handle_signal)

    # Main scheduling loop
    last_status = 0.0
    while (any(queues.values()) or active) and not terminated:
        # Fill available model slots (at most one active per model, global cap honored)
        for model in sorted(queues.keys()):
            if terminated:
                break
            if model in active:
                continue  # already running one for this model
            if len(active) >= args.max_models_parallel:
                break  # wait until a slot frees up
            if queues[model]:
                launch_next(model)

        # Poll active processes
        time.sleep(args.poll_interval)
        for model in list(active.keys()):
            proc, log_f, cfg = active[model]
            rc = proc.poll()
            if rc is None:
                continue
            _print(f"END   {cfg} (rc={rc})")
            # Close log and remove from active
            if log_f is not None:
                try:
                    log_f.flush()
                    log_f.close()
                except Exception:
                    pass
            del active[model]
            done_count += 1  # Count as done regardless of rc
            if rc != 0:
                failures.append((cfg, rc))
            # Keep the model busy by launching the next config immediately (if any)
            if queues.get(model):
                launch_next(model)

            # Immediate status update after each completion
            _print(_status_string(active, queues, done_count, total_configs, args.max_models_parallel))
            _print_queue_lines(active, queues, configs_root_abs)

        # Periodic status line for quick monitoring
        now = time.time()
        if now - last_status >= args.status_interval:
            _print(_status_string(active, queues, done_count, total_configs, args.max_models_parallel))
            _print_queue_lines(active, queues, configs_root_abs)
            last_status = now

    # If terminated by signal, try to wait a moment for children to exit
    if terminated:
        time.sleep(0.5)

    # Final report
    if failures:
        _print("Some configs failed:")
        for cfg, rc in failures:
            _print(f"- {cfg}: rc={rc}")
        return 1

    _print("All configs completed.")
    return 0


if __name__ == "__main__":
    sys.exit(main())


