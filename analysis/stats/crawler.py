#!/usr/bin/env python3
import os
import sys
import json
import argparse
import re
import glob
import statistics
from typing import Dict, List, Tuple, Any, Optional, DefaultDict
import yaml
try:
    from analysis.stats.advanced import compute_advanced_metrics
except ImportError:
    compute_advanced_metrics = None  # type: ignore


SUCCESS_PATTERNS = [
    "Reached within 50 meters of destination after",
    "Reached destination polygon!"
]

# Error categorization regexes (best-effort; tolerant to varied logs)
ERROR_CATEGORIES: List[Tuple[str, re.Pattern]] = [
    ("llm_api_error", re.compile(r"LLM API error", re.IGNORECASE)),
    ("decision_parsing_error", re.compile(r"Decision parsing error", re.IGNORECASE)),
    ("simulation_loop_error", re.compile(r"ERROR during simulation loop step", re.IGNORECASE)),
    ("checkpoint_error", re.compile(r"ERROR saving checkpoint|Checkpoint save failed|ERROR saving final coordinates", re.IGNORECASE)),
    ("checkpoint_load_error", re.compile(r"ERROR loading checkpoint", re.IGNORECASE)),
    ("playwright_error", re.compile(r"ERROR re-initializing Playwright|Playwright", re.IGNORECASE)),
    ("env_sync_error", re.compile(r"ERROR synchronizing environment state", re.IGNORECASE)),
    ("image_fetch_error", re.compile(r"Failed to fetch image|requests\.RequestException", re.IGNORECASE)),
    ("json_decode_error", re.compile(r"JSONDecodeError|Expecting value: line", re.IGNORECASE)),
    ("network_error", re.compile(r"Connection reset|ECONNRESET|timed out|Timeout|502 Bad Gateway|429 Too Many Requests", re.IGNORECASE)),
]

# Heuristic signals for random fallbacks
RANDOM_CHOICE_SIGNALS: List[re.Pattern] = [
    re.compile(r"chose randomly", re.IGNORECASE),
    re.compile(r"defaulting to random choice", re.IGNORECASE),
]


def is_experiment_folder(folder_path: str) -> bool:
    """
    Determine if a directory represents an experiment.
    Mirrors logic in logs/server.py: presence of a file starting with
    'visited_coordinates_' or presence of 'openai_calls'/'gemini_calls' subdirs.
    """
    if not os.path.isdir(folder_path):
        return False

    try:
        entries = os.listdir(folder_path)
    except Exception:
        return False

    has_visited = any(name.startswith("visited_coordinates_") or name == "visited_coordinates.json" for name in entries)
    has_api_calls = (
        os.path.exists(os.path.join(folder_path, "openai_calls"))
        or os.path.exists(os.path.join(folder_path, "gemini_calls"))
        or os.path.exists(os.path.join(folder_path, "self_position_calls"))
    )
    return has_visited or has_api_calls


def is_successful_experiment(folder_path: str) -> bool:
    """
    Check if an experiment was successful by scanning tail of terminal logs
    for success messages. Success is indicated by either:
    - "Reached within 50 meters of destination after"
    - "Reached destination polygon!"
    Mirrors logic in logs/server.py.
    """
    if not os.path.isdir(folder_path):
        return False

    try:
        entries = os.listdir(folder_path)
    except Exception:
        return False

    log_files = [
        f for f in entries if f.startswith("terminal_output_") and f.endswith(".log")
    ]

    for log_file in log_files:
        log_path = os.path.join(folder_path, log_file)
        try:
            with open(log_path, "r", encoding="utf-8", errors="ignore") as f:
                lines = f.readlines()
            tail = lines[-50:] if len(lines) > 50 else lines
            if any(any(pattern in line for pattern in SUCCESS_PATTERNS) for line in tail):
                return True
        except Exception:
            # Ignore unreadable files but continue searching others
            continue

    return False


def _read_text_file(path: str) -> List[str]:
    try:
        with open(path, "r", encoding="utf-8", errors="ignore") as f:
            return f.readlines()
    except Exception:
        return []


def _safe_json_load(path: str) -> Optional[Any]:
    try:
        with open(path, "r", encoding="utf-8", errors="ignore") as f:
            return json.load(f)
    except Exception:
        return None


def _percentiles(values: List[float], ps: List[float]) -> Dict[str, float]:
    if not values:
        return {f"p{int(p*100)}": 0.0 for p in ps}
    sorted_vals = sorted(values)
    out: Dict[str, float] = {}
    for p in ps:
        idx = min(len(sorted_vals) - 1, max(0, int(round(p * (len(sorted_vals) - 1)))))
        out[f"p{int(p*100)}"] = float(sorted_vals[idx])
    return out


def _sum_usage(usage: Any) -> Dict[str, int]:
    # Robustly sum token usage regardless of provider naming
    totals = {
        "input_tokens": 0,
        "output_tokens": 0,
        "total_tokens": 0,
    }
    if not isinstance(usage, dict):
        return totals
    # Common key variants
    input_keys = ["prompt_tokens", "input_tokens", "prompt", "input"]
    output_keys = ["completion_tokens", "output_tokens", "completion", "output"]
    total_keys = ["total_tokens", "total"]
    def _acc(keys: List[str]) -> int:
        for k in keys:
            if k in usage and isinstance(usage[k], (int, float)):
                return int(usage[k])
        return 0
    totals["input_tokens"] = _acc(input_keys)
    totals["output_tokens"] = _acc(output_keys)
    totals["total_tokens"] = _acc(total_keys) or (totals["input_tokens"] + totals["output_tokens"]) or 0
    return totals


def _categorize_line(line: str) -> Optional[str]:
    for name, pat in ERROR_CATEGORIES:
        if pat.search(line):
            return name
    return None


def _extract_exception_type_from_trace(lines: List[str], start_idx: int) -> Optional[str]:
    # Look forward from 'Traceback' to find the exception summary line
    for i in range(start_idx + 1, min(len(lines), start_idx + 20)):
        m = re.search(r"^([\w\.]+):", lines[i].strip())
        if m:
            return m.group(1)
    return None


def analyze_terminal_logs(exp_dir: str) -> Dict[str, Any]:
    entries = []
    try:
        entries = os.listdir(exp_dir)
    except Exception:
        pass

    log_files = [
        os.path.join(exp_dir, f)
        for f in entries
        if f.startswith("terminal_output_") and f.endswith(".log")
    ]

    errors: DefaultDict[str, int] = DefaultDict(int)
    exceptions: DefaultDict[str, int] = DefaultDict(int)
    last_distance: Optional[float] = None
    random_choice_count = 0
    total_steps_seen = 0

    for lp in log_files:
        lines = _read_text_file(lp)
        for idx, line in enumerate(lines):
            cat = _categorize_line(line)
            if cat:
                errors[cat] += 1

            if "Traceback (most recent call last):" in line:
                ex_type = _extract_exception_type_from_trace(lines, idx)
                if ex_type:
                    exceptions[ex_type] += 1

            m = re.search(r"Distance to destination: ([0-9]+\.?[0-9]*) m", line)
            if m:
                try:
                    last_distance = float(m.group(1))
                except Exception:
                    pass

            # Count steps by markers
            if re.search(r"\b--- Step\s+([0-9]+)\b", line):
                total_steps_seen += 1

            # Random fallback hints (rare in terminal, but keep heuristic)
            if any(p.search(line) for p in RANDOM_CHOICE_SIGNALS):
                random_choice_count += 1

    return {
        "files": log_files,
        "error_counts": dict(errors),
        "exception_counts": dict(exceptions),
        "last_reported_distance_m": last_distance,
        "random_choice_signals": random_choice_count,
        "approx_step_markers": total_steps_seen,
    }


def analyze_strategy_logs(exp_dir: str) -> Dict[str, Any]:
    strategy_log = os.path.join(exp_dir, "strategy.log")
    error_log = os.path.join(exp_dir, "error.log")
    errors: DefaultDict[str, int] = DefaultDict(int)
    examples: Dict[str, str] = {}

    for p in [strategy_log, error_log]:
        if os.path.exists(p):
            lines = _read_text_file(p)
            for line in lines:
                cat = _categorize_line(line)
                if cat:
                    errors[cat] += 1
                    if cat not in examples:
                        examples[cat] = line.strip()[:300]

    return {
        "present": os.path.exists(strategy_log) or os.path.exists(error_log),
        "files": [p for p in [strategy_log, error_log] if os.path.exists(p)],
        "error_counts": dict(errors),
        "examples": examples,
    }


def analyze_decision_calls(exp_dir: str) -> Dict[str, Any]:
    calls_dirs = [d for d in [
        os.path.join(exp_dir, "gemini_calls"),
        os.path.join(exp_dir, "openai_calls"),
    ] if os.path.isdir(d)]

    latencies: List[float] = []
    finish_reasons: DefaultDict[str, int] = DefaultDict(int)
    usage_totals = {"input_tokens": 0, "output_tokens": 0, "total_tokens": 0}
    models: DefaultDict[str, int] = DefaultDict(int)
    num_calls = 0

    for calls_dir in calls_dirs:
        for path in sorted(glob.glob(os.path.join(calls_dir, "decision_*.json"))):
            data = _safe_json_load(path)
            if not isinstance(data, dict):
                continue
            num_calls += 1

            # latency
            perf = data.get("performance", {}) if isinstance(data.get("performance"), dict) else {}
            lat = perf.get("api_latency") or perf.get("elapsed_time_seconds")
            try:
                if isinstance(lat, (int, float)):
                    latencies.append(float(lat))
            except Exception:
                pass

            # usage
            usage = data.get("response", {}).get("usage") if isinstance(data.get("response"), dict) else None
            totals = _sum_usage(usage)
            for k in usage_totals:
                usage_totals[k] += int(totals.get(k, 0))

            # finish reason
            fr = data.get("response", {}).get("finish_reason") if isinstance(data.get("response"), dict) else None
            if isinstance(fr, str) and fr:
                finish_reasons[fr] += 1

            # model
            model = None
            req = data.get("request") if isinstance(data.get("request"), dict) else {}
            if isinstance(req, dict):
                model = req.get("model")
            if isinstance(model, str) and model:
                models[model] += 1

    latency_summary: Dict[str, float] = {}
    if latencies:
        latency_summary = {
            "count": float(len(latencies)),
            "min": float(min(latencies)),
            "max": float(max(latencies)),
            "mean": float(statistics.fmean(latencies)),
            **_percentiles(latencies, [0.5, 0.9, 0.95, 0.99]),
        }

    return {
        "calls_dirs": calls_dirs,
        "num_calls": num_calls,
        "latency": latency_summary,
        "finish_reasons": dict(finish_reasons),
        "usage_tokens": usage_totals,
        "models": dict(models),
    }


def analyze_decision_history(exp_dir: str) -> Dict[str, Any]:
    # Detailed decision history written by Simulation
    history_files = sorted(glob.glob(os.path.join(exp_dir, "detailed_decision_log_*.json")))
    random_choices = 0
    api_failure_defaults = 0
    try:
        for hp in history_files:
            data = _safe_json_load(hp)
            if isinstance(data, list):
                for entry in data:
                    if not isinstance(entry, dict):
                        continue
                    analysis = str(entry.get("analysis", ""))
                    if any(p.search(analysis) for p in RANDOM_CHOICE_SIGNALS):
                        random_choices += 1
                    if re.search(r"LLM API failed", analysis, re.IGNORECASE):
                        api_failure_defaults += 1
    except Exception:
        pass
    return {
        "files": history_files,
        "random_choice_count": random_choices,
        "api_failure_default_count": api_failure_defaults,
    }


def analyze_evaluations(exp_dir: str) -> Dict[str, Any]:
    eval_files = sorted([path for path in glob.glob(os.path.join(exp_dir, "decision_evaluations_*.json")) if "recalculated" not in os.path.basename(path)])
    counts: DefaultDict[str, int] = DefaultDict(int)
    total = 0
    last_decision_number: Optional[int] = None
    try:
        for ep in eval_files:
            arr = _safe_json_load(ep)
            if isinstance(arr, list):
                for item in arr:
                    if not isinstance(item, dict):
                        continue
                    status = str(item.get("status", "UNKNOWN")).upper()
                    counts[status] += 1
                    total += 1
                    if isinstance(item.get("decision_number"), int):
                        last_decision_number = item["decision_number"]
    except Exception:
        pass
    
    # Try to load recalculated evaluations if available
    recalc_counts: DefaultDict[str, int] = DefaultDict(int)
    recalc_total = 0
    recalc_file = os.path.join(exp_dir, "decision_evaluations_recalculated.json")
    if os.path.exists(recalc_file):
        try:
            arr = _safe_json_load(recalc_file)
            if isinstance(arr, list):
                for item in arr:
                    if not isinstance(item, dict):
                        continue
                    status = str(item.get("status", "UNKNOWN")).upper()
                    recalc_counts[status] += 1
                    recalc_total += 1
        except Exception:
            pass
    
    result = {
        "files": eval_files,
        "counts": dict(counts),
        "total": total,
        "last_decision_number": last_decision_number,
        "accuracy": (counts.get("RIGHT", 0) / total) if total > 0 else 0.0,
    }
    
    if recalc_total > 0:
        result["recalc_counts"] = dict(recalc_counts)
        result["recalc_total"] = recalc_total
        result["recalc_accuracy"] = (recalc_counts.get("RIGHT", 0) / recalc_total) if recalc_total > 0 else 0.0
    
    return result


def analyze_coordinates(exp_dir: str) -> Dict[str, Any]:
    coord_path = os.path.join(exp_dir, "visited_coordinates.json")
    steps = 0
    last_coord: Optional[Dict[str, Any]] = None
    if os.path.exists(coord_path):
        data = _safe_json_load(coord_path)
        if isinstance(data, list):
            steps = len(data)
            if data:
                last_coord = data[-1]
    return {
        "file": coord_path if os.path.exists(coord_path) else None,
        "steps": steps,
        "last_coordinate": last_coord,
    }


def analyze_score_history(exp_dir: str) -> Dict[str, Any]:
    path = os.path.join(exp_dir, "score_history.json")
    scores: List[float] = []
    if os.path.exists(path):
        arr = _safe_json_load(path)
        if isinstance(arr, list):
            for x in arr:
                try:
                    if isinstance(x, (int, float)):
                        scores.append(float(x))
                except Exception:
                    pass
    summary: Dict[str, float] = {}
    if scores:
        summary = {
            "count": float(len(scores)),
            "min": float(min(scores)),
            "max": float(max(scores)),
            "mean": float(statistics.fmean(scores)),
            **_percentiles(scores, [0.5, 0.9, 0.95]),
            "last": float(scores[-1]),
        }
    return {
        "file": path if os.path.exists(path) else None,
        "summary": summary,
    }


def analyze_self_positioning(exp_dir: str) -> Dict[str, Any]:
    sp_dir = os.path.join(exp_dir, "self_position_calls")
    count = 0
    latest_final_answer: Optional[Dict[str, Any]] = None
    if os.path.isdir(sp_dir):
        files = sorted(glob.glob(os.path.join(sp_dir, "self_position_*.json")))
        count = len(files)
        # Find latest 'final_answer' step if exists
        for fp in reversed(files):
            data = _safe_json_load(fp)
            if isinstance(data, dict):
                steps = data.get("steps")
                if isinstance(steps, list):
                    for st in reversed(steps):
                        if isinstance(st, dict) and st.get("type") == "final_answer":
                            latest_final_answer = {
                                "location_guess": st.get("location_guess"),
                                "meta": {k: v for k, v in st.items() if k != "location_guess"},
                                "file": fp,
                            }
                            break
                if latest_final_answer:
                    break
    return {
        "dir_present": os.path.isdir(sp_dir),
        "count": count,
        "latest_final_answer": latest_final_answer,
    }


def analyze_experiment_folder(exp_dir: str) -> Dict[str, Any]:
    terminal = analyze_terminal_logs(exp_dir)
    strategy = analyze_strategy_logs(exp_dir)
    api = analyze_decision_calls(exp_dir)
    history = analyze_decision_history(exp_dir)
    evals = analyze_evaluations(exp_dir)
    coords = analyze_coordinates(exp_dir)
    scores = analyze_score_history(exp_dir)
    selfpos = analyze_self_positioning(exp_dir)

    # Try to discover per-run config limits from nearest _orchestrator config
    def _find_limits_from_task_config(start_dir: str) -> Dict[str, Optional[int]]:
        cur = os.path.abspath(start_dir)
        for _ in range(6):
            orch_dir = os.path.join(cur, "_orchestrator")
            if os.path.isdir(orch_dir):
                try:
                    ymls = [p for p in glob.glob(os.path.join(orch_dir, "config_*.yml"))]
                    if ymls:
                        # pick latest modified
                        ymls.sort(key=lambda p: os.path.getmtime(p), reverse=True)
                        with open(ymls[0], "r", encoding="utf-8", errors="ignore") as f:
                            cfg = yaml.safe_load(f) or {}
                            mdp = cfg.get("max_decision_points")
                            ms = cfg.get("max_steps")
                            return {
                                "max_decision_points": int(mdp) if isinstance(mdp, int) else None,
                                "max_steps": int(ms) if isinstance(ms, int) else None,
                            }
                except Exception:
                    pass
            parent = os.path.dirname(cur)
            if parent == cur:
                break
            cur = parent
        return {"max_decision_points": None, "max_steps": None}

    limits = _find_limits_from_task_config(exp_dir)

    # Merge error counts
    def _merge_counts(dst: Dict[str, int], src: Dict[str, int]) -> None:
        for k, v in src.items():
            dst[k] = int(dst.get(k, 0)) + int(v)

    error_counts: Dict[str, int] = {}
    _merge_counts(error_counts, terminal.get("error_counts", {}))
    _merge_counts(error_counts, strategy.get("error_counts", {}))

    # Random choice total combining sources
    random_choices_total = int(history.get("random_choice_count", 0)) + int(terminal.get("random_choice_signals", 0))

    # Success flag
    was_success = is_successful_experiment(exp_dir)

    # Status classification: success / finished_no_success / aborted
    def _is_aborted() -> bool:
        # Heuristics: exceptions, explicit loop error, ctrl+c, severe errors
        if any((terminal.get("exception_counts") or {}).values()):
            return True
        ec = error_counts
        if any(ec.get(k, 0) > 0 for k in [
            "simulation_loop_error", "checkpoint_load_error", "playwright_error"
        ]):
            return True
        # Ctrl+C or shutdown lines
        ctrlc = False
        for p in terminal.get("files", []):
            for line in _read_text_file(p)[-200:]:
                if ("Ctrl+C detected" in line) or ("Shutdown signal received again" in line):
                    ctrlc = True
                    break
            if ctrlc:
                break
        return ctrlc

    status: str
    if was_success:
        status = "success"
    else:
        status = "aborted" if _is_aborted() else "finished_no_success"

    # Failure reason classification (for UI details)
    failure_reason = None
    if not was_success:
        if status == "aborted":
            # Find a specific reason if possible
            if error_counts.get("simulation_loop_error", 0):
                failure_reason = "runtime_error"
            elif error_counts.get("checkpoint_load_error", 0):
                failure_reason = "checkpoint_load_error"
            elif error_counts.get("playwright_error", 0):
                failure_reason = "playwright_error"
            else:
                failure_reason = "aborted"
        else:
            # finished_no_success: likely exhausted max steps/decisions without success
            # Decide more specific: if api_calls.num_calls < limits.max_decision_points
            mdp = limits.get("max_decision_points")
            num_calls = api.get("num_calls") if isinstance(api, dict) else None
            if isinstance(mdp, int) and isinstance(num_calls, int):
                if num_calls >= mdp:
                    failure_reason = "decisions_exhausted"
                else:
                    failure_reason = "stopped_before_max_decisions"
            else:
                failure_reason = "no_success"

    # Model guess
    model_guess = None
    if api.get("models"):
        # Choose most frequent
        try:
            model_guess = max(api["models"].items(), key=lambda kv: kv[1])[0]
        except Exception:
            model_guess = None

    # Completed all decisions?
    completed_all_decisions: Optional[bool] = None
    mdp = limits.get("max_decision_points")
    if isinstance(mdp, int) and mdp > 0:
        try:
            completed_all_decisions = bool(int(api.get("num_calls", 0)) >= mdp)
        except Exception:
            completed_all_decisions = None

    # Advanced per-run metrics (guard if module missing)
    advanced: Optional[Dict[str, Any]] = None
    try:
        if compute_advanced_metrics is not None:
            advanced = compute_advanced_metrics(exp_dir, was_success)
    except Exception:
        advanced = None

    return {
        "path": exp_dir,
        "successful": was_success,
        "status": status,
        "failure_reason": failure_reason,
        "model": model_guess,
        "terminal": terminal,
        "strategy_logs": strategy,
        "api_calls": api,
        "decision_history": history,
        "evaluations": evals,
        "coordinates": coords,
        "scores": scores,
        "self_positioning": selfpos,
        "limits": limits,
        "completed_all_decisions": completed_all_decisions,
        "random_choices": random_choices_total,
        "error_counts": error_counts,
        "advanced": advanced,
    }


def crawl_directory(base_dir: str) -> Dict[str, object]:
    """
    Recursively crawl base_dir, summarize experiments and successes.
    Returns a dict with summary and details.
    Structure:
        {
            "base_dir": str,
            "total_experiments": int,
            "successful_experiments": int,
            "failed_experiments": int,
            "experiments": [
                {"path": str, "successful": bool}
            ]
        }
    """
    experiments: List[Dict[str, object]] = []

    # Walk the directory tree and decide on experiment directories at each level
    for root, dirs, files in os.walk(base_dir):
        # Quickly skip heavy subtrees that are clearly not experiments
        # But we still allow nested experiments to be found
        if is_experiment_folder(root):
            success = is_successful_experiment(root)
            experiments.append({
                "path": os.path.relpath(root, base_dir),
                "successful": success,
            })
            # Do not descend further into known experiment folder to avoid counting nested
            # runs twice (if any). If nested experiments need counting, remove this line.
            dirs[:] = []
            continue

    total = len(experiments)
    successful = sum(1 for e in experiments if e.get("successful"))
    failed = total - successful

    return {
        "base_dir": os.path.abspath(base_dir),
        "total_experiments": total,
        "successful_experiments": successful,
        "failed_experiments": failed,
        "experiments": sorted(experiments, key=lambda x: x["path"]),
    }


def crawl_directory_deep(base_dir: str) -> Dict[str, Any]:
    """
    Deeply analyze each experiment directory and compute aggregate metrics.
    """
    base_dir_abs = os.path.abspath(base_dir)
    per_runs: List[Dict[str, Any]] = []
    for root, dirs, files in os.walk(base_dir):
        if is_experiment_folder(root):
            per_runs.append(analyze_experiment_folder(root))
            # do not recurse inside experiment to avoid double count
            dirs[:] = []
            continue

    # Aggregates
    total = len(per_runs)
    successful = sum(1 for r in per_runs if r.get("successful"))
    failed = total - successful

    # Error aggregates
    error_agg: DefaultDict[str, int] = DefaultDict(int)
    exc_agg: DefaultDict[str, int] = DefaultDict(int)
    random_choices_total = 0
    right = wrong = unknown = 0
    recalc_right = recalc_wrong = recalc_unknown = 0
    recalc_available_count = 0
    latencies: List[float] = []
    usage_totals = {"input_tokens": 0, "output_tokens": 0, "total_tokens": 0}

    completed_all_decisions_count = 0
    finished_no_success_count = 0
    aborted_count = 0

    api_failures_total = 0

    # Advanced aggregates containers
    decisions_per_run: List[float] = []
    path_lengths_m: List[float] = []
    min_dists_m: List[float] = []
    spl_values: List[float] = []

    for r in per_runs:
        # errors
        for k, v in (r.get("error_counts") or {}).items():
            error_agg[k] += int(v)

        # exceptions from terminal
        term = r.get("terminal") or {}
        for k, v in (term.get("exception_counts") or {}).items():
            exc_agg[k] += int(v)

        # random
        random_choices_total += int(r.get("random_choices", 0))

        # status breakdown
        st = r.get("status")
        if st == "finished_no_success":
            finished_no_success_count += 1
        elif st == "aborted":
            aborted_count += 1

        # completed all decisions
        cad = r.get("completed_all_decisions")
        if cad is True:
            completed_all_decisions_count += 1

        # evals
        ev = r.get("evaluations") or {}
        c = (ev.get("counts") or {})
        right += int(c.get("RIGHT", 0))
        wrong += int(c.get("WRONG", 0))
        unknown += int(c.get("UNKNOWN", 0))
        
        # recalc evals (if available)
        if "recalc_counts" in ev:
            rc = ev.get("recalc_counts") or {}
            recalc_right += int(rc.get("RIGHT", 0))
            recalc_wrong += int(rc.get("WRONG", 0))
            recalc_unknown += int(rc.get("UNKNOWN", 0))
            recalc_available_count += 1

        # api failures: combine strategy LLM API errors + decision history fallbacks
        api_failures_total += int((r.get("strategy_logs") or {}).get("error_counts", {}).get("llm_api_error", 0))
        api_failures_total += int((r.get("decision_history") or {}).get("api_failure_default_count", 0))

        # api latencies
        api = r.get("api_calls") or {}
        # We don't store per-call latencies here; use summary count weights to approx mean via combining lists is easier.
        # For accurate percentiles, we would need raw latencies; keep aggregate sums only.
        # Here we approximate by aggregating means weighted by count if available.
        # Instead, if counts are small, we can re-open files, but we already did per-run.
        # So just compute simple mean of means weighted by counts.
        lat = api.get("latency") or {}
        if lat and isinstance(lat.get("mean"), (int, float)) and isinstance(lat.get("count"), (int, float)):
            # expand approximate samples: mean replicated 'count' times—not ideal but gives rough combined mean and percentiles won't be accurate.
            try:
                n = int(lat.get("count", 0))
                m = float(lat.get("mean", 0.0))
                # store n copies only if small, else just store m once (avoid memory blow-up)
                if n <= 50:
                    latencies.extend([m] * n)
                else:
                    latencies.append(m)
            except Exception:
                pass

        # usage
        ut = api.get("usage_tokens") or {}
        for k in usage_totals:
            usage_totals[k] += int(ut.get(k, 0))

        # Advanced per-run rollup
        adv = r.get("advanced") or {}
        # decisions
        dc = adv.get("decisions_count")
        if isinstance(dc, (int, float)):
            decisions_per_run.append(float(dc))
        elif isinstance(api.get("num_calls"), (int, float)):
            decisions_per_run.append(float(api.get("num_calls", 0)))
        # path length
        pl = adv.get("path_length_m")
        if isinstance(pl, (int, float)):
            path_lengths_m.append(float(pl))
        # min dist
        mdp = adv.get("min_distance_to_polygon_m")
        if isinstance(mdp, (int, float)):
            min_dists_m.append(float(mdp))
        # spl
        spl = adv.get("spl")
        if isinstance(spl, (int, float)):
            spl_values.append(float(spl))

    # latency summary (approx)
    latency_summary: Dict[str, float] = {}
    if latencies:
        latency_summary = {
            "count": float(len(latencies)),
            "min": float(min(latencies)),
            "max": float(max(latencies)),
            "mean": float(statistics.fmean(latencies)),
            **_percentiles(latencies, [0.5, 0.9, 0.95, 0.99]),
        }

    decision_stats = {
        "RIGHT": right,
        "WRONG": wrong,
        "UNKNOWN": unknown,
        "accuracy": (right / (right + wrong + unknown)) if (right + wrong + unknown) > 0 else 0.0,
    }
    
    # Add recalculated stats if available
    recalc_decision_stats: Optional[Dict[str, Any]] = None
    if recalc_available_count > 0:
        recalc_total = recalc_right + recalc_wrong + recalc_unknown
        recalc_decision_stats = {
            "RIGHT": recalc_right,
            "WRONG": recalc_wrong,
            "UNKNOWN": recalc_unknown,
            "accuracy": (recalc_right / recalc_total) if recalc_total > 0 else 0.0,
            "available_count": recalc_available_count,
        }

    # Helpers for advanced summaries
    def _summary(values: List[float]) -> Dict[str, float]:
        if not values:
            return {}
        return {
            "count": float(len(values)),
            "min": float(min(values)),
            "max": float(max(values)),
            "mean": float(statistics.fmean(values)),
            **_percentiles(values, [0.5, 0.9, 0.95])
        }

    def _hist_exact(values: List[float]) -> Dict[str, int]:
        h: DefaultDict[str, int] = DefaultDict(int)
        for v in values:
            k = str(int(round(v)))
            h[k] += 1
        return dict(h)

    def _hist_bins(values: List[float], bins: int = 10) -> Tuple[List[int], List[float]]:
        if not values:
            return [], []
        vmin, vmax = min(values), max(values)
        if vmax <= vmin:
            return [len(values)] + [0] * (bins - 1), [vmin] * (bins + 1)
        width = (vmax - vmin) / bins
        edges = [vmin + i * width for i in range(bins + 1)]
        counts = [0] * bins
        for v in values:
            idx = int((v - vmin) / width)
            if idx == bins:
                idx = bins - 1
            counts[idx] += 1
        return counts, edges

    adv_agg: Dict[str, Any] = {}
    if decisions_per_run:
        bins10, edges = _hist_bins(decisions_per_run, 10)
        adv_agg["decisions_per_run"] = {
            "summary": _summary(decisions_per_run),
            "exact_hist": _hist_exact(decisions_per_run),
            "bins_10": bins10,
            "bin_edges": edges,
        }
    if min_dists_m:
        adv_agg["min_distance_to_polygon_m"] = _summary(min_dists_m)
    if path_lengths_m:
        adv_agg["path_length_m"] = _summary(path_lengths_m)
    if spl_values:
        adv_agg["spl"] = {
            "count": float(len(spl_values)),
            "mean": float(statistics.fmean(spl_values)),
            "successes": float(successful),
        }

    return {
        "base_dir": base_dir_abs,
        "total_experiments": total,
        "successful_experiments": successful,
        "failed_experiments": failed,
        "success_rate": (successful / total) if total > 0 else 0.0,
        "aggregate": {
            "error_counts": dict(error_agg),
            "exception_counts": dict(exc_agg),
            "random_choices_total": random_choices_total,
            "status_breakdown": {
                "success": successful,
                "finished_no_success": finished_no_success_count,
                "aborted": aborted_count,
            },
            "completed_all_decisions_count": completed_all_decisions_count,
            "api_failures_total": api_failures_total,
            "api_latency": latency_summary,
            "api_usage_tokens": usage_totals,
            "decision_stats": decision_stats,
            "recalc_decision_stats": recalc_decision_stats,
            "advanced": adv_agg,
        },
        "experiments": sorted(per_runs, key=lambda r: r.get("path", "")),
    }


def format_summary(stats: Dict[str, object]) -> str:
    lines: List[str] = []
    lines.append(f"Base directory: {stats['base_dir']}")
    lines.append(f"Total experiments: {stats['total_experiments']}")
    lines.append(f"Successful: {stats['successful_experiments']}")
    lines.append(f"Failed: {stats['failed_experiments']}")
    return "\n".join(lines)


def parse_args(argv: List[str]) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Crawl experiment logs and report run statistics"
    )
    parser.add_argument(
        "base_dir",
        nargs="?",
        default=".",
        help="Base directory to crawl (default: current directory)",
    )
    parser.add_argument(
        "--json",
        dest="output_json",
        action="store_true",
        help="Output full JSON details instead of a text summary",
    )
    parser.add_argument(
        "--deep",
        dest="deep",
        action="store_true",
        help="Perform deep analysis of runs (errors, API, decisions, scores, self-positioning)",
    )
    parser.add_argument(
        "--list-failed",
        action="store_true",
        help="Print the list of failed experiment paths",
    )
    parser.add_argument(
        "--list-success",
        action="store_true",
        help="Print the list of successful experiment paths",
    )
    parser.add_argument(
        "--list-api-failures",
        action="store_true",
        help="Print the list of experiments that had API failures (requires --deep)",
    )
    parser.add_argument(
        "--out",
        dest="out_path",
        default=None,
        help="Write JSON report to this file (works with or without --json)",
    )
    return parser.parse_args(argv)


def main(argv: List[str]) -> int:
    args = parse_args(argv)
    base_dir = args.base_dir

    if not os.path.exists(base_dir):
        print(json.dumps({
            "error": f"Base directory does not exist: {base_dir}"
        }, indent=2))
        return 2

    if args.deep:
        stats = crawl_directory_deep(base_dir)
        if args.out_path:
            try:
                with open(args.out_path, "w", encoding="utf-8") as f:
                    json.dump(stats, f, indent=2)
                print(f"Saved deep analysis JSON to {args.out_path}")
            except Exception as e:
                print(json.dumps({"error": f"Failed to write {args.out_path}: {e}"}, indent=2))
                return 1
        if args.output_json and not args.out_path:
            print(json.dumps(stats, indent=2))
        else:
            # Text summary for deep mode
            summary = {
                "Base directory": stats["base_dir"],
                "Total experiments": stats["total_experiments"],
                "Successful": stats["successful_experiments"],
                "Failed": stats["failed_experiments"],
                "Success rate": f"{stats['success_rate']*100:.2f}%",
                "Random choices (total)": stats.get("aggregate", {}).get("random_choices_total", 0),
                "Completed all decisions": stats.get("aggregate", {}).get("completed_all_decisions_count", 0),
                "Aborted runs": stats.get("aggregate", {}).get("status_breakdown", {}).get("aborted", 0),
                "Finished w/o success": stats.get("aggregate", {}).get("status_breakdown", {}).get("finished_no_success", 0),
                "API failures (total)": stats.get("aggregate", {}).get("api_failures_total", 0),
            }
            for k, v in summary.items():
                print(f"{k}: {v}")

            if args.list_success:
                success_paths = [e.get("path", "") for e in stats.get("experiments", []) if e.get("successful")]
                print("\nSuccessful experiments:")
                for p in success_paths:
                    print(os.path.relpath(p, base_dir))

            if args.list_failed:
                failed_paths = [e.get("path", "") for e in stats.get("experiments", []) if not e.get("successful")]
                print("\nFailed experiments:")
                for p in failed_paths:
                    print(os.path.relpath(p, base_dir))

            if args.list_api_failures:
                # Calculate API failures for each experiment
                api_failure_paths = []
                for exp in stats.get("experiments", []):
                    # API failures = llm_api_error from strategy logs + api_failure_default_count from decision history
                    strategy_llm_errors = (exp.get("strategy_logs") or {}).get("error_counts", {}).get("llm_api_error", 0)
                    history_api_failures = (exp.get("decision_history") or {}).get("api_failure_default_count", 0)
                    total_api_failures = strategy_llm_errors + history_api_failures
                    if total_api_failures > 0:
                        api_failure_paths.append(exp.get("path", ""))

                print("\nExperiments with API failures:")
                if api_failure_paths:
                    for p in sorted(api_failure_paths):
                        print(os.path.relpath(p, base_dir))
                else:
                    print("(none)")
    else:
        stats = crawl_directory(base_dir)
        if args.out_path:
            try:
                with open(args.out_path, "w", encoding="utf-8") as f:
                    json.dump(stats, f, indent=2)
                print(f"Saved summary JSON to {args.out_path}")
            except Exception as e:
                print(json.dumps({"error": f"Failed to write {args.out_path}: {e}"}, indent=2))
                return 1
        if args.output_json and not args.out_path:
            print(json.dumps(stats, indent=2))
        else:
            print(format_summary(stats))

            if args.list_success:
                success_paths = [e["path"] for e in stats["experiments"] if e["successful"]]
                print("\nSuccessful experiments:")
                for p in success_paths:
                    print(p)

            if args.list_failed:
                failed_paths = [e["path"] for e in stats["experiments"] if not e["successful"]]
                print("\nFailed experiments:")
                for p in failed_paths:
                    print(p)

    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))


