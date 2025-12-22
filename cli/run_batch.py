import os
import sys
import time
import json
import yaml
import uuid
import signal
import argparse
import subprocess
from datetime import datetime
from typing import Dict, List, Optional, Tuple
import shutil


def load_paths(json_path: str) -> List[Dict]:
    with open(json_path, "r", encoding="utf-8") as fp:
        data = json.load(fp)

    # Handle different JSON formats
    paths = data.get("paths", [])
    if not paths:
        # Check if it's test_routes.json format with "routes" key
        routes = data.get("routes", [])
        if routes:
            # Convert selected_routes_1.json format to expected format
            paths = []
            for route in routes:
                # Use first polygon point as destination coordinates if polygon exists
                polygon = route.get("destinationPolygon", [])
                if polygon and len(polygon) > 0:
                    dest_lat, dest_lng = polygon[0]  # First polygon point
                else:
                    # Fallback to start coords if no polygon (shouldn't happen)
                    dest_lat, dest_lng = route["start"]["lat"], route["start"]["lng"]

                path = {
                    "start": {
                        "position": route["start"],
                        "startPanoId": route.get("startPanoId")  # Add startPanoId if present
                    },
                    "end": {
                        "position": {
                            "lat": dest_lat,
                            "lng": dest_lng
                        },
                        "destination": route.get("destinationName", "Unknown Destination"),
                        "destinationPolygon": route.get("destinationPolygon")  # Add polygon if present
                    }
                }
                paths.append(path)

    # Require explicit destination; forbid falling back to any description fields
    for idx, p in enumerate(paths, 1):
        if not isinstance(p, dict) or "end" not in p or not isinstance(p.get("end"), dict) or not p["end"].get("destination"):
            raise ValueError(f"Path {idx} missing required field 'end.destination'")
    return paths


def load_prompts(path: Optional[str]) -> List[str]:
    if not path:
        raise ValueError("prompts_file must be provided for controlled runs")
    if not os.path.exists(path):
        raise FileNotFoundError(f"prompts_file not found: {path}")
    with open(path, "r", encoding="utf-8") as fp:
        content = fp.read().strip()
    if not content:
        raise ValueError("prompts_file is empty; provide an explicit prompt")
    return [content]


def ensure_dir(path: str) -> None:
    os.makedirs(path, exist_ok=True)


def write_single_path_file(dest_path: str, path_def: Dict) -> None:
    ensure_dir(os.path.dirname(dest_path))
    with open(dest_path, "w", encoding="utf-8") as fp:
        json.dump({"paths": [path_def]}, fp, indent=2)


def write_config_file(dest_path: str, cfg: Dict) -> None:
    ensure_dir(os.path.dirname(dest_path))
    with open(dest_path, "w", encoding="utf-8") as fp:
        yaml.safe_dump(cfg, fp, sort_keys=False)


def format_dest_name(path_def: Dict) -> str:
    dest_name = path_def.get("end", {}).get("destination", "Unknown")
    return str(dest_name).replace(" ", "_").replace("/", "_")


def get_simulation_status(log_folder: str) -> Dict:
    status = {
        "step": 0,
        "decision": 0,
        "self_positioning_active": False,
        "current_location": "Unknown",
    }
    try:
        coords_file = os.path.join(log_folder, "visited_coordinates.json")
        if os.path.exists(coords_file):
            with open(coords_file, "r", encoding="utf-8") as f:
                coords_data = json.load(f)
                if coords_data:
                    latest = coords_data[-1]
                    status["step"] = latest.get("step", len(coords_data))

        max_decision = 0
        for calls_dir in [
            os.path.join(log_folder, "openai_calls"),
            os.path.join(log_folder, "gemini_calls"),
        ]:
            if os.path.exists(calls_dir):
                for name in os.listdir(calls_dir):
                    if name.startswith("decision_") and name.endswith(".json"):
                        try:
                            num = int(name.split("_")[1].split(".")[0])
                            max_decision = max(max_decision, num)
                        except Exception:
                            pass
        status["decision"] = max_decision
    except Exception:
        pass
    return status


class Task:
    def __init__(self, idx: int, path_def: Dict, run_index: int, base_log_dir: str, session_id: str, group_label: str) -> None:
        self.idx = idx
        self.path_def = path_def
        self.run_index = run_index
        self.task_id = f"{idx:05d}_{path_def.get('uid', f'path_{idx:02d}')}_r{run_index}"
        self.unique_base_dir = os.path.join(base_log_dir, group_label, session_id, self.task_id)
        self.proc: Optional[subprocess.Popen] = None
        self.config_path: Optional[str] = None
        self.paths_path: Optional[str] = None
        self.experiment_folder: Optional[str] = None
        self.start_time: Optional[float] = None
        self.end_time: Optional[float] = None
        self.returncode: Optional[int] = None
        self.stdout_path: Optional[str] = None
        self.stderr_path: Optional[str] = None

    def elapsed(self) -> float:
        if self.end_time is not None and self.start_time is not None:
            return max(0.0, self.end_time - self.start_time)
        if self.start_time is not None:
            return time.time() - self.start_time
        return 0.0


class Orchestrator:
    def __init__(
        self,
        config_file: str,
        max_parallel: int,
        stop_on_failure: bool,
    ) -> None:
        self.config_file = config_file
        self.max_parallel = max_parallel
        self.stop_on_failure = stop_on_failure
        self.session_id = datetime.now().strftime("%Y%m%d_%H%M%S") + f"_{uuid.uuid4().hex[:8]}"
        self.cfg: Dict = {}
        self.original_paths: List[Dict] = []
        self.tasks: List[Task] = []
        self.active: List[Task] = []
        self.completed: List[Task] = []
        self.failed: List[Task] = []
        self.terminated = False
        self.group_label = "multi"

    def load_config(self) -> None:
        with open(self.config_file, "r", encoding="utf-8") as f:
            self.cfg = yaml.safe_load(f)
        # Fall back to environment variables for API keys if not in config
        if not self.cfg.get('maps_api_key'):
            self.cfg['maps_api_key'] = os.environ.get('GOOGLE_MAPS_API_KEY', '')
        if not self.cfg.get('streetview_signing_secret'):
            self.cfg['streetview_signing_secret'] = os.environ.get('STREETVIEW_SIGNING_SECRET', '')
        self.original_paths = load_paths(self.cfg["paths_file"])  # type: ignore[index]
        # Save the base config at the session root for provenance
        base_log_dir = self.cfg.get("base_log_dir", "logs")
        # Build group label: <pathsfile>_<model>, replacing '/' in model with '_'
        paths_file_value = str(self.cfg.get("paths_file", "paths.json"))
        paths_label = os.path.splitext(os.path.basename(paths_file_value))[0]
        safe_paths_label = paths_label.replace(" ", "_").replace("/", "_")
        model_label = str(self.cfg.get("model_name", "model")).replace("/", "_")
        self.group_label = f"{safe_paths_label}_{model_label}"
        session_root = os.path.join(base_log_dir, self.group_label, self.session_id)
        ensure_dir(session_root)
        try:
            with open(self.config_file, "r", encoding="utf-8") as src:
                content = src.read()
            with open(os.path.join(session_root, "config.yml"), "w", encoding="utf-8") as dst:
                dst.write(content)
        except Exception:
            pass

    def build_tasks(self) -> None:
        runs_per_path = int(self.cfg.get("runs_per_path", 1))
        base_log_dir = self.cfg.get("base_log_dir", "logs")
        idx = 0
        for path in self.original_paths:
            for run_idx in range(runs_per_path):
                self.tasks.append(Task(idx, path, run_idx, base_log_dir, self.session_id, self.group_label))
                idx += 1

    def make_per_task_config(self, task: Task) -> None:
        # Prepare single-path file
        cfg_dir = os.path.join(task.unique_base_dir, "_orchestrator")
        ensure_dir(cfg_dir)
        task.paths_path = os.path.join(cfg_dir, f"path_{task.task_id}.json")
        write_single_path_file(task.paths_path, task.path_def)

        # Prepare derived config
        task_config = dict(self.cfg)
        task_config["batch_mode"] = False
        task_config["runs_per_path"] = 1
        task_config["paths_file"] = task.paths_path
        task_config["base_log_dir"] = task.unique_base_dir

        task.config_path = os.path.join(cfg_dir, f"config_{task.task_id}.yml")
        write_config_file(task.config_path, task_config)

        # Copy original paths and prompts files into the run folder for provenance
        try:
            orig_paths_file = str(self.cfg.get("paths_file", ""))
            if orig_paths_file and os.path.exists(orig_paths_file):
                shutil.copy2(orig_paths_file, os.path.join(cfg_dir, os.path.basename(orig_paths_file)))
        except Exception:
            pass
        try:
            orig_prompts_file = str(self.cfg.get("prompts_file", ""))
            if orig_prompts_file and os.path.exists(orig_prompts_file):
                shutil.copy2(orig_prompts_file, os.path.join(cfg_dir, os.path.basename(orig_prompts_file)))
        except Exception:
            pass

    def start_task(self, task: Task, run_py_path: str) -> None:
        ensure_dir(task.unique_base_dir)
        self.make_per_task_config(task)
        task.stdout_path = os.path.join(task.unique_base_dir, "_orchestrator", "stdout.log")
        task.stderr_path = os.path.join(task.unique_base_dir, "_orchestrator", "stderr.log")
        stdout_f = open(task.stdout_path, "ab")
        stderr_f = open(task.stderr_path, "ab")
        args = [sys.executable, run_py_path, task.config_path]  # type: ignore[list-item]
        task.proc = subprocess.Popen(args, stdout=stdout_f, stderr=stderr_f)
        task.start_time = time.time()

    def discover_experiment_folder(self, task: Task) -> Optional[str]:
        """Find the deepest experiment folder created by run.py.

        Some model names contain '/', which causes nested subfolders (e.g., 'trapi/gpt_5').
        We search recursively under the task base, ignoring '_orchestrator', and prefer
        a directory containing typical run artifacts.
        """
        base = task.unique_base_dir
        if not os.path.exists(base):
            return None

        sentinel_names = {"visited_coordinates.json", "strategy.log"}
        sentinel_dirs = {"openai_calls", "gemini_calls"}

        best_match: Optional[Tuple[float, str]] = None  # (mtime, path)
        try:
            for dirpath, dirnames, filenames in os.walk(base):
                # Skip orchestrator internals
                if os.path.basename(dirpath) == "_orchestrator" or "_orchestrator" in dirpath:
                    continue

                # If this dir looks like an experiment folder, return it immediately
                if (sentinel_names & set(filenames)) or (sentinel_dirs & set(dirnames)):
                    return dirpath

                # Track most recently modified directory as a fallback candidate
                try:
                    mtime = os.path.getmtime(dirpath)
                    if best_match is None or mtime > best_match[0]:
                        best_match = (mtime, dirpath)
                except Exception:
                    pass
        except Exception:
            return None

        return best_match[1] if best_match else None

    def update_status(self) -> None:
        # Attach experiment folders for active tasks
        for task in self.active:
            if task.experiment_folder is None:
                task.experiment_folder = self.discover_experiment_folder(task)

        # Move finished tasks out of active
        still_active: List[Task] = []
        for task in self.active:
            if task.proc is None:
                continue
            rc = task.proc.poll()
            if rc is None:
                still_active.append(task)
            else:
                task.returncode = rc
                task.end_time = time.time()
                if rc == 0:
                    self.completed.append(task)
                else:
                    self.failed.append(task)
        self.active = still_active

    def summary_line(self) -> str:
        total = len(self.tasks)
        completed = len(self.completed)
        failed = len(self.failed)
        active = len(self.active)
        queued = total - completed - failed - active
        return f"Active {active}/{self.max_parallel} | Done {completed} (+{failed} failed) | Queued {queued} | Total {total}"

    def handle_signal(self, signum: int, frame) -> None:  # type: ignore[no-untyped-def]
        if self.terminated:
            return
        self.terminated = True
        print(f"Received signal {signum}. Terminating child processes...", flush=True)
        for task in list(self.active):
            if task.proc and task.proc.poll() is None:
                try:
                    task.proc.terminate()
                except Exception:
                    pass
        # Give them a moment to exit
        time.sleep(3.0)
        for task in list(self.active):
            if task.proc and task.proc.poll() is None:
                try:
                    task.proc.kill()
                except Exception:
                    pass

    def run(self) -> int:
        self.load_config()
        self.build_tasks()
        run_py_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "run.py")

        next_to_start = 0

        # Register signal handlers
        signal.signal(signal.SIGINT, self.handle_signal)
        signal.signal(signal.SIGTERM, self.handle_signal)

        while True:
            if self.terminated:
                break

            # Start new tasks if capacity available
            while (
                next_to_start < len(self.tasks)
                and len(self.active) < self.max_parallel
                and not (self.stop_on_failure and len(self.failed) > 0)
            ):
                task = self.tasks[next_to_start]
                next_to_start += 1
                self.start_task(task, run_py_path)
                self.active.append(task)

            # Update task status periodically
            self.update_status()

            # Exit condition
            if (
                len(self.completed) + len(self.failed) == len(self.tasks)
                and len(self.active) == 0
            ):
                break

            time.sleep(0.5)

        print("All tasks finished.")
        print(self.summary_line())
        return 0 if len(self.failed) == 0 else 1


def parse_args(argv: List[str]) -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Run multiple navigation experiments in parallel using subprocesses.")
    p.add_argument("--config", default=os.path.join(os.path.dirname(os.path.abspath(__file__)), "config.yml"), help="Path to base config.yml")
    p.add_argument("--max-parallel", type=int, default=4, help="Maximum number of runs in parallel")
    p.add_argument("--stop-on-failure", action="store_true", help="Stop launching new tasks if any task fails")
    return p.parse_args(argv)


def main(argv: List[str]) -> int:
    args = parse_args(argv)
    orch = Orchestrator(
        config_file=args.config,
        max_parallel=args.max_parallel,
        stop_on_failure=args.stop_on_failure,
    )
    return orch.run()


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
