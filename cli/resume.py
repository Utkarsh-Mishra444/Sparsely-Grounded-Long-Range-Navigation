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


def ensure_dir(path: str) -> None:
    os.makedirs(path, exist_ok=True)


def discover_experiment_folder(task_base_dir: str) -> Optional[str]:
    """Find the deepest experiment folder created by run.py under a task folder.

    Mirrors run_multi.py logic: prefer a directory containing typical run artifacts.
    Skips the orchestrator internals.
    """
    if not os.path.exists(task_base_dir):
        return None

    sentinel_names = {"visited_coordinates.json", "strategy.log"}
    sentinel_dirs = {"openai_calls", "gemini_calls"}

    best_match: Optional[Tuple[float, str]] = None  # (mtime, path)
    try:
        for dirpath, dirnames, filenames in os.walk(task_base_dir):
            if os.path.basename(dirpath) == "_orchestrator" or "_orchestrator" in dirpath:
                continue
            if (sentinel_names & set(filenames)) or (sentinel_dirs & set(dirnames)):
                return dirpath
            try:
                mtime = os.path.getmtime(dirpath)
                if best_match is None or mtime > best_match[0]:
                    best_match = (mtime, dirpath)
            except Exception:
                pass
    except Exception:
        return None

    return best_match[1] if best_match else None


def has_final_arrival(stdout_path: str) -> bool:
    try:
        if not os.path.exists(stdout_path):
            return False
        with open(stdout_path, "r", encoding="utf-8", errors="ignore") as f:
            for line in f:
                if "Reached destination" in line:
                    return True
        return False
    except Exception:
        return False


def read_yaml(path: str) -> Dict:
    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def write_yaml(path: str, data: Dict) -> None:
    ensure_dir(os.path.dirname(path))
    with open(path, "w", encoding="utf-8") as f:
        yaml.safe_dump(data, f, sort_keys=False)


def list_latest_checkpoint(exp_dir: str) -> Optional[str]:
    latest_num = -1
    latest_path: Optional[str] = None
    try:
        for name in os.listdir(exp_dir):
            if name.startswith("checkpoint_decision_") and name.endswith(".pkl"):
                mid = name[len("checkpoint_decision_"):-len(".pkl")]
                if mid.isdigit():
                    num = int(mid)
                    if num > latest_num:
                        latest_num = num
                        latest_path = os.path.join(exp_dir, name)
    except Exception:
        return None
    return latest_path


def is_incomplete_run(task_dir: str, exp_dir: Optional[str]) -> Tuple[bool, str]:
    """Heuristic to decide if a run is incomplete.

    - Completed if stdout has "Reached destination".
    - Else incomplete if a checkpoint exists but the process ended (we're resuming crashed/terminated runs).
    - Else incomplete if no terminal signal of success and experiment folder exists.
    - Else unknown/skip with message.
    """
    orch_dir = os.path.join(task_dir, "_orchestrator")
    stdout_path = os.path.join(orch_dir, "stdout.log")
    if has_final_arrival(stdout_path):
        return (False, "completed: destination reached")

    if not exp_dir or not os.path.exists(exp_dir):
        return (False, "skip: experiment folder not found")

    chk = list_latest_checkpoint(exp_dir)
    if chk:
        return (True, "incomplete: checkpoint present, will resume from latest")

    # If no checkpoint but experiment folder exists, likely died before first decision
    return (False, "skip: no checkpoints found")


class ResumeTask:
    def __init__(self, task_dir: str, exp_dir: str, base_cfg_path: str) -> None:
        self.task_dir = task_dir
        self.exp_dir = exp_dir
        self.base_cfg_path = base_cfg_path
        self.stdout_path = os.path.join(task_dir, "_orchestrator", "stdout.log")
        self.stderr_path = os.path.join(task_dir, "_orchestrator", "stderr.log")
        self.resume_cfg_path = os.path.join(task_dir, "_orchestrator", f"resume_{uuid.uuid4().hex[:8]}.yml")
        self.proc: Optional[subprocess.Popen] = None
        self.start_time: Optional[float] = None
        self.end_time: Optional[float] = None
        self.returncode: Optional[int] = None


class ResumeOrchestrator:
    def __init__(self, session_root: str, maps_key: str, signing_secret: str, max_parallel: int, into_new_folder: bool) -> None:
        self.session_root = session_root
        self.maps_key = maps_key
        self.signing_secret = signing_secret
        self.max_parallel = max_parallel
        self.into_new_folder = into_new_folder
        self.tasks: List[ResumeTask] = []
        self.active: List[ResumeTask] = []
        self.completed: List[ResumeTask] = []
        self.failed: List[ResumeTask] = []
        self.terminated = False

    def discover_tasks(self) -> None:
        # Task directories are immediate children like 00000_path_00_r0
        try:
            for name in sorted(os.listdir(self.session_root)):
                task_dir = os.path.join(self.session_root, name)
                if not os.path.isdir(task_dir):
                    continue
                orch_dir = os.path.join(task_dir, "_orchestrator")
                if not os.path.isdir(orch_dir):
                    continue
                base_cfg_path = None
                # Find the saved per-run config from original run_multi
                for f in os.listdir(orch_dir):
                    if f.startswith("config_") and f.endswith(".yml"):
                        base_cfg_path = os.path.join(orch_dir, f)
                        break
                exp_dir = discover_experiment_folder(task_dir)
                if not base_cfg_path:
                    continue

                # Decide if incomplete
                incomplete, reason = is_incomplete_run(task_dir, exp_dir)
                if incomplete and exp_dir:
                    self.tasks.append(ResumeTask(task_dir, exp_dir, base_cfg_path))
                else:
                    # Print reason for transparency
                    print(f"[SKIP] {task_dir}: {reason}")
        except Exception as e:
            print(f"ERROR scanning session root {self.session_root}: {e}")

    def make_resume_config(self, task: ResumeTask) -> None:
        base_cfg = read_yaml(task.base_cfg_path)
        # Override only what is necessary
        base_cfg["is_resume"] = True
        base_cfg["resume_folder"] = task.exp_dir
        base_cfg["resume_from_decision"] = "latest"
        base_cfg["resume_into_new_folder"] = bool(self.into_new_folder)
        # Credentials overrides
        base_cfg["maps_api_key"] = self.maps_key
        base_cfg["streetview_signing_secret"] = self.signing_secret
        write_yaml(task.resume_cfg_path, base_cfg)

    def start_task(self, task: ResumeTask) -> None:
        stdout_f = open(task.stdout_path, "ab")
        stderr_f = open(task.stderr_path, "ab")
        self.make_resume_config(task)
        run_py = os.path.join(os.path.dirname(os.path.abspath(__file__)), "run.py")
        args = [sys.executable, run_py, task.resume_cfg_path]
        task.proc = subprocess.Popen(args, stdout=stdout_f, stderr=stderr_f)
        task.start_time = time.time()

    def update_status(self) -> None:
        still_active: List[ResumeTask] = []
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
        time.sleep(3.0)
        for task in list(self.active):
            if task.proc and task.proc.poll() is None:
                try:
                    task.proc.kill()
                except Exception:
                    pass

    def summary_line(self) -> str:
        total = len(self.tasks)
        completed = len(self.completed)
        failed = len(self.failed)
        active = len(self.active)
        queued = total - completed - failed - active
        return f"Active {active}/{self.max_parallel} | Done {completed} (+{failed} failed) | Queued {queued} | Total {total}"

    def run(self) -> int:
        self.discover_tasks()
        if not self.tasks:
            print("No incomplete runs found to resume.")
            return 0

        signal.signal(signal.SIGINT, self.handle_signal)
        signal.signal(signal.SIGTERM, self.handle_signal)

        next_idx = 0
        while True:
            if self.terminated:
                break

            while next_idx < len(self.tasks) and len(self.active) < self.max_parallel:
                task = self.tasks[next_idx]
                next_idx += 1
                try:
                    self.start_task(task)
                    self.active.append(task)
                except Exception as e:
                    print(f"[ERROR] Failed to start resume for {task.task_dir}: {e}")
                    self.failed.append(task)

            self.update_status()

            if len(self.completed) + len(self.failed) == len(self.tasks) and len(self.active) == 0:
                break

            time.sleep(0.5)

        print("All resume tasks finished.")
        print(self.summary_line())
        return 0 if len(self.failed) == 0 else 1


def parse_args(argv: List[str]) -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Resume incomplete navigation experiments in parallel using subprocesses.")
    p.add_argument("--session-root", required=True, help="Path to timestamped session root created by run_multi.py")
    p.add_argument("--maps-key", default=os.environ.get("GOOGLE_MAPS_API_KEY", ""), 
                   help="Google Maps API key (default: GOOGLE_MAPS_API_KEY env var)")
    p.add_argument("--signing-secret", default=os.environ.get("STREETVIEW_SIGNING_SECRET", ""),
                   help="Street View signing secret (default: STREETVIEW_SIGNING_SECRET env var)")
    p.add_argument("--max-parallel", type=int, default=4, help="Maximum resumes in parallel")
    p.add_argument("--into-new-folder", action="store_true", help="Resume into a new folder (branching)")
    return p.parse_args(argv)


def main(argv: List[str]) -> int:
    args = parse_args(argv)
    orch = ResumeOrchestrator(
        session_root=args.session_root,
        maps_key=args.maps_key,
        signing_secret=args.signing_secret,
        max_parallel=args.max_parallel,
        into_new_folder=args.into_new_folder,
    )
    return orch.run()


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))


