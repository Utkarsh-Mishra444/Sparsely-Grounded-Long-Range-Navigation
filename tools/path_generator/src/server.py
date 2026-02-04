import os
import sys
import logging
import traceback

# Use a stable project root so path resolution doesn't depend on process CWD.
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

# Configure detailed logging
logging.basicConfig(
    level=logging.DEBUG,
    format='%(asctime)s [%(levelname)s] %(name)s:%(lineno)d - %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S'
)
logger = logging.getLogger("RouteService")
logger.setLevel(logging.DEBUG)

# Load .env file if present
try:
    from dotenv import load_dotenv
    load_dotenv()
    logger.info("Loaded .env file successfully")
except ImportError:
    logger.warning("python-dotenv not installed, using system env vars")
    pass  # python-dotenv not installed, use system env vars

import threading
import time
import uuid
from typing import Dict, Any, List, Optional
import json
from datetime import datetime

from flask import Flask, request, jsonify
try:
    from flask_cors import CORS
except Exception:
    CORS = None

# Import PanoCache directly to avoid launching browsers in environment
try:
    from .cache import PanoCache
except Exception as e:
    raise RuntimeError(f"Failed to import PanoCache: {e}")
try:
    from .cache import PathRunDB
except Exception:
    PathRunDB = None
try:
    # Import environment for remote fetch on cache miss
    from .environment import StreetViewEnvironment
except Exception:
    StreetViewEnvironment = None

# Import proxy server components
PANORAMA_PROXY_AVAILABLE = False
try:
    from .proxy import BrowserPool
    import subprocess
    PANORAMA_PROXY_AVAILABLE = True
except Exception as e:
    print(f"Warning: Panorama proxy not available: {e}")
    BrowserPool = None


# ----------------------
# Config (hardcoded)
# ----------------------
HOST = "127.0.0.1"
PORT = 8765
CACHE_PATH = os.path.join(PROJECT_ROOT, "cache", "pano")
STEP_DELAY_SEC = 0.01
ENABLE_REMOTE_FETCH = True   # enable fetch-and-cache on miss via environmentpro
USE_PROXY_SERVER = False     # Set to False to use direct browser, True to use proxy server
API_KEY = (os.environ.get("GOOGLE_MAPS_API_KEY") or "").strip()
ENV_INIT_COORDS = (40.7580, -73.9855)  # Times Square, known-good pano area
HEADING_SOFTMAX_TEMP_DEG = 200.0  # annealing: 45°→200° (moderate→exploratory)
WALK_DEBUG = False  # emit backend walk logs

# Log configuration details
logger.info("=" * 80)
logger.info("ROUTE SERVICE CONFIGURATION")
logger.info("=" * 80)
logger.info(f"HOST: {HOST}")
logger.info(f"PORT: {PORT}")
logger.info(f"CACHE_PATH: {CACHE_PATH}")
logger.info(f"ENABLE_REMOTE_FETCH: {ENABLE_REMOTE_FETCH}")
logger.info(f"USE_PROXY_SERVER: {USE_PROXY_SERVER}")
logger.info(f"API_KEY present: {bool(API_KEY)}")
logger.info(f"API_KEY length: {len(API_KEY)}")
logger.info(f"API_KEY first 10 chars: {API_KEY[:10] if API_KEY else 'EMPTY'}")
logger.info(f"ENV_INIT_COORDS: {ENV_INIT_COORDS}")
logger.info("=" * 80)

# Panorama proxy configuration
PANORAMA_PROXY_HOST = "127.0.0.1"
PANORAMA_PROXY_PORT = 12345
PANORAMA_PROXY_URL = f"http://{PANORAMA_PROXY_HOST}:{PANORAMA_PROXY_PORT}"
PANORAMA_PROXY_PROCESS = None  # Global proxy server process


# ----------------------
# Proxy Server Management
# ----------------------
def start_panorama_proxy():
    """Start the panorama proxy server if available."""
    global PANORAMA_PROXY_PROCESS

    # Only start the proxy server when explicitly requested.
    if not USE_PROXY_SERVER:
        return False

    if not PANORAMA_PROXY_AVAILABLE:
        print("Panorama proxy not available, falling back to direct browser access")
        return False

    try:
        # Check if proxy is already running
        import requests
        try:
            response = requests.get(f"{PANORAMA_PROXY_URL}/health", timeout=2)
            if response.status_code == 200:
                print("Panorama proxy server already running")
                return True
        except:
            pass

        print("Starting panorama proxy server...")
        project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        PANORAMA_PROXY_PROCESS = subprocess.Popen(
            [sys.executable, "-m", "src.proxy"],
            cwd=project_root,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True
        )

        # Wait for proxy to start
        import time
        for _ in range(10):  # Wait up to 10 seconds
            try:
                response = requests.get(f"{PANORAMA_PROXY_URL}/health", timeout=2)
                if response.status_code == 200:
                    health = response.json()
                    print(f"✓ Panorama proxy server started successfully with {health.get('browsers', 0)} browsers")
                    return True
            except:
                pass
            time.sleep(1)

        print("✗ Failed to start panorama proxy server")
        if PANORAMA_PROXY_PROCESS:
            PANORAMA_PROXY_PROCESS.terminate()
            PANORAMA_PROXY_PROCESS = None
        return False

    except Exception as e:
        print(f"Error starting panorama proxy: {e}")
        return False


def stop_panorama_proxy():
    """Stop the panorama proxy server."""
    global PANORAMA_PROXY_PROCESS

    if PANORAMA_PROXY_PROCESS:
        print("Stopping panorama proxy server...")
        try:
            PANORAMA_PROXY_PROCESS.terminate()
            PANORAMA_PROXY_PROCESS.wait(timeout=5)
            print("✓ Panorama proxy server stopped")
        except Exception as e:
            print(f"Error stopping proxy server: {e}")
            try:
                PANORAMA_PROXY_PROCESS.kill()
            except:
                pass
        PANORAMA_PROXY_PROCESS = None


# ----------------------
# App and Cache
# ----------------------
WEB_DIR = os.path.join(PROJECT_ROOT, 'web')
DATA_DIR = os.path.join(PROJECT_ROOT, 'data')

app = Flask(__name__)
if CORS is not None:
    # Allow browser calls from file:// or other localhost ports
    CORS(app, resources={r"/api/*": {"origins": "*"}})
cache = PanoCache(CACHE_PATH)

# Frontend port (separate from API to avoid request overhead)
FRONTEND_PORT = 8766

# Also handle shutdown via atexit for other deployment scenarios
import atexit
atexit.register(stop_panorama_proxy)
FETCH_LOCK = threading.Lock()
ENV_TL = threading.local()  # per-thread StreetViewEnvironment
LAST_FETCH_ERROR: Optional[str] = None
PATH_DB = PathRunDB(os.path.join(PROJECT_ROOT, "cache", "paths")) if PathRunDB is not None else None

# Multi-job/batch registries
JOBS_LOCK = threading.Lock()
JOBS: Dict[str, 'RouteJob'] = {}
BATCHES: Dict[str, List[str]] = {}
LAST_JOB_ID: Optional[str] = None


# ----------------------
# Single Job State
# ----------------------
class RouteJob:
    def __init__(self, seed_pano_id: str, target_m: float, random_forward: bool,
                 regress_window: int = 25, step_limit: Optional[int] = None, min_final_degree: int = 3,
                 heading_temp_deg: Optional[float] = None,
                 forced_first_child_id: Optional[str] = None,
                 batch_repulsion_data: Optional[Dict[str, float]] = None,
                 persist_to_db: bool = True,
                 bias_direction: Optional[float] = None,
                 bias_decision_count: int = 0,
                 visit_penalty_factor: float = 2.0):
        self.job_id = str(uuid.uuid4())
        self.seed_pano_id = seed_pano_id
        self.target_m = float(target_m)
        self.random_forward = bool(random_forward)
        self.regress_window = int(regress_window)
        self.step_limit = int(step_limit) if step_limit is not None else int(target_m * 2)
        self.min_final_degree = int(min_final_degree)
        self.heading_temp_deg = float(heading_temp_deg) if heading_temp_deg is not None else float(HEADING_SOFTMAX_TEMP_DEG)
        self.forced_first_child_id = forced_first_child_id
        self.bias_direction = bias_direction  # Direction in degrees (0-360) to bias towards, or None for no bias
        self.bias_decision_count = int(bias_decision_count)  # Number of decisions to apply bias to
        self.visit_penalty_factor = float(visit_penalty_factor)  # Exponential penalty factor for revisiting panos

        self.points: List[Dict[str, Any]] = []  # [{panoId, position:{lat,lng}}]
        self.visited: Dict[str, int] = {}  # panoId -> visit_count
        self.prev_pano_id: Optional[str] = None
        self.last_deltas: List[float] = []

        self.total_distance_m: float = 0.0
        self.done: bool = False
        self.stopped: bool = False
        self.error: Optional[str] = None
        self.message: str = ""

        # Annealing temperature parameters
        self.max_temperature_deg = float(heading_temp_deg) if heading_temp_deg is not None else float(HEADING_SOFTMAX_TEMP_DEG)
        self.min_temperature_deg = 45.0  # Early decisions: moderate focus

        # Decision tracking for first pano randomization
        self.decision_count = 0

        self._thread: Optional[threading.Thread] = None
        self.dead_edges_marked: int = 0
        self.events: List[str] = []

        # Batch repulsion: track current direction and taken edges for divergence
        self.current_bearing_deg: Optional[float] = None
        self.batch_repulsion_data: Optional[Dict[str, float]] = None  # job_id -> bearing_deg
        self.batch_id: Optional[str] = None  # which batch this job belongs to
        self.taken_edges: set[str] = set()  # edges taken by this job: "parent->child"
        # Whether this job should persist to LMDB when finished
        self.persist_to_db: bool = bool(persist_to_db)
        # Created timestamp (for optional commit metadata)
        try:
            self.created_at_utc: str = datetime.utcnow().isoformat(timespec='seconds') + 'Z'
        except Exception:
            self.created_at_utc = ''
        # Polygon metadata (filled by API layer if available)
        self.polygon_key: Optional[str] = None
        self.polygon_file: Optional[str] = None
        # Allow one-time automatic replacement if run ends at seed only
        self.allow_seed_replacement: bool = True

    def _event(self, message: str):
        try:
            logger.debug(f"[JOB:{self.job_id[:8]}] EVENT: {message}")
            if WALK_DEBUG:
                print(f"[WALK] {message}")
            self.events.append(message)
            if len(self.events) > 500:
                self.events = self.events[-500:]
        except Exception:
            pass

    def start(self):
        logger.info(f"[JOB:{self.job_id[:8]}] start() called")
        logger.info(f"[JOB:{self.job_id[:8]}] seed_pano_id: {self.seed_pano_id}")
        logger.info(f"[JOB:{self.job_id[:8]}] target_m: {self.target_m}")
        logger.info(f"[JOB:{self.job_id[:8]}] random_forward: {self.random_forward}")
        logger.info(f"[JOB:{self.job_id[:8]}] step_limit: {self.step_limit}")

        if self._thread and self._thread.is_alive():
            logger.warning(f"[JOB:{self.job_id[:8]}] Thread already running, skipping start")
            return

        logger.info(f"[JOB:{self.job_id[:8]}] Creating daemon thread for _run...")
        self._thread = threading.Thread(target=self._run, daemon=True)
        self._thread.start()
        logger.info(f"[JOB:{self.job_id[:8]}] Thread started: {self._thread.is_alive()}")

    def stop(self):
        logger.info(f"[JOB:{self.job_id[:8]}] stop() called")
        self.stopped = True

    def _run(self):
        logger.info(f"[JOB:{self.job_id[:8]}] _run() started in thread")
        try:
            logger.info(f"[JOB:{self.job_id[:8]}] About to call _walk()...")
            self._walk()
            logger.info(f"[JOB:{self.job_id[:8]}] _walk() completed normally")
        except Exception as e:
            logger.error(f"[JOB:{self.job_id[:8]}] !!! EXCEPTION IN _walk() !!!")
            logger.error(f"[JOB:{self.job_id[:8]}] Error: {type(e).__name__}: {e}")
            logger.error(f"[JOB:{self.job_id[:8]}] Traceback:\n{traceback.format_exc()}")
            self.error = str(e)
        finally:
            logger.info(f"[JOB:{self.job_id[:8]}] _run() finalizing, setting done=True")
            self.done = True
            # Persist points and finalize run in LMDB (if enabled)
            try:
                if PATH_DB is not None and self.persist_to_db:
                    # points
                    pts_payload = []
                    for idx, pt in enumerate(self.points):
                        pid = pt.get("panoId")
                        pos = pt.get("position", {})
                        lat = pos.get("lat")
                        lng = pos.get("lng")
                        lnks = links_for(pid) or []
                        lc = len(lnks)
                        if lc >= 3:
                            cls = "decision"
                        elif lc == 2:
                            cls = "corridor"
                        elif lc == 1:
                            cls = "dead_end"
                        else:
                            cls = "isolated"
                        pts_payload.append({
                            "idx": idx,
                            "panoId": pid,
                            "lat": lat,
                            "lng": lng,
                            "linksCount": lc,
                            "classification": cls,
                            "links": lnks,
                        })
                    PATH_DB.put_points(self.job_id, pts_payload)
                    # run summary
                    status_str = 'error' if self.error else ('stopped' if self.stopped else ('done' if self.done else 'running'))
                    PATH_DB.update_run(self.job_id, {
                        "finishedAt": datetime.utcnow().isoformat(timespec='seconds') + 'Z',
                        "status": status_str,
                        "message": self.error or self.message,
                        "totalDistanceMeters": self.total_distance_m,
                        "stepCount": len(self.points)
                    })
            except Exception:
                pass

            # If the job completed immediately at the seed (<= 1 point) without error or stop,
            # start a replacement job with a different nearby seed pano (one attempt only).
            try:
                if (not self.stopped and not self.error and len(self.points) < 2 and getattr(self, 'allow_seed_replacement', True)):
                    lnks = links_for(self.seed_pano_id) or []
                    candidates = [lnk.get('pano') for lnk in lnks if lnk.get('pano') and lnk.get('pano') != self.seed_pano_id]
                    if candidates:
                        import random
                        new_seed = random.choice(candidates)
                        repl = RouteJob(
                            seed_pano_id=new_seed,
                            target_m=self.target_m,
                            random_forward=self.random_forward,
                            regress_window=self.regress_window,
                            step_limit=self.step_limit,
                            min_final_degree=self.min_final_degree,
                            heading_temp_deg=self.heading_temp_deg,
                            persist_to_db=self.persist_to_db,
                            bias_direction=self.bias_direction,
                            bias_decision_count=self.bias_decision_count,
                            visit_penalty_factor=self.visit_penalty_factor,
                        )
                        # Prevent cascading replacements
                        repl.allow_seed_replacement = False
                        # Carry over polygon and batch metadata
                        try:
                            repl.polygon_key = self.polygon_key
                            repl.polygon_file = self.polygon_file
                        except Exception:
                            pass
                        repl.batch_id = self.batch_id

                        # Persist initial run record for replacement
                        try:
                            if PATH_DB is not None and repl.persist_to_db:
                                sc = coord_for(new_seed)
                                PATH_DB.put_run(repl.job_id, {
                                    "jobId": repl.job_id,
                                    "seedPanoId": new_seed,
                                    "seedLat": sc[0] if sc else None,
                                    "seedLng": sc[1] if sc else None,
                                    "startedAt": datetime.utcnow().isoformat(timespec='seconds') + 'Z',
                                    "status": "running",
                                    "message": "",
                                    "targetMeters": self.target_m,
                                    "randomForward": int(self.random_forward),
                                    "regressWindow": self.regress_window,
                                    "stepLimit": self.step_limit,
                                    "minFinalDegree": self.min_final_degree,
                                    "headingTempDeg": repl.heading_temp_deg,
                                    "polygonKey": getattr(repl, 'polygon_key', None),
                                    "polygonFile": repl.polygon_file,
                                })
                        except Exception:
                            pass

                        # Register and start
                        try:
                            global LAST_JOB_ID
                            with JOBS_LOCK:
                                JOBS[repl.job_id] = repl
                                if self.batch_id:
                                    try:
                                        BATCHES.setdefault(self.batch_id, []).append(repl.job_id)
                                    except Exception:
                                        pass
                                LAST_JOB_ID = repl.job_id
                            repl.start()
                        except Exception:
                            pass
            except Exception:
                pass

    # ----------------------
    # Temperature annealing
    # ----------------------
    def _current_temperature(self, current_distance_m: float) -> float:
        """Calculate current temperature based on distance progress from start.

        FIRST DECISION (decision_count == 0): Pure randomness (temp = 1000.0)
        SUBSEQUENT DECISIONS: Annealing from min_temperature_deg to max_temperature_deg
        based on distance progress from start.
        """
        if not self.random_forward:
            return self.max_temperature_deg

        # First decision after seed: pure randomness (very high temperature)
        # This ensures the initial direction is completely unbiased
        # UNLESS bias is active for the first decision
        if self.decision_count == 0:
            if self.bias_direction is not None and self.bias_decision_count > 0:
                return self.max_temperature_deg  # Use normal temperature for biased decisions
            else:
                return 1000.0  # Pure randomness when no bias

        if current_distance_m <= 0:
            return self.min_temperature_deg

        # Linear annealing: temperature increases with distance progress (0-500m range)
        progress_ratio = min(current_distance_m / 500.0, 1.0)
        current_temp = self.min_temperature_deg + (self.max_temperature_deg - self.min_temperature_deg) * progress_ratio

        return current_temp

    # ----------------------
    # Core walker (cache-only)
    # ----------------------
    def _walk(self):
        logger.info(f"[JOB:{self.job_id[:8]}] ========== _walk() STARTED ==========")
        logger.info(f"[JOB:{self.job_id[:8]}] seed_pano_id: {self.seed_pano_id}")
        logger.info(f"[JOB:{self.job_id[:8]}] target_m: {self.target_m}")
        logger.info(f"[JOB:{self.job_id[:8]}] step_limit: {self.step_limit}")
        logger.info(f"[JOB:{self.job_id[:8]}] min_final_degree: {self.min_final_degree}")
        logger.info(f"[JOB:{self.job_id[:8]}] random_forward: {self.random_forward}")

        # Ensure seed is cached
        logger.info(f"[JOB:{self.job_id[:8]}] Checking if seed is in cache...")
        start_coords = coord_for(self.seed_pano_id)
        logger.info(f"[JOB:{self.job_id[:8]}] Initial coord_for result: {start_coords}")

        if not start_coords:
            logger.info(f"[JOB:{self.job_id[:8]}] Seed not in cache, calling ensure_cached...")
            ensure_cached(self.seed_pano_id)
            start_coords = coord_for(self.seed_pano_id)
            logger.info(f"[JOB:{self.job_id[:8]}] After ensure_cached, coord_for result: {start_coords}")

        if not start_coords:
            # Attach last fetch error if available for easier diagnosis
            global LAST_FETCH_ERROR
            logger.error(f"[JOB:{self.job_id[:8]}] !!! FATAL: Seed pano not in cache !!!")
            logger.error(f"[JOB:{self.job_id[:8]}] seed_pano_id: {self.seed_pano_id}")
            logger.error(f"[JOB:{self.job_id[:8]}] LAST_FETCH_ERROR: {LAST_FETCH_ERROR}")
            detail = f"; fetch_error={LAST_FETCH_ERROR}" if LAST_FETCH_ERROR else ""
            self.error = f"Seed pano not in cache: {self.seed_pano_id}{detail}"
            logger.error(f"[JOB:{self.job_id[:8]}] Setting error: {self.error}")
            return

        logger.info(f"[JOB:{self.job_id[:8]}] Seed coords found: {start_coords}")
        logger.info(f"[JOB:{self.job_id[:8]}] Adding seed to points...")

        self.points.append({
            "panoId": self.seed_pano_id,
            "position": {"lat": start_coords[0], "lng": start_coords[1]}
        })
        self.visited[self.seed_pano_id] = 1
        self._event(f"SEED {self.seed_pano_id} at {start_coords}")
        logger.info(f"[JOB:{self.job_id[:8]}] Seed added to points. Points count: {len(self.points)}")

        current_id = self.seed_pano_id
        prev_id: Optional[str] = None
        step = 0
        move_count = 0
        junction_stack: List[Dict[str, Any]] = []

        logger.info(f"[JOB:{self.job_id[:8]}] ===== PHASE 1: CORRIDOR WALKING =====")
        # 1) Walk deterministically through corridors until first junction (degree ≥ 3)
        curr_coords = start_coords
        while not self.stopped:
            step += 1
            logger.debug(f"[JOB:{self.job_id[:8]}] Corridor step {step}: current_id={current_id}, move_count={move_count}")

            if move_count >= self.step_limit:
                logger.info(f"[JOB:{self.job_id[:8]}] Step limit reached: {move_count} >= {self.step_limit}")
                self.message = "Step limit reached"
                break

            logger.debug(f"[JOB:{self.job_id[:8]}] Calling ensure_cached({current_id})...")
            ensure_cached(current_id)

            links = links_for(current_id) or []
            logger.debug(f"[JOB:{self.job_id[:8]}] links_for({current_id}): {len(links)} links")

            fwd = forward_children(current_id, prev_id)
            logger.debug(f"[JOB:{self.job_id[:8]}] forward_children: {len(fwd)} forward options: {fwd}")

            if len(fwd) >= self.min_final_degree or len(fwd) >= 2:
                logger.info(f"[JOB:{self.job_id[:8]}] DECISION POINT reached! {len(fwd)} forward choices")
                # decision point (≥2 forward choices)
                break

            if len(fwd) == 0:
                logger.warning(f"[JOB:{self.job_id[:8]}] NO forward options at step {step}! Stopping corridor walk.")
                # corridor forces backtrack; nothing to do at seed
                self.message = "Seed corridor returns; no forward options"
                break

            # len(fwd) == 1 → forced forward
            nxt = fwd[0]
            logger.debug(f"[JOB:{self.job_id[:8]}] Forced forward to: {nxt}")
            ensure_cached(nxt)
            coords = coord_for(nxt)
            logger.debug(f"[JOB:{self.job_id[:8]}] Coords for {nxt}: {coords}")

            if not coords:
                logger.error(f"[JOB:{self.job_id[:8]}] !!! Missing coords for {nxt} during corridor !!!")
                # fetch failed; stop early
                self.message = "Missing coords during initial corridor"
                break

            prev_id, current_id = current_id, nxt
            self.taken_edges.add(f"{prev_id}->{current_id}")  # Track taken edge for batch repulsion
            self.visited[current_id] = self.visited.get(current_id, 0) + 1
            self.points.append({"panoId": current_id, "position": {"lat": coords[0], "lng": coords[1]}})
            move_count += 1
            delta = distance_m(start_coords, coords) - distance_m(start_coords, curr_coords)
            self.last_deltas.append(delta)
            if len(self.last_deltas) > self.regress_window:
                self.last_deltas.pop(0)
            curr_coords = coords
            self._event(f"CORRIDOR step {prev_id} -> {current_id}")
            logger.debug(f"[JOB:{self.job_id[:8]}] Corridor move: {prev_id} -> {current_id}, delta={delta:.1f}m")

            # target check during corridor
            dist_from_start = distance_m(start_coords, curr_coords)
            if dist_from_start >= self.target_m:
                logger.info(f"[JOB:{self.job_id[:8]}] Target reached during corridor! dist={dist_from_start:.1f}m >= target={self.target_m}m")
                break

        logger.info(f"[JOB:{self.job_id[:8]}] ===== PHASE 2: DECISION-BASED WALKING =====")
        logger.info(f"[JOB:{self.job_id[:8]}] After corridor: move_count={move_count}, points={len(self.points)}, stopped={self.stopped}")

        # 2) Decision-based walking using DFS with backtracking and dead-end marking
        while not self.stopped:
            step += 1
            if move_count >= self.step_limit:
                self.message = "Step limit reached"
                break
            cur_coords = coord_for(current_id)
            if not cur_coords:
                ensure_cached(current_id)
                cur_coords = coord_for(current_id)
                if not cur_coords:
                    self.message = "Missing coords for current pano"
                    break

            current_straight = distance_m(start_coords, cur_coords)

            # Current forward choices (excluding back and known dead edges)
            kids = forward_children(current_id, prev_id)

            # --- Dead end (leaf): backtrack and mark incoming edge dead ---
            if len(kids) == 0:
                if prev_id is not None:
                    if mark_dead_edge(prev_id, current_id):
                        self.dead_edges_marked += 1
                        self._event(f"LEAF at {current_id}; mark {prev_id}->{current_id} dead")
                # Backtrack to last decision
                if not junction_stack:
                    self.message = "Exhausted: no decisions left to backtrack"
                    break
                # Consider the top frame; remove the path to that frame
                frame = junction_stack[-1]
                self.points = self.points[:frame.get("path_len", len(self.points))]
                # Remove last chosen from available options and climb until options remain
                while junction_stack:
                    frame = junction_stack[-1]
                    # Normalize an options set on frame
                    opts = frame.get("options")
                    if opts is None:
                        frame["options"] = set(forward_children(frame["node"], frame.get("prev")))
                        opts = frame["options"]
                    last = frame.get("last_chosen")
                    if last in opts:
                        opts.discard(last)
                    # If options remain, choose next from here
                    if opts:
                        # Pick next candidate at this junction
                        import random
                        # Evaluate remaining candidates by distance from start
                        ev = []
                        for nxt in list(opts):
                            coords = coord_for(nxt)
                            if not coords:
                                ensure_cached(nxt)
                                coords = coord_for(nxt)
                                if not coords:
                                    # Allow candidates even without coords - will fetch when moving
                                    # Use a fallback distance estimate
                                    fallback_dist = current_straight + 10.0  # Assume ~10m ahead
                                    ev.append({"id": nxt, "coords": None, "dist": fallback_dist})
                                    continue
                            ev.append({"id": nxt, "coords": coords, "dist": distance_m(start_coords, coords)})
                        if not ev:
                            # No evaluable options; treat as exhausted
                            opts.clear()
                            continue
                        all_recent_neg = len(self.last_deltas) >= self.regress_window and all(d < 0 for d in self.last_deltas[-self.regress_window:])
                        def pick_candidate(evlist):
                            if self.random_forward:
                                # For random mode, prefer candidates with coords for heading calculation
                                candidates_with_coords = [e for e in evlist if e["coords"] is not None]
                                if candidates_with_coords:
                                    current_temp = self._current_temperature(current_straight)
                                    nid = softmax_sample_by_heading(frame["node"], [e["id"] for e in candidates_with_coords], start_coords, current_temp,
                                                                   self.bias_direction, self.decision_count, self.bias_decision_count,
                                                                   self.visited, self.visit_penalty_factor)
                                    cand = next((e for e in candidates_with_coords if e["id"] == nid), None)
                                    if cand:
                                        return cand
                                return random.choice(evlist)
                            if all_recent_neg:
                                return max(evlist, key=lambda e: e["dist"] - current_straight)
                            improving = [e for e in evlist if e["dist"] > current_straight]
                            return max(improving, key=lambda e: e["dist"]) if improving else random.choice(evlist)
                        # Respect forced first decision if applicable
                        if self.forced_first_child_id is not None:
                            forced = next((e for e in ev if e["id"] == self.forced_first_child_id), None)
                            cand = forced if forced is not None else pick_candidate(ev)
                            self.forced_first_child_id = None
                        else:
                            cand = pick_candidate(ev)
                        nxt_id = cand["id"]
                        frame["last_chosen"] = nxt_id
                        self._event(f"BACKTRACK choose from {frame['node']} -> {nxt_id}")
                        # Move from the frame node to nxt_id
                        prev_id, current_id = frame["node"], nxt_id
                        # Increment decision counter after successful decision
                        self.decision_count += 1
                        ensure_cached(current_id)
                        ncoords = coord_for(current_id)
                        if not ncoords:
                            # Try one more time with fresh fetch
                            ensure_cached(current_id)
                            ncoords = coord_for(current_id)
                            if not ncoords:
                                # Still no coords; treat as transient miss; skip persisting dead-edge
                                self._event(f"MISS coords at {current_id}; skip marking {prev_id}->{current_id}")
                                continue
                        self.visited[current_id] = self.visited.get(current_id, 0) + 1
                        self.points.append({"panoId": current_id, "position": {"lat": ncoords[0], "lng": ncoords[1]}})
                        move_count += 1
                        delta = distance_m(start_coords, ncoords) - current_straight
                        self.last_deltas.append(delta)
                        if len(self.last_deltas) > self.regress_window:
                            self.last_deltas.pop(0)
                        current_straight = distance_m(start_coords, ncoords)
                        # Proceed to next loop iteration after advancing
                        break
                    # No options remain: logical dead end at this junction
                    # Mark incoming edge to this junction as dead and pop
                    parent = frame.get("prev")
                    if parent is not None:
                        if mark_dead_edge(parent, frame["node"]):
                            self.dead_edges_marked += 1
                            self._event(f"JX EXHAUSTED at {frame['node']}; mark {parent}->{frame['node']} dead")
                    junction_stack.pop()
                    # Trim path to previous frame (if any)
                    if junction_stack:
                        prev_frame = junction_stack[-1]
                        self.points = self.points[:prev_frame.get("path_len", len(self.points))]
                else:
                    # No frames left
                    self.message = "Exhausted: all decisions are dead ends"
                    break
                # Continue outer while loop after backtrack/advance
                time.sleep(STEP_DELAY_SEC)
                continue

            # --- Corridor: forced advance ---
            if len(kids) == 1:
                nxt = kids[0]
                ensure_cached(nxt)
                coords = coord_for(nxt)
                if not coords:
                    # Try one more time with fresh fetch
                    ensure_cached(nxt)
                    coords = coord_for(nxt)
                    if not coords:
                        # Still no coords; treat as transient; skip persisting dead-edge and retry loop
                        self._event(f"TRANSIENT miss at {nxt}; skip marking {current_id}->{nxt}")
                        time.sleep(STEP_DELAY_SEC)
                        continue
                prev_id, current_id = current_id, nxt
                self.taken_edges.add(f"{prev_id}->{current_id}")  # Track taken edge for batch repulsion
                self.visited[current_id] = self.visited.get(current_id, 0) + 1
                self.points.append({"panoId": current_id, "position": {"lat": coords[0], "lng": coords[1]}})
                move_count += 1
                delta = distance_m(start_coords, coords) - current_straight
                self.last_deltas.append(delta)
                if len(self.last_deltas) > self.regress_window:
                    self.last_deltas.pop(0)
                current_straight = distance_m(start_coords, coords)
                self._event(f"CORRIDOR step {prev_id} -> {current_id}")
                # Target check
                if current_straight >= self.target_m:
                    # Post-target: ensure final node has required degree
                    import random
                    guard = 20
                    while guard > 0:
                        guard -= 1
                        ensure_cached(current_id)
                        all_links = links_for(current_id) or []
                        if len(all_links) >= self.min_final_degree:
                            break
                        fwd_links = forward_children(current_id, prev_id)
                        if not fwd_links:
                            break
                        current_temp = self._current_temperature(current_straight)
                        nxt2 = softmax_sample_by_heading(current_id, fwd_links, start_coords, current_temp,
                                                       self.bias_direction, self.decision_count, self.bias_decision_count,
                                                       self.visited, self.visit_penalty_factor)
                        ensure_cached(nxt2)
                        coords2 = coord_for(nxt2)
                        if not coords2:
                            continue
                        prev_id, current_id = current_id, nxt2
                        # Increment decision counter after successful decision
                        self.decision_count += 1
                        self.taken_edges.add(f"{prev_id}->{current_id}")  # Track taken edge for batch repulsion
                        self.visited[current_id] = self.visited.get(current_id, 0) + 1
                        self.points.append({"panoId": current_id, "position": {"lat": coords2[0], "lng": coords2[1]}})
                        delta = distance_m(start_coords, coords2) - current_straight
                        self.last_deltas.append(delta)
                        if len(self.last_deltas) > self.regress_window:
                            self.last_deltas.pop(0)
                        current_straight = distance_m(start_coords, coords2)
                    # Target reached - mark as done and exit main loop
                    self.done = True
                    self.message = f"Target distance {self.target_m/1000:.1f}km reached"
                    break
                time.sleep(STEP_DELAY_SEC)
                continue

            # --- Decision point: push frame and choose an option ---
            # Ensure top frame corresponds to this junction
            top = junction_stack[-1] if junction_stack else None
            if not top or top.get("node") != current_id:
                junction_stack.append({
                    "node": current_id,
                    "prev": prev_id,
                    "path_len": len(self.points),
                    "options": set(kids),
                    "last_chosen": None,
                })
                self._event(f"PUSH JX at {current_id} with {len(kids)} options")
            else:
                # Refresh options to current kids (preserve removals)
                opts = top.get("options")
                if opts is None:
                    top["options"] = set(kids)
                else:
                    top["options"] &= set(kids)

            frame = junction_stack[-1]
            opts = frame.get("options") or set()
            # Evaluate candidates for picking
            import random
            ev = []
            for nid in list(opts):
                coords = coord_for(nid)
                if not coords:
                    ensure_cached(nid)
                    coords = coord_for(nid)
                    if not coords:
                        # Allow candidates even without coords - will fetch when moving
                        # Use a fallback distance estimate (could be improved)
                        fallback_dist = current_straight + 10.0  # Assume ~10m ahead
                        ev.append({"id": nid, "coords": None, "dist": fallback_dist})
                        continue
                ev.append({"id": nid, "coords": coords, "dist": distance_m(start_coords, coords)})
            if not ev:
                # No evaluable options; treat as dead end and trigger backtrack next loop
                kids = []
                time.sleep(STEP_DELAY_SEC)
                continue

            all_recent_neg = len(self.last_deltas) >= self.regress_window and all(d < 0 for d in self.last_deltas[-self.regress_window:])
            def pick_candidate(evlist):
                if self.random_forward:
                    # For random mode, prefer candidates with coords for heading calculation
                    candidates_with_coords = [e for e in evlist if e["coords"] is not None]
                    if candidates_with_coords:
                        current_temp = self._current_temperature(current_straight)
                        nid = softmax_sample_by_heading(current_id, [e["id"] for e in candidates_with_coords], start_coords, current_temp,
                                                       self.bias_direction, self.decision_count, self.bias_decision_count,
                                                       self.visited, self.visit_penalty_factor)
                        cand = next((e for e in candidates_with_coords if e["id"] == nid), None)
                        if cand:
                            return cand
                    return random.choice(evlist)
                if all_recent_neg:
                    return max(evlist, key=lambda e: e["dist"] - current_straight)
                improving = [e for e in evlist if e["dist"] > current_straight]
                return max(improving, key=lambda e: e["dist"]) if improving else random.choice(evlist)

            if self.forced_first_child_id is not None:
                forced = next((e for e in ev if e["id"] == self.forced_first_child_id), None)
                cand = forced if forced is not None else pick_candidate(ev)
                self.forced_first_child_id = None
            else:
                cand = pick_candidate(ev)
            nxt_id = cand["id"]
            frame["last_chosen"] = nxt_id
            self._event(f"CHOOSE at {current_id} -> {nxt_id}")

            # Advance one step into the chosen option
            ensure_cached(nxt_id)
            ncoords = coord_for(nxt_id)
            if not ncoords:
                # Try one more time with fresh fetch
                ensure_cached(nxt_id)
                ncoords = coord_for(nxt_id)
                if not ncoords:
                    # Still no coords - treat as transient; skip persisting dead-edge and retry next iteration
                    self._event(f"MISS coords at {nxt_id}; skip marking {current_id}->{nxt_id}")
                    time.sleep(STEP_DELAY_SEC)
                    continue
            prev_id, current_id = current_id, nxt_id
            # Increment decision counter after successful decision
            self.decision_count += 1
            self.taken_edges.add(f"{prev_id}->{current_id}")  # Track taken edge for batch repulsion
            self.visited[current_id] = self.visited.get(current_id, 0) + 1
            self.points.append({"panoId": current_id, "position": {"lat": ncoords[0], "lng": ncoords[1]}})
            move_count += 1
            delta = distance_m(start_coords, ncoords) - current_straight
            self.last_deltas.append(delta)
            if len(self.last_deltas) > self.regress_window:
                self.last_deltas.pop(0)
            current_straight = distance_m(start_coords, ncoords)

            # Target check after advancing from decision
            if current_straight >= self.target_m:
                import random
                guard = 20
                while guard > 0:
                    guard -= 1
                    ensure_cached(current_id)
                    all_links = links_for(current_id) or []
                    if len(all_links) >= self.min_final_degree:
                        break
                    fwd_links = forward_children(current_id, prev_id, get_batch_repulsion_edges(self))
                    if not fwd_links:
                        break
                    current_temp = self._current_temperature(current_straight)
                    nxt2 = softmax_sample_by_heading(current_id, fwd_links, start_coords, current_temp,
                                                   self.bias_direction, self.decision_count, self.bias_decision_count,
                                                   self.visited, self.visit_penalty_factor)
                    ensure_cached(nxt2)
                    coords2 = coord_for(nxt2)
                    if not coords2:
                        continue
                    prev_id, current_id = current_id, nxt2
                    # Increment decision counter after successful decision
                    self.decision_count += 1
                    self.taken_edges.add(f"{prev_id}->{current_id}")  # Track taken edge for batch repulsion
                    self.visited[current_id] = self.visited.get(current_id, 0) + 1
                    self.points.append({"panoId": current_id, "position": {"lat": coords2[0], "lng": coords2[1]}})
                    move_count += 1
                    delta = distance_m(start_coords, coords2) - current_straight
                    self.last_deltas.append(delta)
                    if len(self.last_deltas) > self.regress_window:
                        self.last_deltas.pop(0)
                    current_straight = distance_m(start_coords, coords2)
                # Target reached - mark as done and exit main loop
                self.done = True
                self.message = f"Target distance {self.target_m/1000:.1f}km reached"
                break

            time.sleep(STEP_DELAY_SEC)

        # Set a default stop reason if none was recorded
        if not self.message:
            if self.stopped:
                self.message = "Stopped by user"
            elif step >= self.step_limit:
                self.message = "Step limit reached"
            else:
                self.message = "Finished"
        try:
            end_pos = self.points[-1]["position"] if self.points else None
            dist = distance_m(start_coords, (end_pos["lat"], end_pos["lng"])) if end_pos else 0.0
            self._event(f"DONE: {self.message}; steps={step}; dist={dist:.1f}m")
        except Exception:
            pass

        # Finalize
        if self.points:
            end_pos = self.points[-1]["position"]
            self.total_distance_m = distance_m(start_coords, (end_pos["lat"], end_pos["lng"]))
        else:
            self.total_distance_m = 0.0


# Global single job
JOB_LOCK = threading.Lock()
JOB: Optional[RouteJob] = None


# ----------------------
# Helpers
# ----------------------
def coord_for(pano_id: str) -> Optional[tuple[float, float]]:
    try:
        lat, lng = cache.coord_for(pano_id)
        if lat is None or lng is None:
            return None
        return (float(lat), float(lng))
    except Exception:
        return None


def links_for(pano_id: str) -> Optional[List[Dict[str, Any]]]:
    try:
        return cache.links_for(pano_id) or []
    except Exception:
        return []


# ----- Polygon validation -----
def point_in_polygon(lat: float, lng: float, polygon: List[List[float]]) -> bool:
    inside = False
    n = len(polygon)
    if n < 3:
        return False

    # Check if polygon is clockwise (most GIS polygons are clockwise)
    # If clockwise, we'll reverse the logic
    sum_val = 0
    for i in range(n):
        j = (i + 1) % n
        sum_val += (polygon[j][1] - polygon[i][1]) * (polygon[j][0] + polygon[i][0])

    is_clockwise = sum_val > 0

    for i in range(n):
        j = (i - 1) % n
        yi, xi = polygon[i][0], polygon[i][1]
        yj, xj = polygon[j][0], polygon[j][1]
        intersect = ((xi > lng) != (xj > lng)) and (lat < (yj - yi) * (lng - xi) / (xj - xi + 1e-12) + yi)
        if intersect:
            inside = not inside

    # If polygon is clockwise, flip the result
    return inside if not is_clockwise else not inside


def load_valid_polygons(file_path: Optional[str] = None) -> Dict[str, List[List[float]]]:
    """Load polygons for validation.

    If file_path is provided, load from that JSON file. The file may be in one of two formats:
    - Grouped: { editedPolygons: { city: { landmarkName: { index, coordinates, polygon } } } }
    - Flat:    { editedPolygons: { key: polygon } }

    If file_path is not provided, fall back to the legacy example/dated files.
    """
    def _resolve_path(p: str) -> str:
        return p if os.path.isabs(p) else os.path.join(PROJECT_ROOT, p)

    def _load_from_json(path: str) -> Dict[str, List[List[float]]]:
        out: Dict[str, List[List[float]]] = {}
        try:
            with open(path, 'r', encoding='utf-8') as f:
                data = json.load(f)
        except Exception:
            return out
        ep = data.get('editedPolygons') or {}
        # Detect grouped vs flat
        if ep and isinstance(next(iter(ep.values()), None), dict) and any(
            isinstance(v, dict) and any(isinstance(x, dict) and 'polygon' in x for x in v.values())
            for v in ep.values()
        ):
            # Grouped by city -> landmarkName -> { index, coordinates, polygon }
            for city, landmarks in ep.items():
                if not isinstance(landmarks, dict):
                    continue
                for name, obj in landmarks.items():
                    if not isinstance(obj, dict):
                        continue
                    poly = obj.get('polygon')
                    if not poly:
                        continue
                    # Use name-based keys instead of index-based for robustness
                    normalized_name = name.replace(' ', '_').replace('-', '_').lower()
                    key = f"{city}_{normalized_name}"
                    out[key] = poly
        else:
            # Flat mapping key -> polygon
            for key, poly in ep.items():
                if isinstance(poly, list):
                    out[str(key)] = poly
        return out

    # Preferred path (provided by client)
    if file_path:
        return _load_from_json(_resolve_path(file_path))

    # Load polygons.json from data/ - no fallbacks
    latest_path = os.path.join(os.path.dirname(__file__), '..', 'data', 'polygons.json')
    polygons = _load_from_json(latest_path)

    if not polygons:
        raise RuntimeError(
            f"Failed to load polygon data from data/polygons.json. "
            f"Please ensure the file exists and contains valid polygon data. "
            f"File path: {latest_path}"
        )

    return polygons


def load_landmark_name_mapping() -> Dict[str, str]:
    """Load mapping from polygon keys to proper landmark names from data/landmarks.js"""
    import re
    mapping = {}

    try:
        # Read the landmarks.js file
        landmarks_path = os.path.join(os.path.dirname(__file__), '..', 'data', 'landmarks.js')
        with open(landmarks_path, 'r', encoding='utf-8') as f:
            content = f.read()

        # Extract the JSON part (everything between window.ALL_DATA = { and };)
        json_match = re.search(r'window\.ALL_DATA\s*=\s*({.*?});', content, re.DOTALL)
        if not json_match:
            return mapping

        landmarks_data = json.loads(json_match.group(1))

        # Build mapping from polygon keys to proper names
        for city, landmarks in landmarks_data.items():
            for landmark in landmarks:
                if len(landmark) >= 4:  # [name, lat, lng, polygon]
                    name = landmark[0]
                    # Create normalized polygon key format
                    normalized_name = name.replace(' ', '_').replace('-', '_').lower()
                    polygon_key = f"{city}_{normalized_name}"
                    mapping[polygon_key] = name

        return mapping

    except Exception as e:
        print(f"Warning: Failed to load landmark name mapping: {e}")
        return mapping


def load_polygon_metadata(file_path: Optional[str] = None) -> Dict[str, Dict[str, Any]]:
    """Return mapping polygonKey -> { polygon, coordinates, city, name }.

    Supports both grouped and flat formats. If file_path is None, falls back to example/dated files.
    """
    def _resolve_path(p: str) -> str:
        return p if os.path.isabs(p) else os.path.join(PROJECT_ROOT, p)

    def _load_meta_from_json(path: str) -> Dict[str, Dict[str, Any]]:
        out: Dict[str, Dict[str, Any]] = {}
        try:
            with open(path, 'r', encoding='utf-8') as f:
                data = json.load(f)
        except Exception:
            return out

        # Load landmark name mapping for proper names
        landmark_mapping = load_landmark_name_mapping()

        ep = data.get('editedPolygons') or {}
        # Grouped format
        if ep and isinstance(next(iter(ep.values()), None), dict) and any(
            isinstance(v, dict) and any(isinstance(x, dict) and 'polygon' in x for x in v.values())
            for v in ep.values()
        ):
            for city, landmarks in ep.items():
                if not isinstance(landmarks, dict):
                    continue
                for name, obj in landmarks.items():
                    if not isinstance(obj, dict):
                        continue
                    poly = obj.get('polygon')
                    coords = obj.get('coordinates')
                    idx = obj.get('index', 0)

                    # Create normalized key for mapping lookup
                    normalized_name = name.replace(' ', '_').replace('-', '_').lower()
                    key = f"{city}_{normalized_name}"

                    # Try to get proper landmark name from mapping, fallback to original name
                    proper_name = landmark_mapping.get(key, name)

                    out[key] = {
                        'polygon': poly,
                        'coordinates': coords,
                        'city': city,
                        'name': proper_name,
                    }
        else:
            # Flat mapping - try to map keys to proper names
            for key, poly in ep.items():
                key_str = str(key)
                # Try to get proper landmark name from mapping, fallback to key
                proper_name = landmark_mapping.get(key_str, key_str)
                out[key_str] = {
                    'polygon': poly,
                    'coordinates': None,
                    'city': None,
                    'name': proper_name,
                }
        return out

    if file_path:
        meta = _load_meta_from_json(_resolve_path(file_path))
        if not meta:
            raise RuntimeError(
                f"Failed to load polygon metadata from {file_path}. "
                f"Please ensure the file exists and contains valid polygon data."
            )
        return meta

    # Load polygons.json from data/ - no fallbacks
    latest_path = os.path.join(os.path.dirname(__file__), '..', 'data', 'polygons.json')
    meta = _load_meta_from_json(latest_path)

    if not meta:
        raise RuntimeError(
            f"Failed to load polygon metadata from data/polygons.json. "
            f"Please ensure the file exists and contains valid polygon data. "
            f"File path: {latest_path}"
        )

    return meta


def ensure_cached(pano_id: str):
    global LAST_FETCH_ERROR
    logger.debug(f"[ENSURE_CACHED] Called for pano_id={pano_id}")
    logger.debug(f"[ENSURE_CACHED] ENABLE_REMOTE_FETCH={ENABLE_REMOTE_FETCH}, StreetViewEnvironment={StreetViewEnvironment is not None}")

    if not ENABLE_REMOTE_FETCH or StreetViewEnvironment is None:
        logger.warning(f"[ENSURE_CACHED] Skipping - ENABLE_REMOTE_FETCH={ENABLE_REMOTE_FETCH}, StreetViewEnvironment available={StreetViewEnvironment is not None}")
        return

    if not API_KEY:
        logger.error("[ENSURE_CACHED] !!! MISSING GOOGLE_MAPS_API_KEY !!!")
        logger.error(f"[ENSURE_CACHED] API_KEY value: '{API_KEY}'")
        logger.error(f"[ENSURE_CACHED] Environment variable GOOGLE_MAPS_API_KEY: '{os.environ.get('GOOGLE_MAPS_API_KEY', 'NOT SET')}'")
        LAST_FETCH_ERROR = "Missing GOOGLE_MAPS_API_KEY"
        return

    logger.debug(f"[ENSURE_CACHED] API_KEY is present (length={len(API_KEY)})")

    try:
        # fast path: already have both pieces
        latlng = cache.coord_for(pano_id)
        links = cache.links_for(pano_id)
        logger.debug(f"[ENSURE_CACHED] Cache check - latlng={latlng}, links={links is not None and len(links) if links else 'None'} links")

        if latlng and links:
            logger.debug(f"[ENSURE_CACHED] CACHE HIT for {pano_id}")
            return

        logger.info(f"[ENSURE_CACHED] CACHE MISS for {pano_id}, will fetch from API")

        # per-thread environment
        env = getattr(ENV_TL, 'env', None)
        logger.debug(f"[ENSURE_CACHED] Thread-local env exists: {env is not None}")

        if env is None:
            logger.info(f"[ENSURE_CACHED] Creating new StreetViewEnvironment for this thread")
            logger.debug(f"[ENSURE_CACHED] ENV_INIT_COORDS={ENV_INIT_COORDS}")
            logger.debug(f"[ENSURE_CACHED] API_KEY (first 10 chars)={API_KEY[:10] if API_KEY else 'EMPTY'}")
            logger.debug(f"[ENSURE_CACHED] USE_PROXY_SERVER={USE_PROXY_SERVER}, PANORAMA_PROXY_AVAILABLE={PANORAMA_PROXY_AVAILABLE}")
            proxy_url_to_use = PANORAMA_PROXY_URL if USE_PROXY_SERVER and PANORAMA_PROXY_AVAILABLE else None
            logger.debug(f"[ENSURE_CACHED] proxy_url to use: {proxy_url_to_use}")

            # destination coords are unused; pass seed same as init
            with FETCH_LOCK:
                env = getattr(ENV_TL, 'env', None)
                if env is None:
                    logger.info(f"[ENSURE_CACHED] About to instantiate StreetViewEnvironment...")
                    logger.debug(f"[ENSURE_CACHED] Parameters: initial_coords={ENV_INIT_COORDS}, initial_pano_id={pano_id}, proxy_url={proxy_url_to_use}")
                    try:
                        env = StreetViewEnvironment(
                            initial_coords=ENV_INIT_COORDS,
                            destination_coords=ENV_INIT_COORDS,
                            api_key=API_KEY,
                            initial_pano_id=pano_id,
                            proxy_url=PANORAMA_PROXY_URL if USE_PROXY_SERVER and PANORAMA_PROXY_AVAILABLE else None
                        )
                        logger.info(f"[ENSURE_CACHED] StreetViewEnvironment created successfully")
                        logger.debug(f"[ENSURE_CACHED] Environment state: {env.state if hasattr(env, 'state') else 'N/A'}")
                    except Exception as env_err:
                        logger.error(f"[ENSURE_CACHED] !!! FAILED TO CREATE StreetViewEnvironment !!!")
                        logger.error(f"[ENSURE_CACHED] Error: {type(env_err).__name__}: {env_err}")
                        logger.error(f"[ENSURE_CACHED] Traceback:\n{traceback.format_exc()}")
                        raise
                    ENV_TL.env = env

        # Fetch via environment; this inserts into cache
        logger.info(f"[ENSURE_CACHED] Calling env.fetch_pano_data({pano_id})...")
        with FETCH_LOCK:
            result = env.fetch_pano_data(pano_id)
            logger.info(f"[ENSURE_CACHED] fetch_pano_data result: {result}")
            if result:
                logger.debug(f"[ENSURE_CACHED] Result keys: {result.keys() if isinstance(result, dict) else 'not a dict'}")
                if isinstance(result, dict):
                    logger.debug(f"[ENSURE_CACHED] links count: {len(result.get('links', [])) if result.get('links') else 0}")
                    logger.debug(f"[ENSURE_CACHED] location: {result.get('location')}")
            else:
                logger.warning(f"[ENSURE_CACHED] fetch_pano_data returned None/empty for {pano_id}")

    except Exception as e:
        # Surface the error so callers/UI can see why cache population failed
        logger.error(f"[ENSURE_CACHED] !!! EXCEPTION DURING CACHE OPERATION !!!")
        logger.error(f"[ENSURE_CACHED] Exception type: {type(e).__name__}")
        logger.error(f"[ENSURE_CACHED] Exception message: {e}")
        logger.error(f"[ENSURE_CACHED] Full traceback:\n{traceback.format_exc()}")
        LAST_FETCH_ERROR = f"{type(e).__name__}: {e}"
        # Keep traversal robust: on any error, fallback to existing cache state
        pass


def forward_children(current_id: str, prev_id: Optional[str], batch_repulsion_edges: Optional[set[str]] = None) -> List[str]:
    """Return forward candidate pano IDs from current_id excluding:
    - immediate back edge (prev_id)
    - any link marked dead-end in cache for current_id
    - edges taken by other jobs in the same batch (batch_repulsion_edges)
    """
    ensure_cached(current_id)
    raw = links_for(current_id) or []
    try:
        dead = set(cache.dead_children_for(current_id) or [])
    except Exception:
        dead = set()
    kids = []
    for ln in raw:
        nid = ln.get("pano")
        if not nid:
            continue
        if prev_id and nid == prev_id:
            continue
        if nid in dead:
            continue
        # Exclude edges taken by other jobs in the batch
        if batch_repulsion_edges and f"{current_id}->{nid}" in batch_repulsion_edges:
            continue
        kids.append(nid)
    return kids


def mark_dead_edge(parent_id: str, child_id: str) -> bool:
    try:
        cache.mark_dead_edge(parent_id, child_id)
        return True
    except Exception:
        return False


def get_batch_repulsion_edges(job: 'RouteJob') -> Optional[set[str]]:
    """Get edges taken by other jobs in the same batch as the given job."""
    if not job.batch_id:
        return None

    with JOBS_LOCK:
        batch_jobs = BATCHES.get(job.batch_id, [])
        taken_edges = set()
        for jid in batch_jobs:
            if jid != job.job_id and jid in JOBS:
                other_job = JOBS[jid]
                taken_edges.update(other_job.taken_edges)
        return taken_edges if taken_edges else None


def corridor_probe(parent_id: str, child_id: str, start_coords: tuple[float, float], max_steps: int = 100) -> Dict[str, Any]:
    """Deterministically walk from parent->child until:
    - return to parent before hitting a junction (≥ 2 forward choices) => return_to_parent (dead edge)
    - hit a junction (≥ 2 forward choices) => junction (path and tip)
    - hit a leaf (0 forward children) => leaf (path and tip)
    - exceed step budget or missing data => unknown
    Probe allows revisits and ignores global visited; only immediate back is disallowed.
    """
    path: List[Dict[str, Any]] = []
    prev = parent_id
    curr = child_id
    steps = 0
    while steps < max_steps:
        steps += 1
        ensure_cached(curr)
        coords = coord_for(curr)
        if not coords:
            return {"status": "unknown", "reason": "no_coords"}
        path.append({"id": curr, "coords": coords})
        if curr == parent_id:
            return {"status": "return_to_parent", "path": path}
        kids = forward_children(curr, prev)
        if len(kids) >= 2:
            return {"status": "junction", "path": path, "tipCoords": coords, "tipDist": distance_m(start_coords, coords)}
        if len(kids) == 0:
            return {"status": "leaf", "path": path, "tipCoords": coords, "tipDist": distance_m(start_coords, coords)}
        # forced corridor step (len=1 or 2 but one is back, so effective one)
        nxt = kids[0]
        prev, curr = curr, nxt
    return {"status": "unknown", "reason": "budget"}


def job_status(J: 'RouteJob') -> Dict[str, Any]:
    return {
        "jobId": J.job_id,
        "status": "stopped" if J.stopped else ("error" if J.error else ("done" if J.done else "running")),
        "message": J.message,
        "error": J.error,
        "points": J.points,
        "totalDistanceMeters": J.total_distance_m,
        "stepCount": len(J.points),
        "deadEdgesMarked": getattr(J, "dead_edges_marked", 0),
        "events": getattr(J, "events", [])[-100:]
    }


def first_junction_children(seed_pano_id: str) -> tuple[Optional[str], List[str]]:
    prev = None
    curr = seed_pano_id
    steps = 0
    while steps < 200:
        steps += 1
        ensure_cached(curr)
        kids = forward_children(curr, prev)
        if len(kids) >= 2:
            return curr, kids
        if len(kids) == 0:
            return curr, []
        # forced corridor advance
        prev, curr = curr, kids[0]
    return curr, []

def distance_m(a: tuple[float, float], b: tuple[float, float]) -> float:
    import math
    R = 6371000.0
    lat1 = math.radians(a[0])
    lat2 = math.radians(b[0])
    dlat = math.radians(b[0] - a[0])
    dlng = math.radians(b[1] - a[1])
    sa = math.sin(dlat/2) ** 2 + math.cos(lat1) * math.cos(lat2) * math.sin(dlng/2) ** 2
    c = 2 * math.atan2(math.sqrt(sa), math.sqrt(1 - sa))
    return R * c


def bearing_deg(a: tuple[float, float], b: tuple[float, float]) -> float:
    import math
    lat1 = math.radians(a[0])
    lat2 = math.radians(b[0])
    dlon = math.radians(b[1] - a[1])
    y = math.sin(dlon) * math.cos(lat2)
    x = math.cos(lat1) * math.sin(lat2) - math.sin(lat1) * math.cos(lat2) * math.cos(dlon)
    brng = math.degrees(math.atan2(y, x))
    return (brng + 360.0) % 360.0


def angle_diff_deg(a: float, b: float) -> float:
    d = abs((a - b) % 360.0)
    return 360.0 - d if d > 180.0 else d


def link_heading(current_id: str, child_id: str) -> float | None:
    # Try cached link heading
    try:
        links = links_for(current_id) or []
        for ln in links:
            if ln.get("pano") == child_id and "heading" in ln:
                h = ln.get("heading")
                if isinstance(h, (int, float)):
                    return float(h)
    except Exception:
        pass
    # Derive from coords
    a = coord_for(current_id)
    b = coord_for(child_id)
    if not a or not b:
        ensure_cached(child_id)
        b = coord_for(child_id)
    if a and b:
        return bearing_deg(a, b)
    return None


def softmax_sample_by_heading(current_id: str, candidates: List[str], start_coords: tuple[float, float], tau_deg: float,
                              bias_direction: Optional[float] = None, decision_count: int = 0, bias_decision_count: int = 0,
                              visited_counts: Optional[Dict[str, int]] = None, visit_penalty_factor: float = 2.0) -> Optional[str]:
    import math, random

    # Apply bias if within the biased decision count and bias direction is set
    if bias_direction is not None and decision_count < bias_decision_count:
        desired = bias_direction
    else:
        # Desired is away from start: from current toward away from start means bearing from current to start plus 180
        curr = coord_for(current_id)
        if not curr:
            ensure_cached(current_id)
            curr = coord_for(current_id)
            if not curr: #
                return random.choice(candidates) if candidates else None
        to_start = bearing_deg(curr, start_coords)
        desired = (to_start + 180.0) % 360.0

    # If we don't have coordinates for bias calculation, fall back to random
    if bias_direction is not None and decision_count < bias_decision_count:
        # For bias mode, we can use coordinates if available, but bias overrides
        pass
    elif not curr:
        return random.choice(candidates) if candidates else None

    scores = []
    for nid in candidates:
        h = link_heading(current_id, nid)
        if h is None:
            # fallback neutral weight
            base_w = 1.0
        else:
            ang = angle_diff_deg(h, desired)
            base_w = math.exp(-(ang / max(1e-3, tau_deg)))

        # Apply exponential penalty for previously visited panos
        if visited_counts is not None:
            visit_count = visited_counts.get(nid, 0)
            if visit_count > 0:
                # Exponential penalty: exp(-visit_count * penalty_factor)
                penalty = math.exp(-visit_count * visit_penalty_factor)
                base_w *= penalty

        scores.append(base_w)
    s = sum(scores)
    if s <= 0:
        return random.choice(candidates) if candidates else None
    r = random.random() * s
    acc = 0.0
    for nid, w in zip(candidates, scores):
        acc += w
        if r <= acc:
            return nid
    return candidates[-1] if candidates else None


# ----------------------
# Batch Job Management
# ----------------------
def start_batch_with_limit(job_configs, batch_id, max_parallel, params, no_persist, polygon_file):
    """Start batch jobs with parallel execution limit."""
    with JOBS_LOCK:
        job_ids = []
        pending_jobs = []  # List of (job_id, job) tuples waiting to start

        # Create all jobs
        for config in job_configs:
            job = RouteJob(
                seed_pano_id=config['seed_pano_id'],
                target_m=params['target_m'],
                random_forward=params['random_forward'],
                regress_window=params['regress_window'],
                step_limit=params['step_limit'],
                min_final_degree=params['min_final_degree'],
                heading_temp_deg=config['heading_temp'],
                forced_first_child_id=config['forced'],
                persist_to_db=not no_persist,
                bias_direction=params.get('bias_direction'),
                bias_decision_count=params.get('bias_decision_count', 0),
                visit_penalty_factor=params.get('visit_penalty_factor', 2.0),
            )

            # Set batch information for repulsion
            job.batch_id = batch_id

            try:
                job.polygon_key = config['valid_key']
                job.polygon_file = polygon_file or None
            except Exception:
                pass

            job_id = job.job_id
            JOBS[job_id] = job
            job_ids.append(job_id)
            pending_jobs.append((job_id, job))

        BATCHES[batch_id] = job_ids

    # Start initial batch of jobs
    running_count = 0
    for job_id, job in pending_jobs[:max_parallel]:
        try:
            # Persist initial run record
            if PATH_DB is not None and job.persist_to_db:
                PATH_DB.put_run(job_id, {
                    "jobId": job_id,
                    "seedPanoId": job.seed_pano_id,
                    "seedLat": job.points[0]['position']['lat'] if job.points else None,
                    "seedLng": job.points[0]['position']['lng'] if job.points else None,
                    "startedAt": datetime.utcnow().isoformat(timespec='seconds') + 'Z',
                    "status": "running",
                    "message": "",
                    "targetMeters": params['target_m'],
                    "randomForward": int(params['random_forward']),
                    "regressWindow": params['regress_window'],
                    "stepLimit": params['step_limit'],
                    "minFinalDegree": params['min_final_degree'],
                    "headingTempDeg": job.heading_temp_deg,
                    "polygonKey": getattr(job, 'polygon_key', None),
                    "polygonFile": polygon_file or None,
                })
        except Exception:
            pass

        job.start()
        running_count += 1

    # Remove started jobs from pending
    pending_jobs = pending_jobs[max_parallel:]

    # Start background monitor thread if there are pending jobs
    if pending_jobs:
        def monitor_and_start_jobs():
            while pending_jobs:
                # Check how many jobs are currently running
                with JOBS_LOCK:
                    running_jobs = [jid for jid, job in pending_jobs if JOBS.get(jid) and not JOBS[jid].done]
                    current_running = sum(1 for jid in job_ids if jid in JOBS and JOBS[jid]._thread and JOBS[jid]._thread.is_alive())

                # Start new jobs if we have capacity
                while current_running < max_parallel and pending_jobs:
                    job_id, job = pending_jobs.pop(0)
                    try:
                        # Persist run record
                        if PATH_DB is not None and job.persist_to_db:
                            PATH_DB.put_run(job_id, {
                                "jobId": job_id,
                                "seedPanoId": job.seed_pano_id,
                                "seedLat": job.points[0]['position']['lat'] if job.points else None,
                                "seedLng": job.points[0]['position']['lng'] if job.points else None,
                                "startedAt": datetime.utcnow().isoformat(timespec='seconds') + 'Z',
                                "status": "running",
                                "message": "",
                                "targetMeters": params['target_m'],
                                "randomForward": int(params['random_forward']),
                                "regressWindow": params['regress_window'],
                                "stepLimit": params['step_limit'],
                                "minFinalDegree": params['min_final_degree'],
                                "headingTempDeg": job.heading_temp_deg,
                                "polygonKey": getattr(job, 'polygon_key', None),
                                "polygonFile": polygon_file or None,
                            })
                    except Exception:
                        pass

                    job.start()
                    current_running += 1

                # Wait before checking again
                time.sleep(2)

        monitor_thread = threading.Thread(target=monitor_and_start_jobs, daemon=True)
        monitor_thread.start()

    return {"batch_id": batch_id, "job_ids": job_ids}



# ----------------------
# Config API (serves API key to frontend)
# ----------------------
@app.get("/api/config")
def api_config():
    """Return configuration including API key for frontend."""
    return jsonify({
        "googleMapsApiKey": API_KEY,
        "polygonFile": "data/polygons.json"
    })


# ----------------------
# API
# ----------------------
@app.post("/api/route/start")
def api_start():
    logger.info("=" * 80)
    logger.info("[API_START] /api/route/start ENDPOINT CALLED")
    logger.info("=" * 80)

    data = request.get_json(force=True, silent=True) or {}
    logger.info(f"[API_START] Raw request data: {data}")

    seed = data.get("seedPanoId")
    logger.info(f"[API_START] seedPanoId from request: {seed}")

    if not seed:
        logger.error("[API_START] ERROR: seedPanoId is missing or empty!")
        return jsonify({"error": "seedPanoId required"}), 400

    logger.info(f"[API_START] Looking up coordinates for seed: {seed}")

    # Validate seed is inside a valid polygon with landmark
    seed_coords = coord_for(seed)
    logger.info(f"[API_START] Initial coord_for({seed}) result: {seed_coords}")

    if not seed_coords:
        logger.info(f"[API_START] Seed not in cache, calling ensure_cached({seed})...")
        ensure_cached(seed)
        seed_coords = coord_for(seed)
        logger.info(f"[API_START] After ensure_cached, coord_for({seed}) result: {seed_coords}")

        if not seed_coords:
            logger.error(f"[API_START] !!! CRITICAL: Still no coords for seed {seed} after ensure_cached !!!")
            logger.error(f"[API_START] LAST_FETCH_ERROR: {LAST_FETCH_ERROR}")

    polygon_file = data.get("polygonFile")
    logger.info(f"[API_START] polygonFile: {polygon_file}")
    polygons = load_valid_polygons(polygon_file)
    expected_key = data.get("expectedPolygonKey")
    expected_landmark = data.get("expectedLandmarkName")

    # If we know the expected landmark, just use it directly - no polygon validation needed
    if expected_key and expected_key in polygons:
        valid_key = expected_key
    elif expected_landmark:
        # Dynamically map landmark names to polygon keys using data/landmarks.js structure
        valid_key = None

        try:
            # Load landmarks data from the JavaScript file
            landmarks_path = os.path.join(os.path.dirname(__file__), '..', 'data', 'landmarks.js')
            with open(landmarks_path, 'r') as f:
                content = f.read()

            # Extract the JSON part from the JS file
            json_start = content.find('{')
            json_end = content.rfind('}') + 1
            if json_start >= 0 and json_end > json_start:
                import json
                landmarks_data = json.loads(content[json_start:json_end])

                # Find the landmark and create the polygon key
                for city, landmarks in landmarks_data.items():
                    if isinstance(landmarks, list):
                        for i, landmark in enumerate(landmarks):
                            if isinstance(landmark, list) and len(landmark) >= 1:
                                name = landmark[0]
                                if name == expected_landmark:
                                    # Use name-based keys for robustness
                                    normalized_name = name.replace(' ', '_').replace('-', '_').lower()
                                    polygon_key = f"{city}_{normalized_name}"
                                    if polygon_key in polygons:
                                        valid_key = polygon_key
                                        break
                        if valid_key:
                            break

        except Exception as e:
            # If we can't load the landmarks data, fall back to polygon validation
            pass

        if not valid_key:
            # Fallback to polygon validation for unknown landmarks
            pass

    # Only do polygon validation if we don't have an expected assignment
    if not valid_key and seed_coords and polygons:
        # Sort keys to ensure deterministic ordering (prefer higher indices)
        sorted_keys = sorted(polygons.keys(), key=lambda k: int(k.split('_')[-1]) if '_' in k and k.split('_')[-1].isdigit() else 0, reverse=True)
        for key in sorted_keys:
            poly = polygons[key]
            if point_in_polygon(seed_coords[0], seed_coords[1], poly):
                valid_key = key
                break
    if not valid_key:
        return jsonify({"error": "Seed start is not inside a valid polygon"}), 400
    no_persist = bool(data.get("noPersist", False))
    job = RouteJob(
        seed_pano_id=seed,
        target_m=float(data.get("targetMeters", 2000)),
        random_forward=bool(data.get("randomForward", True)),
        regress_window=int(data.get("regressWindow", 25)),
        step_limit=int(data.get("stepLimit")) if data.get("stepLimit") is not None else None,
        min_final_degree=int(data.get("minFinalDegree", 3)),
        heading_temp_deg=float(data.get("headingTempDeg", HEADING_SOFTMAX_TEMP_DEG)),
        persist_to_db=not no_persist,
        bias_direction=data.get("biasDirection"),
        bias_decision_count=int(data.get("biasDecisionCount", 0)),
        visit_penalty_factor=float(data.get("visitPenaltyFactor", 2.0)),
    )
    try:
        job.polygon_key = valid_key
        job.polygon_file = polygon_file or None
    except Exception:
        pass
    # Persist initial run record
    try:
        if PATH_DB is not None and job.persist_to_db:
            PATH_DB.put_run(job.job_id, {
                "jobId": job.job_id,
                "seedPanoId": seed,
                "seedLat": seed_coords[0] if seed_coords else None,
                "seedLng": seed_coords[1] if seed_coords else None,
                "startedAt": datetime.utcnow().isoformat(timespec='seconds') + 'Z',
                "status": "running",
                "message": "",
                "targetMeters": float(data.get("targetMeters", 2000)),
                "randomForward": int(bool(data.get("randomForward", True))),
                "regressWindow": int(data.get("regressWindow", 25)),
                "stepLimit": int(data.get("stepLimit")) if data.get("stepLimit") is not None else None,
                "minFinalDegree": int(data.get("minFinalDegree", 3)),
                "headingTempDeg": float(data.get("headingTempDeg", HEADING_SOFTMAX_TEMP_DEG)),
                "polygonKey": valid_key,
                "polygonFile": polygon_file or None,
            })
    except Exception as db_err:
        logger.warning(f"[API_START] Failed to persist run record to DB: {db_err}")
        pass

    logger.info(f"[API_START] Registering job {job.job_id} in JOBS registry...")
    with JOBS_LOCK:
        JOBS[job.job_id] = job
        global LAST_JOB_ID
        LAST_JOB_ID = job.job_id
        logger.info(f"[API_START] Job registered. Total jobs in registry: {len(JOBS)}")

    logger.info(f"[API_START] Calling job.start() for job {job.job_id}...")
    job.start()
    logger.info(f"[API_START] job.start() returned. Thread started: {job._thread is not None and job._thread.is_alive()}")
    logger.info("=" * 80)
    logger.info(f"[API_START] SUCCESS - Returning jobId: {job.job_id}")
    logger.info("=" * 80)
    return jsonify({"status": "started", "jobId": job.job_id})


@app.get("/api/route/status")
def api_status():
    job_id = request.args.get('jobId')
    with JOBS_LOCK:
        if job_id:
            J = JOBS.get(job_id)
            if not J:
                return jsonify({"status": "unknown_job"}), 404
            return jsonify(job_status(J))
        # default to last job
        if not LAST_JOB_ID or LAST_JOB_ID not in JOBS:
            return jsonify({"status": "idle"})
        return jsonify(job_status(JOBS[LAST_JOB_ID]))


@app.post("/api/route/stop")
def api_stop():
    data = request.get_json(force=True, silent=True) or {}
    job_id = data.get('jobId')
    with JOBS_LOCK:
        if job_id:
            J = JOBS.get(job_id)
            if not J:
                return jsonify({"status": "unknown_job"}), 404
            J.stop()
            return jsonify({"status": "stopping", "jobId": job_id})
        if not LAST_JOB_ID or LAST_JOB_ID not in JOBS:
            return jsonify({"status": "idle"})
        JOBS[LAST_JOB_ID].stop()
        return jsonify({"status": "stopping", "jobId": LAST_JOB_ID})


@app.post("/api/route/start_batch")
def api_start_batch():
    data = request.get_json(force=True, silent=True) or {}
    seed = data.get("seedPanoId")
    available_seeds = data.get("availableSeeds", [])  # List of seed objects with panoId, position, etc.

    if not seed and not available_seeds:
        return jsonify({"error": "Either seedPanoId or availableSeeds required"}), 400

    # If available_seeds provided, randomly select different seeds for each run
    use_random_seeds = bool(available_seeds)
    if use_random_seeds:
        import random
        # Create a pool of seeds to draw from (include the primary seed if not in available_seeds)
        seed_pool = available_seeds.copy()
        if seed and not any(s.get('panoId') == seed for s in seed_pool):
            # Add primary seed to pool if not already there
            primary_seed_info = {'panoId': seed}
            seed_pool.append(primary_seed_info)
    else:
        # Fallback to original behavior with single seed
        seed_pool = [{'panoId': seed}]

    num_runs = max(1, int(data.get("numRuns", 10)))
    max_parallel = max(1, int(data.get("maxParallel", num_runs)))  # Default to num_runs (start all)
    params = {
        'target_m': float(data.get("targetMeters", 2000)),
        'random_forward': bool(data.get("randomForward", True)),
        'regress_window': int(data.get("regressWindow", 25)),
        'step_limit': int(data.get("stepLimit")) if data.get("stepLimit") is not None else None,
        'min_final_degree': int(data.get("minFinalDegree", 3)),
        'heading_temp_deg': float(data.get("headingTempDeg", HEADING_SOFTMAX_TEMP_DEG)),
        'bias_direction': data.get("biasDirection"),
        'bias_decision_count': int(data.get("biasDecisionCount", 0)),
        'visit_penalty_factor': float(data.get("visitPenaltyFactor", 2.0)),
    }

    # Validate that at least one seed is inside a polygon
    polygon_file = data.get("polygonFile")
    polygons = load_valid_polygons(polygon_file)
    valid_seeds = []
    for seed_info in seed_pool:
        seed_id = seed_info.get('panoId')
        if not seed_id:
            continue
        seed_coords = coord_for(seed_id)
        if not seed_coords:
            ensure_cached(seed_id)
            seed_coords = coord_for(seed_id)
        expected_key = seed_info.get('expectedPolygonKey')
        expected_landmark = seed_info.get('expectedLandmarkName')
        valid_key = None

        # If we know the expected landmark, just use it directly - no polygon validation needed
        if expected_key and expected_key in polygons:
            valid_key = expected_key
        elif expected_landmark:
            # Dynamically map landmark names to polygon keys using data/landmarks.js structure
            try:
                # Load landmarks data from the JavaScript file
                landmarks_path = os.path.join(os.path.dirname(__file__), '..', 'data', 'landmarks.js')
                with open(landmarks_path, 'r') as f:
                    content = f.read()

                # Extract the JSON part from the JS file
                json_start = content.find('{')
                json_end = content.rfind('}') + 1
                if json_start >= 0 and json_end > json_start:
                    import json
                    landmarks_data = json.loads(content[json_start:json_end])

                    # Find the landmark and create the polygon key
                    for city, landmarks in landmarks_data.items():
                        if isinstance(landmarks, list):
                            for i, landmark in enumerate(landmarks):
                                if isinstance(landmark, list) and len(landmark) >= 1:
                                    name = landmark[0]
                                    if name == expected_landmark:
                                        # Use name-based keys for robustness
                                        normalized_name = name.replace(' ', '_').replace('-', '_').lower()
                                        polygon_key = f"{city}_{normalized_name}"
                                        if polygon_key in polygons:
                                            valid_key = polygon_key
                                            break
                            if valid_key:
                                break

            except Exception as e:
                # If we can't load the landmarks data, continue to polygon validation
                pass

        # Only do polygon validation if we don't have an expected assignment
        if not valid_key and seed_coords and polygons:
            # Sort keys to ensure deterministic ordering (prefer higher indices)
            sorted_keys = sorted(polygons.keys(), key=lambda k: int(k.split('_')[-1]) if '_' in k and k.split('_')[-1].isdigit() else 0, reverse=True)
            for key in sorted_keys:
                poly = polygons[key]
                if point_in_polygon(seed_coords[0], seed_coords[1], poly):
                    valid_key = key
                    break

        if valid_key:
            valid_seeds.append((seed_id, valid_key, seed_coords))

    if not valid_seeds:
        return jsonify({"error": "No valid seeds found inside polygons"}), 400

    no_persist = bool(data.get("noPersist", False))

    # Create job configurations but don't start them yet
    job_configs = []
    for i in range(num_runs):
        # Randomly select a seed for this run
        if use_random_seeds and len(valid_seeds) > 1:
            selected_seed_id, valid_key, seed_coords = random.choice(valid_seeds)
        else:
            # Use first valid seed (maintains backward compatibility)
            selected_seed_id, valid_key, seed_coords = valid_seeds[0]

        # Diversify first decision by sampling distinct headings where possible
        jnode, children = first_junction_children(selected_seed_id)
        forced = None
        if children:
            forced = children[i % len(children)]

        # Vary temperature slightly per run for spread
        ht = params['heading_temp_deg'] * (0.9 + 0.2 * (i / max(1, num_runs - 1))) if params['random_forward'] else params['heading_temp_deg']

        job_configs.append({
            'seed_pano_id': selected_seed_id,
            'valid_key': valid_key,
            'seed_coords': seed_coords,
            'forced': forced,
            'heading_temp': ht,
        })

    # Start jobs with parallel limit
    result = start_batch_with_limit(job_configs, batch_id=str(uuid.uuid4()), max_parallel=max_parallel,
                                   params=params, no_persist=no_persist, polygon_file=polygon_file)

    return jsonify({
        "status": "started",
        "batchId": result["batch_id"],
        "jobIds": result["job_ids"],
        "randomSeedsUsed": use_random_seeds,
        "validSeedsFound": len(valid_seeds),
        "maxParallel": max_parallel
    })


@app.get("/api/route/batch_status")
def api_batch_status():
    batch_id = request.args.get('batchId')
    if not batch_id:
        return jsonify({"error": "batchId required"}), 400
    with JOBS_LOCK:
        job_ids = BATCHES.get(batch_id, [])
        jobs = [JOBS[jid] for jid in job_ids if jid in JOBS]
    job_dicts = [job_status(j) for j in jobs]
    any_running = any(j['status'] == 'running' for j in job_dicts)
    return jsonify({"status": "running" if any_running else "done", "jobs": job_dicts})


@app.post("/api/route/stop_batch")
def api_stop_batch():
    data = request.get_json(force=True, silent=True) or {}
    batch_id = data.get('batchId')
    if not batch_id:
        return jsonify({"error": "batchId required"}), 400
    with JOBS_LOCK:
        for jid in BATCHES.get(batch_id, []):
            if jid in JOBS:
                JOBS[jid].stop()
    return jsonify({"status": "stopping"})


# ----------------------
# Paths/DB viewing API
# ----------------------

@app.get('/api/paths/runs')
def api_list_runs():
    if PATH_DB is None:
        return jsonify({"error": "Path DB unavailable"}), 500
    runs = []
    try:
        with PATH_DB.env.begin(db=PATH_DB.runs_db) as txn:
            cur = txn.cursor()
            for _, v in cur:
                try:
                    runs.append(json.loads(v))
                except Exception:
                    continue
    except Exception as e:
        return jsonify({"error": str(e)}), 500
    # Sort by startedAt desc when available
    try:
        runs.sort(key=lambda r: r.get('startedAt', ''), reverse=True)
    except Exception:
        pass
    return jsonify({"runs": runs})


@app.get('/api/paths/points')
def api_list_points():
    job_id = request.args.get('jobId')
    if not job_id:
        return jsonify({"error": "jobId required"}), 400
    # Prefer DB if available; otherwise fall back to in-memory job state
    pts = []
    if PATH_DB is not None:
        try:
            with PATH_DB.env.begin(db=PATH_DB.points_db) as txn:
                cur = txn.cursor()
                prefix = f"pt:{job_id}:".encode()
                it = cur.set_range(prefix)
                # Iterate keys with prefix
                while cur.key() and cur.key().startswith(prefix):
                    try:
                        pts.append(json.loads(cur.value()))
                    except Exception:
                        pass
                    if not cur.next():
                        break
        except Exception as e:
            return jsonify({"error": str(e)}), 500
    if not pts:
        # Try in-memory (noPersist jobs or not yet committed)
        with JOBS_LOCK:
            J = JOBS.get(job_id)
            if J is not None and isinstance(J.points, list):
                try:
                    for idx, pt in enumerate(J.points):
                        pos = pt.get("position") or {}
                        lat = pos.get("lat")
                        lng = pos.get("lng")
                        lnks = links_for(pt.get("panoId")) or []
                        lc = len(lnks)
                        if lc >= 3:
                            cls = "decision"
                        elif lc == 2:
                            cls = "corridor"
                        elif lc == 1:
                            cls = "dead_end"
                        else:
                            cls = "isolated"
                        pts.append({
                            "idx": idx,
                            "panoId": pt.get("panoId"),
                            "lat": lat,
                            "lng": lng,
                            "linksCount": lc,
                            "classification": cls,
                            "links": lnks,
                        })
                except Exception:
                    pts = []
    # Ensure ordered by idx
    try:
        pts.sort(key=lambda p: p.get('idx', 0))
    except Exception:
        pass
    return jsonify({"points": pts})


@app.post('/api/paths/delete')
def api_delete_run():
    if PATH_DB is None:
        return jsonify({"error": "Path DB unavailable"}), 500
    data = request.get_json(force=True, silent=True) or {}
    job_id = data.get('jobId')
    if not job_id:
        return jsonify({"error": "jobId required"}), 400
    try:
        removed = PATH_DB.delete_run(job_id)
        return jsonify({"status": "ok", "removed": int(removed)})
    except Exception as e:
        return jsonify({"error": str(e)}), 500


@app.get('/api/paths/polygons')
def api_polygons():
    try:
        file_arg = request.args.get('file')
        full = request.args.get('full', 'false').lower() == 'true'

        if full:
            # Return full polygon data for seedPanos extraction
            def _resolve_path(p: str) -> str:
                return p if os.path.isabs(p) else os.path.join(PROJECT_ROOT, p)

            source_file = file_arg or "data/polygons.json"
            path = _resolve_path(source_file)

            with open(path, 'r', encoding='utf-8') as f:
                data = json.load(f)

            return jsonify({"data": data, "sourceFile": source_file})
        else:
            # Return metadata format for backward compatibility
            meta = load_polygon_metadata(file_arg)
            source_file = file_arg or "data/polygons.json"
            return jsonify({"polygons": meta, "sourceFile": source_file})
    except Exception as e:
        return jsonify({"error": str(e)}), 500


# ----------------------
# Pano Cache Visualization API
# ----------------------
@app.get('/api/pano/cache_data')
def api_pano_cache_data():
    """Serve pano cache data for visualization - returns nodes, links, and statistics."""
    try:
        if cache is None:
            return jsonify({"error": "Pano cache not available"}), 500

        nodes = []
        links = []
        stats = {
            "total_panos": 0,
            "total_links": 0,
            "isolated_panos": 0,
            "dead_end_panos": 0,
            "corridor_panos": 0,
            "decision_panos": 0,
            "dead_edges_count": 0
        }

        # Collect all panos and their coordinates
        with cache.env.begin(db=cache.coords) as txn:
            cur = txn.cursor()
            for pano_id_bytes, coord_bytes in cur:
                pano_id = pano_id_bytes.decode()
                lat, lng = map(float, coord_bytes.split(b","))
                nodes.append({
                    "id": pano_id,
                    "lat": lat,
                    "lng": lng
                })

        stats["total_panos"] = len(nodes)

        # Collect all links and build graph data
        link_counts = {}
        with cache.env.begin(db=cache.links) as txn:
            cur = txn.cursor()
            for pano_id_bytes, links_json_bytes in cur:
                pano_id = pano_id_bytes.decode()
                try:
                    pano_links = json.loads(links_json_bytes)
                    link_counts[pano_id] = len(pano_links)
                    for link in pano_links:
                        link_pano_id = link.get("pano")
                        if link_pano_id:
                            links.append({
                                "source": pano_id,
                                "target": link_pano_id,
                                "heading": link.get("heading"),
                                "description": link.get("description", "")
                            })
                except json.JSONDecodeError:
                    link_counts[pano_id] = 0

        stats["total_links"] = len(links)

        # Classify panos by link count
        for pano_id, count in link_counts.items():
            if count == 0:
                stats["isolated_panos"] += 1
            elif count == 1:
                stats["dead_end_panos"] += 1
            elif count == 2:
                stats["corridor_panos"] += 1
            else:  # count >= 3
                stats["decision_panos"] += 1

        # Count dead edges
        try:
            with cache.env.begin(db=cache.dead_edges) as txn:
                cur = txn.cursor()
                for _, dead_list_bytes in cur:
                    try:
                        dead_list = json.loads(dead_list_bytes)
                        stats["dead_edges_count"] += len(dead_list)
                    except json.JSONDecodeError:
                        pass
        except Exception:
            pass

        return jsonify({
            "nodes": nodes,
            "links": links,
            "stats": stats
        })

    except Exception as e:
        return jsonify({"error": str(e)}), 500


# ----------------------------------------------------
# Manual commit of locally crafted paths (frontend)
# ----------------------------------------------------
@app.post('/api/paths/manual_commit')
def api_manual_commit():
    if PATH_DB is None:
        return jsonify({"error": "Path DB unavailable"}), 500
    data = request.get_json(force=True, silent=True) or {}
    paths = data.get('paths')
    job_ids = data.get('jobIds') or []
    if not isinstance(paths, list) and not (isinstance(job_ids, list) and job_ids):
        return jsonify({"error": "paths list or jobIds required"}), 400
    try:
        if isinstance(paths, list) and paths:
            batch_id = str(uuid.uuid4())
            meta = {
                "committedAt": datetime.utcnow().isoformat(timespec='seconds') + 'Z',
                "count": len(paths)
            }
            PATH_DB.put_manual_batch(batch_id, paths, meta)
            return jsonify({"status": "ok", "batchId": batch_id, "count": len(paths)})
        # Commit in-memory jobs into runs/points DBs
        committed = 0
        for jid in job_ids:
            with JOBS_LOCK:
                J = JOBS.get(jid)
            if not J or not isinstance(J.points, list) or not J.points:
                continue
            # Create or update run record
            try:
                if PATH_DB is not None:
                    PATH_DB.put_run(jid, {
                        "jobId": jid,
                        "seedPanoId": J.seed_pano_id,
                        "seedLat": J.points[0]['position']['lat'] if J.points and 'position' in J.points[0] else None,
                        "seedLng": J.points[0]['position']['lng'] if J.points and 'position' in J.points[0] else None,
                        "startedAt": getattr(J, 'created_at_utc', datetime.utcnow().isoformat(timespec='seconds') + 'Z'),
                        "finishedAt": datetime.utcnow().isoformat(timespec='seconds') + 'Z',
                        "status": 'done' if J.done and not J.error else ('error' if J.error else ('stopped' if J.stopped else 'running')),
                        "message": J.error or J.message,
                        "targetMeters": J.target_m,
                        "randomForward": int(J.random_forward),
                        "regressWindow": J.regress_window,
                        "stepLimit": J.step_limit,
                        "minFinalDegree": J.min_final_degree,
                        "headingTempDeg": J.heading_temp_deg,
                        "polygonKey": getattr(J, 'polygon_key', None),
                        "polygonFile": getattr(J, 'polygon_file', None),
                        "totalDistanceMeters": getattr(J, 'total_distance_m', 0.0),
                        "stepCount": len(J.points),
                    })
                    # Points
                    pts_payload = []
                    for idx, pt in enumerate(J.points):
                        pid = pt.get('panoId')
                        pos = pt.get('position', {})
                        lat = pos.get('lat')
                        lng = pos.get('lng')
                        lnks = links_for(pid) or []
                        lc = len(lnks)
                        if lc >= 3:
                            cls = 'decision'
                        elif lc == 2:
                            cls = 'corridor'
                        elif lc == 1:
                            cls = 'dead_end'
                        else:
                            cls = 'isolated'
                        pts_payload.append({
                            'idx': idx,
                            'panoId': pid,
                            'lat': lat,
                            'lng': lng,
                            'linksCount': lc,
                            'classification': cls,
                            'links': lnks,
                        })
                    PATH_DB.put_points(jid, pts_payload)
                    committed += 1
            except Exception:
                continue
        return jsonify({"status": "ok", "committed": committed})
    except Exception as e:
        return jsonify({"error": str(e)}), 500


# Simple test function to demonstrate annealing behavior (no Flask dependencies)
def test_temperature_annealing():
    """Demonstrate the temperature annealing logic"""

    print("Temperature Annealing Test:")
    print("=" * 40)

    # Simulate the annealing logic
    max_temp = 45.0
    min_temp = 0.1
    target_distance = 2000.0

    def current_temperature(decision_count, current_distance_m):
        if decision_count == 0:
            return 1000.0  # Pure randomness for first decision

        if current_distance_m <= 0:
            return min_temp

        progress_ratio = min(current_distance_m / target_distance, 1.0)
        return min_temp + (max_temp - min_temp) * progress_ratio

    # Test first decision (should be pure random)
    temp1 = current_temperature(0, 0)
    print(f"First decision (distance=0m): {temp1:.1f}° (pure random)")

    # Test subsequent decisions at different distances
    distances = [0, 250, 500, 1000, 1500, 2000]
    for dist in distances:
        temp = current_temperature(1, dist)
        print(f"Distance {dist:4d}m: {temp:.2f}°")

    print("=" * 40)
    print("✓ First decision: Pure random (temp=1000.0°)")
    print("✓ Subsequent decisions: Annealing from ~0.1° to 45.0°")


def start_frontend_server(host, port, web_dir, data_dir):
    """Start a simple HTTP server for the frontend on a separate port."""
    import http.server
    import socketserver

    # Copy landmarks.js to web dir temporarily for serving
    import shutil
    landmarks_src = os.path.join(data_dir, 'landmarks.js')
    landmarks_dst = os.path.join(web_dir, 'landmarks.js')
    if os.path.exists(landmarks_src) and not os.path.exists(landmarks_dst):
        shutil.copy(landmarks_src, landmarks_dst)
    
    class QuietHandler(http.server.SimpleHTTPRequestHandler):
        def __init__(self, *args, **kwargs):
            # Serve from web_dir without changing the process CWD (thread-safe).
            super().__init__(*args, directory=web_dir, **kwargs)

        def log_message(self, format, *args):
            pass  # Suppress logging
    
    try:
        with socketserver.TCPServer((host, port), QuietHandler) as httpd:
            httpd.serve_forever()
    except Exception as e:
        print(f"Frontend server error: {e}")


def main():
    """Main entry point with CLI argument parsing."""
    import argparse
    
    parser = argparse.ArgumentParser(
        description='Street View Pathfinding Server',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python -m src.server
  python -m src.server --port 8080
        """
    )
    parser.add_argument('--host', default=HOST, help=f'Host to bind (default: {HOST})')
    parser.add_argument('--port', type=int, default=PORT, help=f'API port (default: {PORT})')
    parser.add_argument('--frontend-port', type=int, default=FRONTEND_PORT, help=f'Frontend port (default: {FRONTEND_PORT})')
    parser.add_argument('--debug', action='store_true', help='Enable debug mode')
    
    args = parser.parse_args()
    
    # Start frontend server in background thread
    frontend_thread = threading.Thread(
        target=start_frontend_server,
        args=(args.host, args.frontend_port, WEB_DIR, DATA_DIR),
        daemon=True
    )
    frontend_thread.start()
    
    # Start panorama proxy server
    proxy_started = start_panorama_proxy()

    try:
        print("")
        print("=" * 50)
        print(f"  Frontend: http://{args.host}:{args.frontend_port}")
        print(f"  API:      http://{args.host}:{args.port}")
        print("=" * 50)
        print("")
        print(f"Open http://{args.host}:{args.frontend_port} in your browser")
        print("")
        if USE_PROXY_SERVER and proxy_started:
            print("✓ Using panorama proxy server")
        elif USE_PROXY_SERVER and not proxy_started:
            print("⚠ Proxy server requested but failed to start")
        else:
            print("✓ Using direct browser access")

        app.run(host=args.host, port=args.port, debug=args.debug)
    finally:
        stop_panorama_proxy()


if __name__ == "__main__":
    main()
