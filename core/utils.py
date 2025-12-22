from playwright.sync_api import sync_playwright, Page
from contextlib import contextmanager
import typing as t
import math
from math import radians, sin, cos, sqrt, asin
import os
import json
from datetime import datetime
import pickle
import traceback
import io  # Import io for Tee class
import sys  # Import sys for Tee class
import base64
import hashlib
import hmac
import urllib.parse as urlparse
_log_file_handle = None
from typing import Any, Dict, List, Callable, cast

def dump_coordinates(visited_coordinates, filename, experiment_folder=None):
    """Write the visited coordinates to a JSON file."""
    filename = os.path.join(experiment_folder, filename) if experiment_folder else filename
    with open(filename, 'w') as f:
        json.dump(visited_coordinates, f, indent=4)

def dump_evaluations(decision_evaluations, filename, experiment_folder=None):
    """Write the decision evaluations to a JSON file."""
    filename = os.path.join(experiment_folder, filename) if experiment_folder else filename
    with open(filename, 'w') as f:
        json.dump(decision_evaluations, f, indent=4)

# Legacy StreetViewBrowser helpers retained for reference only.
#
# @contextmanager
#
# def StreetViewBrowser(api_key: str) -> t.Generator[Page, None, None]:
#     """
#     Context manager to handle the lifecycle of a headless browser for Street View API calls.
#
#     Args:
#         api_key (str): Your Google Maps API key.
#
#     Yields:
#         Page: A Playwright page object with the Google Maps API loaded.
#
#     Example:
#         ```python
#         with StreetViewBrowser("YOUR_API_KEY") as page:
#             pano_data = get_panorama_from_coords(page, 40.742077, -73.982914)
#             print(pano_data['location']['pano'])
#         ```
#     """
#     playwright = sync_playwright().start()
#     browser = playwright.chromium.launch(headless=True)
#     page = browser.new_page()
#     page.goto('about:blank')
#     page.add_script_tag(url=f'https://maps.googleapis.com/maps/api/js?key={api_key}')
#     page.wait_for_load_state('networkidle')
#     try:
#         yield page
#     finally:
#         browser.close()
#         playwright.stop()
#
# def _fetch_panorama_data(page: Page, pano_id: str) -> dict:
#     """
#     Internal helper function to fetch panorama data for a given panorama ID.
#
#     Args:
#         page (Page): The Playwright page object from StreetViewBrowser.
#         pano_id (str): The panorama ID to fetch data for.
#
#     Returns:
#         dict: The panorama data fetched from the Street View API.
#     """
#     data = page.evaluate(f'''
#         (async () => {{
#             const streetViewService = new google.maps.StreetViewService();
#             const data = await new Promise((resolve, reject) => {{
#                 streetViewService.getPanorama({{ pano: '{pano_id}' }}, (data, status) => {{
#                     if (status === google.maps.StreetViewStatus.OK) {{
#                         resolve(data);
#                     }} else {{
#                         reject(status);
#                     }}
#                 }});
#             }});
#             return data;
#         }})()
#     ''')
#     return data
#
# def get_panorama_from_coords(page: Page, lat: float, lng: float, radius: int = 50, max_radius: int = 3000) -> dict:
#     """
#     Retrieves a Street View panorama near the given coordinates with at least one link.
#
#     Args:
#         page (Page): The Playwright page object from StreetViewBrowser.
#         lat (float): Latitude of the starting location.
#         lng (float): Longitude of the starting location.
#         radius (int, optional): Initial search radius in meters. Defaults to 50.
#         max_radius (int, optional): Maximum search radius in meters. Defaults to 3000.
#
#     Returns:
#         dict: The panorama data, including location and links.
#
#     Raises:
#         Exception: If no panorama with links is found within max_radius or an error occurs.
#
#     Example:
#         ```python
#         with StreetViewBrowser("YOUR_API_KEY") as page:
#             pano_data = get_panorama_from_coords(page, 40.742077, -73.982914)
#             print(pano_data['location']['pano'])
#         ```
#     """
#     result = page.evaluate(f'''
#         (async () => {{
#             const streetViewService = new google.maps.StreetViewService();
#             let currentRadius = {radius};
#             const maxRadius = {max_radius};
#             const latLng = {{ lat: {lat}, lng: {lng} }};
#
#             while (currentRadius <= maxRadius) {{
#                 const data = await new Promise((resolve, reject) => {{
#                     streetViewService.getPanorama({{ location: latLng, radius: currentRadius }}, (data, status) => {{
#                         if (status === google.maps.StreetViewStatus.OK) {{
#                             resolve(data);
#                         }} else {{
#                             reject(status);
#                         }}
#                     }});
#                 }});
#                 if (data.links && data.links.length > 0 && data.location && data.location.pano) {{
#                     return data;
#                 }}
#                 currentRadius += 50;
#             }}
#             throw new Error("No panorama with links found within max radius");
#         }})()
#     ''')
#     return result
#
# def get_panorama(page: Page, pano_id: str) -> dict:
#     """
#     Retrieves the full panorama data for a given panorama ID.
#
#     Args:
#         page (Page): The Playwright page object from StreetViewBrowser.
#         pano_id (str): The panorama ID.
#
#     Returns:
#         dict: The panorama data, including location, links, and other metadata.
#
#     Raises:
#         Exception: If the panorama is not found or an error occurs.
#
#     Example:
#         ```python
#         with StreetViewBrowser("YOUR_API_KEY") as page:
#             pano_data = get_panorama(page, "some_pano_id")
#             print(pano_data)
#         ```
#     """
#     return _fetch_panorama_data(page, pano_id)
#
# def get_panorama_links(page: Page, pano_id: str) -> list:
#     """
#     Retrieves the list of links for a given panorama ID.
#
#     Args:
#         page (Page): The Playwright page object from StreetViewBrowser.
#         pano_id (str): The panorama ID.
#
#     Returns:
#         list: A list of link dictionaries, each containing 'pano' and 'heading'.
#
#     Raises:
#         Exception: If the panorama is not found or an error occurs.
#
#     Example:
#         ```python
#         with StreetViewBrowser("YOUR_API_KEY") as page:
#             links = get_panorama_links(page, "some_pano_id")
#             for link in links:
#                 print(link['pano'], link['heading'])
#         ```
#     """
#     data = _fetch_panorama_data(page, pano_id)
#     return data.get('links', [])


def sign_streetview_url(input_url: str, secret: str) -> str:
    """Return the signed Street View Static API URL using Google-supplied logic."""
    if not input_url or not secret:
        raise ValueError("Both input_url and secret are required for signing")

    parsed = urlparse.urlparse(input_url)
    url_to_sign = f"{parsed.path}?{parsed.query}"
    decoded_key = base64.urlsafe_b64decode(secret)
    signature = hmac.new(decoded_key, url_to_sign.encode("utf-8"), hashlib.sha1)
    encoded_signature = base64.urlsafe_b64encode(signature.digest()).decode("utf-8")
    return f"{input_url}&signature={encoded_signature}"

def haversine(lat1, lon1, lat2, lon2):
    """Calculate the great circle distance in meters between two points."""
    lat1, lon1, lat2, lon2 = map(radians, [lat1, lon1, lat2, lon2])
    dlon = lon2 - lon1
    dlat = lat2 - lat1
    a = sin(dlat/2)**2 + cos(lat1) * cos(lat2) * sin(dlon/2)**2
    c = 2 * asin(sqrt(a))
    r = 6371000  # Earth radius in meters
    return c * r

 


def save_checkpoint(agent, env, strategy, config, checkpoint_path, action=None, visited_coordinates=None):


    original_action_in_config = None  # <-- ADD THIS LINE
    print(f"Saving checkpoint to {checkpoint_path}...", flush=True)

    # --- DEBUG: Print state BEFORE saving --- 
    print("--- State Before Saving Checkpoint: ---")
    print(f"  - Agent Step Counter: {getattr(agent, 'step_counter', 'Not Found')}")
    print(f"  - Env State (pano_id): {getattr(env, 'state', 'Not Found')}")
    print(f"-------------------------------------")

    # Store non-pickleable objects temporarily
    original_playwright = getattr(env, 'playwright', None)
    original_browser = getattr(env, 'browser', None)
    original_page = getattr(env, 'page', None)
    original_cache = getattr(env, 'cache', None)
    original_distance_cache = getattr(env, 'distance_cache', None)
    original_api_client = None
    strategy_client_attr_name = None

    # --- detach loggers --------------------------------------------------------
    original_env_logger      = getattr(env,      'logger', None)
    original_strategy_logger = getattr(strategy, 'logger', None)
    original_agent_logger    = getattr(agent,    'logger', None)
    original_pos_logger      = getattr(getattr(agent, 'pos_agent', None), 'logger', None)

    if original_env_logger      is not None: env.logger = None
    if original_strategy_logger is not None: strategy.logger = None
    if original_agent_logger    is not None: agent.logger = None
    if original_pos_logger      is not None: agent.pos_agent.logger = None
    # ---------------------------------------------------------------------------




    if hasattr(strategy, 'model') and strategy.model is not None: # Gemini
        original_api_client = strategy.model
        strategy_client_attr_name = 'model'
    elif hasattr(strategy, 'client') and strategy.client is not None: # OpenAI
        original_api_client = strategy.client
        strategy_client_attr_name = 'client'

    # Set non-pickleable attributes to None
    env.playwright = None
    env.browser = None
    env.page = None
    env.cache = None  # LMDB environment cannot be pickled
    env.distance_cache = None  # LMDB distance cache cannot be pickled
    if strategy_client_attr_name:
        # Temporarily add action to config if provided (for decision checkpoints)
        original_action_in_config = config.pop('saved_action', None) # Clear any old one
        if action is not None:
            config['saved_action'] = action
        setattr(strategy, strategy_client_attr_name, None)

    # Prepare data structure to save (include timestamp)
    config['checkpoint_timestamp'] = datetime.now().isoformat()
    data_to_save = {
        'config': config,
        'agent_state': agent,
        'env_state': env,
        'strategy_state': strategy,
        'visited_coordinates': visited_coordinates if visited_coordinates is not None else [] # Add coordinates
    }

    save_success = False
    # Pickle the data
    try:
        # Ensure directory exists
        os.makedirs(os.path.dirname(checkpoint_path), exist_ok=True)
        # Write atomically (write to temp file, then rename)
        temp_checkpoint_path = checkpoint_path + ".tmp"
        with open(temp_checkpoint_path, 'wb') as f:
            pickle.dump(data_to_save, f)
        os.replace(temp_checkpoint_path, checkpoint_path) # Atomic rename
        print("Checkpoint saved successfully.", flush=True)
        save_success = True
    except Exception as e:
        print(f"ERROR saving checkpoint: {e}", flush=True)
        traceback.print_exc() # Print traceback for debugging
        # Don't restore objects if saving failed, state might be inconsistent

    # IMPORTANT: Restore original objects to the instances in memory
    # This allows the *current* run (if not exiting) to continue.
    # Only restore if saving seemed successful.
    try:
        print("Restoring live objects to in-memory instances...", flush=True)
        env.playwright = original_playwright
        env.browser = original_browser
        env.page = original_page
        env.cache = original_cache
        env.distance_cache = original_distance_cache

        # ── add to the RESTORE section --------------------------------------------
        if original_env_logger      is not None: env.logger = original_env_logger
        if original_strategy_logger is not None: strategy.logger = original_strategy_logger
        if original_agent_logger    is not None: agent.logger = original_agent_logger
        if original_pos_logger      is not None: agent.pos_agent.logger = original_pos_logger
        # ---------------------------------------------------------------------------

        if strategy_client_attr_name and original_api_client is not None:
            setattr(strategy, strategy_client_attr_name, original_api_client)
        print("Live objects restored.", flush=True)
    except Exception as restore_error:
        print(f"ERROR restoring live objects: {restore_error}", flush=True)
        traceback.print_exc()
    
    finally:
        # --- IMPORTANT: Restore config state --- 
        # Remove the saved_action we might have added
        config.pop('saved_action', None)
        # Put back any action that was originally there (should be None normally)
        if original_action_in_config is not None:
            config['saved_action'] = original_action_in_config

    return save_success

# --- Tee class for duplicating stdout/stderr ---
class Tee(io.TextIOBase):
    """Duplicates output to multiple streams (e.g., stdout and a file)."""
    def __init__(self, *streams):
        self.streams = streams

    def write(self, data):
        for stream in self.streams:
            stream.write(data)
            # Flush frequently to ensure logs are written in case of crash
            if hasattr(stream, 'flush'):
                stream.flush()

    def flush(self):
        for stream in self.streams:
            if hasattr(stream, 'flush'):
                stream.flush()

def setup_logging_tee(log_filepath):
    """Redirects stdout and stderr to both console and a file."""
    global _log_file_handle
    # Close previous handle if exists
    if _log_file_handle and not _log_file_handle.closed:
        _log_file_handle.close()
    
    try:
        os.makedirs(os.path.dirname(log_filepath), exist_ok=True)
        _log_file_handle = open(log_filepath, 'a', encoding='utf-8') # Append mode
        # Keep original streams
        original_stdout = sys.stdout
        original_stderr = sys.stderr
        # Create Tee objects
        sys.stdout = Tee(original_stdout, _log_file_handle)
        sys.stderr = Tee(original_stderr, _log_file_handle)
        print(f"--- Logging stdout/stderr to {log_filepath} ---")
        return original_stdout, original_stderr # Return originals for potential restoration
    except Exception as e:
        print(f"ERROR setting up logging Tee to {log_filepath}: {e}", file=sys.__stderr__) # Use original stderr
        # Attempt to close file handle if opened
        if _log_file_handle:
            _log_file_handle.close()
            _log_file_handle = None
        return sys.__stdout__, sys.__stderr__ # Return originals if setup fails
    


def close_log_file():
    """Closes the global log file handle."""
    global _log_file_handle
    if _log_file_handle and not _log_file_handle.closed:
        print(f"\n--- Closing log file: {_log_file_handle.name} ---")
        sys.stdout.flush() # Ensure buffers are flushed before closing
        sys.stderr.flush()
        _log_file_handle.close()
        _log_file_handle = None
        # It's generally not safe/recommended to restore stdout/stderr globally
        # after redirection, especially in complex applications or libraries.
        # If restoration is needed, manage the original streams carefully.



def load_checkpoint_and_prepare(checkpoint_path, new_experiment_folder, is_branching=False):
    print(f"Loading checkpoint from {checkpoint_path}...", flush=True)

    # Load the pickled data
    try:
        with open(checkpoint_path, 'rb') as f:
            loaded_data = pickle.load(f)
    except FileNotFoundError:
        print(f"ERROR: Checkpoint file not found at {checkpoint_path}", flush=True)
        raise
    except Exception as e:
        print(f"ERROR loading checkpoint: {e}", flush=True)
        traceback.print_exc()
        raise

    config = loaded_data['config']
    agent = loaded_data['agent_state']
    env = loaded_data['env_state']
    strategy = loaded_data['strategy_state']

    # --- DEBUG: Print loaded environment state ---
    print("--- Loaded Environment State (from checkpoint): ---")
    print(f"  - env.state (pano_id): {getattr(env, 'state', 'Not Found')}")
    print(f"  - env.initial_coords: {getattr(env, 'initial_coords', 'Not Found')}")
    print(f"  - env.destination_coords: {getattr(env, 'destination_coords', 'Not Found')}")
    print(f"  - env.experiment_folder: {getattr(env, 'experiment_folder', 'Not Found')}")
    print(f"  - env.playwright (should be None): {getattr(env, 'playwright', 'Attribute Missing')}")
    print(f"  - env.browser (should be None): {getattr(env, 'browser', 'Attribute Missing')}")
    print(f"  - env.page (should be None): {getattr(env, 'page', 'Attribute Missing')}")
    print(f"-------------------------------------------------")

    print(f"Checkpoint data loaded. Original run state:")
    print(f"  - Checkpoint Timestamp: {config.get('checkpoint_timestamp', 'N/A')}")
    print(f"  - Agent Step: {agent.step_counter}")

    # --- Re-initialize non-pickleable parts ---
    print("Re-initializing Playwright...", flush=True)
    playwright = None
    browser = None
    try:
        maps_api_key = config['maps_api_key']
        playwright = sync_playwright().start()
        browser = playwright.chromium.launch(headless=True) # Consider adding config options (e.g., channel)
        page = browser.new_page()
        page.goto('about:blank')
        page.add_script_tag(url=f'https://maps.googleapis.com/maps/api/js?key={maps_api_key}&libraries=places,geometry,directions')
        page.wait_for_load_state('networkidle')
        # Assign back to the loaded env object
        env.playwright = playwright
        env.browser = browser
        env.page = page
        print("Playwright re-initialized.", flush=True)
        # Re-initialize LMDB caches (they were removed before pickling)
        try:
            from infrastructure.cache import PanoCache, DistanceCache
            env.cache = PanoCache(os.path.join("cache", "pano"))
            print("StreetView pano cache re-initialized.", flush=True)
            env.distance_cache = DistanceCache(path=os.path.join("cache", "distances"), map_size=8 << 30)
            print("Distance cache re-initialized.", flush=True)
        except Exception as cache_e:
            print(f"WARNING: Could not re-initialize cache(s): {cache_e}", flush=True)
            env.cache = None
            env.distance_cache = None
    except Exception as e:
        print(f"ERROR re-initializing Playwright: {e}", flush=True)
        traceback.print_exc()
        # Cleanup partially created playwright resources if possible before raising
        if browser: browser.close()
        if playwright: playwright.stop()
        raise

    # Skipping direct LLM SDK re-initialization.
    # Strategy uses `agent_core.llm_wrapper.llm_call` (LiteLLM) which reads keys from its own config.
    # This avoids requiring provider keys inside checkpoint config during resume.

    # --- Update paths and configuration for the new run context ---
    print(f"Updating paths for experiment folder: {new_experiment_folder}", flush=True)
    os.makedirs(new_experiment_folder, exist_ok=True)
    env.experiment_folder = new_experiment_folder
    strategy.call_folder = new_experiment_folder
    strategy.intersection_memory_file = os.path.join(new_experiment_folder, "intersection_memory.json")

    # Re-initialize strategy logger (was detached for pickling)
    try:
        import logging
        strategy.logger = strategy._setup_logger(logging.INFO)
        print("Strategy logger re-initialized.", flush=True)
    except Exception as log_e:
        print(f"WARNING: Could not re-initialize strategy logger: {log_e}", flush=True)

    # Clear intersection memory ONLY when branching
    if is_branching:
        strategy.intersection_memory = {}
        # Also clear related state variables to avoid KeyError
        if hasattr(strategy, 'last_intersection_hash'):
            delattr(strategy, 'last_intersection_hash')
        if hasattr(strategy, 'last_chosen_index'):
            delattr(strategy, 'last_chosen_index')
        print(f"Cleared intersection memory and state for branch exploration")
    else:
        print(f"Preserved intersection memory for resume")

    # --- Sync environment state ---
    print("Synchronizing environment state (running get_observation)...", flush=True)
    try:
        _ = env.get_observation() # Run for side effects (populates links, alias_map)
        print("Environment state synchronized.", flush=True)
    except Exception as e:
         print(f"ERROR synchronizing environment state after load: {e}", flush=True)
         # Cleanup playwright and API client?
         raise

    # --- Extract resume action if present ---
    resume_action = config.get('saved_action')
    if resume_action:
        print(f"Found saved action in checkpoint: {resume_action}", flush=True)
        config.pop('saved_action', None) # Remove from config after extracting

    # --- Extract visited coordinates ---
    loaded_coords = loaded_data.get('visited_coordinates', [])
    print(f"Found {len(loaded_coords)} visited coordinates in checkpoint.", flush=True)

    # --- Reconstruct coordinate file --- 
    coord_filepath = os.path.join(new_experiment_folder, "visited_coordinates.json")
    try:
        print(f"Reconstructing {coord_filepath} from checkpoint data...", flush=True)
        with open(coord_filepath, 'w') as f:
            json.dump(loaded_coords, f, indent=4)
        print(f"Successfully reconstructed {coord_filepath}", flush=True)
    except Exception as e:
        print(f"ERROR reconstructing coordinate file {coord_filepath}: {e}", flush=True)
        traceback.print_exc()

    # --- Return the ready-to-run objects and the config ---
    print("Checkpoint loaded and re-initialized successfully.", flush=True)
    return agent, env, strategy, config, resume_action, loaded_coords




def generate_run_folder(path_def: dict, model_name: str, use_azure: bool, config: dict) -> str:
    """
    Generate a unique run folder based on model, destination, and timestamp.
    Format: YYYY-MM-DD_HH-MM-SS_MODEL_DESTINATION_pathXX_maxSteps_maxDecisions
    """
    timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    dest_name = path_def["end"].get("destination", "Unknown").replace(" ", "_").replace("/", "_")
    path_id = path_def.get("uid", "path_00")  # Default fallback if uid not found
    
    # Clean model name for folder
    clean_model = model_name.replace(".", "_").replace("-", "_")
    if use_azure:
        clean_model = f"azure_{clean_model}"
    
    # Get config values with defaults
    max_steps = config.get('max_steps', 700)
    max_decision_points = config.get('max_decision_points', 50)
    base_log_dir = config.get('base_log_dir', "logs")
    
    # Add simulation parameters to folder name
    folder_name = f"{timestamp}_{clean_model}_{dest_name}_{path_id}_s{max_steps}_d{max_decision_points}"

    # if path not there, create it
    if not os.path.exists(base_log_dir):
        os.makedirs(base_log_dir)

    return os.path.join(base_log_dir, folder_name)


def get_simulation_status(log_folder: str) -> dict:
    """
    Extract current simulation status from log files.
    Returns dict with step, decision, self_positioning_active, and current_location.
    """
    status = {
        'step': 0,
        'decision': 0,
        'self_positioning_active': False,
        'current_location': 'Unknown'
    }
    
    try:
        # Check for visited coordinates (gives us current step)
        coords_file = os.path.join(log_folder, "visited_coordinates.json")
        if os.path.exists(coords_file):
            with open(coords_file, 'r') as f:
                coords_data = json.load(f)
                if coords_data and len(coords_data) > 0:
                    # Get the latest coordinate entry
                    latest_coord = coords_data[-1]
                    status['step'] = latest_coord.get('step', len(coords_data))
        
        # Check for decision files (gives us current decision count)
        # With unified approach, files are stored in openai_calls or gemini_calls based on use_openai flag
        openai_calls_dir = os.path.join(log_folder, "openai_calls")
        gemini_calls_dir = os.path.join(log_folder, "gemini_calls")
        
        max_decision = 0
        for calls_dir in [openai_calls_dir, gemini_calls_dir]:
            if os.path.exists(calls_dir):
                decision_files = [f for f in os.listdir(calls_dir) if f.startswith("decision_") and f.endswith(".json")]
                for decision_file in decision_files:
                    try:
                        decision_num = int(decision_file.split("_")[1].split(".")[0])
                        max_decision = max(max_decision, decision_num)
                    except (ValueError, IndexError):
                        continue
        
        status['decision'] = max_decision
        
        # Check for active self-positioning
        self_pos_dir = os.path.join(log_folder, "self_position_calls")
        if os.path.exists(self_pos_dir):
            # Look for recent self-positioning files (within last 30 seconds)
            import time
            current_time = time.time()
            recent_files = []
            
            for f in os.listdir(self_pos_dir):
                if f.startswith("self_position_") and f.endswith(".json"):
                    file_path = os.path.join(self_pos_dir, f)
                    try:
                        file_mtime = os.path.getmtime(file_path)
                        if current_time - file_mtime < 30:  # Within last 30 seconds
                            recent_files.append(f)
                    except OSError:
                        continue
            
            if recent_files:
                status['self_positioning_active'] = True
                
                # Try to get location from the most recent self-positioning file
                recent_files.sort(reverse=True)
                latest_self_pos_file = os.path.join(self_pos_dir, recent_files[0])
                try:
                    with open(latest_self_pos_file, 'r') as f:
                        self_pos_data = json.load(f)
                        # Look for location in the final answer or steps
                        for step in self_pos_data.get('steps', []):
                            if step.get('type') == 'final_answer':
                                location = step.get('location_guess', '')
                                if location and location.lower() not in ['unknown', 'uncertain', 'unsure']:
                                    status['current_location'] = location
                                    break
                except (json.JSONDecodeError, KeyError):
                    pass
        
        # If no self-positioning location found, try to get location from decision files
        if status['current_location'] == 'Unknown' and max_decision > 0:
            for calls_dir in [openai_calls_dir, gemini_calls_dir]:
                if os.path.exists(calls_dir):
                    latest_decision_file = os.path.join(calls_dir, f"decision_{max_decision}.json")
                    if os.path.exists(latest_decision_file):
                        try:
                            with open(latest_decision_file, 'r') as f:
                                decision_data = json.load(f)
                                # Look for location info in the request or response
                                request_data = decision_data.get('request', {})
                                if 'Estimated position:' in str(request_data):
                                    # Extract estimated position from the prompt
                                    prompt_text = str(request_data)
                                    import re
                                    match = re.search(r'Estimated position:\s*([^(]+)', prompt_text)
                                    if match:
                                        location = match.group(1).strip()
                                        if location and len(location) < 100:  # Reasonable length
                                            status['current_location'] = location
                                            break
                        except (json.JSONDecodeError, KeyError):
                            continue
                    break
                    
    except Exception as e:
        # Don't let status checking break the main loop
        pass
    
    return status

def load_paths(json_path: str) -> List[Dict[str, Any]]:
    """Load paths from JSON file and assign unique IDs."""
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
                path = {
                    "start": {
                        "position": route["start"],
                        "startPanoId": route.get("startPanoId")  # Add startPanoId if present
                    },
                    "end": {
                        "position": route["end"],  # Use the explicit end coordinates
                        "destination": route.get("destinationName", "Unknown Destination"),
                        "destinationPolygon": route.get("destinationPolygon")  # Add polygon if present
                    }
                }
                paths.append(path)

    # Enforce explicit destination naming; do not rely on any description fields
    for i, p in enumerate(paths, 1):
        if not isinstance(p, dict):
            raise ValueError(f"Path {i} is not a dictionary")

        # Handle both old format (with "end") and new format (direct properties)
        if "end" in p:
            if not isinstance(p.get("end"), dict) or not p["end"].get("destination"):
                raise ValueError(f"Path {i} missing required field 'end.destination'")
        elif not p.get("end", {}).get("destination"):
            raise ValueError(f"Path {i} missing required destination field")

        p["uid"] = f"path_{i:02d}"
    return paths


def load_prompts(path: str | None) -> List[str]:
    """Load prompts from file; require explicit prompt for controlled runs."""
    if not path:
        raise ValueError("prompts_file must be provided for controlled runs")
    if not os.path.exists(path):
        raise FileNotFoundError(f"prompts_file not found: {path}")
    with open(path, "r", encoding="utf-8") as fp:
        content = fp.read().strip()
    if not content:
        raise ValueError("prompts_file is empty; provide an explicit prompt")
    return [content]
