#!/usr/bin/env python3
"""
Post-hoc decision evaluation recalculation script.

This script recalculates RIGHT/WRONG decision evaluations from saved run data
(visited_coordinates.json and detailed_decision_log.json) without needing the
walking distance API.

Usage:
    python recalculate_evaluations.py <run_folder>
    
Example:
    python recalculate_evaluations.py "logs/100_paths_full_trapi_gpt-4o/20250918_001718_ad9f1a42/00011_path_11_r0/2025-09-18_00-17-21_trapi/gpt_4o_Times_Square_path_01_s2000_d150"
"""

import json
import os
import sys
from math import radians, cos, sin, asin, sqrt
import glob
from typing import Optional, Dict, Any, List, Tuple
import re
from pathlib import Path

from infrastructure.cache import DistanceCache
from analysis.stats import advanced as run_stats_advanced
GOOGLE_MAPS_API_KEY = os.environ.get("GOOGLE_MAPS_API_KEY", "")

run_stats_advanced.GOOGLE_MAPS_API_KEY = GOOGLE_MAPS_API_KEY

_DISTANCE_CACHE: Optional[DistanceCache] = None
_WALKING_DISTANCE_API_CALLS = 0


def _get_distance_cache() -> DistanceCache:
    global _DISTANCE_CACHE
    if _DISTANCE_CACHE is None:
        _DISTANCE_CACHE = DistanceCache()
    return _DISTANCE_CACHE


def reset_walking_distance_stats() -> None:
    global _WALKING_DISTANCE_API_CALLS
    _WALKING_DISTANCE_API_CALLS = 0


def get_walking_distance_api_calls() -> int:
    return _WALKING_DISTANCE_API_CALLS


def _fetch_walking_distance(lat1: float, lon1: float, lat2: float, lon2: float) -> Optional[float]:
    cache = _get_distance_cache()
    try:
        cached = cache.get(lat1, lon1, lat2, lon2)
    except Exception:
        cached = None

    if isinstance(cached, dict):
        distance_value = cached.get('distance')
        if isinstance(distance_value, (int, float)):
            return float(distance_value)
        # If cached result exists but has no usable distance, fall through to attempt API call

    distance_value = None
    try:
        result = run_stats_advanced._get_walking_distance_from_api(lat1, lon1, lat2, lon2)
        # Only count as an API call if a request module is available and the helper attempted a call
        if run_stats_advanced.requests is not None:
            global _WALKING_DISTANCE_API_CALLS
            _WALKING_DISTANCE_API_CALLS += 1
        if isinstance(result, (int, float)):
            distance_value = float(result)
    except Exception:
        distance_value = None

    try:
        cache.put(lat1, lon1, lat2, lon2, {
            'distance': distance_value,
            'status': 'OK' if isinstance(distance_value, float) else 'ERROR'
        })
    except Exception:
        pass

    return distance_value


def _distance_with_walking_fallback(lat1: float, lon1: float, lat2: float, lon2: float) -> float:
    distance = _fetch_walking_distance(lat1, lon1, lat2, lon2)
    if isinstance(distance, float):
        return distance
    return haversine(lat1, lon1, lat2, lon2)


def haversine(lat1, lon1, lat2, lon2):
    """
    Calculate the great circle distance between two points 
    on the earth (specified in decimal degrees).
    Returns distance in meters.
    """
    # Convert decimal degrees to radians 
    lon1, lat1, lon2, lat2 = map(radians, [lon1, lat1, lon2, lat2])
    
    # Haversine formula 
    dlon = lon2 - lon1 
    dlat = lat2 - lat1 
    a = sin(dlat/2)**2 + cos(lat1) * cos(lat2) * sin(dlon/2)**2
    c = 2 * asin(sqrt(a)) 
    
    # Radius of earth in meters
    r = 6371000
    
    return c * r


def _build_alias_map(decision_entry: Dict[str, Any]) -> Dict[str, str]:
    """Construct alias_to_id_map from decision entry observation."""
    observation = decision_entry.get('observation') or {}
    reverse_map = observation.get('reverse_alias_map') or {}
    alias_to_id_map: Dict[str, str] = {}
    for pano_id, alias in reverse_map.items():
        alias_to_id_map[alias] = f"step{decision_entry.get('step')}_option{alias.replace('link_', '')}"
    return alias_to_id_map


def _build_correct_ids(decision_entry: Dict[str, Any], alias_map: Dict[str, str]) -> List[str]:
    observation = decision_entry.get('observation') or {}
    eval_alias = observation.get('evaluations_alias') or {}
    correct = []
    for alias, data in eval_alias.items():
        if isinstance(data, dict) and data.get('label') == 'RIGHT':
            correct.append(alias_map.get(alias, alias))
    return correct


def _load_gemini_decisions(run_folder: str) -> List[Dict[str, Any]]:
    gemini_dir = Path(run_folder) / "gemini_calls"
    if not gemini_dir.is_dir():
        return []
    decision_files: List[Tuple[int, Path]] = []
    pattern = re.compile(r"decision_(\d+)\.json$")
    for entry in gemini_dir.iterdir():
        if not entry.is_file():
            continue
        match = pattern.match(entry.name)
        if not match:
            continue
        try:
            decision_idx = int(match.group(1))
        except ValueError:
            continue
        decision_files.append((decision_idx, entry))
    decision_files.sort(key=lambda t: t[0])

    decisions: List[Dict[str, Any]] = []
    for _, path in decision_files:
        try:
            with path.open("r") as f:
                data = json.load(f)
        except Exception:
            continue
        step = data.get("agent_context", {}).get("step_counter")
        decision_info = data.get("response", {}).get("parsed_json") or {}
        if step is None or not isinstance(step, int):
            continue
        option_id = decision_info.get("decision")
        if not option_id or not isinstance(option_id, str):
            continue
        match_option = re.match(r"step(\d+)_option(\d+)", option_id)
        if not match_option:
            continue
        option_step = int(match_option.group(1))
        option_suffix = match_option.group(2)
        # Build an observation stub to align with existing processing helpers
        observation = {
            "reverse_alias_map": {
                option_id: f"link_{option_suffix}"
            },
            "evaluations_alias": {}
        }
        decisions.append({
            "step": step,
            "action": f"link_{option_suffix}",
            "observation": observation,
            "source": "gemini"
        })
    return decisions


def load_run_data(run_folder):
    """Load all necessary data from a run folder."""
    
    # Find visited_coordinates.json
    coords_file = os.path.join(run_folder, "visited_coordinates.json")
    if not os.path.exists(coords_file):
        raise FileNotFoundError(f"Cannot find visited_coordinates.json in {run_folder}")
    
    with open(coords_file, 'r') as f:
        visited_coordinates = json.load(f)
    
    # Find detailed_decision_log
    decision_log_files = glob.glob(os.path.join(run_folder, "detailed_decision_log_*.json"))
    if not decision_log_files:
        raise FileNotFoundError(f"Cannot find detailed_decision_log_*.json in {run_folder}")
    
    decision_log_file = decision_log_files[0]
    with open(decision_log_file, 'r') as f:
        detailed_decision_log = json.load(f)
    
    # Find destination coordinates - look in config or path JSON
    destination_coords = None
    
    # Try to find config/path JSON files
    parent_dirs = run_folder.split(os.sep)
    
    # Look for path_*.json in _orchestrator folder
    for i in range(len(parent_dirs), 0, -1):
        test_path = os.path.join(os.sep.join(parent_dirs[:i]), "_orchestrator")
        if os.path.exists(test_path):
            path_files = glob.glob(os.path.join(test_path, "path_*.json"))
            if path_files:
                with open(path_files[0], 'r') as f:
                    path_data = json.load(f)
                    if 'paths' in path_data and len(path_data['paths']) > 0:
                        end_pos = path_data['paths'][0]['end']['position']
                        destination_coords = (end_pos['lat'], end_pos['lng'])
                        break
    
    if destination_coords is None:
        raise ValueError(f"Cannot find destination coordinates for {run_folder}")
    
    return visited_coordinates, detailed_decision_log, destination_coords


def recalculate_evaluations(visited_coordinates, detailed_decision_log, destination_coords, run_folder):
    """
    Recalculate decision evaluations based on the trajectory.
    
    Algorithm:
    - For each decision at step S:
      1. Get coords_before from visited_coordinates[S]
      2. Find evaluation step = min(S+3, next_decision_step)
      3. Get coords_after from visited_coordinates[evaluation_step]
      4. Compare distances to destination
      5. Status = "RIGHT" if distance decreased, else "WRONG"
    """
    
    # Build lookup: step -> coordinate
    coords_by_step = {coord['step']: coord for coord in visited_coordinates}
    trajectory = sorted(coords_by_step.values(), key=lambda c: c['step'])

    # Combine simulator decision log entries with Gemini call fallbacks
    decision_map: Dict[int, Dict[str, Any]] = {}

    for entry in detailed_decision_log:
        step = int(entry.get('step'))
        copied = dict(entry)
        copied['step'] = step
        copied['source'] = 'log'
        decision_map[step] = copied

    for entry in _load_gemini_decisions(run_folder):
        step = int(entry.get('step'))
        if step not in decision_map:
            decision_map[step] = entry

    # Build ordered list of decisions
    decision_entries = [decision_map[step] for step in sorted(decision_map.keys())]
    for idx, entry in enumerate(decision_entries, start=1):
        entry['decision_number'] = idx

    decision_steps_set = {entry['step'] for entry in decision_entries}

    pending_queue: List[Dict[str, Any]] = []
    decision_results: List[Dict[str, Any]] = []

    def compute_distance(lat, lng):
        return _distance_with_walking_fallback(lat, lng, destination_coords[0], destination_coords[1])

    # Iterate through steps to simulate queue processing
    for idx, coord in enumerate(trajectory):
        step = coord['step']
        is_decision_point = step in decision_steps_set
        # Increment hops and check completion for pending
        still_pending = []
        for pending in pending_queue:
            pending['hops'] += 1
            should_finalize = pending['hops'] >= 3 or is_decision_point
            if should_finalize:
                after_distance = compute_distance(coord['lat'], coord['lng'])
                before_distance = pending['distance_before']
                status = "RIGHT" if after_distance < before_distance else "WRONG"
                decision_entry = pending['decision_entry']
                decision_eval = {
                    'decision_number': decision_entry['decision_number'],
                    'action': decision_entry.get('action'),
                    'status': status,
                    'decision_step': pending['decision_step'],
                    'evaluation_step': step,
                    'distance_before': round(before_distance, 2),
                    'distance_after': round(after_distance, 2),
                    'distance_change': round(after_distance - before_distance, 2)
                }
                if pending['correct_ids']:
                    decision_eval['correct_ids'] = pending['correct_ids']
                if pending['alias_to_id_map']:
                    decision_eval['alias_to_id_map'] = pending['alias_to_id_map']
                decision_results.append(decision_eval)
            else:
                still_pending.append(pending)
        pending_queue = still_pending

        # If this step is a decision point, enqueue its evaluation
        matching_decisions = [d for d in decision_entries if d['step'] == step]
        for decision_entry in matching_decisions:
            distance_before = compute_distance(coord['lat'], coord['lng'])
            decision_entry = dict(decision_entry)
            if decision_entry.get('source') == 'log':
                alias_map = _build_alias_map(decision_entry)
                correct_ids = _build_correct_ids(decision_entry, alias_map)
            else:
                alias_map = _build_alias_map(decision_entry)
                correct_ids = []
            pending_queue.append({
                'decision_entry': decision_entry,
                'distance_before': distance_before,
                'hops': 0,
                'correct_ids': correct_ids,
                'alias_to_id_map': alias_map,
                'decision_step': step,
                'evaluation_step': None
            })

    # Finalize remaining pending evaluations at end
    if trajectory:
        final_coord = trajectory[-1]
        for pending in pending_queue:
            after_distance = compute_distance(final_coord['lat'], final_coord['lng'])
            before_distance = pending['distance_before']
            decision_entry = pending['decision_entry']
            status = "RIGHT" if after_distance < before_distance else "WRONG"
            decision_eval = {
                'decision_number': decision_entry['decision_number'],
                'action': decision_entry.get('action'),
                'status': status,
                'decision_step': pending['decision_step'],
                'evaluation_step': final_coord['step'],
                'distance_before': round(before_distance, 2),
                'distance_after': round(after_distance, 2),
                'distance_change': round(after_distance - before_distance, 2),
                'max_steps_reached': True
            }
            if pending['correct_ids']:
                decision_eval['correct_ids'] = pending['correct_ids']
            if pending['alias_to_id_map']:
                decision_eval['alias_to_id_map'] = pending['alias_to_id_map']
            decision_results.append(decision_eval)

    return decision_results


def compare_with_ground_truth(run_folder, recalculated_evals):
    """Compare recalculated evaluations with existing decision_evaluations.json."""
    
    # Find existing decision_evaluations file
    eval_files = glob.glob(os.path.join(run_folder, "decision_evaluations_*.json"))
    
    if not eval_files:
        print("No existing decision_evaluations.json found - cannot compare with ground truth")
        return None
    
    with open(eval_files[0], 'r') as f:
        ground_truth = json.load(f)
    
    print("\n" + "="*80)
    print("COMPARISON WITH GROUND TRUTH")
    print("="*80)
    
    matches = 0
    mismatches = 0
    
    for gt_entry in ground_truth:
        decision_num = gt_entry['decision_number']
        gt_status = gt_entry['status']
        
        # Find corresponding recalculated entry
        recalc_entry = next((e for e in recalculated_evals if e['decision_number'] == decision_num), None)
        
        if recalc_entry:
            recalc_status = recalc_entry['status']
            match = gt_status == recalc_status
            
            if match:
                matches += 1
                symbol = "✓"
            else:
                mismatches += 1
                symbol = "✗"
            
            print(f"{symbol} Decision {decision_num:2d}: GT={gt_status:5s} | Recalc={recalc_status:5s} | "
                  f"Step={recalc_entry['decision_step']:3d} | Eval@={recalc_entry['evaluation_step']:3d} | "
                  f"ΔDist={recalc_entry['distance_change']:+7.1f}m")
    
    print("="*80)
    accuracy = (matches / (matches + mismatches) * 100) if (matches + mismatches) > 0 else 0
    print(f"Accuracy: {matches}/{matches + mismatches} = {accuracy:.1f}%")
    print("="*80 + "\n")
    
    return accuracy


def main():
    if len(sys.argv) < 2:
        print(__doc__)
        sys.exit(1)
    
    run_folder = sys.argv[1]
    
    if not os.path.exists(run_folder):
        print(f"ERROR: Run folder does not exist: {run_folder}")
        sys.exit(1)
    
    print(f"Loading data from: {run_folder}")
    
    try:
        visited_coordinates, detailed_decision_log, destination_coords = load_run_data(run_folder)
        
        print(f"  - Loaded {len(visited_coordinates)} coordinate points")
        print(f"  - Loaded {len(detailed_decision_log)} decisions")
        print(f"  - Destination: ({destination_coords[0]:.6f}, {destination_coords[1]:.6f})")
        
        print("\nRecalculating evaluations...")
        recalculated_evals = recalculate_evaluations(visited_coordinates, detailed_decision_log, destination_coords, run_folder)
        
        print(f"  - Recalculated {len(recalculated_evals)} decision evaluations")
        
        # Compare with ground truth if available
        accuracy = compare_with_ground_truth(run_folder, recalculated_evals)
        
        # Save recalculated evaluations
        output_file = os.path.join(run_folder, "decision_evaluations_recalculated.json")
        with open(output_file, 'w') as f:
            json.dump(recalculated_evals, f, indent=2)
        
        print(f"Saved recalculated evaluations to: {output_file}")
        
    except Exception as e:
        print(f"ERROR: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()
