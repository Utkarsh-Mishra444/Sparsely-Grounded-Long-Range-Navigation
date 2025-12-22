#!/usr/bin/env python3
"""
Comprehensive navigation analysis script for polygon-based destination distance.

This script analyzes ALL runs in a navigation experiment and determines:
1. Navigation success rates (agents that reached destination polygons)
2. Distance to destination for all runs (polygon-based calculation)
3. 403 forbidden error analysis
4. Complete performance metrics

Usage:
    conda activate sys2
    python analyze_403_errors.py <log_directory>

Example:
    python analyze_403_errors.py logs/Vienna_2_trapi_gpt-4o
    python analyze_403_errors.py logs/Sao_Paulo_2_trapi_gpt-5
"""

import os
import json
import glob
import math
import sys
from pathlib import Path
import argparse
from datetime import datetime

def haversine_distance(lat1, lon1, lat2, lon2):
    """Calculate the great circle distance between two points in meters."""
    # Convert decimal degrees to radians
    lat1, lon1, lat2, lon2 = map(math.radians, [lat1, lon1, lat2, lon2])

    # Haversine formula
    dlat = lat2 - lat1
    dlon = lon2 - lon1
    a = math.sin(dlat/2)**2 + math.cos(lat1) * math.cos(lat2) * math.sin(dlon/2)**2
    c = 2 * math.asin(math.sqrt(a))
    r = 6371000  # Radius of earth in meters
    return c * r

def point_in_polygon(px, py, polygon):
    """Ray casting algorithm to determine if point is inside polygon."""
    n = len(polygon)
    inside = False

    for i in range(n):
        ax, ay = polygon[i]
        bx, by = polygon[(i + 1) % n]

        # Check if point is on the edge
        if min(ax, bx) <= px <= max(ax, bx) and min(ay, by) <= py <= max(ay, by):
            # Point is on horizontal edge
            if ax == bx and px == ax:
                return True
            # Point is on vertical edge
            if ay == by and py == ay:
                return True

        # Ray casting: shoot ray to the right
        if ((ay > py) != (by > py)) and (px < ax + (bx - ax) * (py - ay) / (by - ay)):
            inside = not inside

    return inside

def point_to_segment_distance(px, py, ax, ay, bx, by):
    """Calculate minimum distance from point to line segment using geographic coordinates."""
    # Handle degenerate case (same point)
    if ax == bx and ay == by:
        return haversine_distance(px, py, ax, ay)

    # Convert all coordinates to radians for calculation
    px_rad, py_rad = math.radians(px), math.radians(py)
    ax_rad, ay_rad = math.radians(ax), math.radians(ay)
    bx_rad, by_rad = math.radians(bx), math.radians(by)

    # Vector AB in radians
    ab_x = bx_rad - ax_rad
    ab_y = by_rad - ay_rad

    # Vector AP in radians
    ap_x = px_rad - ax_rad
    ap_y = py_rad - ay_rad

    # Projection scalar
    ab_len_squared = ab_x * ab_x + ab_y * ab_y
    if ab_len_squared == 0:  # Degenerate segment
        return haversine_distance(px, py, ax, ay)

    proj = (ap_x * ab_x + ap_y * ab_y) / ab_len_squared

    # Clamp projection to segment bounds [0,1]
    proj = max(0.0, min(1.0, proj))

    # Find closest point on segment
    closest_x_rad = ax_rad + proj * ab_x
    closest_y_rad = ay_rad + proj * ab_y

    # Convert back to degrees for haversine calculation
    closest_x = math.degrees(closest_x_rad)
    closest_y = math.degrees(closest_y_rad)

    return haversine_distance(px, py, closest_x, closest_y)

def point_to_polygon_distance(px, py, polygon):
    """Calculate distance from point to polygon (0 if inside, min edge distance if outside)."""
    if not polygon or len(polygon) < 3:
        return None

    # First check if point is inside polygon
    if point_in_polygon(px, py, polygon):
        return 0.0

    # Point is outside, find minimum distance to any edge
    min_distance = float('inf')
    n = len(polygon)

    for i in range(n):
        ax, ay = polygon[i]
        bx, by = polygon[(i + 1) % n]  # Wrap around to first point
        edge_distance = point_to_segment_distance(px, py, ax, ay, bx, by)
        min_distance = min(min_distance, edge_distance)

    return min_distance

def analyze_run(run_dir):
    """Analyze a single run directory for navigation performance and errors."""
    run_path = Path(run_dir)
    run_name = run_path.name

    # Find the trapi subdirectory
    trapi_dirs = list(run_path.glob("*trapi*"))
    if not trapi_dirs:
        return None

    trapi_dir = trapi_dirs[0]

    # Find the actual run directory (gpt_*_location_path_*)
    run_subdirs = list(trapi_dir.glob("gpt_*"))
    if not run_subdirs:
        return None

    actual_run_dir = run_subdirs[0]

    # Check for errors (but don't filter - analyze all runs)
    error_file = actual_run_dir / "error.log"
    has_403 = False
    has_other_error = False
    completed_successfully = True

    if error_file.exists():
        with open(error_file, 'r') as f:
            error_content = f.read()
            if error_content.strip():  # Has some error content
                completed_successfully = False
                has_403 = '403' in error_content and 'image url can not be accessed' in error_content
                has_other_error = not has_403

    # Get destination coordinates and polygon
    points_file = actual_run_dir / "points.json"
    destination_lat = None
    destination_lng = None
    destination_polygon = None

    if points_file.exists():
        try:
            with open(points_file, 'r') as f:
                points_data = json.load(f)
                destination_lat = points_data['end']['lat']
                destination_lng = points_data['end']['lng']
                destination_polygon = points_data.get('destinationPolygon')
        except:
            pass

    # Count decisions from strategy.log
    strategy_file = actual_run_dir / "strategy.log"
    decisions_made = 0

    if strategy_file.exists():
        try:
            with open(strategy_file, 'r') as f:
                for line in f:
                    if 'Recorded decision' in line:
                        decisions_made += 1
        except:
            pass

    # Get last position from visited_coordinates.json
    visited_file = actual_run_dir / "visited_coordinates.json"
    last_lat = None
    last_lng = None

    if visited_file.exists():
        try:
            with open(visited_file, 'r') as f:
                visited_data = json.load(f)
                if visited_data:
                    last_entry = visited_data[-1]
                    last_lat = last_entry['lat']
                    last_lng = last_entry['lng']
        except:
            pass

    # Calculate distance to destination (polygon-based)
    distance_to_dest = None
    if last_lat is not None and last_lng is not None:
        if destination_polygon is not None:
            # Use polygon distance (preferred)
            distance_to_dest = point_to_polygon_distance(last_lat, last_lng, destination_polygon)
        elif destination_lat is not None and destination_lng is not None:
            # Fallback to point distance if no polygon
            distance_to_dest = haversine_distance(last_lat, last_lng, destination_lat, destination_lng)

    # Extract location name
    location_name = actual_run_dir.name.replace('gpt_', '').replace('_path_01_s2000_d150', '')

    # Determine run status
    if completed_successfully and distance_to_dest == 0.0:
        status = 'SUCCESS'
    elif completed_successfully:
        status = 'COMPLETED_NO_ERROR'
    elif has_403:
        status = 'FAILED_403'
    elif has_other_error:
        status = 'FAILED_OTHER'
    else:
        status = 'UNKNOWN'

    return {
        'run_name': run_name,
        'location': location_name,
        'decisions_made': decisions_made,
        'distance_to_destination_m': distance_to_dest,
        'status': status,
        'last_position': (last_lat, last_lng) if last_lat else None,
        'destination': (destination_lat, destination_lng) if destination_lat else None
    }

def main():
    parser = argparse.ArgumentParser(description='Comprehensive navigation analysis with polygon-based distance calculations')
    parser.add_argument('log_directory', help='Path to the log directory (e.g., logs/Vienna_2_trapi_gpt-4o)')
    args = parser.parse_args()

    log_dir = Path(args.log_directory)
    if not log_dir.exists():
        print(f"Error: Directory {log_dir} does not exist")
        return

    # Create timestamp and output file
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    city_name = log_dir.name
    output_file = log_dir / f"403_error_analysis_{city_name}_{timestamp}.txt"

    print(f"Analyzing 403 forbidden errors in: {log_dir}")
    print(f"Results will be saved to: {output_file}")
    print("=" * 60)

    # Open file for writing
    with open(output_file, 'w') as f:
        f.write(f"Comprehensive Navigation Analysis - {timestamp}\n")
        f.write(f"Analyzed directory: {log_dir}\n")
        f.write("=" * 60 + "\n\n")

    # Find all run directories
    run_dirs = []
    for item in log_dir.glob("*"):
        if item.is_dir() and any(subdir.name.startswith(('000', '001')) for subdir in item.iterdir() if subdir.is_dir()):
            # This is a timestamped run directory
            for run_subdir in item.iterdir():
                if run_subdir.is_dir() and run_subdir.name.startswith(('000', '001')):
                    run_dirs.append(run_subdir)

    if not run_dirs:
        print("No run directories found")
        with open(output_file, 'w') as f:
            f.write("No run directories found\n")
        return

    print(f"Found {len(run_dirs)} run directories")

    with open(output_file, 'a') as f:
        f.write(f"Found {len(run_dirs)} run directories\n\n")

    # Analyze ALL runs
    all_results = []
    for run_dir in run_dirs:
        result = analyze_run(run_dir)
        if result:
            all_results.append(result)

    if not all_results:
        print("No analyzable runs found")
        with open(output_file, 'a') as f:
            f.write("No analyzable runs found\n")
        return

    print(f"\nAnalyzed {len(all_results)} runs")
    print("Results saved to:", output_file)

    # Categorize results
    success_runs = [r for r in all_results if r['status'] == 'SUCCESS']
    completed_no_error_runs = [r for r in all_results if r['status'] == 'COMPLETED_NO_ERROR']
    failed_403_runs = [r for r in all_results if r['status'] == 'FAILED_403']
    failed_other_runs = [r for r in all_results if r['status'] == 'FAILED_OTHER']

    with open(output_file, 'a') as f:
        f.write("OVERALL NAVIGATION PERFORMANCE:\n")
        f.write("=" * 50 + "\n")
        f.write(f"Total runs analyzed: {len(all_results)}\n")
        f.write(f"✅ SUCCESS (reached destination): {len(success_runs)}/{len(all_results)} ({len(success_runs)/len(all_results)*100:.1f}%)\n")
        f.write(f"⚠️  COMPLETED (no error but didn't reach dest): {len(completed_no_error_runs)}/{len(all_results)} ({len(completed_no_error_runs)/len(all_results)*100:.1f}%)\n")
        f.write(f"❌ FAILED_403 (API access error): {len(failed_403_runs)}/{len(all_results)} ({len(failed_403_runs)/len(all_results)*100:.1f}%)\n")
        f.write(f"❓ FAILED_OTHER (other errors): {len(failed_other_runs)}/{len(all_results)} ({len(failed_other_runs)/len(all_results)*100:.1f}%)\n\n")

        # Distance analysis for all runs
        all_distances = [r['distance_to_destination_m'] for r in all_results if r['distance_to_destination_m'] is not None]
        if all_distances:
            reached_count = sum(1 for d in all_distances if d == 0.0)
            close_count = sum(1 for d in all_distances if 0 < d < 1000)
            far_count = sum(1 for d in all_distances if d > 3000)

            f.write("DISTANCE TO DESTINATION ANALYSIS (ALL RUNS):\n")
            f.write("-" * 50 + "\n")
            f.write(f"Agents that REACHED destination polygon: {reached_count}/{len(all_distances)} ({reached_count/len(all_distances)*100:.1f}%)\n")
            f.write(f"Agents within 1km of destination: {close_count}/{len(all_distances)} ({close_count/len(all_distances)*100:.1f}%)\n")
            f.write(f"Agents more than 3km from destination: {far_count}/{len(all_distances)} ({far_count/len(all_distances)*100:.1f}%)\n")
            f.write(f"Average distance to destination: {sum(all_distances)/len(all_distances):.1f} meters\n\n")

        # Detailed table of ALL runs
        f.write("DETAILED RESULTS - ALL RUNS:\n")
        f.write("-" * 160 + "\n")
        f.write(f"{'Run Name':<20} {'Status':<15} {'Location':<40} {'Decisions':<12} {'Distance to Dest (m)':<20}\n")
        f.write("-" * 160 + "\n")

        for result in sorted(all_results, key=lambda x: (x['status'], -x['decisions_made'])):
            location_display = result['location']
            if len(location_display) > 37:  # Leave room for "..."
                location_display = location_display[:37] + "..."

            distance_str = f"{result['distance_to_destination_m']:.1f}" if result['distance_to_destination_m'] is not None else "N/A"
            status_display = result['status'].replace('_', ' ')
            f.write(f"{result['run_name']:<20} {status_display:<15} {location_display:<40} {result['decisions_made']:<12} {distance_str:<20}\n")

        f.write("-" * 160 + "\n\n")

        # 403 Error specific analysis
        if failed_403_runs:
            f.write("403 FORBIDDEN ERROR ANALYSIS:\n")
            f.write("-" * 50 + "\n")

            # 403 specific stats
            distances_403 = [r['distance_to_destination_m'] for r in failed_403_runs if r['distance_to_destination_m'] is not None]
            if distances_403:
                reached_403 = sum(1 for d in distances_403 if d == 0.0)
                close_403 = sum(1 for d in distances_403 if 0 < d < 1000)
                far_403 = sum(1 for d in distances_403 if d > 3000)

                f.write("403 Error Proximity Analysis:\n")
                f.write(f"  Agents that reached destination BEFORE 403 error: {reached_403}/{len(failed_403_runs)} ({reached_403/len(failed_403_runs)*100:.1f}%)\n")
                f.write(f"  Agents within 1km when 403 hit: {close_403}/{len(failed_403_runs)} ({close_403/len(failed_403_runs)*100:.1f}%)\n")
                f.write(f"  Agents >3km away when 403 hit: {far_403}/{len(failed_403_runs)} ({far_403/len(failed_403_runs)*100:.1f}%)\n")
                f.write(f"  Average distance when 403 occurred: {sum(distances_403)/len(distances_403):.1f} meters\n\n")

        f.write("\n" + "=" * 60 + "\n")
        f.write("NOTES:\n")
        f.write("- Distance = 0 means agent was INSIDE the destination polygon\n")
        f.write("- SUCCESS = completed without errors AND reached destination\n")
        f.write("- COMPLETED_NO_ERROR = finished without errors but didn't reach destination\n")
        f.write("- FAILED_403 = hit 403 forbidden image access error\n")
        f.write("- All distance calculations use polygon boundaries for accuracy\n")

if __name__ == "__main__":
    main()
