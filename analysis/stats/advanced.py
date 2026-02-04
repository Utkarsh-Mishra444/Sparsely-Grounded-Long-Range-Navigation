#!/usr/bin/env python3
import os
import json
import glob
import math
from typing import Any, Dict, List, Optional, Tuple

try:
    import requests
except ImportError:
    requests = None  # type: ignore


EARTH_RADIUS_M = 6371000.0
GOOGLE_MAPS_API_KEY = os.environ.get("GOOGLE_MAPS_API_KEY", "")

def _safe_json_load(path: str) -> Optional[Any]:
    try:
        with open(path, "r", encoding="utf-8", errors="ignore") as f:
            return json.load(f)
    except Exception:
        return None


def _haversine_m(lat1: float, lon1: float, lat2: float, lon2: float) -> float:
    lat1_rad = math.radians(lat1)
    lat2_rad = math.radians(lat2)
    dlat = lat2_rad - lat1_rad
    dlon = math.radians(lon2 - lon1)
    a = math.sin(dlat / 2) ** 2 + math.cos(lat1_rad) * math.cos(lat2_rad) * math.sin(dlon / 2) ** 2
    c = 2 * math.atan2(math.sqrt(a), math.sqrt(1 - a))
    return EARTH_RADIUS_M * c


def _project_equirectangular(lat: float, lon: float, lat0: float) -> Tuple[float, float]:
    x = math.radians(lon) * EARTH_RADIUS_M * math.cos(math.radians(lat0))
    y = math.radians(lat) * EARTH_RADIUS_M
    return x, y


def _point_in_polygon(point: Tuple[float, float], polygon_xy: List[Tuple[float, float]]) -> bool:
    x, y = point
    inside = False
    n = len(polygon_xy)
    if n < 3:
        return False
    for i in range(n):
        x1, y1 = polygon_xy[i]
        x2, y2 = polygon_xy[(i + 1) % n]
        if ((y1 > y) != (y2 > y)) and (x < (x2 - x1) * (y - y1) / (y2 - y1 + 1e-12) + x1):
            inside = not inside
    return inside


def _point_to_segment_distance_xy(px: float, py: float, ax: float, ay: float, bx: float, by: float) -> float:
    abx, aby = bx - ax, by - ay
    apx, apy = px - ax, py - ay
    ab_len2 = abx * abx + aby * aby
    if ab_len2 == 0:
        dx, dy = px - ax, py - ay
        return math.hypot(dx, dy)
    t = (apx * abx + apy * aby) / ab_len2
    t = max(0.0, min(1.0, t))
    cx, cy = ax + t * abx, ay + t * aby
    dx, dy = px - cx, py - cy
    return math.hypot(dx, dy)


def _min_distance_point_polygon_m(lat: float, lon: float, polygon_latlon: List[Tuple[float, float]]) -> float:
    if not polygon_latlon:
        return float("inf")
    lat0 = polygon_latlon[0][0]
    poly_xy = [_project_equirectangular(plat, plon, lat0) for plat, plon in polygon_latlon]
    px, py = _project_equirectangular(lat, lon, lat0)
    if _point_in_polygon((px, py), poly_xy):
        return 0.0
    n = len(poly_xy)
    best = float("inf")
    for i in range(n):
        ax, ay = poly_xy[i]
        bx, by = poly_xy[(i + 1) % n]
        d = _point_to_segment_distance_xy(px, py, ax, ay, bx, by)
        if d < best:
            best = d
    return best


def _load_polygon_and_points(points_path: str) -> Tuple[Optional[List[Tuple[float, float]]], Optional[Tuple[float, float]], Optional[Tuple[float, float]]]:
    data = _safe_json_load(points_path)
    if not isinstance(data, dict):
        return None, None, None

    start = None
    if isinstance(data.get("start"), dict) and "lat" in data["start"] and "lng" in data["start"]:
        start = (float(data["start"]["lat"]), float(data["start"]["lng"]))

    end = None
    if isinstance(data.get("end"), dict) and "lat" in data["end"] and "lng" in data["end"]:
        end = (float(data["end"]["lat"]), float(data["end"]["lng"]))

    poly_coords: Optional[List[Tuple[float, float]]] = None
    # Primary key as used by viewer
    if isinstance(data.get("destinationPolygon"), list):
        try:
            poly_coords = [(float(p[0]), float(p[1])) for p in data["destinationPolygon"] if isinstance(p, (list, tuple)) and len(p) >= 2]
        except Exception:
            poly_coords = None
    # Common fallbacks
    if poly_coords is None:
        for key in ("destination_polygon", "polygon"):
            arr = data.get(key)
            if isinstance(arr, list):
                try:
                    poly_coords = [(float(p[0]), float(p[1])) for p in arr if isinstance(p, (list, tuple)) and len(p) >= 2]
                except Exception:
                    poly_coords = None
            if poly_coords:
                break
    # Nested destination object
    if poly_coords is None:
        dest = data.get("destination")
        if isinstance(dest, dict):
            arr = dest.get("polygon") or dest.get("destinationPolygon")
            if isinstance(arr, list):
                try:
                    poly_coords = [(float(p[0]), float(p[1])) for p in arr if isinstance(p, (list, tuple)) and len(p) >= 2]
                except Exception:
                    poly_coords = None

    return poly_coords, start, end


def _get_walking_distance_from_api(start_lat: float, start_lng: float, end_lat: float, end_lng: float) -> Optional[float]:
    """Get optimal walking distance from Google Maps Directions API."""
    if requests is None:
        return None
    try:
        url = "https://maps.googleapis.com/maps/api/directions/json"
        params = {
            "origin": f"{start_lat},{start_lng}",
            "destination": f"{end_lat},{end_lng}",
            "mode": "walking",
            "key": GOOGLE_MAPS_API_KEY
        }
        resp = requests.get(url, params=params, timeout=10)
        data = resp.json()
        
        if data.get("status") == "OK" and data.get("routes"):
            return float(data["routes"][0]["legs"][0]["distance"]["value"])
        return None
    except Exception:
        return None


def _closest_point_on_polygon_boundary_latlon(start_lat: float, start_lng: float, polygon: List[Tuple[float, float]]) -> Tuple[float, float, bool]:
    """Return the closest point ON THE BOUNDARY (edges) in lat/lon and whether start is inside.

    Uses equirectangular projection around polygon latitude for stable XY calculations,
    then maps the closest XY point back to lat/lon.
    """
    if not polygon:
        return start_lat, start_lng, False

    lat0 = polygon[0][0]
    poly_xy = [_project_equirectangular(plat, plon, lat0) for plat, plon in polygon]
    px, py = _project_equirectangular(start_lat, start_lng, lat0)

    # If inside polygon, distance is 0 and any boundary point would be along a ray; we can treat l* as 0.
    inside = _point_in_polygon((px, py), poly_xy)
    if inside:
        return start_lat, start_lng, True

    best_d2 = float("inf")
    best_cx = poly_xy[0][0]
    best_cy = poly_xy[0][1]

    n = len(poly_xy)
    for i in range(n):
        ax, ay = poly_xy[i]
        bx, by = poly_xy[(i + 1) % n]
        abx, aby = bx - ax, by - ay
        apx, apy = px - ax, py - ay
        ab_len2 = abx * abx + aby * aby
        if ab_len2 == 0.0:
            cx, cy = ax, ay
        else:
            t = (apx * abx + apy * aby) / ab_len2
            if t < 0.0:
                t = 0.0
            elif t > 1.0:
                t = 1.0
            cx = ax + t * abx
            cy = ay + t * aby
        dx, dy = px - cx, py - cy
        d2 = dx * dx + dy * dy
        if d2 < best_d2:
            best_d2 = d2
            best_cx = cx
            best_cy = cy

    # Map closest XY back to lat/lon
    closest_lat = math.degrees(best_cy / EARTH_RADIUS_M)
    # avoid divide by zero on cos(lat0)
    denom = max(1e-12, math.cos(math.radians(lat0)))
    closest_lon = math.degrees(best_cx / (EARTH_RADIUS_M * denom))
    return closest_lat, closest_lon, False


def _read_all_terminal_distances(exp_dir: str) -> List[float]:
    values: List[float] = []
    try:
        for name in os.listdir(exp_dir):
            if not (name.startswith("terminal_output_") and name.endswith(".log")):
                continue
            path = os.path.join(exp_dir, name)
            try:
                with open(path, "r", encoding="utf-8", errors="ignore") as f:
                    for line in f:
                        if "Distance to destination:" in line:
                            # Expect pattern: Distance to destination: <num> m
                            try:
                                idx = line.index(":")
                                text = line[idx + 1 :].strip()
                                num = text.split(" ")[0]
                                values.append(float(num))
                            except Exception:
                                continue
            except Exception:
                continue
    except Exception:
        pass
    return values


def _count_decisions(exp_dir: str) -> int:
    total = 0
    for sub in ("openai_calls", "gemini_calls"):
        d = os.path.join(exp_dir, sub)
        if os.path.isdir(d):
            try:
                total += sum(1 for p in os.listdir(d) if p.startswith("call_") and p.endswith(".json"))
            except Exception:
                pass
    return total


def _read_coordinates(exp_dir: str) -> List[Tuple[float, float]]:
    path = os.path.join(exp_dir, "visited_coordinates.json")
    coords: List[Tuple[float, float]] = []
    data = _safe_json_load(path)
    if isinstance(data, list):
        for item in data:
            if isinstance(item, dict) and "lat" in item and "lng" in item:
                try:
                    coords.append((float(item["lat"]), float(item["lng"])) )
                except Exception:
                    continue
    return coords


def compute_advanced_metrics(exp_dir: str, successful: bool) -> Dict[str, float]:
    points_path = os.path.join(exp_dir, "points.json")
    polygon, start, end = _load_polygon_and_points(points_path)

    coords = _read_coordinates(exp_dir)
    path_length_m = 0.0
    if len(coords) >= 2:
        for i in range(1, len(coords)):
            lat1, lon1 = coords[i - 1]
            lat2, lon2 = coords[i]
            path_length_m += _haversine_m(lat1, lon1, lat2, lon2)

    # L* calculation: Use Google Maps Directions API for optimal walking distance
    l_star_m = None
    start_for_metrics = coords[0] if coords else start
    if start_for_metrics is not None:
        # Determine destination point
        dest_lat, dest_lng = None, None
        inside = False
        if polygon:
            dest_lat, dest_lng, inside = _closest_point_on_polygon_boundary_latlon(start_for_metrics[0], start_for_metrics[1], polygon)
        elif end is not None:
            dest_lat, dest_lng = end[0], end[1]

        if inside:
            l_star_m = 0.0
        else:
            if dest_lat is None or dest_lng is None:
                raise RuntimeError(f"Unable to determine destination point for SPL in {exp_dir}")
            l_star_m = _get_walking_distance_from_api(start_for_metrics[0], start_for_metrics[1], dest_lat, dest_lng)
            if not isinstance(l_star_m, (int, float)):
                raise RuntimeError(f"Directions API failed to provide walking distance for {exp_dir}")
    else:
        raise RuntimeError(f"Missing start point for SPL in {exp_dir}")

    # Min distance to polygon over the path
    min_distance_to_polygon_m = None
    if polygon:
        md = float("inf")
        for (lat, lon) in coords or []:
            d = _min_distance_point_polygon_m(lat, lon, polygon)
            if d < md:
                md = d
        if md != float("inf"):
            min_distance_to_polygon_m = md
    elif end is not None and coords:
        # Fallback: min distance to final point
        md = float("inf")
        for (lat, lon) in coords:
            d = _haversine_m(lat, lon, end[0], end[1])
            if d < md:
                md = d
        if md != float("inf"):
            min_distance_to_polygon_m = md

    decisions_count = _count_decisions(exp_dir)

    spl = 0.0
    if successful and l_star_m > 0.0:
        denom = max(l_star_m, path_length_m) if path_length_m > 0 else l_star_m
        if denom > 0:
            spl = l_star_m / denom

    return {
        "decisions_count": float(decisions_count),
        "path_length_m": float(path_length_m),
        "l_star_m": float(l_star_m),
        "min_distance_to_polygon_m": float(min_distance_to_polygon_m) if min_distance_to_polygon_m is not None else None,  # type: ignore
        "spl": float(spl),
    }


