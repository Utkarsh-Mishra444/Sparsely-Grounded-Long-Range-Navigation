from core.base import Environment
from playwright.sync_api import sync_playwright
import json
import glob
import os
from infrastructure.scoring import score_run
import requests
import sys
import json, math, requests, urllib.parse
import polyline   # pip install polyline
import sqlite3
from datetime import datetime
from shapely.geometry import Point, Polygon
from infrastructure.cache import PanoCache
from infrastructure.cache import DistanceCache

class StreetViewEnvironment(Environment):
    def __init__(self, initial_coords, destination_coords, api_key, *, enable_evaluations: bool = True, init_from_pano_id: bool = False, initial_pano_id: str = None, destination_polygon: list = None):
        super().__init__(initial_state="Init")
        initial_lat, initial_lng = initial_coords
        self.api_key = api_key
        self.init_from_pano_id = init_from_pano_id
        self.initial_pano_id = initial_pano_id
        self.destination_polygon = destination_polygon
        self.playwright = sync_playwright().start()
        # Add server-safe flags to avoid sandbox/dev-shm/gpu issues which can hang on new_page()
        self.browser = self.playwright.chromium.launch(
            headless=True,
            args=[
                '--no-sandbox',
                '--disable-setuid-sandbox',
                '--disable-dev-shm-usage',
                '--disable-gpu'
            ]
        )
        print("[ENVPRO] after launch, before new_page")
        self.page = self.browser.new_page()
        print("[ENVPRO] after new_page, before maps script")
        self.page.goto('about:blank')
        self.page.add_script_tag(url=f'https://maps.googleapis.com/maps/api/js?key={api_key}&libraries=places,geometry,directions')
        self.page.wait_for_load_state('networkidle')
        self.alias_map = {}
        self.current_lat = initial_lat
        self.current_lng = initial_lng
        self.score = 0
        self.experiment_folder = None
        # Control whether evaluate_first_level_actions is executed
        self.enable_evaluations = enable_evaluations
        # Track missing evaluation files: allow some steps before error
        self._missing_eval_steps = 0
        self._missing_eval_threshold = 5

        self.last_heading = None  # Track the heading of the last move
        self.current_links = []  # Store current panorama links
        
        self.previous_state = None
        self.cache = PanoCache(os.path.join("cache", "pano"))

        # ── distance cache for Google Maps Directions API results ────────
        self.distance_cache = DistanceCache(path=os.path.join("cache", "distances"), map_size=8 << 30)  # Creates cache/distances/ with data.mdb and lock.mdb

        # Determine initial panorama based on initialization mode
        if self.init_from_pano_id:
            if self.initial_pano_id is None:
                raise ValueError("init_from_pano_id is True but initial_pano_id is not provided.")
            # Verify the pano exists by fetching its data
            pano_data = self.fetch_pano_data(self.initial_pano_id)
            if pano_data is None:
                raise ValueError(f"Could not fetch data for initial pano ID: {self.initial_pano_id}")
            pano_id = self.initial_pano_id
            print(f"Initial panorama ID (from config): {pano_id}")
        else:
            pano_id = self.get_pano_from_coords(initial_lat, initial_lng, skip_cache=True)
            print(f"Initial panorama ID (from coords): {pano_id}")
            if pano_id is None:
                raise ValueError(f"No valid Street View pano found near ({initial_lat}, {initial_lng}).")

        self.state = pano_id
        self.destination_lat, self.destination_lng = destination_coords
        # Dead-end pruning toggle
        self.enable_deadend_pruning = False  # Disabled to prevent false dead ends from blocking valid paths

        # Create polygon object for destination area if provided
        self.destination_polygon_obj = None
        if self.destination_polygon:
            try:
                # Convert [lat, lng] to (lng, lat) for Shapely (x=lng, y=lat)
                self.destination_polygon_obj = Polygon([(lng, lat) for (lat, lng) in self.destination_polygon])
            except Exception as e:
                print(f"Warning: Could not create destination polygon: {e}")
                self.destination_polygon_obj = None

    def get_observation(self):
        """
        Fetch panorama data and return an observation with headings and aliases.
        
        Returns:
            list of dict: A list of observations, where each observation is a dictionary with:
                - 'heading' (float): The heading direction of the link.
                - 'alias' (str): A generated alias (e.g., 'link_0', 'link_1') mapping to a panorama ID.

        If no panorama data is available or there are no links, an empty list is returned.
        """
        pano_data = self.fetch_pano_data(self.state)
        if not pano_data or 'links' not in pano_data:
            self.current_lat = None
            self.current_lng = None
            return []
        #print(f"Panorama data: {pano_data}")
        # Optionally prune links that are marked dead-end from this node
        raw_links = pano_data['links']
        if self.enable_deadend_pruning:
            try:
                dead_set = set(self.cache.dead_children_for(self.state))
            except Exception:
                dead_set = set()
            pruned = [ln for ln in raw_links if ln.get('pano') not in dead_set]
            if not pruned and raw_links:
                msg = f"[DEADEND-PRUNE] All links for pano {self.state} would be pruned based on dead edges {sorted(dead_set)}. Stopping simulation."
                print(msg)
                raise RuntimeError(msg)
            self.current_links = pruned if pruned else raw_links
        else:
            self.current_links = raw_links
        self.current_lat = pano_data['location']['lat']
        self.current_lng = pano_data['location']['lng']
        links = self.current_links
        links_real = self.current_links
        self.alias_map = {f'link_{i}': link['pano'] for i, link in enumerate(links)}
        reverse_alias_map = {v: k for k, v in self.alias_map.items()}

        #links = [{'heading': link['heading'], 'alias': f'link_{i}'} for i, link in enumerate(self.current_links)]
        links = []
        for i, link in enumerate(self.current_links):
            alias = f'link_{i}'
            robust = self.stable_heading(self.state, link['pano'])
            links.append({
                'heading': robust if robust is not None else link['heading'],
                'alias': alias,
                'raw_heading': link['heading']      # keep for debugging / analysis
            })
        nav_stats = nav_results = None
        if len(links) > 2 and self.experiment_folder:
            try:
                # Attempt to load navigation score
                nav_stats, nav_results = score_run(self.experiment_folder)
                self.score = nav_results.get('nav_asym', {}).get('mean', self.score)
                # reset missing counter on success
                self._missing_eval_steps = 0
            except FileNotFoundError:
                # evaluation file not yet present: increment and check threshold
                self._missing_eval_steps += 1
                if self._missing_eval_steps > self._missing_eval_threshold:
                    raise FileNotFoundError(f"No evaluation file in {self.experiment_folder} after {self._missing_eval_steps} observations.")
                # skip scoring until evaluations appear
                nav_stats = nav_results = None
            except Exception as e:
                print(f"Error computing navigation score: {e}")
                nav_stats = nav_results = None
        elif len(links) > 2:
            # fallback to legacy geometric mean
            self.score = self.geom_mean_score(self.load_outcomes(self.experiment_folder))

        observation = {
            'pano_id': self.state,
            'previous_pano_id': self.previous_state,
            'arrival_heading': self.last_heading,
            'links': links,
            'links_real': links_real,
            'reverse_alias_map': reverse_alias_map,
            'score': self.score
        }
        # If more than two links, embed evaluation directly in each link entry
        if len(links) > 2 and self.enable_evaluations:
            direct_evals = self.evaluate_first_level_actions(self.state, self.destination_lat, self.destination_lng)
            for link_item in links:
                alias = link_item.get('alias')
                pano_id = self.alias_map.get(alias)
                stats = direct_evals.get(pano_id)
                if stats:
                    link_item['avg_delta'] = stats.get('avg_delta')
                    link_item['label'] = stats.get('label')
            # Also keep separate mappings for backward compatibility
            alias_evals = {}
            for pano_id, stats in direct_evals.items():
                alias = reverse_alias_map.get(pano_id)
                if alias:
                    alias_evals[alias] = stats
            observation['evaluations_real'] = direct_evals
            observation['evaluations_alias'] = alias_evals
            # attach navigation score details if computed (backward compatible)
            if nav_stats is not None and nav_results is not None:
                observation['nav_stats'] = nav_stats
                observation['nav_results'] = nav_results
        elif len(links) > 2 and not self.enable_evaluations:
            # Populate placeholders when evaluations are disabled
            for link_item in links:
                link_item['avg_delta'] = None
                link_item['label'] = 'NA'
            observation['evaluations_real'] = {}
            observation['evaluations_alias'] = {}
            observation['nav_stats'] = 'eval disabled'
            observation['nav_results'] = 'eval disabled'
        #print(f"Observation In function: {observation}")
        return observation
    
    def get_observation_legacy(self):
        """
        Fetch panorama data and return an observation with headings and aliases.
        
        Returns:
            list of dict: A list of observations, where each observation is a dictionary with:
                - 'heading' (float): The heading direction of the link.
                - 'alias' (str): A generated alias (e.g., 'link_0', 'link_1') mapping to a panorama ID.

        If no panorama data is available or there are no links, an empty list is returned.
        """
        pano_data = self.fetch_pano_data(self.state)
        if not pano_data or 'links' not in pano_data:
            self.current_lat = None
            self.current_lng = None
            return []
        #print(f"Panorama data: {pano_data}")
        # Legacy path: apply same pruning if enabled
        raw_links = pano_data['links']
        if self.enable_deadend_pruning:
            try:
                dead_set = set(self.cache.dead_children_for(self.state))
            except Exception:
                dead_set = set()
            pruned = [ln for ln in raw_links if ln.get('pano') not in dead_set]
            if not pruned and raw_links:
                msg = f"[DEADEND-PRUNE] All links for pano {self.state} would be pruned based on dead edges {sorted(dead_set)}. Stopping simulation."
                print(msg)
                raise RuntimeError(msg)
            self.current_links = pruned if pruned else raw_links
        else:
            self.current_links = raw_links
        self.current_lat = pano_data['location']['lat']
        self.current_lng = pano_data['location']['lng']
        links = self.current_links
        links_real = self.current_links
        self.alias_map = {f'link_{i}': link['pano'] for i, link in enumerate(links)}
        reverse_alias_map = {v: k for k, v in self.alias_map.items()}
        # observation = [
        #     {'heading': link['heading'], 'alias': f'link_{i}'}
        #     for i, link in enumerate(links)
        # ]
        # return observation
        links = [{'heading': link['heading'], 'alias': f'link_{i}'} for i, link in enumerate(self.current_links)]
        observation = {
            'pano_id': self.state,
            'previous_pano_id': self.previous_state,
            'arrival_heading': self.last_heading,
            'links': links,
            'links_real': links_real,
            'reverse_alias_map' : reverse_alias_map       
        }
        #print(f"Observation In function: {observation}")
        return observation

    # def apply_action(self, action):
    #     """Apply the agent's selected alias to move to a new panorama."""
    #     if action in self.alias_map:
    #         self.state = self.alias_map[action]
    #     else:
    #         raise ValueError(f"Invalid action: {action}")

    def apply_action(self, action):
        """Apply the agent's selected alias to move to a new panorama and update last_heading."""
        if action in self.alias_map:
            for i, link in enumerate(self.current_links):
                if f'link_{i}' == action:
                    self.last_heading = link['heading']
                    break
            self.previous_state = str(self.state)    
            self.state = self.alias_map[action]
        else:
            raise ValueError(f"Invalid action: {action}")    

    # --- Dead-end registration (from strategy via simulation) ---
    def register_dead_end_edge(self, parent_pano_id: str, child_pano_id: str):
        """Mark parent→child as dead-end in the shared LMDB cache.

        Safe across concurrent runs; idempotent.
        """
        if not parent_pano_id or not child_pano_id:
            return
        try:
            self.cache.mark_dead_edge(parent_pano_id, child_pano_id)
            print(f"[DEADEND] Registered dead edge {parent_pano_id} -> {child_pano_id}")
        except Exception as e:
            print(f"[DEADEND] Failed to register dead edge {parent_pano_id}->{child_pano_id}: {e}")

    def _fetch_pano_data_remote(self, pano_id):
        """Fetch panorama data using Playwright and the Google Maps API."""
        try:
            data = self.page.evaluate(f'''
                new Promise((resolve, reject) => {{
                    const streetViewService = new google.maps.StreetViewService();
                    streetViewService.getPanorama({{ pano: '{pano_id}' }}, (data, status) => {{
                        if (status === google.maps.StreetViewStatus.OK) {{
                            resolve({{
                                links: data.links || [],
                                location: {{
                                    lat: data.location.latLng.lat(),
                                    lng: data.location.latLng.lng()
                                }}
                            }});
                        }} else {{
                            reject(status);
                        }}
                    }});
                }})
            ''')
            return data
        except Exception as e:
            print(f"Error fetching pano data: {e}")
            return None

    def fetch_pano_data(self, pano_id):
        # local helpers kept inside for minimal surface change
        def _log_sanitation(action, info):
            try:
                if getattr(self, 'experiment_folder', None):
                    os.makedirs(self.experiment_folder, exist_ok=True)
                    log_path = os.path.join(self.experiment_folder, 'sanitation.log')
                    record = {
                        'ts': datetime.utcnow().isoformat(timespec='seconds') + 'Z',
                        'action': action,
                        **info
                    }
                    with open(log_path, 'a', encoding='utf-8') as f:
                        f.write(json.dumps(record) + "\n")
            except Exception:
                # Never let logging break navigation
                pass

        def _maybe_add_backlink(returning_links, b_lat, b_lng):
            # Ensure we can escape from B by linking back to A if missing
            a_id = getattr(self, 'previous_state', None)
            if not a_id:
                print(f"[SANITATION] {pano_id}: No previous pano; skipping")
                _log_sanitation('skip_no_previous', {'pano': pano_id})
                return returning_links, False
            links_list = list(returning_links or [])
            if any((ln.get('pano') == a_id) for ln in links_list):
                print(f"[SANITATION] {pano_id}: Back-link exists to {a_id}; no sanitation")
                _log_sanitation('exists', {'pano': pano_id, 'prev': a_id})
                return links_list, False

            # Prefer in-memory coords from last step for A
            a_lat, a_lng = getattr(self, 'current_lat', None), getattr(self, 'current_lng', None)
            if a_lat is None or a_lng is None:
                a_lat, a_lng = self.cache.coord_for(a_id)

            if a_lat is None or a_lng is None or b_lat is None or b_lng is None:
                print(f"[SANITATION] {pano_id}: Missing coords for backlink to {a_id}; skipping")
                _log_sanitation('skip_no_coords', {'pano': pano_id, 'prev': a_id})
                return links_list, False

            bearing = self._bearing_between(b_lat, b_lng, a_lat, a_lng)
            links_list.append({'pano': a_id, 'heading': bearing})
            print(f"[SANITATION] {pano_id}: Added back-link to {a_id} (heading={bearing:.1f})")
            _log_sanitation('added_backlink', {
                'pano': pano_id,
                'prev': a_id,
                'heading': bearing
            })
            return links_list, True

        # 1) try cache
        links = self.cache.links_for(pano_id)
        if links:
            lat,lng = self.cache.coord_for(pano_id)
            # Only sanitize when fetching the live current pano after a move A->B
            if pano_id == getattr(self, 'state', None) and getattr(self, 'previous_state', None) and self.previous_state != pano_id:
                sanitized_links, changed = _maybe_add_backlink(links, lat, lng)
                if changed:
                    try:
                        self.cache.update_links(pano_id, sanitized_links)
                    except Exception:
                        pass
            else:
                sanitized_links = links
            return {'links': sanitized_links, 'location': {'lat': lat, 'lng': lng}}

        # 2) fall back to Google
        print(f"[CACHE] {pano_id} – miss, querying API")
        data = self._fetch_pano_data_remote(pano_id)
        if data and 'links' in data:
            # Sanitize remote links before caching — but only for the active pano after a move
            b_lat = data.get('location', {}).get('lat')
            b_lng = data.get('location', {}).get('lng')
            remote_links = data.get('links') or []
            if pano_id == getattr(self, 'state', None) and getattr(self, 'previous_state', None) and self.previous_state != pano_id:
                sanitized_links, _ = _maybe_add_backlink(remote_links, b_lat, b_lng)
            else:
                sanitized_links = remote_links
            # Insert (sanitized or original)
            self.cache.insert_pano(pano_id,
                                b_lat,
                                b_lng,
                                sanitized_links)
            data = {'links': sanitized_links, 'location': {'lat': b_lat, 'lng': b_lng}}
        return data

    def get_current_coordinates(self):
        """Return the current coordinates and panorama ID."""
        return {
            'pano_id': self.state,
            'lat': self.current_lat,
            'lng': self.current_lng
        }
    

    def cleanup(self):
        if self.browser is not None:
            try:
                self.browser.close()
            except:
                pass
            self.browser = None
        if self.playwright is not None:
            try:
                self.playwright.stop()
            except:
                pass
            self.playwright = None
        if getattr(self, "cache", None):
            # Prefer a dedicated close() method when available to hide implementation details.
            try:
                if hasattr(self.cache, "close") and callable(getattr(self.cache, "close")):
                    self.cache.close()
                elif hasattr(self.cache, "conn"):
                    # Legacy SQLite‐based cache
                    self.cache.conn.close()
                elif hasattr(self.cache, "env"):
                    # LMDB environment handle
                    self.cache.env.close()
            except Exception:
                # Suppress any errors during shutdown to avoid masking the original flow
                pass
            self.cache = None    

    def __del__(self):
        self.cleanup()

    def get_pano_from_coords(self, lat: float, lng: float, radius: int = 50, max_radius: int = 3000, *, skip_cache: bool = False) -> str:
        """
        Retrieve a Street View panorama ID near the given coordinates using the environment's browser instance.

        Args:
            lat (float): Latitude of the location.
            lng (float): Longitude of the location.
            radius (int, optional): Initial search radius in meters. Defaults to 50.
            max_radius (int, optional): Maximum search radius in meters. Defaults to 3000.

        Returns:
            str: The panorama ID if found, otherwise None.

        Example:
            ```python
            env = StreetViewEnvironment((40.742077, -73.982914), (40.748817, -73.985428), "YOUR_API_KEY")
            pano_id = env.get_pano_from_coords(40.742077, -73.982914)
            if pano_id:
                env.state = pano_id  # Update the environment state if desired
            ```
        """


        # 0) cache lookup
        if not skip_cache:
            pid = self.cache.nearest_pano(lat, lng, radius_m=radius)
            if pid:
                print(f"[CACHE] start pano – hit within {radius} m")
                return pid

        try:
            result = self.page.evaluate(f'''
                (async () => {{
                    const streetViewService = new google.maps.StreetViewService();
                    let currentRadius = {radius};
                    const maxRadius = {max_radius};
                    const latLng = {{ lat: {lat}, lng: {lng} }};
                    while (currentRadius <= maxRadius) {{
                        const data = await new Promise((resolve, reject) => {{
                            streetViewService.getPanorama({{ location: latLng, radius: currentRadius }}, (data, status) => {{
                                if (status === google.maps.StreetViewStatus.OK) {{
                                    resolve(data);
                                }} else {{
                                    reject(status);
                                }}
                            }});
                        }});
                        if (data.links && data.links.length > 0 && data.location && data.location.pano) {{
                            return data.location.pano;
                        }}
                        currentRadius += 50;
                    }}
                    throw new Error("No panorama with links found within max radius");
                }})()
            ''')
            #return result
        except Exception as e:
            print(f"Error fetching panorama: {e}")
            return None
        
        if result:
            data = self._fetch_pano_data_remote(result)    # one JS call here
            if data and 'links' in data:
                self.cache.insert_pano(
                    result,
                    data['location']['lat'],
                    data['location']['lng'],
                    data['links']
                )     

        return result

    # ── robust-heading helpers ──────────────────────────────────────────────
    def _bearing_between(self, lat0: float, lng0: float,
                        lat1: float, lng1: float) -> float:
        """Great-circle bearing (deg) from (lat0,lng0) → (lat1,lng1)."""
        import math
        φ1, φ2 = map(math.radians, (lat0, lat1))
        Δλ = math.radians(lng1 - lng0)
        y = math.sin(Δλ) * math.cos(φ2)
        x = (math.cos(φ1) * math.sin(φ2)
            - math.sin(φ1) * math.cos(φ2) * math.cos(Δλ))
        return (math.degrees(math.atan2(y, x)) + 360) % 360


    def stable_heading(self,
                    start_pano_id: str,
                    child_pano_id: str,
                    *,
                    max_depth: int = 5) -> float | None:
        """
        Walk forward from *child_pano_id* up to *max_depth* hops **unless**
        another junction (≥ 3 links) is hit.  
        Returns a derived bearing [0-360) ° or *None* (⇒ keep raw heading).
        """
        try:
            base_coords = self.cache.coord_for(start_pano_id)
            if not base_coords:
                return None
            lat0, lng0 = base_coords

            prev, curr, depth = start_pano_id, child_pano_id, 0
            while depth < max_depth:
                data = self.fetch_pano_data(curr)
                if not data or 'links' not in data:
                    break
                # Abort if we hit another decision point
                if len(data['links']) >= 3:
                    return None
                # Follow the one link that isn’t pointing back
                nxt = next((ln['pano'] for ln in data['links'] if ln['pano'] != prev), None)
                if not nxt:
                    break
                prev, curr, depth = curr, nxt, depth + 1

            leaf_coords = self.cache.coord_for(curr)
            if not leaf_coords:
                return None
            lat1, lng1 = leaf_coords
            return self._bearing_between(lat0, lng0, lat1, lng1)
        except Exception:
            return None





    def calculate_walking_distance(self, start_lat: float, start_lng: float, end_lat: float, end_lng: float) -> dict:
        """
        Calculate the walking distance and duration between two points using Google Maps Directions API.

        Args:
            start_lat (float): Starting point latitude
            start_lng (float): Starting point longitude
            end_lat (float): Ending point latitude
            end_lng (float): Ending point longitude

        Returns:
            dict: A dictionary containing:
                - 'distance': The walking distance in meters
                - 'duration': The estimated walking duration in seconds
                - 'status': The status of the request
                - 'error_details': Detailed error information if the request fails
        """
        # Check cache first
        cached_result = self.distance_cache.get(start_lat, start_lng, end_lat, end_lng)
        if cached_result is not None:
            return cached_result

        try:
            result = self.page.evaluate(f'''
                (async () => {{
                    try {{
                        return await new Promise((resolve, reject) => {{
                            if (!google || !google.maps || !google.maps.DirectionsService) {{
                                reject(new Error("Google Maps DirectionsService not available"));
                                return;
                            }}
                            
                            const directionsService = new google.maps.DirectionsService();
                            const origin = new google.maps.LatLng({start_lat}, {start_lng});
                            const destination = new google.maps.LatLng({end_lat}, {end_lng});
                            
                            const request = {{
                                origin: origin,
                                destination: destination,
                                travelMode: google.maps.TravelMode.WALKING,
                                provideRouteAlternatives: false,
                                optimizeWaypoints: false
                            }};
                            
                            directionsService.route(request, (result, status) => {{
                                if (status === google.maps.DirectionsStatus.OK && result && 
                                    result.routes && result.routes.length > 0 && 
                                    result.routes[0].legs && result.routes[0].legs.length > 0) {{
                                    
                                    const leg = result.routes[0].legs[0];
                                    
                                    if (leg.distance && leg.duration) {{
                                        resolve({{
                                            status: 'OK',
                                            distance: leg.distance.value,  // Value in meters
                                            duration: leg.duration.value,  // Value in seconds
                                            distance_text: leg.distance.text,
                                            duration_text: leg.duration.text,
                                            route: result.routes[0]  // Include the full route data
                                        }});
                                    }} else {{
                                        resolve({{
                                            status: 'MISSING_DATA',
                                            distance: null,
                                            duration: null,
                                            error_details: 'Distance or duration missing in response'
                                        }});
                                    }}
                                }} else {{
                                    resolve({{ 
                                        status: status || 'ERROR',
                                        distance: null,
                                        duration: null,
                                        error_details: status
                                    }});
                                }}
                            }});
                        }});
                    }} catch (error) {{
                        return {{
                            status: 'ERROR',
                            distance: null,
                            duration: null,
                            error_details: error.message || 'Unknown error'
                        }};
                    }}
                }})()
            ''')
            # Cache the result before returning
            self.distance_cache.put(start_lat, start_lng, end_lat, end_lng, result)
            return result
        except Exception as e:
            result = {
                'status': 'PYTHON_ERROR',
                'distance': None,
                'duration': None,
                'error_details': str(e)
            }
            # Cache error results too to avoid repeated failures
            self.distance_cache.put(start_lat, start_lng, end_lat, end_lng, result)
            return result

    def is_point_in_destination_polygon(self, lat: float, lng: float) -> bool:
        """
        Check if a point is inside the destination polygon.

        Args:
            lat (float): Latitude of the point
            lng (float): Longitude of the point

        Returns:
            bool: True if the point is inside the destination polygon, False otherwise
        """
        if self.destination_polygon_obj is None:
            return False

        point = Point(lng, lat)  # Note: shapely uses (x, y) = (lng, lat)
        # Use covers to include boundary points as inside
        return self.destination_polygon_obj.covers(point)

    def get_link_tree(self, start_pano_id: str, levels: int = 3) -> dict:
        """
        Retrieve a tree of linked panoramas up to a specified depth.
        Each node contains 'pano', its coordinates, and its 'links' as child nodes.
        """
        def helper(pano_id, depth, exclude_pano=None):
            data = self.fetch_pano_data(pano_id)
            lat = data['location']['lat'] if data and 'location' in data else None
            lng = data['location']['lng'] if data and 'location' in data else None
            node = {'pano': pano_id, 'lat': lat, 'lng': lng, 'links': []}
            if not data or 'links' not in data or depth <= 0:
                return node
            for link in data['links']:
                child_pano = link.get('pano')
                if exclude_pano is not None and child_pano == exclude_pano:
                    continue
                subtree = helper(child_pano, depth - 1, pano_id)
                node['links'].append(subtree)
            return node
        return helper(start_pano_id, levels, None)

    def evaluate_first_level_actions(self, start_pano_id: str, dest_lat: float, dest_lng: float, levels: int = 3) -> dict:
        """
        Evaluate immediate child link choices by average change in walking distance to the destination.
        Returns a dict mapping each child pano to {'avg_delta': float, 'label': 'RIGHT'/'WRONG'}.
        """
        # Build the link tree
        tree = self.get_link_tree(start_pano_id, levels)
        root = tree
        root_lat = root.get('lat')
        root_lng = root.get('lng')
        # Calculate walking distance from root to destination
        root_res = self.calculate_walking_distance(root_lat, root_lng, dest_lat, dest_lng)
        root_dist = root_res.get('distance')
        evaluations = {}
        # Helper to collect all leaf nodes in a subtree
        def collect_leaves(node):
            if not node.get('links'):
                return [node]
            leaves = []
            for child in node.get('links', []):
                leaves.extend(collect_leaves(child))
            return leaves
        # Evaluate each immediate child of the root
        for child in root.get('links', []):
            leaves = collect_leaves(child)
            deltas = []
            for leaf in leaves:
                leaf_lat = leaf.get('lat')
                leaf_lng = leaf.get('lng')
                res = self.calculate_walking_distance(leaf_lat, leaf_lng, dest_lat, dest_lng)
                leaf_dist = res.get('distance')
                if root_dist is not None and leaf_dist is not None:
                    deltas.append(root_dist - leaf_dist)
            avg_delta = sum(deltas) / len(deltas) if deltas else 0
            label = 'RIGHT' if avg_delta > 0 else 'WRONG'
            evaluations[child.get('pano')] = {'avg_delta': avg_delta, 'label': label}
        return evaluations
    
    def load_outcomes(self,folder):
        """→ list[bool]  (True = RIGHT, False = WRONG)."""
        f = glob.glob(os.path.join(folder, 'decision_evaluations_*.json'))
        if not f:
            return []
        with open(f[0], 'r', encoding='utf-8') as h:
            data = json.load(h)
        return [(d['status'] == 'RIGHT') for d in data]

    # ── geometric mean score ────────────────────────────────────────────

    def geom_mean_score(self,outcomes, g=2, cap=256):
        """
        outcomes : iterable of booleans
        g        : growth base (>1)
        cap      : max magnitude of any single increment
        returns  : per-step geometric score (float)
        """
        streak = 0
        total  = 0.0
        for ok in outcomes:
            if ok:
                streak = streak + 1 if streak >= 0 else 1
                inc =  min(g ** (streak - 1), cap)
            else:
                streak = streak - 1 if streak <= 0 else -1
                inc = -min(g ** (abs(streak) - 1), cap)
            total += inc
        return total / len(outcomes) if outcomes else 0.0

    def close_browser(self):
        self.browser.close()    


if __name__ == "__main__":
    import os
    import json
    
    # Load API key from environment
    api_key = os.environ.get("GOOGLE_MAPS_API_KEY", "")
    
    # Test coordinates in New York City
    initial_coords = (40.742077, -73.982914)  # Near NYU, NYC
    destination_coords = (40.748066, -73.984822)  # Near Empire State Building

    #try:
    env = StreetViewEnvironment(
        initial_coords=initial_coords,
        destination_coords=destination_coords,
        api_key=api_key
    )
        
        # Make sure Directions service is loaded
        # env.page.add_script_tag(url=f'https://maps.googleapis.com/maps/api/js?key={api_key}&libraries=places,geometry,directions')
        # env.page.wait_for_load_state('networkidle')

        # eval_results = env.evaluate_first_level_actions(env.state, destination_coords[0], destination_coords[1])

        #print(f"Evaluation results: {eval_results}")
        # with open('evaluation.json', 'w') as f:
        #     json.dump(eval_results, f, indent=2)


        # output = env.get_observation()
        # print(f"Output: {output}")

        # with open('observation.json', 'w') as f:
        #     json.dump(output, f, indent=2)
      
    # except Exception as e:
    #     print(f"Error: {e}")
    #     import traceback
    #     traceback.print_exc()
    # finally:
    #     if 'env' in locals():
    #         env.cleanup()
    
    # Test the new plot_route_map helper
    #if len(sys.argv) > 1:
    visited_file = r"X:\logs\OptUltra\0503_182704_Run\Branch_Branch_Branch_Branch_Run_Depth0_Initial_Dec4_DirectPrompt_20250503_183535_Dec5_DirectPrompt_20250503_184832_Dec22_DirectPrompt_20250503_190120_Dec24_DirectPrompt_20250503_190949\visited_coordinates.json" #sys.argv[1]
    image_path = 'custom_route_mapZ2Extreme.png'  # Specify the path of the image
    try:
        #png = env.plot_route_map(visited_file)
        png = env.render_route_map_auto(
            visited_file,
            width=640,
            height=640,
            margin_px=5
            #max_iter=7,
            # scale_step=0.1
        )
        with open(image_path, 'wb') as imgf:
            imgf.write(png)
        print(f"Route map saved to {image_path} from {visited_file}")
    except Exception as e:
        print(f"Error plotting route map: {e}")
    # else:
    #     print("Usage: python environmentpro.py <visited_coordinates.json>")
    

