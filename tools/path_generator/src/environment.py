from .base.environment import Environment
from typing import Optional
import json
import glob
import os
import requests
import sys
import json, math, requests, urllib.parse
import polyline   # pip install polyline
import sqlite3
from datetime import datetime
import logging
import traceback

# Use a stable project root so caches don't depend on process CWD.
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

# Configure logging
logging.basicConfig(
    level=logging.DEBUG,
    format='%(asctime)s [%(levelname)s] %(name)s:%(lineno)d - %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S'
)
env_logger = logging.getLogger("Environment")
env_logger.setLevel(logging.DEBUG)

# Proxy server support - can use either browser or proxy
try:
    from playwright.sync_api import sync_playwright
    PLAYWRIGHT_AVAILABLE = True
    env_logger.info("Playwright is AVAILABLE")
except ImportError:
    PLAYWRIGHT_AVAILABLE = False
    env_logger.warning("Playwright is NOT AVAILABLE")

# Cache imports
from .cache import PanoCache

class StreetViewEnvironment(Environment):
    def __init__(self, initial_coords, destination_coords, api_key, *, enable_evaluations: bool = True, initial_pano_id: Optional[str] = None, proxy_url: Optional[str] = None):
        env_logger.info("=" * 80)
        env_logger.info("[ENV] StreetViewEnvironment __init__ STARTING")
        env_logger.info("=" * 80)
        env_logger.info(f"[ENV] initial_coords: {initial_coords}")
        env_logger.info(f"[ENV] destination_coords: {destination_coords}")
        env_logger.info(f"[ENV] api_key present: {bool(api_key)}")
        env_logger.info(f"[ENV] api_key length: {len(api_key) if api_key else 0}")
        env_logger.info(f"[ENV] api_key first 10 chars: {api_key[:10] if api_key else 'EMPTY'}")
        env_logger.info(f"[ENV] enable_evaluations: {enable_evaluations}")
        env_logger.info(f"[ENV] initial_pano_id: {initial_pano_id}")
        env_logger.info(f"[ENV] proxy_url: {proxy_url}")
        env_logger.info(f"[ENV] PLAYWRIGHT_AVAILABLE: {PLAYWRIGHT_AVAILABLE}")

        super().__init__(initial_state="Init")
        initial_lat, initial_lng = initial_coords
        self.api_key = api_key
        self.proxy_url = proxy_url or "http://localhost:12345"  # Default proxy URL
        env_logger.info(f"[ENV] Effective proxy_url: {self.proxy_url}")

        # Initialize browser only if not using proxy and playwright is available
        self.use_proxy = proxy_url is not None
        env_logger.info(f"[ENV] use_proxy: {self.use_proxy}")

        if not self.use_proxy and PLAYWRIGHT_AVAILABLE:
            env_logger.info("[ENV] Initializing local browser (not using proxy)...")
            print("[ENVPRO] Initializing local browser...")

            try:
                env_logger.info("[ENV] Starting Playwright...")
                self.playwright = sync_playwright().start()
                env_logger.info("[ENV] Playwright started successfully")

                # Add server-safe flags to avoid sandbox/dev-shm/gpu issues which can hang on new_page()
                env_logger.info("[ENV] Launching Chromium browser...")
                self.browser = self.playwright.chromium.launch(
                    headless=True,
                    args=[
                        '--no-sandbox',
                        '--disable-setuid-sandbox',
                        '--disable-dev-shm-usage',
                        '--disable-gpu'
                    ]
                )
                env_logger.info("[ENV] Chromium launched successfully")

                print("[ENVPRO] after launch, before new_page")
                env_logger.info("[ENV] Creating new page...")
                self.page = self.browser.new_page()
                env_logger.info("[ENV] Page created successfully")

                print("[ENVPRO] after new_page, before maps script")
                env_logger.info("[ENV] Navigating to about:blank...")
                self.page.goto('about:blank')
                env_logger.info("[ENV] Navigation complete")

                maps_url = f'https://maps.googleapis.com/maps/api/js?key={api_key}&libraries=places,geometry,directions'
                env_logger.info(f"[ENV] Adding Google Maps script tag...")
                env_logger.info(f"[ENV] Maps URL (first 100 chars): {maps_url[:100]}...")
                self.page.add_script_tag(url=maps_url)
                env_logger.info("[ENV] Script tag added, waiting for networkidle...")

                self.page.wait_for_load_state('networkidle')
                env_logger.info("[ENV] Network idle reached")

                # Verify Google Maps loaded
                try:
                    has_google = self.page.evaluate('typeof google !== "undefined"')
                    has_maps = self.page.evaluate('typeof google !== "undefined" && typeof google.maps !== "undefined"')
                    has_sv = self.page.evaluate('typeof google !== "undefined" && typeof google.maps !== "undefined" && typeof google.maps.StreetViewService !== "undefined"')
                    env_logger.info(f"[ENV] Google Maps verification: google={has_google}, maps={has_maps}, StreetViewService={has_sv}")
                    if not has_sv:
                        env_logger.error("[ENV] !!! GOOGLE MAPS NOT LOADED PROPERLY - CHECK API KEY !!!")
                except Exception as verify_err:
                    env_logger.error(f"[ENV] Failed to verify Google Maps: {verify_err}")

                print("[ENVPRO] Local browser initialized")
                env_logger.info("[ENV] Local browser initialization COMPLETE")

            except Exception as browser_err:
                env_logger.error("[ENV] !!! BROWSER INITIALIZATION FAILED !!!")
                env_logger.error(f"[ENV] Error: {type(browser_err).__name__}: {browser_err}")
                env_logger.error(f"[ENV] Traceback:\n{traceback.format_exc()}")
                raise

        elif self.use_proxy:
            print(f"[ENVPRO] Using proxy server at {self.proxy_url}")
            env_logger.info(f"[ENV] Using proxy server at {self.proxy_url}")
            self.playwright = None
            self.browser = None
            self.page = None
        else:
            env_logger.error("[ENV] !!! FATAL: Neither proxy server nor local browser available !!!")
            raise RuntimeError("Neither proxy server nor local browser available. Install playwright or provide proxy_url.")
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
        self.cache = PanoCache(os.path.join(PROJECT_ROOT, "cache", "pano"))#PanoCache("pano_cache")      # one shared conn

        # Require explicit panoId for initialization; do not probe by coordinates
        if initial_pano_id is None:
            raise ValueError("initial_pano_id is required for StreetViewEnvironment initialization")
        self.state = initial_pano_id
        data = self.fetch_pano_data(initial_pano_id)
        if data and 'location' in data:
            self.current_lat = data['location'].get('lat')
            self.current_lng = data['location'].get('lng')
        print(f"Initial panorama ID (provided): {initial_pano_id}")
        self.destination_lat, self.destination_lng = destination_coords
        # Dead-end pruning toggle
        self.enable_deadend_pruning = True

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
        # Navigation scoring disabled (score_navigation module not included)

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
        """Fetch panorama data using proxy server or local browser."""
        env_logger.info(f"[ENV] _fetch_pano_data_remote({pano_id}) called")
        env_logger.info(f"[ENV] use_proxy: {self.use_proxy}")

        if self.use_proxy:
            env_logger.info(f"[ENV] Fetching via PROXY server...")
            return self._fetch_pano_data_via_proxy(pano_id)
        else:
            env_logger.info(f"[ENV] Fetching via local BROWSER...")
            return self._fetch_pano_data_via_browser(pano_id)

    def _fetch_pano_data_via_proxy(self, pano_id):
        """Fetch panorama data using the proxy server."""
        env_logger.info(f"[ENV] _fetch_pano_data_via_proxy({pano_id})")
        env_logger.info(f"[ENV] proxy_url: {self.proxy_url}")
        env_logger.info(f"[ENV] api_key present: {bool(self.api_key)}")

        try:
            request_url = f"{self.proxy_url}/fetch_pano"
            request_body = {"pano_id": pano_id, "api_key": self.api_key}
            env_logger.debug(f"[ENV] POST {request_url}")
            env_logger.debug(f"[ENV] Request body: pano_id={pano_id}, api_key length={len(self.api_key) if self.api_key else 0}")

            response = requests.post(
                request_url,
                json=request_body,
                timeout=30
            )
            env_logger.info(f"[ENV] Response status: {response.status_code}")
            env_logger.debug(f"[ENV] Response headers: {dict(response.headers)}")

            response.raise_for_status()
            result = response.json()
            env_logger.info(f"[ENV] Response JSON keys: {result.keys() if isinstance(result, dict) else type(result)}")
            if isinstance(result, dict):
                env_logger.info(f"[ENV] links count: {len(result.get('links', []))}")
                env_logger.info(f"[ENV] location: {result.get('location')}")

            return result

        except requests.RequestException as e:
            env_logger.error(f"[ENV] !!! PROXY REQUEST FAILED !!!")
            env_logger.error(f"[ENV] Error: {type(e).__name__}: {e}")
            env_logger.error(f"[ENV] Traceback:\n{traceback.format_exc()}")
            print(f"Error fetching pano data via proxy: {e}")
            return None

    def _fetch_pano_data_via_browser(self, pano_id):
        """Fetch panorama data using local Playwright browser."""
        env_logger.info(f"[ENV] _fetch_pano_data_via_browser({pano_id})")
        env_logger.debug(f"[ENV] page state: {self.page is not None}")

        try:
            # Verify Google Maps is available
            try:
                has_sv = self.page.evaluate('typeof google !== "undefined" && typeof google.maps !== "undefined" && typeof google.maps.StreetViewService !== "undefined"')
                env_logger.debug(f"[ENV] StreetViewService available: {has_sv}")
                if not has_sv:
                    env_logger.error("[ENV] !!! StreetViewService NOT AVAILABLE !!!")
                    return None
            except Exception as check_err:
                env_logger.error(f"[ENV] Failed to check StreetViewService: {check_err}")

            js_code = f'''
                new Promise((resolve, reject) => {{
                    console.log("ENV: Starting getPanorama for: {pano_id}");
                    const streetViewService = new google.maps.StreetViewService();
                    streetViewService.getPanorama({{ pano: '{pano_id}' }}, (data, status) => {{
                        console.log("ENV: getPanorama callback - status:", status);
                        if (status === google.maps.StreetViewStatus.OK) {{
                            console.log("ENV: SUCCESS - links:", data.links ? data.links.length : 0);
                            resolve({{
                                links: data.links || [],
                                location: {{
                                    lat: data.location.latLng.lat(),
                                    lng: data.location.latLng.lng()
                                }}
                            }});
                        }} else {{
                            console.log("ENV: FAILED - status:", status);
                            reject(status);
                        }}
                    }});
                }})
            '''

            env_logger.debug(f"[ENV] Executing page.evaluate for pano {pano_id}...")
            data = self.page.evaluate(js_code)

            env_logger.info(f"[ENV] page.evaluate SUCCESS")
            env_logger.info(f"[ENV] links count: {len(data.get('links', []))}")
            env_logger.info(f"[ENV] location: {data.get('location')}")
            return data

        except Exception as e:
            env_logger.error(f"[ENV] !!! page.evaluate FAILED !!!")
            env_logger.error(f"[ENV] Error: {type(e).__name__}: {e}")
            env_logger.error(f"[ENV] Traceback:\n{traceback.format_exc()}")
            print(f"Error fetching pano data via browser: {e}")
            return None

    def fetch_pano_data(self, pano_id):
        env_logger.info(f"[ENV] fetch_pano_data({pano_id}) CALLED")

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
        env_logger.debug(f"[ENV] Checking cache for {pano_id}...")
        links = self.cache.links_for(pano_id)
        env_logger.debug(f"[ENV] cache.links_for({pano_id}): {len(links) if links else 0} links")

        if links:
            lat, lng = self.cache.coord_for(pano_id)
            env_logger.info(f"[ENV] CACHE HIT for {pano_id}: {len(links)} links, coords=({lat}, {lng})")
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
        env_logger.info(f"[ENV] CACHE MISS for {pano_id}, will query API")
        print(f"[CACHE] {pano_id} – miss, querying API")

        env_logger.info(f"[ENV] Calling _fetch_pano_data_remote({pano_id})...")
        data = self._fetch_pano_data_remote(pano_id)
        env_logger.info(f"[ENV] _fetch_pano_data_remote result: {data is not None}")

        if data and 'links' in data:
            env_logger.info(f"[ENV] Got data from API: {len(data.get('links', []))} links")
            # Sanitize remote links before caching — but only for the active pano after a move
            b_lat = data.get('location', {}).get('lat')
            b_lng = data.get('location', {}).get('lng')
            remote_links = data.get('links') or []
            env_logger.debug(f"[ENV] Remote location: ({b_lat}, {b_lng})")

            if pano_id == getattr(self, 'state', None) and getattr(self, 'previous_state', None) and self.previous_state != pano_id:
                sanitized_links, _ = _maybe_add_backlink(remote_links, b_lat, b_lng)
            else:
                sanitized_links = remote_links

            # Insert (sanitized or original)
            env_logger.debug(f"[ENV] Inserting into cache...")
            self.cache.insert_pano(pano_id,
                                b_lat,
                                b_lng,
                                sanitized_links)
            env_logger.info(f"[ENV] Cache updated for {pano_id}")
            data = {'links': sanitized_links, 'location': {'lat': b_lat, 'lng': b_lng}}
        else:
            env_logger.warning(f"[ENV] !!! NO DATA returned from API for {pano_id} !!!")
            if data:
                env_logger.warning(f"[ENV] data keys: {data.keys() if isinstance(data, dict) else type(data)}")

        return data

    def get_current_coordinates(self):
        """Return the current coordinates and panorama ID."""
        return {
            'pano_id': self.state,
            'lat': self.current_lat,
            'lng': self.current_lng
        }
    

    def cleanup(self):
        # Only clean up browser resources if not using proxy
        if not self.use_proxy:
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
            env = StreetViewEnvironment((40.742077, -73.982914), (40.748817, -73.985428), os.environ["GOOGLE_MAPS_API_KEY"])
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

        if self.use_proxy:
            try:
                response = requests.post(
                    f"{self.proxy_url}/fetch_pano_coords",
                    json={
                        "lat": lat,
                        "lng": lng,
                        "radius": radius,
                        "max_radius": max_radius,
                        "api_key": self.api_key
                    },
                    timeout=30
                )
                response.raise_for_status()
                result = response.json().get("pano_id")
            except requests.RequestException as e:
                print(f"Error fetching panorama via proxy: {e}")
                return None
        else:
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
            except Exception as e:
                print(f"Error fetching panorama via browser: {e}")
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
        Calculate the walking distance and duration between two points using mock values.
        No Google Maps API calls - uses Haversine distance calculation with walking speed estimate.

        Args:
            start_lat (float): Starting point latitude
            start_lng (float): Starting point longitude
            end_lat (float): Ending point latitude
            end_lng (float): Ending point longitude

        Returns:
            dict: A dictionary containing:
                - 'distance': The walking distance in meters (Haversine calculation)
                - 'duration': The estimated walking duration in seconds (based on 5 km/h walking speed)
                - 'status': Always 'OK' for mock implementation
                - 'distance_text': Human-readable distance string
                - 'duration_text': Human-readable duration string
                - 'route': Mock route data (empty dict)
        """
        try:
            # Calculate Haversine distance (straight-line distance)
            import math

            def haversine_distance(lat1, lng1, lat2, lng2):
                R = 6371000.0  # Earth radius in meters
                lat1_rad, lat2_rad = map(math.radians, (lat1, lat2))
                dlat = math.radians(lat2 - lat1)
                dlng = math.radians(lng2 - lng1)

                a = math.sin(dlat/2)**2 + math.cos(lat1_rad) * math.cos(lat2_rad) * math.sin(dlng/2)**2
                c = 2 * math.atan2(math.sqrt(a), math.sqrt(1 - a))
                return R * c

            # Calculate distance in meters
            distance_meters = haversine_distance(start_lat, start_lng, end_lat, end_lng)

            # Estimate walking time: assume 5 km/h walking speed (1.39 m/s)
            walking_speed_mps = 1.39  # meters per second (5 km/h)
            duration_seconds = distance_meters / walking_speed_mps

            # Format distance text
            if distance_meters < 1000:
                distance_text = f"{distance_meters:.1f} m"
            else:
                distance_text = f"{distance_meters/1000:.1f} km"

            # Format duration text
            hours = int(duration_seconds // 3600)
            minutes = int((duration_seconds % 3600) // 60)

            if hours > 0:
                duration_text = f"{hours} hour{'s' if hours > 1 else ''} {minutes} min"
            else:
                duration_text = f"{minutes} min"

            return {
                'status': 'OK',
                'distance': int(distance_meters),
                'duration': int(duration_seconds),
                'distance_text': distance_text,
                'duration_text': duration_text,
                'route': {}  # Mock empty route data
            }

        except Exception as e:
            return {
                'status': 'MOCK_ERROR',
                'distance': None,
                'duration': None,
                'error_details': f'Mock calculation error: {str(e)}'
            }

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
    api_key = os.environ.get("GOOGLE_MAPS_API_KEY")
    if not api_key:
        raise SystemExit("Missing GOOGLE_MAPS_API_KEY in environment")
    
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
    

