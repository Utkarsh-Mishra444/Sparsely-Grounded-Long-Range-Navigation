#!/usr/bin/env python3
"""
Panorama Proxy Server

A FastAPI server that manages a pool of Playwright browsers to handle
Google Street View panorama fetch requests concurrently.

Usage:
    python -m src.proxy

The server will start on http://localhost:8000 and provide endpoints for:
- POST /fetch_pano: Fetch panorama data by ID
- POST /fetch_pano_coords: Fetch panorama ID by coordinates
"""

import os
import traceback

# Load .env file if present
try:
    from dotenv import load_dotenv
    load_dotenv()
    print("[PROXY] Loaded .env file successfully")
except ImportError:
    print("[PROXY] python-dotenv not installed, using system env vars")
    pass  # python-dotenv not installed, use system env vars
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from typing import Optional, Dict, Any, List
import asyncio
from concurrent.futures import ThreadPoolExecutor
import threading
import queue
import time
import logging
from playwright.sync_api import sync_playwright
import uvicorn

# Configure detailed logging
logging.basicConfig(
    level=logging.DEBUG,
    format='%(asctime)s [%(levelname)s] %(name)s:%(lineno)d - %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S'
)
logger = logging.getLogger("PanoProxy")
logger.setLevel(logging.DEBUG)

# Log startup info
logger.info("=" * 80)
logger.info("PANORAMA PROXY SERVER STARTING")
logger.info("=" * 80)

app = FastAPI(title="Panorama Proxy Server", version="1.0.0")

class PanoRequest(BaseModel):
    pano_id: str
    api_key: str

class CoordsRequest(BaseModel):
    lat: float
    lng: float
    radius: int = 50
    max_radius: int = 3000
    api_key: str

class BrowserWorker:
    """Manages a single browser instance and its request queue."""

    def __init__(self, worker_id: int, api_key: str):
        logger.info(f"[WORKER:{worker_id}] __init__ called")
        logger.info(f"[WORKER:{worker_id}] api_key present: {bool(api_key)}")
        logger.info(f"[WORKER:{worker_id}] api_key length: {len(api_key) if api_key else 0}")
        logger.info(f"[WORKER:{worker_id}] api_key first 10 chars: {api_key[:10] if api_key else 'EMPTY'}")

        self.worker_id = worker_id
        self.api_key = api_key
        self.playwright = None
        self.browser = None
        self.page = None
        self.request_queue = queue.Queue()
        self.response_queues = {}  # request_id -> response_queue
        self.running = False
        self.thread = None
        self.request_counter = 0

    def start(self):
        """Start the browser worker thread."""
        logger.info(f"[WORKER:{self.worker_id}] start() called")
        self.running = True
        self.thread = threading.Thread(target=self._run_worker, daemon=True)
        self.thread.start()
        logger.info(f"[WORKER:{self.worker_id}] Thread started: {self.thread.is_alive()}")

    def stop(self):
        """Stop the browser worker."""
        self.running = False
        if self.thread:
            self.thread.join(timeout=5)
        if self.browser:
            try:
                self.browser.close()
            except:
                pass
        if self.playwright:
            try:
                self.playwright.stop()
            except:
                pass
        logger.info(f"Browser worker {self.worker_id} stopped")

    def submit_request(self, request_type: str, **kwargs) -> str:
        """Submit a request to this worker and return a request ID."""
        request_id = f"{self.worker_id}_{self.request_counter}"
        self.request_counter += 1

        # Create response queue for this request
        response_queue = queue.Queue()
        self.response_queues[request_id] = response_queue

        # Submit request to worker queue
        self.request_queue.put({
            'request_id': request_id,
            'type': request_type,
            **kwargs
        })

        return request_id

    def get_response(self, request_id: str, timeout: float = 30.0):
        """Get response for a request ID."""
        if request_id not in self.response_queues:
            raise ValueError(f"Unknown request ID: {request_id}")

        try:
            response = self.response_queues[request_id].get(timeout=timeout)
            del self.response_queues[request_id]
            return response
        except queue.Empty:
            del self.response_queues[request_id]
            raise TimeoutError(f"Request {request_id} timed out")

    def _initialize_browser(self):
        """Initialize the browser instance."""
        logger.info(f"[WORKER:{self.worker_id}] _initialize_browser() starting...")
        logger.info(f"[WORKER:{self.worker_id}] API key check: present={bool(self.api_key)}, len={len(self.api_key) if self.api_key else 0}")

        try:
            logger.info(f"[WORKER:{self.worker_id}] Starting Playwright...")
            self.playwright = sync_playwright().start()
            logger.info(f"[WORKER:{self.worker_id}] Playwright started successfully")

            logger.info(f"[WORKER:{self.worker_id}] Launching Chromium browser...")
            self.browser = self.playwright.chromium.launch(
                headless=True,
                args=[
                    '--no-sandbox',
                    '--disable-setuid-sandbox',
                    '--disable-dev-shm-usage',
                    '--disable-gpu'
                ]
            )
            logger.info(f"[WORKER:{self.worker_id}] Chromium launched successfully")

            logger.info(f"[WORKER:{self.worker_id}] Creating new page...")
            self.page = self.browser.new_page()
            logger.info(f"[WORKER:{self.worker_id}] Page created")

            logger.info(f"[WORKER:{self.worker_id}] Navigating to about:blank...")
            self.page.goto('about:blank')
            logger.info(f"[WORKER:{self.worker_id}] Navigation complete")

            maps_url = f'https://maps.googleapis.com/maps/api/js?key={self.api_key}&libraries=places,geometry,directions'
            logger.info(f"[WORKER:{self.worker_id}] Adding Google Maps script tag...")
            logger.info(f"[WORKER:{self.worker_id}] Maps URL (first 100 chars): {maps_url[:100]}...")

            self.page.add_script_tag(url=maps_url)
            logger.info(f"[WORKER:{self.worker_id}] Script tag added, waiting for networkidle...")

            self.page.wait_for_load_state('networkidle')
            logger.info(f"[WORKER:{self.worker_id}] Network idle reached")

            # Verify Google Maps loaded
            try:
                has_google = self.page.evaluate('typeof google !== "undefined"')
                has_maps = self.page.evaluate('typeof google !== "undefined" && typeof google.maps !== "undefined"')
                has_sv = self.page.evaluate('typeof google !== "undefined" && typeof google.maps !== "undefined" && typeof google.maps.StreetViewService !== "undefined"')
                logger.info(f"[WORKER:{self.worker_id}] Google Maps verification: google={has_google}, maps={has_maps}, StreetViewService={has_sv}")

                if not has_sv:
                    logger.error(f"[WORKER:{self.worker_id}] !!! GOOGLE MAPS NOT LOADED PROPERLY !!!")
                    logger.error(f"[WORKER:{self.worker_id}] This may indicate an API key issue!")
            except Exception as verify_err:
                logger.error(f"[WORKER:{self.worker_id}] Failed to verify Google Maps: {verify_err}")

            logger.info(f"[WORKER:{self.worker_id}] Browser initialization COMPLETE")
        except Exception as e:
            logger.error(f"[WORKER:{self.worker_id}] !!! BROWSER INITIALIZATION FAILED !!!")
            logger.error(f"[WORKER:{self.worker_id}] Error: {type(e).__name__}: {e}")
            logger.error(f"[WORKER:{self.worker_id}] Traceback:\n{traceback.format_exc()}")
            raise

    def _run_worker(self):
        """Main worker loop that processes requests."""
        try:
            self._initialize_browser()

            while self.running:
                try:
                    # Get next request with timeout to allow checking self.running
                    request = self.request_queue.get(timeout=1.0)
                    request_id = request['request_id']

                    try:
                        if request['type'] == 'fetch_pano':
                            result = self._fetch_pano_data_remote(request['pano_id'])
                        elif request['type'] == 'fetch_pano_coords':
                            result = self._fetch_pano_from_coords(
                                request['lat'], request['lng'],
                                request['radius'], request['max_radius']
                            )
                        else:
                            result = {'error': f'Unknown request type: {request["type"]}'}

                        # Send response back
                        self.response_queues[request_id].put(result)

                    except Exception as e:
                        logger.error(f"Error processing request {request_id}: {e}")
                        self.response_queues[request_id].put({'error': str(e)})

                    finally:
                        self.request_queue.task_done()

                except queue.Empty:
                    continue

        except Exception as e:
            logger.error(f"Worker {self.worker_id} crashed: {e}")
        finally:
            self._cleanup()

    def _fetch_pano_data_remote(self, pano_id: str) -> Dict[str, Any]:
        """Fetch panorama data using the browser."""
        logger.info(f"[WORKER:{self.worker_id}] _fetch_pano_data_remote({pano_id}) called")
        logger.debug(f"[WORKER:{self.worker_id}] Page state: {self.page is not None}")

        try:
            # First verify Google Maps is available
            logger.debug(f"[WORKER:{self.worker_id}] Verifying Google Maps availability...")
            try:
                has_sv = self.page.evaluate('typeof google !== "undefined" && typeof google.maps !== "undefined" && typeof google.maps.StreetViewService !== "undefined"')
                logger.debug(f"[WORKER:{self.worker_id}] StreetViewService available: {has_sv}")
                if not has_sv:
                    logger.error(f"[WORKER:{self.worker_id}] !!! StreetViewService NOT AVAILABLE !!!")
                    return {'error': 'Google Maps StreetViewService not available - check API key'}
            except Exception as check_err:
                logger.error(f"[WORKER:{self.worker_id}] Failed to check StreetViewService: {check_err}")

            js_code = f'''
                new Promise((resolve, reject) => {{
                    console.log("Starting getPanorama for: {pano_id}");
                    const streetViewService = new google.maps.StreetViewService();
                    streetViewService.getPanorama({{ pano: '{pano_id}' }}, (data, status) => {{
                        console.log("getPanorama callback - status:", status);
                        if (status === google.maps.StreetViewStatus.OK) {{
                            console.log("getPanorama SUCCESS - links:", data.links ? data.links.length : 0);
                            resolve({{
                                links: data.links || [],
                                location: {{
                                    lat: data.location.latLng.lat(),
                                    lng: data.location.latLng.lng()
                                }}
                            }});
                        }} else {{
                            console.log("getPanorama FAILED - status:", status);
                            reject(status);
                        }}
                    }});
                }})
            '''

            logger.debug(f"[WORKER:{self.worker_id}] Executing page.evaluate for pano {pano_id}...")
            data = self.page.evaluate(js_code)

            logger.info(f"[WORKER:{self.worker_id}] page.evaluate SUCCESS for {pano_id}")
            logger.debug(f"[WORKER:{self.worker_id}] Result: links={len(data.get('links', []))}, location={data.get('location')}")
            return data

        except Exception as e:
            logger.error(f"[WORKER:{self.worker_id}] !!! page.evaluate FAILED for {pano_id} !!!")
            logger.error(f"[WORKER:{self.worker_id}] Error type: {type(e).__name__}")
            logger.error(f"[WORKER:{self.worker_id}] Error message: {e}")
            logger.error(f"[WORKER:{self.worker_id}] Traceback:\n{traceback.format_exc()}")
            return {'error': str(e)}

    def _fetch_pano_from_coords(self, lat: float, lng: float, radius: int, max_radius: int) -> str:
        """Fetch panorama ID from coordinates."""
        logger.info(f"[WORKER:{self.worker_id}] _fetch_pano_from_coords(lat={lat}, lng={lng}, radius={radius}, max_radius={max_radius})")

        try:
            # Verify Google Maps is available
            try:
                has_sv = self.page.evaluate('typeof google !== "undefined" && typeof google.maps !== "undefined" && typeof google.maps.StreetViewService !== "undefined"')
                logger.debug(f"[WORKER:{self.worker_id}] StreetViewService available: {has_sv}")
                if not has_sv:
                    logger.error(f"[WORKER:{self.worker_id}] !!! StreetViewService NOT AVAILABLE for coords fetch !!!")
                    return {'error': 'Google Maps StreetViewService not available - check API key'}
            except Exception as check_err:
                logger.error(f"[WORKER:{self.worker_id}] Failed to check StreetViewService: {check_err}")

            js_code = f'''
                (async () => {{
                    console.log("Starting coords search at:", {lat}, {lng});
                    const streetViewService = new google.maps.StreetViewService();
                    let currentRadius = {radius};
                    const maxRadius = {max_radius};
                    const latLng = {{ lat: {lat}, lng: {lng} }};
                    while (currentRadius <= maxRadius) {{
                        console.log("Trying radius:", currentRadius);
                        try {{
                            const data = await new Promise((resolve, reject) => {{
                                streetViewService.getPanorama({{ location: latLng, radius: currentRadius }}, (data, status) => {{
                                    console.log("Coords search callback - status:", status, "radius:", currentRadius);
                                    if (status === google.maps.StreetViewStatus.OK) {{
                                        resolve(data);
                                    }} else {{
                                        reject(status);
                                    }}
                                }});
                            }});
                            console.log("Got data - links:", data.links ? data.links.length : 0);
                            if (data.links && data.links.length > 0 && data.location && data.location.pano) {{
                                console.log("Found pano:", data.location.pano);
                                return data.location.pano;
                            }}
                        }} catch (e) {{
                            console.log("Radius", currentRadius, "failed:", e);
                        }}
                        currentRadius += 50;
                    }}
                    throw new Error("No panorama with links found within max radius");
                }})()
            '''

            logger.debug(f"[WORKER:{self.worker_id}] Executing page.evaluate for coords ({lat}, {lng})...")
            result = self.page.evaluate(js_code)

            logger.info(f"[WORKER:{self.worker_id}] page.evaluate SUCCESS for coords - pano: {result}")
            return result

        except Exception as e:
            logger.error(f"[WORKER:{self.worker_id}] !!! page.evaluate FAILED for coords ({lat}, {lng}) !!!")
            logger.error(f"[WORKER:{self.worker_id}] Error type: {type(e).__name__}")
            logger.error(f"[WORKER:{self.worker_id}] Error message: {e}")
            logger.error(f"[WORKER:{self.worker_id}] Traceback:\n{traceback.format_exc()}")
            return {'error': str(e)}

    def _cleanup(self):
        """Clean up browser resources."""
        if self.browser:
            try:
                self.browser.close()
            except:
                pass
        if self.playwright:
            try:
                self.playwright.stop()
            except:
                pass


class BrowserPool:
    """Manages a pool of browser workers."""

    def __init__(self, num_browsers: int = 5, api_key: str = None):
        logger.info(f"[POOL] __init__ called with num_browsers={num_browsers}")

        self.num_browsers = num_browsers
        raw_key = api_key or os.environ.get("GOOGLE_MAPS_API_KEY") or ""
        self.api_key = raw_key.strip()

        logger.info(f"[POOL] API key from param: {bool(api_key)}")
        logger.info(f"[POOL] API key from env GOOGLE_MAPS_API_KEY: {bool(os.environ.get('GOOGLE_MAPS_API_KEY'))}")
        logger.info(f"[POOL] Final API key present: {bool(self.api_key)}")
        logger.info(f"[POOL] Final API key length: {len(self.api_key)}")
        logger.info(f"[POOL] Final API key first 10 chars: {self.api_key[:10] if self.api_key else 'EMPTY'}")

        if not self.api_key:
            logger.error("[POOL] !!! WARNING: NO API KEY CONFIGURED !!!")

        self.workers = []
        self.request_counter = 0
        self.executor = ThreadPoolExecutor(max_workers=num_browsers)

    def start(self):
        """Start all browser workers."""
        logger.info(f"[POOL] start() called - starting {self.num_browsers} workers...")
        for i in range(self.num_browsers):
            logger.info(f"[POOL] Creating worker {i}...")
            worker = BrowserWorker(i, self.api_key)
            worker.start()
            self.workers.append(worker)
            logger.info(f"[POOL] Worker {i} started")
        logger.info(f"[POOL] All {self.num_browsers} workers started successfully")

    def stop(self):
        """Stop all browser workers."""
        for worker in self.workers:
            worker.stop()
        self.executor.shutdown(wait=True)
        logger.info("Browser pool stopped")

    def submit_request(self, request_type: str, **kwargs) -> str:
        """Submit a request to the least busy worker."""
        # Simple round-robin assignment (could be improved with load balancing)
        worker = self.workers[self.request_counter % self.num_browsers]
        self.request_counter += 1

        return worker.submit_request(request_type, **kwargs)

    def get_response(self, request_id: str, timeout: float = 30.0):
        """Get response for a request ID."""
        worker_id = int(request_id.split('_')[0])
        return self.workers[worker_id].get_response(request_id, timeout)


# Global browser pool
browser_pool = BrowserPool(num_browsers=5)  # Configurable number of browsers

@app.on_event("startup")
async def startup_event():
    """Start the browser pool when the server starts."""
    logger.info("[API] startup_event called")
    logger.info(f"[API] browser_pool.api_key present: {bool(getattr(browser_pool, 'api_key', ''))}")
    logger.info(f"[API] browser_pool.api_key length: {len(getattr(browser_pool, 'api_key', ''))}")

    if not getattr(browser_pool, "api_key", ""):
        logger.error("[API] !!! FATAL: MISSING GOOGLE_MAPS_API_KEY !!!")
        raise RuntimeError("Missing GOOGLE_MAPS_API_KEY in environment")

    logger.info("[API] Starting browser pool...")
    browser_pool.start()
    logger.info("[API] Browser pool started successfully")

@app.on_event("shutdown")
async def shutdown_event():
    """Stop the browser pool when the server shuts down."""
    logger.info("[API] shutdown_event called")
    browser_pool.stop()
    logger.info("[API] Browser pool stopped")

@app.post("/fetch_pano")
async def fetch_pano(request: PanoRequest):
    """Fetch panorama data by ID."""
    logger.info(f"[API] /fetch_pano called with pano_id={request.pano_id}")
    logger.debug(f"[API] api_key from request present: {bool(request.api_key)}")

    try:
        logger.debug(f"[API] Submitting request to browser pool...")
        request_id = browser_pool.submit_request('fetch_pano', pano_id=request.pano_id)
        logger.debug(f"[API] Request submitted, request_id={request_id}")

        logger.debug(f"[API] Waiting for response...")
        result = browser_pool.get_response(request_id)
        logger.info(f"[API] Got response for {request.pano_id}: {result.keys() if isinstance(result, dict) else type(result)}")

        if 'error' in result:
            logger.error(f"[API] Error in result: {result['error']}")
            raise HTTPException(status_code=400, detail=result['error'])

        logger.debug(f"[API] Returning result with {len(result.get('links', []))} links")
        return result

    except TimeoutError:
        logger.error(f"[API] Request timed out for pano_id={request.pano_id}")
        raise HTTPException(status_code=504, detail="Request timed out")
    except Exception as e:
        logger.error(f"[API] Exception in /fetch_pano: {type(e).__name__}: {e}")
        logger.error(f"[API] Traceback:\n{traceback.format_exc()}")
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/fetch_pano_coords")
async def fetch_pano_coords(request: CoordsRequest):
    """Fetch panorama ID from coordinates."""
    logger.info(f"[API] /fetch_pano_coords called with lat={request.lat}, lng={request.lng}, radius={request.radius}")

    try:
        logger.debug(f"[API] Submitting coords request to browser pool...")
        request_id = browser_pool.submit_request(
            'fetch_pano_coords',
            lat=request.lat,
            lng=request.lng,
            radius=request.radius,
            max_radius=request.max_radius
        )
        logger.debug(f"[API] Request submitted, request_id={request_id}")

        logger.debug(f"[API] Waiting for response...")
        result = browser_pool.get_response(request_id)
        logger.info(f"[API] Got response for coords ({request.lat}, {request.lng}): {result}")

        if 'error' in result:
            logger.error(f"[API] Error in result: {result['error']}")
            raise HTTPException(status_code=400, detail=result['error'])

        logger.debug(f"[API] Returning pano_id: {result}")
        return {"pano_id": result}

    except TimeoutError:
        logger.error(f"[API] Request timed out for coords ({request.lat}, {request.lng})")
        raise HTTPException(status_code=504, detail="Request timed out")
    except Exception as e:
        logger.error(f"[API] Exception in /fetch_pano_coords: {type(e).__name__}: {e}")
        logger.error(f"[API] Traceback:\n{traceback.format_exc()}")
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/health")
async def health_check():
    """Health check endpoint."""
    logger.debug("[API] /health called")
    return {"status": "healthy", "browsers": len(browser_pool.workers), "api_key_configured": bool(browser_pool.api_key)}

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=12345)
