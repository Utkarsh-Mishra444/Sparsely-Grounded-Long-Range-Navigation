"""
agents/self_positioning.py
Self-positioning agent that builds a pano tile grid (using pano_grid)
and asks the LLM "where exactly is this place?".

Includes Markovian memory - tracks the last N position estimates
and includes them in the prompt to help prevent drastic location jumps.

Interface: compatible with strategy usage:
    loc_guess, loc_evidence = pos_agent(image_urls, step_number=...)
"""

import logging
import json
import os
import time
from datetime import datetime
from typing import List, Tuple, Dict, Any
import base64
import requests
from urllib.parse import urlparse, parse_qs

from infrastructure.llm_wrapper import llm_call
from agents.pano_grid import build_fixed_grid_urls, apply_grid_selection
from core.utils import sign_streetview_url


class SelfPositioningAgent:
    """
    Self-positioning agent with Markovian memory: tracks the last N position estimates
    and includes them in the prompt to help maintain spatial consistency.
    """

    # Sensible defaults for geo-localisation (trim sky/road; equal-area rows)
    DEFAULT_X = 4
    DEFAULT_Y = 3
    DEFAULT_TUP = 20.0
    DEFAULT_TDOWN = 20.0
    DEFAULT_EQUAL_AREA = True
    DEFAULT_TILE_SIZE = (512, 512)
    DEFAULT_MAX_IMAGES = 240

    # Memory settings
    DEFAULT_MEMORY_SIZE = 5  # Track last 5 position estimates

    def __init__(
        self,
        api_key: str | None = None,              # Optional: if None, try to parse from input URLs
        model: str = "gemini/gemini-2.5-flash",
        *,
        logger_instance: logging.Logger | None = None,
        log_dir: str | None = None,              # Folder where logs will be written
        llm_params: Dict[str, Any] | None = None,
        grid_params: Dict[str, Any] | None = None,
        grid_selection: str | List[int] | None = None,  # Selection pattern: "all", "middle_row", "first_row", "last_row", or list of indices
        streetview_signing_secret: str | None = None,
        use_signed_streetview: bool = False,
        memory_size: int | None = None,  # Number of past estimates to track
    ) -> None:
        self.logger = logger_instance or logging.getLogger(__name__)
        self.model = model
        self.llm_params: Dict[str, Any] = llm_params or {}
        self.api_key = api_key
        self.log_dir = log_dir or "."
        self._call_index = 0
        self.streetview_signing_secret = streetview_signing_secret
        self.use_signed_streetview = bool(streetview_signing_secret) and bool(use_signed_streetview)

        gp = grid_params or {}
        self.X = int(gp.get("X", self.DEFAULT_X))
        self.Y = int(gp.get("Y", self.DEFAULT_Y))
        self.Tup = float(gp.get("Tup", self.DEFAULT_TUP))
        self.Tdown = float(gp.get("Tdown", self.DEFAULT_TDOWN))
        self.equal_area = bool(gp.get("equal_area", self.DEFAULT_EQUAL_AREA))
        self.tile_size = tuple(gp.get("tile_size", self.DEFAULT_TILE_SIZE))  # (W, H)
        self.max_images = int(gp.get("max_images", self.DEFAULT_MAX_IMAGES))
        self.grid_selection = grid_selection if grid_selection is not None else "middle_row"

        # Memory settings
        self.memory_size = memory_size if memory_size is not None else self.DEFAULT_MEMORY_SIZE
        # List of past position estimates: [(step_number, location_guess, evidence), ...]
        self.position_history: List[Tuple[int, str, str]] = []

    def _get_position_history_prompt(self) -> str:
        """
        Generate a prompt section describing the last N position estimates.
        Returns empty string if no history is available.
        """
        if not self.position_history:
            return ""

        history_lines = []
        for step_num, loc_guess, evidence in self.position_history:
            # Truncate evidence if too long
            short_evidence = evidence[:150] + "..." if len(evidence) > 150 else evidence
            history_lines.append(f"  - Step {step_num}: {loc_guess} (reason: {short_evidence})")

        history_text = "\n".join(history_lines)

        prompt_section = (
            "\n\n--- IMPORTANT: Previous Position Estimates ---\n"
            "Your last few position estimates were:\n"
            f"{history_text}\n\n"
            "NOTES ON PREVIOUS ESTIMATES:\n"
            "- These estimates may NOT be 100% accurate - they are your previous guesses based on visual cues.\n"
            "- Use them as a reference to maintain spatial consistency - you should be moving along a path, "
            "not jumping between distant cities or neighborhoods.\n"
            "- If the current view looks dramatically different from the last estimate's area, "
            "consider whether you might be:\n"
            "  a) Still in the same general area but at a different spot (more likely)\n"
            "  b) Actually in a completely different location (less likely unless there's strong evidence)\n"
            "- Drastic location changes (e.g., different city, different country) should only be reported "
            "if there is overwhelming visual evidence (different language signs, distinct architecture, etc.)\n"
            "--- End of Previous Estimates ---\n\n"
        )

        return prompt_section

    def _add_to_history(self, step_number: int, location_guess: str, evidence: str) -> None:
        """
        Add a position estimate to the history, maintaining the maximum size.
        Only adds valid estimates (not 'unknown').
        """
        if location_guess.lower() == "unknown":
            return

        self.position_history.append((step_number, location_guess, evidence))

        # Keep only the last N entries
        if len(self.position_history) > self.memory_size:
            self.position_history = self.position_history[-self.memory_size:]

    def __call__(
        self,
        image_urls: List[str],
        max_iterations: int = 1,   # ignored; kept for compatibility
        *,
        step_number: int | None = None,
    ) -> Tuple[str, str]:
        """Return (location_guess, evidence)."""
        print(f"[SELF-POS-MEMORY] Starting self-positioning with {len(image_urls)} input URLs")
        print(f"[SELF-POS-MEMORY] Position history size: {len(self.position_history)}")
        print(f"[SELF-POS-MEMORY] API key configured: {self.api_key[:10] if self.api_key else 'None'}...")
        print(f"[SELF-POS-MEMORY] Use signed Street View: {self.use_signed_streetview}")
        print(f"[SELF-POS-MEMORY] Signing secret configured: {bool(self.streetview_signing_secret)}")
        try:
            if not image_urls:
                return "unknown", "No images provided to self-positioning agent."

            # Parse pano id and API key from the first provided URL
            pano_id, api_key_in_url, size_wh = self._extract_from_url(image_urls[0])
            if not pano_id:
                return "unknown", "Could not extract pano id from provided image URLs."

            print(f"[SELF-POS-MEMORY] API key in input URL: {api_key_in_url[:10] if api_key_in_url else 'None'}...")
            api_key = self.api_key or api_key_in_url
            print(f"[SELF-POS-MEMORY] Using API key: {api_key[:10] if api_key else 'None'}...")
            if not api_key:
                return "unknown", "Missing Google Maps API key (not in URLs and not configured)."

            # Size is extracted from input navigation URLs to ensure consistency
            # with navigation images. Falls back to DEFAULT_TILE_SIZE (512x512)
            # only if URL has no size param (shouldn't happen - API requires it).
            W, H = size_wh or self.tile_size

            # Base URL built from pano id. build_fixed_grid_urls will override heading/pitch/fov/size
            base = (
                "https://maps.googleapis.com/maps/api/streetview"
                f"?size={W}x{H}&pano={pano_id}&key={api_key}"
            )

            # Construct grid with optional selection
            urls = build_fixed_grid_urls(
                base,
                self.X,
                self.Y,
                self.Tup,
                self.Tdown,
                size=(W, H),
                equal_area=self.equal_area,
                yaw0_deg=0.0,
                grid_selection=self.grid_selection if self.grid_selection != "all" else None,
            )

            # Sample to limit payload (if max_images is set and selection didn't already limit it)
            urls_for_llm = self._evenly_sample(urls, max(1, self.max_images))

            # Convert URLs to data URIs by fetching directly (no caching)
            fetched = 0
            data_uris: List[str] = []
            for u in urls_for_llm:
                data_uri = self._url_to_data_uri(u)
                if data_uri:
                    fetched += 1
                    data_uris.append(data_uri)
                else:
                    # Fallback to original URL if fetch failed
                    data_uris.append(u)
            print(f"[SELF-POS-MEMORY] grid prepared: fetched={fetched} total={len(data_uris)}")

            # Compose and call LLM using data-URIs (or original URLs when fetch failed)
            messages = self._compose_messages(data_uris)

            # Build log object (same shape as original agent)
            start_ts = time.time()
            step_id = step_number or self._next_id()
            log = {
                "meta": {
                    "step_number": step_id,
                    "timestamp_utc": datetime.utcnow().isoformat(timespec="seconds") + "Z",
                    "pano_id": pano_id,
                    "position_history_size": len(self.position_history),
                },
                "request": {
                    "system_prompt": messages[0]["content"],
                    "image_urls": list(urls_for_llm),
                    "max_iterations": 1,
                    "model": self.model,
                    "llm_params": self.llm_params,
                    "position_history": [
                        {"step": s, "location": l, "evidence": e[:100]}
                        for s, l, e in self.position_history
                    ],
                },
                "steps": [],
            }

            raw = llm_call(
                model=self.model,
                messages=messages,
                max_tokens=8000,
                **(self.llm_params or {}),
            )

            # Extract textual answer
            content = self._extract_content(raw)
            guess, evidence = self._split_guess_and_evidence(content)

            # Add this estimate to history (only if valid)
            self._add_to_history(step_id, guess, evidence)
            print(f"[SELF-POS-MEMORY] Position estimate: {guess}")
            print(f"[SELF-POS-MEMORY] Updated history size: {len(self.position_history)}")

            # Token usage and latency
            token_prompt_sum = token_comp_sum = 0
            try:
                usage = raw.get("usage", {}) or {}
                if hasattr(usage, "model_dump"):
                    usage = usage.model_dump()
                elif hasattr(usage, "dict"):
                    usage = usage.dict()
                token_prompt_sum = usage.get("prompt_tokens") or 0
                token_comp_sum = usage.get("completion_tokens") or 0
            except Exception:
                pass

            log["meta"]["latency_ms"] = int((time.time() - start_ts) * 1000)
            log["meta"]["token_usage"] = {
                "prompt": token_prompt_sum,
                "completion": token_comp_sum,
                "total": token_prompt_sum + token_comp_sum,
            }
            # final section to mirror legacy file content
            log["final"] = {
                "location_guess": guess,
                "evidence": evidence,
                "raw_text": content,
            }
            self._write_log_json(log)
            return guess, evidence

        except Exception as e:
            # Log error if logger is available, otherwise just print
            if self.logger is not None:
                self.logger.error(f"Self-positioning error: {e}", exc_info=True)
            else:
                print(f"[SELF-POS-MEMORY] ERROR: Self-positioning error: {e}", flush=True)
            # Best-effort log on failure if we have minimal context
            try:
                step_id = step_number or self._next_id()
                fail_log = {
                    "meta": {
                        "step_number": step_id,
                        "timestamp_utc": datetime.utcnow().isoformat(timespec="seconds") + "Z",
                        "pano_id": pano_id if 'pano_id' in locals() else None,
                        "latency_ms": 0,
                        "token_usage": {"prompt": 0, "completion": 0, "total": 0},
                    },
                    "request": {
                        "system_prompt": messages[0]["content"] if 'messages' in locals() else "",
                        "image_urls": list(urls_for_llm) if 'urls_for_llm' in locals() else list(image_urls),
                        "max_iterations": 1,
                        "model": self.model,
                        "llm_params": self.llm_params,
                    },
                    "steps": [],
                    "final": {
                        "location_guess": "unknown",
                        "evidence": f"Self-positioning failed: {e}",
                        "raw_text": "",
                    },
                }
                self._write_log_json(fail_log)
            except Exception:
                pass
            return "unknown", f"Self-positioning failed: {e}"

    # ── Helpers ─────────────────────────────────────────────────────
    def _extract_from_url(self, url: str) -> Tuple[str | None, str | None, Tuple[int, int] | None]:
        """Return (pano_id, api_key, (W,H)) from a Street View URL."""
        try:
            pu = urlparse(url)
            q = {k: v[0] for k, v in parse_qs(pu.query).items()}
            pano = q.get("pano")
            key = q.get("key")
            size = q.get("size")
            if size and "x" in size:
                w, h = size.split("x", 1)
                W, H = int(w), int(h)
                return pano, key, (W, H)
            return pano, key, None
        except Exception:
            return None, None, None

    def _evenly_sample(self, items: List[str], max_count: int) -> List[str]:
        if max_count <= 0 or max_count >= len(items):
            return list(items)
        n = len(items)
        idxs = [round(i * (n - 1) / (max_count - 1)) for i in range(max_count)]
        seen = set(); out: List[str] = []
        for i in idxs:
            if i not in seen:
                out.append(items[i]); seen.add(i)
        return out

    def _compose_messages(self, image_urls: List[str]) -> list:
        # Get the position history context
        history_context = self._get_position_history_prompt()

        system_prompt = (
            "You are a precise geo-localisation assistant. Examine all provided Street View crops "
            "and answer in one or two sentences: Where exactly is this place? Include the most "
            "specific landmark/address/neighbourhood visible, down to the last exact intersection if possible. If uncertain, say 'unknown' and explain."
            "We need this information since we are navigating a map and we need to know where we are precisely in order to decide where to go next."
            f"{history_context}"
        )
        content = [{"type": "text", "text": f"Here are {len(image_urls)} camera views. Where exactly is this place?"}]
        for i, url in enumerate(image_urls):
            content.append({"type": "text", "text": f"Image {i+1}:"})
            content.append({"type": "image_url", "image_url": {"url": url}})
        return [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": content},
        ]

    def _extract_content(self, response: Any) -> str:
        try:
            return response["choices"][0]["message"].get("content") or ""
        except Exception:
            try:
                return getattr(response.choices[0].message, "content", "")
            except Exception:
                return ""

    def _split_guess_and_evidence(self, text: str) -> Tuple[str, str]:
        if not text:
            return "unknown", "Empty response"
        # Heuristic: first sentence as guess, rest as evidence
        parts = text.strip().split(". ", 1)
        if len(parts) == 1:
            return parts[0], ""
        return parts[0], parts[1]

    # ─── Logging helpers (same as legacy agent) ─────────────────────────
    def _next_id(self) -> int:
        self._call_index += 1
        return self._call_index

    def _write_log_json(self, data: Dict[str, Any]) -> None:
        base_dir = os.path.normpath(str(self.log_dir or ".").strip())
        if os.path.basename(base_dir) == "self_position_calls":
            folder = base_dir
        else:
            folder = os.path.join(base_dir, "self_position_calls")

        os.makedirs(folder, exist_ok=True)
        fname = f"self_position_{int(data['meta']['step_number']):04d}.json"
        tmp = os.path.join(folder, fname + ".tmp")

        def _default(o):
            if hasattr(o, "model_dump_json"):
                try:
                    return json.loads(o.model_dump_json())
                except Exception:
                    return o.model_dump()
            if hasattr(o, "model_dump"):
                try:
                    return o.model_dump()
                except Exception:
                    pass
            if isinstance(o, (list, tuple)):
                return [ _default(x) for x in o ]
            if isinstance(o, dict):
                return { k: _default(v) for k, v in o.items() }
            return str(o)

        with open(tmp, "w", encoding="utf-8") as f:
            json.dump(data, f, indent=2, default=_default)
        os.replace(tmp, os.path.join(folder, fname))

    # ── Image fetch helper (NO CACHING) ─────────────────────────────────────────────
    def _url_to_data_uri(self, url: str) -> str | None:
        """
        Fetch image from URL and return as data URI.
        Returns None if fetch fails.

        NOTE: This version does NOT use caching - images are fetched directly
        from the Google Street View API on each call.
        """
        try:
            pu = urlparse(url)
            q = {k: v[0] for k, v in parse_qs(pu.query).items()}
            pano = q.get("pano")
            heading = q.get("heading")
            fov = q.get("fov")
            pitch = q.get("pitch")
            size = q.get("size") or f"{self.tile_size[0]}x{self.tile_size[1]}"

            fetch_url = url
            if self.use_signed_streetview and self.streetview_signing_secret:
                try:
                    fetch_url = sign_streetview_url(url, self.streetview_signing_secret)
                    print(f"[SELF-POS-MEMORY][SIGNING] Successfully signed URL for pano {pano} h={heading} fov={fov} pitch={pitch}")
                except Exception as e:
                    print(f"[SELF-POS-MEMORY][SIGNING] FAILED to sign URL for pano {pano} h={heading} fov={fov} pitch={pitch}: {e}")
                    fetch_url = url

            print(f"[SELF-POS-MEMORY] fetching {pano} h={heading} fov={fov} pitch={pitch} size={size}")
            resp = requests.get(fetch_url, timeout=25)
            if resp.status_code == 200 and resp.content:
                data = resp.content
                print(f"[SELF-POS-MEMORY] Successfully fetched {len(data)} bytes")
                b64 = base64.b64encode(data).decode("ascii")
                return f"data:image/jpeg;base64,{b64}"
            else:
                print(f"[SELF-POS-MEMORY] FETCH FAILED: HTTP {resp.status_code} for pano {pano} h={heading} fov={fov} pitch={pitch}")
                return None
        except Exception as e:
            print(f"[SELF-POS-MEMORY] fetch error: {e}")
            return None
