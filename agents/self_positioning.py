"""
agents/self_positioning.py
Self-positioning agent that builds a panorama tile grid and asks the LLM
to estimate the current location based on visual cues.

Usage:
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
from agents.pano_grid import build_fixed_grid_urls
from core.utils import sign_streetview_url


class SelfPositioningAgent:
    """
    Simple, single-shot self-positioning: construct a trimmed, equal-area pano
    grid around the current panorama and ask the LLM for the exact location.
    """

    # Sensible defaults for geo-localisation (trim sky/road; equal-area rows)
    DEFAULT_X = 8
    DEFAULT_Y = 3
    DEFAULT_TUP = 20.0
    DEFAULT_TDOWN = 20.0
    DEFAULT_EQUAL_AREA = True
    DEFAULT_TILE_SIZE = (512, 512)
    DEFAULT_MAX_IMAGES = 240

    def __init__(
        self,
        api_key: str | None = None,              # Optional: if None, try to parse from input URLs
        model: str = "gemini/gemini-2.5-flash",
        *,
        logger_instance: logging.Logger | None = None,
        log_dir: str | None = None,              # Folder where logs will be written
        llm_params: Dict[str, Any] | None = None,
        grid_params: Dict[str, Any] | None = None,
        streetview_signing_secret: str | None = None,
        use_signed_streetview: bool = False,
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

    def __call__(
        self,
        image_urls: List[str],
        max_iterations: int = 1,   # ignored; kept for compatibility
        *,
        step_number: int | None = None,
    ) -> Tuple[str, str]:
        """Return (location_guess, evidence)."""
        print(f"[SELF-POS] Starting self-positioning with {len(image_urls)} input URLs")
        print(f"[SELF-POS] API key configured: {self.api_key[:10] if self.api_key else 'None'}...")
        print(f"[SELF-POS] Use signed Street View: {self.use_signed_streetview}")
        print(f"[SELF-POS] Signing secret configured: {bool(self.streetview_signing_secret)}")
        try:
            if not image_urls:
                return "unknown", "No images provided to self-positioning agent."

            # Parse pano id and API key from the first provided URL
            pano_id, api_key_in_url, size_wh = self._extract_from_url(image_urls[0])
            if not pano_id:
                return "unknown", "Could not extract pano id from provided image URLs."

            print(f"[SELF-POS] API key in input URL: {api_key_in_url[:10] if api_key_in_url else 'None'}...")
            api_key = self.api_key or api_key_in_url
            print(f"[SELF-POS] Using API key: {api_key[:10] if api_key else 'None'}...")
            if not api_key:
                return "unknown", "Missing Google Maps API key (not in URLs and not configured)."

            W, H = size_wh or self.tile_size

            # Base URL built from pano id. build_fixed_grid_urls will override heading/pitch/fov/size
            base = (
                "https://maps.googleapis.com/maps/api/streetview"
                f"?size={W}x{H}&pano={pano_id}&key={api_key}"
            )

            # Construct grid
            urls = build_fixed_grid_urls(
                base,
                self.X,
                self.Y,
                self.Tup,
                self.Tdown,
                size=(W, H),
                equal_area=self.equal_area,
                yaw0_deg=0.0,
            )

            # Sample to limit payload
            urls_for_llm = self._evenly_sample(urls, max(1, self.max_images))

            # Fetch images and convert to data URIs (no caching - complies with Google ToS)
            success_count = fail_count = 0
            data_uris: List[str] = []
            for u in urls_for_llm:
                data_uri, success = self._url_to_data_uri(u)
                if success:
                    success_count += 1
                else:
                    fail_count += 1
                data_uris.append(data_uri)
            print(f"[SELF-POS] grid prepared: success={success_count} failed={fail_count} total={len(data_uris)}")

            # Compose and call LLM using data-URIs
            messages = self._compose_messages(data_uris)

            # Build log object (same shape as original agent)
            start_ts = time.time()
            step_id = step_number or self._next_id()
            log = {
                "meta": {
                    "step_number": step_id,
                    "timestamp_utc": datetime.utcnow().isoformat(timespec="seconds") + "Z",
                    "pano_id": pano_id,
                },
                "request": {
                    "system_prompt": messages[0]["content"],
                    "image_urls": list(urls_for_llm),
                    "max_iterations": 1,
                    "model": self.model,
                    "llm_params": self.llm_params,
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
            self.logger.error(f"Self-positioning error: {e}", exc_info=True)
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
        system_prompt = (
            "You are a precise geo-localisation assistant. Examine all provided Street View crops "
            "and answer in one or two sentences: Where exactly is this place? Include the most "
            "specific landmark/address/neighbourhood visible, down to the last exact intersection if possible. If uncertain, say 'unknown' and explain."
            "We need this information since we are navigating a map and we need to know where we are precisely in order to decide where to go next."
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



    def _url_to_data_uri(self, url: str) -> Tuple[str, bool]:
        """Fetch image from URL and return as data URI.
        Returns (data_uri, success). On failure returns (original_url, False).
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
                    print(f"[SELF-POS][SIGNING] Successfully signed URL for pano {pano} h={heading} fov={fov} pitch={pitch}")
                except Exception as e:
                    print(f"[SELF-POS][SIGNING] FAILED to sign URL: {e}")
                    fetch_url = url

            print(f"[SELF-POS] Fetching {pano} h={heading} fov={fov} pitch={pitch} size={size}")
            resp = requests.get(fetch_url, timeout=25)
            if resp.status_code == 200 and resp.content:
                data = resp.content
                print(f"[SELF-POS] Successfully fetched {len(data)} bytes")
                b64 = base64.b64encode(data).decode("ascii")
                return f"data:image/jpeg;base64,{b64}", True
            else:
                print(f"[SELF-POS] FETCH FAILED: HTTP {resp.status_code}")
                if resp.content:
                    try:
                        error_text = resp.text[:500]
                        print(f"[SELF-POS] Error response: {error_text}")
                    except:
                        pass
                return url, False
        except Exception as e:
            print(f"[SELF-POS] fetch error: {e}")
            return url, False

