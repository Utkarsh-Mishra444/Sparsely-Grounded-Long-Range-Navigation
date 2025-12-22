from strategies.base import Strategy
import requests
import base64
from infrastructure.llm_wrapper import llm_call
import json
from json_repair import repair_json
import random
import os
import logging
import hashlib
from typing import Dict, List, Optional, Any, Union, Tuple
from collections import defaultdict
import time
from datetime import datetime
import numpy as np
from PIL import Image
from tenacity import retry, stop_after_attempt, wait_exponential, retry_if_exception_type
from core.utils import sign_streetview_url
# from agents.memory_manager import MemoryManager  # Disabled per request; leaving code for later
import pdb

class AdvancedStreetViewStrategy(Strategy):
    """
    A strategy for navigating Street View environments using vision AI models
    to make intelligent path decisions at intersections, with memory to avoid
    repeating the same decisions at previously visited intersections.
    """    
    # Single JSON schema for LLM response (no heading intent/steps)
    RESPONSE_SCHEMA_NO_JUMP = {
        "type": "object",
        "required": ["analysis", "decision", "memory"],
        "properties": {
            "analysis": {"type": "string"},
            "decision": {"type": "integer", "minimum": 0},
            "memory": {"type": "string"},
        },
    }

    # Default (class-level) schema
    RESPONSE_SCHEMA = RESPONSE_SCHEMA_NO_JUMP
 
    # ── self-positioning frequency ────────────────────────────
    # Always perform self-positioning on the very first decision.
    # Thereafter, if this interval (N) is > 0, perform it every N decisions.
    # Set to ‑1 to disable self-positioning after the first decision.
    SELF_POSITION_INTERVAL = 3  # ← tweak as desired

    def __init__(self, 
                 maps_api_key: str, 
                 call_folder: str, 
                 navigation_prompt: str,
                 model_name: str,
                 log_level: int = logging.INFO,
                 use_memory_manager: bool = False,
                 use_self_positioning: bool = False,
                 llm_params: Optional[Dict[str, Any]] = None,
                 prompt_components: Optional[Dict[str, Any]] = None,
                 streetview_signing_secret: Optional[str] = None,
                 use_signed_streetview: bool = True,
                 ):

        # Store core fields
        self.maps_api_key = maps_api_key
        self.call_folder = call_folder
        self.navigation_prompt = navigation_prompt
        self.model_name = model_name
        # Optional provider-specific kwargs (e.g., reasoning_effort, top_p, etc.)
        self.llm_params: Dict[str, Any] = llm_params or {}
        # Optional per-component prompt toggles (default True if not provided)
        self.prompt_components: Dict[str, Any] = prompt_components or {}

        self.use_signed_streetview = use_signed_streetview and bool(streetview_signing_secret)
        self.streetview_signing_secret = streetview_signing_secret

        # (heading-lock removed)

        # ── new state for self-positioning cache ────────────────
        self.last_known_position: Optional[str] = None
        self.last_position_check_decision_counter: Optional[int] = None
        self.use_self_positioning = use_self_positioning

        # Fixed schema
        self.RESPONSE_SCHEMA = self.RESPONSE_SCHEMA_NO_JUMP
            
        # Setup logging
        self.logger = self._setup_logger(log_level)
        
        # Initialize Memory Manager if enabled
        self.use_memory_manager = use_memory_manager
        self.memory_manager = None
        if False and self.use_memory_manager:
            self.logger.info("MemoryManager is enabled. Initializing...")
            try:
                self.memory_manager = MemoryManager(
                    call_folder=self.call_folder,
                    model=self.model_name,
                    log_level=log_level
                )
                self.logger.info("MemoryManager initialized successfully.")
            except Exception as e:
                self.logger.error(f"Failed to initialize MemoryManager: {e}", exc_info=True)
                self.use_memory_manager = False # Disable if initialization fails

        self.logger.info(f"Strategy configured to use model: {self.model_name}")
        
        # Ensure call directory exists  
        os.makedirs(f"{self.call_folder}/gemini_calls", exist_ok=True)
            
        # Initialize intersection memory
        self.intersection_memory_file = f"{self.call_folder}/intersection_memory.json"
        self.intersection_memory = self._load_intersection_memory() 
    def _pc(self, name: str, default: bool = True) -> bool:
        """Return whether a prompt component is enabled; defaults to True if unspecified."""
        try:
            val = self.prompt_components.get(name)
            return default if val is None else bool(val)
        except Exception:
            return default
    def _setup_logger(self, log_level: int) -> logging.Logger:
        """Set up and configure logger to write only to a file in the call_folder."""
        logger = logging.getLogger("StreetViewStrategy")
        logger.setLevel(log_level)
        # Remove any existing handlers
        for h in logger.handlers[:]:
            logger.removeHandler(h)
        # Ensure call_folder exists
        os.makedirs(self.call_folder, exist_ok=True)
        # Add file handler
        file_log_path = os.path.join(self.call_folder, 'strategy.log')
        try:
            file_handler = logging.FileHandler(file_log_path, encoding='utf-8')
            formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
            file_handler.setFormatter(formatter)
            logger.addHandler(file_handler)

            # Add a dedicated handler that captures only ERROR (and above) messages
            error_log_path = os.path.join(self.call_folder, 'error.log')
            error_handler = logging.FileHandler(error_log_path, encoding='utf-8')
            error_handler.setLevel(logging.ERROR)
            error_handler.setFormatter(formatter)
            logger.addHandler(error_handler)
            logger.info(f"Logging initialized. General logs: {file_log_path}, error logs: {error_log_path}")
            logger.propagate = False
        except Exception as e:
            # If file handler setup fails, fallback to no handlers
            logger.error(f"Failed to set up file handler for strategy logger: {e}")
        return logger
    
    def _load_intersection_memory(self) -> Dict[str, Dict[int, int]]:
        """Load intersection memory from a JSON file."""
        if os.path.exists(self.intersection_memory_file):
            try:
                with open(self.intersection_memory_file, 'r') as f:
                    memory = json.load(f)
                self.logger.info(f"Loaded intersection memory with {len(memory)} entries")
                return memory
            except Exception as e:
                self.logger.error(f"Error loading intersection memory: {e}", exc_info=True)
                return {}
        else:
            self.logger.info("No existing intersection memory found, starting fresh")
            return {}
    
    def _save_intersection_memory(self) -> None:
        """Save the intersection memory to a JSON file."""
        try:
            with open(self.intersection_memory_file, 'w') as f:
                json.dump(self.intersection_memory, f, indent=4)
            self.logger.info(f"Saved intersection memory with {len(self.intersection_memory)} entries")
        except Exception as e:
            self.logger.error(f"Error saving intersection memory: {e}", exc_info=True)
    
    def _hash_image_set(self, images: List[Dict]) -> str:
        """
        Create a unique hash for a set of images.
        
        Args:
            images: List of image data dictionaries
            
        Returns:
            Hash string uniquely identifying this set of images
        """
        # Concatenate all image data
        combined_data = "".join([img['data'][:1000] for img in images])  # Use first 1000 chars for efficiency
        
        # Create hash
        hash_obj = hashlib.sha256(combined_data.encode())
        return hash_obj.hexdigest()
    
    def _record_decision(self, intersection_hash: str, num_options: int, chosen_index: int, analysis: Optional[str] = None) -> None:
        """
        Record a decision made at an intersection.

        Args:
            intersection_hash: Hash identifying the intersection
            num_options: Number of available options/directions
            chosen_index: Index of the chosen direction
            analysis: LLM analysis explaining why this decision was made
        """
        # Initialize intersection entry if not exists
        if intersection_hash not in self.intersection_memory:
            # Initialise with nested structure ready for per-step counts
            self.intersection_memory[intersection_hash] = {
                "visits": 1,
                "decisions": {
                    str(i): {"total": 0, "steps": {}} for i in range(num_options)
                },
                "dead_ends": [],
                "intersection_memory_summary": []
            }
        else:
            self.intersection_memory[intersection_hash]["visits"] += 1
            # Ensure containers exist for older memories
            if "intersection_memory_summary" not in self.intersection_memory[intersection_hash]:
                self.intersection_memory[intersection_hash]["intersection_memory_summary"] = []

            # Upgrade legacy flat counts to nested structure if encountered
            for key, value in list(self.intersection_memory[intersection_hash]["decisions"].items()):
                if isinstance(value, int):
                    self.intersection_memory[intersection_hash]["decisions"][key] = {
                        "total": value,
                        "steps": {}
                    }

        # ── Update counts for this choice (drop per-step tracking) ─────────
        decision_entry = self.intersection_memory[intersection_hash]["decisions"].get(str(chosen_index))
        if isinstance(decision_entry, dict):
            decision_entry["total"] = decision_entry.get("total", 0) + 1
        else:
            prev = int(decision_entry) if isinstance(decision_entry, int) else 0
            self.intersection_memory[intersection_hash]["decisions"][str(chosen_index)] = prev + 1

        # ── NEW: append human-readable summary ───────────────────────────
        summary_entry = {
            "chosen_index": chosen_index,
            "analysis": analysis or "",
            "timestamp": datetime.utcnow().isoformat() + "Z"
        }
        self.intersection_memory[intersection_hash]["intersection_memory_summary"].append(summary_entry)

        # Save after each update
        self._save_intersection_memory()


        # Store last intersection and choice to detect dead ends
        self.last_intersection_hash = intersection_hash
        self.last_chosen_index = chosen_index
        # Snapshot for env dead-end registration: parent pano and real links
        try:
            if hasattr(self, 'agent_snapshot') and isinstance(self.agent_snapshot, dict):
                pass
        except Exception:
            pass

        self.logger.info(
            f"Recorded decision {chosen_index} for intersection {intersection_hash[:8]}...")

    def _summarize_intersection_history(self, history_list: List[Dict]) -> str:
        """
        Summarize the intersection memory summary list into a concise text via LLM.
        
        Args:
            history_list: List of decision entries from intersection_memory_summary
            
        Returns:
            Concise summary string, or empty if summarization fails
        """
        if not history_list:
            return ""
        
        # Convert list to JSON string for LLM input
        history_json = json.dumps(history_list, indent=2)
        
        # Simple prompt for summarization
        summary_prompt = (
            "Summarize the following list of past decision analyses at this intersection "
            "into a single concise paragraph (max 100 words). Focus on key patterns, "
            "reasons for choices, outcomes, and any recurring themes or warnings:\n\n"
            f"{history_json}\n\n"
            "Output only the summary text, no JSON or extra content."
        )
        
        try:
            # Use the new llm_call wrapper
            raw = llm_call(
                model=self.model_name,
                messages=[{"role": "user", "content": summary_prompt}],
                max_tokens=2500,
                **(self.llm_params or {})
            )
            summary_text = raw["choices"][0]["message"]["content"].strip()
            self.logger.info(f"Generated history summary: {summary_text[:50]}...")
            return summary_text
        except Exception as e:
            self.logger.error(f"Failed to summarize history: {e}")
            return ""

    def _check_for_dead_end(self, intersection_hash: str, num_options: int) -> None:
        """
        Check if we've immediately returned to the same intersection, indicating a dead end.
        
        Args:
            intersection_hash: Hash identifying the current intersection
            num_options: Number of available options/directions
        """
        # Check if we have a record of the last intersection
        if hasattr(self, 'last_intersection_hash') and hasattr(self, 'last_chosen_index'):
            # If we're back at the same intersection immediately
            if self.last_intersection_hash == intersection_hash:
                # Mark the last chosen direction as a dead end
                self.logger.info(f"Dead end detected! Direction {self.last_chosen_index} at intersection {intersection_hash[:8]}...")
                
                # Add to dead_ends if not already there
                if str(self.last_chosen_index) not in self.intersection_memory[intersection_hash].get("dead_ends", []):
                    if "dead_ends" not in self.intersection_memory[intersection_hash]:
                        self.intersection_memory[intersection_hash]["dead_ends"] = []
                    
                    self.intersection_memory[intersection_hash]["dead_ends"].append(str(self.last_chosen_index))
                    self._save_intersection_memory()

                # --- Emit env dead-end event with pano IDs via agent queue ---
                try:
                    agent = getattr(self, '_last_agent_ref', None)
                    if agent and hasattr(agent, 'observation'):
                        parent_pano_id = agent.observation.get('pano_id')
                        # Use the last observation snapshot saved at decision time
                        last_obs = getattr(agent, 'last_decision_observation', {}) or {}
                        links_real = last_obs.get('links_real', [])
                        child_pano_id = None
                        if isinstance(links_real, list) and 0 <= self.last_chosen_index < len(links_real):
                            child_pano_id = links_real[self.last_chosen_index].get('pano')
                        if parent_pano_id and child_pano_id:
                            if not hasattr(agent, 'env_events'):
                                setattr(agent, 'env_events', [])
                            agent.env_events.append({
                                'type': 'dead_end',
                                'parent_pano_id': str(parent_pano_id),
                                'child_pano_id': str(child_pano_id)
                            })
                            self.logger.info(f"Emitted dead_end event {parent_pano_id}->{child_pano_id}")
                except Exception as e:
                    self.logger.error(f"Failed to emit dead_end event: {e}")

    # Heading-lock helper removed
    def _apply_heading_lock(self, links: list[dict]) -> None:
        return None

    def select_action(self, agent: Any, step_counter: Optional[int] = None) -> Optional[str]:
        """
        Select an action based on the current observation.
        
        Args:
            agent: The agent with observation data
            step_counter: Optional step counter for logging
            
        Returns:
            The chosen alias for navigation or None if no action is possible
        """
        # Extract observation data
        observation = agent.observation
        if not observation or 'links' not in observation:
            self.logger.warning("No valid observation or links found.")
            return None
        # # ── 0.  Try short‑circuit first ─────────────────────
        # if self._lock_left > 0:
        #     print(f"Heading lock attempt: left={self._lock_left}, bearing={self._lock_bearing}, drift={self._lock_drift}")
        #     alias = self._apply_heading_lock(observation['links'])
        #     if alias:
        #         print(f"Heading lock chose alias: {alias}")
        #         return alias
        
        links = observation['links']
        links_real = observation.get('links_real', [])
        num_links = len(links)
        previous_pano_id = observation.get('previous_pano_id')

        # Handle different number of available links
        if num_links == 0:
            self.logger.warning("No links available.")
            return None
        elif num_links == 1:
            # Only one path available
            chosen_alias = links[0]['alias']
            self.logger.info(f"Only one link available. Selected alias: {chosen_alias}")
            return chosen_alias
        elif num_links == 2:
            # Two paths: try to avoid going back
            return self._handle_two_links(observation, links_real, previous_pano_id)
        else:
            # More than two links, use vision API to make a true decision
            # Only increment decision counter for actual decisions
            agent.decision_counter += 1
            return self.decide_with_vision_api(agent, links, log=(num_links >= 3))

    def _handle_two_links(self, observation: Dict, links_real: List[Dict], 
                          previous_pano_id: Optional[str]) -> str:
        """
        Handle the case where there are two links (typically forward/backward).
        Try to select the link that doesn't go back to the previous panorama.
        
        Args:
            observation: The current observation
            links_real: The real links data
            previous_pano_id: The previous panorama ID
            
        Returns:
            The chosen alias
        """
        # If we have a previous pano ID, try to avoid going back
        if previous_pano_id:
            for link in links_real:
                link_pano = link.get('pano')
                if link_pano and link_pano != previous_pano_id:
                    self.logger.debug(f"Current link pano: {link_pano}")
                    self.logger.debug(f"Previous pano: {previous_pano_id}")
                    chosen_alias = observation['reverse_alias_map'].get(link_pano)
                    if chosen_alias:
                        self.logger.info(f"Selected link that does not return to previous pano.")
                        return chosen_alias
        
        # Fallback to random choice if we can't determine which link to take
        link_aliases = [link['alias'] for link in observation['links']]
        chosen_alias = random.choice(link_aliases)
        self.logger.info(f"Randomly selected link: {chosen_alias}")
        return chosen_alias

    def decide_with_vision_api(self, agent: Any, links: List[Dict], log: bool = True) -> str:
        """
        Use vision API to analyze images and decide at an intersection.

        Args:
            agent: The agent with observation data
            links: The available links
            log: Whether to log the decision

        Returns:
            The chosen alias
        """
        pano_id = agent.observation['pano_id']
        step_counter = agent.step_counter # Get step counter

        # Fetch images for each possible direction
        self.logger.info(f"Fetching images for {len(links)} directions")
        images, image_urls = self.fetch_images(pano_id, links)

        if not images:
            self.logger.warning("No images fetched, falling back to random choice.")
            chosen_alias = random.choice([link['alias'] for link in links])
            if log:
                self._log_decision(agent, chosen_alias,
                                "No images available, chose randomly.",
                                "Ensure image fetching works for next decision.",
                                image_urls)
            return chosen_alias

        # Generate a hash for this set of images (intersection)
        intersection_hash = self._hash_image_set(images)
        self.logger.info(f"Intersection hash: {intersection_hash[:8]}...")

        # Check if we've returned to the same intersection (dead end detection)
        self._check_for_dead_end(intersection_hash, len(links))

        # Check if we've seen this intersection before
        previous_visits = None
        if intersection_hash in self.intersection_memory:
            previous_visits = self.intersection_memory[intersection_hash]
            self.logger.info(f"Found previous visits: {previous_visits['visits']} times")

        # Extract headings and convert to cardinal directions
        headings = [link['heading'] for link in links]
        cardinal_directions = [self.heading_to_cardinal(h) for h in headings]

        # Save minimal snapshot for env dead-end emission
        try:
            setattr(self, '_last_agent_ref', agent)
            setattr(agent, 'last_decision_observation', dict(agent.observation))
        except Exception:
            pass
        # Generate unique IDs and mapping
        unique_option_ids = [f"step{step_counter}_option{i}" for i in range(len(links))]
        id_to_alias_map = {unique_option_ids[i]: link['alias'] for i, link in enumerate(links)}
        alias_to_index_map = {link['alias']: i for i, link in enumerate(links)} # Map alias back to index
        # ── expose mapping for outer code (no leakage to the LLM) ───────────
        agent.last_decision_info = {
            "step": step_counter,
            "id_to_alias_map": id_to_alias_map,
            "alias_to_id_map": {v: k for k, v in id_to_alias_map.items()}
        }
        # Prepare prompt with unique IDs and cardinal directions
        prompt = self.prepare_prompt(
            agent, len(images), cardinal_directions, headings, previous_visits, unique_option_ids, image_urls
        )

        decision_data = self.call_llm_api(agent, prompt, images, image_urls)

        # Process API response
        try:
            # Expecting a string unique ID now
            chosen_unique_id = decision_data.get('decision')
            analysis = decision_data.get('analysis', "No analysis provided")
            proposed_memory = decision_data.get('memory', "")  # Get the proposed memory

            # heading-intent/steps no longer supported
            h_int  = None
            n_step = None

            # ── NEW: Refine memory using MemoryManager if enabled ───────────
            if False and self.use_memory_manager and self.memory_manager:
                self.logger.info("Refining memory with MemoryManager...")
                # Find chosen direction info for context
                chosen_alias = id_to_alias_map.get(str(chosen_unique_id))
                chosen_index = alias_to_index_map.get(chosen_alias, -1)
                chosen_direction_info = "Unknown"
                if chosen_index != -1 and chosen_index < len(cardinal_directions):
                    chosen_direction_info = (
                        f"Option {chosen_unique_id} "
                        f"(facing {cardinal_directions[chosen_index]} at {headings[chosen_index]:.0f}°)"
                    )
                
                # Prepare full context for the memory manager
                memory_context = {
                    "step_counter": agent.step_counter,
                    "destination": getattr(agent, 'destination', 'the destination'),
                    "previous_memory": getattr(agent, 'memory', ''),
                    "proposed_memory": proposed_memory,
                    "analysis": analysis,
                    "estimated_position": self.last_known_position,
                    "intersection_info": previous_visits,
                    "chosen_direction_info": chosen_direction_info,
                }
                
                # Call the memory manager to get the refined memory
                agent.memory = self.memory_manager.refine_memory(memory_context)
            else:
                # Original behavior: directly use the proposed memory
                agent.memory = proposed_memory


            if chosen_unique_id is None or not isinstance(chosen_unique_id, str):
                raise ValueError(f"Invalid decision ID type: {type(chosen_unique_id)}, expected string.")

            if chosen_unique_id not in id_to_alias_map:
                 raise ValueError(f"Invalid decision ID received: {chosen_unique_id}. Valid IDs: {list(id_to_alias_map.keys())}")

            # Map unique ID back to environment alias
            chosen_alias = id_to_alias_map[chosen_unique_id]

            # Find the original index for memory recording
            if chosen_alias not in alias_to_index_map:
                 # This should not happen if maps are consistent
                 raise ValueError(f"Chosen alias {chosen_alias} not found in original links.")
            chosen_index = alias_to_index_map[chosen_alias]

            # Record this decision using the original index
            self._record_decision(
                intersection_hash,
                len(links),
                chosen_index,
                analysis=analysis,
            )

            # Add chosen image URL to decision data if not already present
            if 'chosen_image_url' not in decision_data and chosen_index < len(image_urls):
                decision_data['chosen_image_url'] = image_urls[chosen_index]

            if log:
                decision_log = {
                    'step': agent.step_counter,
                    'action': chosen_alias,  # env alias
                    'cardinal_direction': cardinal_directions[chosen_index] if chosen_index < len(cardinal_directions) else 'Unknown',
                    'observation': agent.observation,
                    'analysis': analysis,
                    'memory': agent.memory,  # final memory (maybe refined)
                }

                # Add image URLs if available
                if image_urls:
                    decision_log['image_urls'] = image_urls
                    if chosen_index < len(image_urls):
                        decision_log['chosen_image_url'] = image_urls[chosen_index]

                agent.decision_history.append(decision_log)

            self.logger.info(f"Selected unique ID {chosen_unique_id}, mapped to alias: {chosen_alias} (index {chosen_index})")
            time.sleep(10)
            return chosen_alias # Return the alias the environment understands

        except Exception as e:
            self.logger.error(f"Decision parsing error: {e}", exc_info=True)
            # Fallback to random choice using original alias list
            chosen_alias = random.choice([link['alias'] for link in links])
            chosen_index = alias_to_index_map.get(chosen_alias, 0) # Find index for logging if possible

            if log:
                error_note = f"Error in decision process: {str(e)}, chose randomly."
                self._log_decision(agent, chosen_alias,
                                error_note,
                                "Verify API response format (should return string ID) for next decision.",
                                image_urls, chosen_index) # Log index if found

            return chosen_alias

    def _log_decision(self, agent: Any, chosen_alias: str, 
                     analysis: str, memory: str, 
                     image_urls: Optional[List[str]] = None,
                     chosen_index: Optional[int] = None) -> None:
        """Helper method to log decision to agent history.
        
        Args:
            agent: The agent with decision history
            chosen_alias: The chosen alias for navigation
            analysis: Analysis from the decision
            memory: Memory information to remember
            image_urls: Optional list of image URLs used for the decision
            chosen_index: Optional index of the chosen direction
        """
        decision_log = {
            'step': agent.step_counter,
            'action': chosen_alias,
            'observation': agent.observation,
            'analysis': analysis,
            'memory': memory
        }
        
        # Add image URLs if available
        if image_urls:
            decision_log['image_urls'] = image_urls
            # If we have both image_urls and chosen_index, we can log the specific URL used
            if chosen_index is not None and 0 <= chosen_index < len(image_urls):
                decision_log['chosen_image_url'] = image_urls[chosen_index]
        
        agent.decision_history.append(decision_log)

    def fetch_images(self, pano_id: str, links: List[Dict]) -> Tuple[List[Dict], List[str]]:
        """
        Fetch Street View images for each link heading.
        
        Args:
            pano_id: The panorama ID
            links: The available links
            
        Returns:
            Tuple containing:
            - List of image data dictionaries
            - List of image URLs used for fetching
        """
        images: List[Dict] = []
        image_urls: List[str] = []

        for link in links:
            heading = link['heading']

            base_url = (
                "https://maps.googleapis.com/maps/api/streetview"
                f"?size=512x512&pano={pano_id}&heading={heading}"
                f"&fov=90&pitch=30&key={self.maps_api_key}"
            )

            if self.use_signed_streetview and self.streetview_signing_secret:
                try:
                    signed_url = sign_streetview_url(base_url, self.streetview_signing_secret)
                except Exception as e:
                    self.logger.error(f"Street View signing failed for heading {heading}: {e}")
                    return [], []
                fetch_url = signed_url
                log_url = base_url
            else:
                fetch_url = base_url
                log_url = base_url

            image_urls.append(log_url)
            try:
                self.logger.debug(f"Fetching image for heading {heading}")
                resp = requests.get(fetch_url, timeout=10)
                resp.raise_for_status()
                jpeg_bytes = resp.content
            except requests.RequestException as e:
                self.logger.error(f"Failed to fetch image for heading {heading}: {e}")
                continue

            # Add base64‐encoded payload
            if jpeg_bytes:
                images.append({
                    'mime_type': 'image/jpeg',
                    'data': base64.b64encode(jpeg_bytes).decode('utf-8')
                })

        return images, image_urls

    def prepare_prompt(self, agent: Any, num_images: int, cardinal_directions: List[str], headings: List[float] | None = None,
                        previous_visits: Optional[Dict] = None, unique_option_ids: List[str] = None,
                        image_urls: List[str] | None = None
                        ) -> str:
        """Prepare the detailed prompt for the vision API."""

        # ── COMPONENT 1: Self-positioning enrichment ───────────────────
        # This component provides the agent with its estimated current location based on visual analysis
        # It either runs fresh self-positioning (at decision 1 or at regular intervals) or uses cached position
        # ABLATION: Remove this entire block to test without self-positioning information
        position_header = ""  # Only set when self-positioning provides information
        pos_agent = getattr(agent, "pos_agent", None)

        # Perform self-positioning only if the feature is enabled
        if self._pc('include_self_positioning') and self.use_self_positioning and image_urls and pos_agent:
            # Decision number (1-based) from the agent; fallback to step counter.
            decision_num = getattr(agent, "decision_counter", getattr(agent, "step_counter", 1))

            use_self_pos = (
                decision_num == 1 or
                (
                    self.SELF_POSITION_INTERVAL > 0 and
                    (decision_num - 1) % self.SELF_POSITION_INTERVAL == 0
                )
            )

            if use_self_pos:
                self.logger.info(
                    f"Using self-positioning (decision {decision_num}; interval {self.SELF_POSITION_INTERVAL})"
                )
                loc_guess, loc_evidence = pos_agent(image_urls, step_number=agent.step_counter)

                # Only update cache if we got a valid position (not "unknown")
                if loc_guess != "unknown":
                    self.last_known_position = loc_guess
                    self.last_position_check_decision_counter = decision_num
                
                position_header = (
                    f"Estimated position: {loc_guess} "
                    f"(evidence: {loc_evidence})\n\n"
                )
            elif self.last_known_position is not None and self.last_position_check_decision_counter is not None:
                # Use cached position
                age = decision_num - self.last_position_check_decision_counter
                position_header = (
                    f"Estimated position (last checked {age} movement decision{'s' if age != 1 else ''} ago. This is where you were estimated to be {age} decisions ago. You have made {age} decisions since then. Hence don't assume you are here. Use this info combined with everything else to calculate where you are): {self.last_known_position}\n\n"
                )

        # ── COMPONENT 2: Basic task context ───────────────────
        # This component provides the fundamental task description and destination goal
        # ABLATION: This is the minimal prompt - keep this for all ablation studies
        destination = getattr(agent, 'destination', "the destination")

        # ── COMPONENT 3: Movement history (currently disabled) ───────────────────
        # This component would provide statistics about recent panorama movements
        # ABLATION: Currently commented out - uncomment to test movement history impact
        # movement_history = getattr(agent, 'movement_history', [])
        # movement_history_text = ""
        # if movement_history:
        #     last_n_moves = movement_history[-50:]  # Get last 50 moves
        #     move_counts = Counter(last_n_moves)
        #     counts_str = ', '.join(f"{count}× {move}" for move, count in move_counts.items())
        #     movement_history_text = (
        #         f"\nRecent Panorama Movement History (last {len(last_n_moves)} movements):\n"
        #         f"Note: these represent panorama jumps (movements), not decisions at intersections.\n"
        #         f"{counts_str}\n"
        #     )

        # ── COMPONENT 4: Decision history ───────────────────
        # This component provides a sequence of the last 8 navigation decisions made by the agent
        # Shows the path taken through recent intersections to help with backtracking and loop detection
        # ABLATION: Remove this block to test without decision history context
        decision_history = getattr(agent, 'decision_history', [])
        decision_history_text = ""
        if self._pc('include_decision_history') and decision_history:
            last_n_decisions = decision_history[-8:]
            decision_actions = [
                str(
                    d.get('cardinal_direction')
                    or d.get('action')
                    or 'Unknown'
                ) for d in last_n_decisions
            ]
            decision_sequence = " -> ".join(decision_actions)
            decision_history_text = (
                f"\nRecent Decision History (last {len(last_n_decisions)} decisions):\n"
                f"{decision_sequence}\n"
            )

        # ── COMPONENT 5: Agent memory state ───────────────────
        # This component provides the agent's accumulated memory from previous decisions
        # Contains insights, observations, and strategic information gathered during navigation
        # ABLATION: Remove this block to test without memory retention
        memory_text = ""
        if self._pc('include_memory') and hasattr(agent, 'memory') and agent.memory:
            memory_text = (
                f"\nMemory so far:\n"
                f"{'-' * 40}\n"
                f"{agent.memory}\n"
                f"{'-' * 40}\n"
            )

        # ── COMPONENT 6: Intersection history summary ───────────────────
        # This component provides a summary of past analyses performed at this specific intersection
        # Helps the agent understand what it has previously considered at this location
        # ABLATION: Remove this block to test without intersection-specific history
        history_summary = ""
        # Safeguard: previous_visits may be None on first visit, so check it before accessing keys
        if self._pc('include_intersection_summary') and previous_visits and "intersection_memory_summary" in previous_visits and previous_visits["intersection_memory_summary"]:
            history_summary = self._summarize_intersection_history(previous_visits["intersection_memory_summary"])
            if history_summary:
                history_summary = f"\nSummary of past analyses at this intersection: {history_summary}"

        # ── COMPONENT 7: Previous visits information ───────────────────
        # This component provides detailed information about previous visits to this exact intersection
        # Includes visit count, previous direction choices, dead end detection, and exploration guidance
        # ABLATION: Remove this entire block to test without visit tracking and dead end detection
        previous_visits_text = ""
        if self._pc('include_previous_visits') and previous_visits:
            visit_count = previous_visits["visits"]
            decisions = previous_visits["decisions"]
            dead_ends = previous_visits.get("dead_ends", [])

            # Map previous decisions to cardinal directions and unique IDs
            decision_counts: list[str] = []
            for idx_str, data in decisions.items():
                # Support both legacy int and new nested structure
                if isinstance(data, dict):
                    total = data.get("total", 0)
                    if total == 0:
                        continue
                    steps_dict = data.get("steps", {})
                    step_parts = [f"{cnt}\u00D7 step{step}" for step, cnt in steps_dict.items() if cnt > 0]
                    step_summary = f" (by steps: {', '.join(step_parts)})" if step_parts else ""
                    idx = int(idx_str)
                    if idx < len(unique_option_ids):
                        direction = cardinal_directions[idx]
                        option_id = unique_option_ids[idx]
                        decision_counts.append(
                            f"{direction} (Option {option_id}): chosen {total} times{step_summary}")
                else:  # legacy int count
                    count_int = int(data)
                    if count_int == 0:
                        continue
                    idx = int(idx_str)
                    if idx < len(unique_option_ids):
                        direction = cardinal_directions[idx]
                        option_id = unique_option_ids[idx]
                        decision_counts.append(
                            f"{direction} (Option {option_id}): chosen {count_int} times")

            # Format dead ends information using unique IDs
            dead_end_info = ""
            if dead_ends:
                dead_end_directions = []
                for idx_str in dead_ends:
                    idx = int(idx_str)
                    if idx < len(unique_option_ids): # Check index validity
                        direction = cardinal_directions[idx]
                        option_id = unique_option_ids[idx]
                        dead_end_directions.append(f"{direction} (Option {option_id})")

                if dead_end_directions:
                    dead_end_info = f"\nDEAD ENDS DETECTED: {', '.join(dead_end_directions)}. These directions immediately lead back to this same intersection."

            previous_visits_text = (
                f"IMPORTANT: You have visited this exact intersection {visit_count} times before.\n"
                f"Previous direction choices: {', '.join(decision_counts) if decision_counts else 'None'}"
                f"{dead_end_info}\n"
                f"You should try to explore new directions that haven't been taken before or that aren't dead ends, "
                f"unless there's a compelling reason to revisit a previous path.\n"
            )

        # ── COMPONENT 8: Core task prompt ───────────────────
        # This component provides the basic instruction for the navigation task
        # ABLATION: This is the minimal prompt - keep this for all ablation studies
        prompt = (
            f"You are at an intersection with {num_images} possible directions (options). Below are images for each option. "
            f"Analyze the images and think step by step to determine the best direction towards {destination}. "
            f"Use the following information to guide your decision:\n\n"
        )

        # ── COMPONENT 9: Arrival heading information ───────────────────
        # This component identifies which direction the agent came from when arriving at this intersection
        # Helps prevent backtracking and provides spatial context for decision making
        # ABLATION: Remove this block to test without arrival direction awareness
        arrival_heading_info = ""
        if self._pc('include_arrival_heading') and hasattr(agent, 'observation') and 'arrival_heading' in agent.observation:
            arrival_heading = agent.observation['arrival_heading']
            # Make sure arrival_heading is not None and is a valid number before doing calculations
            if arrival_heading is not None and isinstance(arrival_heading, (int, float)):
                # Calculate the opposite heading (opposite of where the agent came from)
                # This accounts for arrival_heading pointing INTO the intersection
                # while link headings point OUT from the intersection
                opposite_heading = (arrival_heading + 180) % 360

                # Find which option corresponds most closely to the opposite of arrival direction
                arrival_option_idx = None
                smallest_angle_diff = 180  # Initialize with maximum possible difference

                # Extract headings from the links in observation
                if 'links' in agent.observation:
                    links = agent.observation['links']
                    for i, link in enumerate(links):
                        heading = link['heading']
                        # Calculate angle difference, considering 360-degree wrap
                        angle_diff = min(abs(heading - opposite_heading), 360 - abs(heading - opposite_heading))
                        if angle_diff < smallest_angle_diff:
                            smallest_angle_diff = angle_diff
                            arrival_option_idx = i

                if arrival_option_idx is not None and arrival_option_idx < len(unique_option_ids):
                    option_id = unique_option_ids[arrival_option_idx]
                    direction = cardinal_directions[arrival_option_idx]
                    arrival_heading_info = f"IMPORTANT: You arrived at this intersection from the direction of Option {option_id} ({direction}). This is the direction you came FROM.\n\n"

        # ── COMPONENT 10: Option/direction mapping ───────────────────
        # This component maps each image to its corresponding cardinal direction and unique option ID
        # Provides the agent with clear labels for each available choice
        # ABLATION: This is essential for basic functionality - keep for all studies
        prompt += "The images correspond to the following options/directions:\n"
        for i, direction in enumerate(cardinal_directions):
            if i < len(unique_option_ids):  # Check index validity
                option_id = unique_option_ids[i]
                # Append numeric heading if available
                heading_text = f" ({headings[i]:.0f}°)" if headings and i < len(headings) else ""
                prompt += f"Option {option_id}: facing {direction}{heading_text}\n"
        prompt += "\n"

        # ── PROMPT ASSEMBLY: Adding components to the base prompt ───────────────────
        # The following section assembles all the components into the final prompt
        # Each component is conditionally added based on availability and configuration
        
        # Add previous visits information (Component 7)
        if previous_visits_text and self._pc('include_previous_visits'):
            prompt += f"{previous_visits_text}\n"
        
        # Add intersection history summary (Component 6)
        if history_summary and self._pc('include_intersection_summary'):
            prompt += f"{history_summary}\n"
        
        # Add arrival heading information (Component 9)
        if arrival_heading_info and self._pc('include_arrival_heading'):
            prompt += arrival_heading_info

        # Add decision history (Component 4)
        if decision_history_text and self._pc('include_decision_history'):
            prompt += decision_history_text

        # Add self-positioning information (Component 1)
        if position_header and self._pc('include_self_positioning'):
            prompt += position_header
        
        # Add agent memory (Component 5)
        if memory_text and self._pc('include_memory'):
            prompt += memory_text

        # ── COMPONENT 11: Response schema and format instructions ───────────────────
        # This component provides the JSON schema and format requirements for the agent's response
        # Includes valid option IDs and example response format
        # ABLATION: This is essential for basic functionality - keep for all studies
        prompt += self.navigation_prompt 

        # **IMPORTANT**: Modify the schema description within the prompt
        # Create a temporary schema dict based on the class schema, but change 'decision' type
        temp_schema = json.loads(json.dumps(self.RESPONSE_SCHEMA))  # Deep copy

        # Always convert `decision` to a string-typed field for the prompt.
        temp_schema['properties']['decision'] = {"type": "string"}
        temp_schema['properties']['decision'].pop('minimum', None)
        temp_schema['properties']['decision'].pop('maximum', None)

        # No heading fields to modify

        prompt += (
            f"Return a JSON object strictly matching this schema. The 'decision' MUST be the unique string ID of your chosen option (e.g., 'stepX_optionY'):\\n"
            f"{json.dumps(temp_schema, indent=2)}\\n"
            "\\nVALID OPTION IDS (choose exactly one and place it in the 'decision' field):\\n"
            + " | ".join(unique_option_ids) +
            "\\n\\nEXAMPLE OF THE EXPECTED JSON FORMAT (fill with your own analysis, decision, and memory):\\n"
            "{\\n  \"analysis\": \"Your reasoning here\",\\n  \"decision\": \"" + unique_option_ids[0] + "\",\\n  \"memory\": \"Any memory to retain for future steps\"\\n}\\n"
        )

        return prompt

    def _validate_json_schema(self, data: Dict) -> bool:
        """
        Validate that the JSON data matches our expected schema.
        Now expects 'decision' to be a string unique ID.

        Args:
            data: Dictionary to validate

        Returns:
            True if valid, False otherwise
        """
        try:
            # Check required fields
            for field in self.RESPONSE_SCHEMA["required"]:
                if field not in data:
                    self.logger.warning(f"Missing required field: {field}")
                    return False

            # Validate field types and constraints based on the original schema
            # but handle 'decision' as a special case (string)
            for field, value in data.items():
                if field not in self.RESPONSE_SCHEMA["properties"]:
                    self.logger.warning(f"Unexpected field: {field}")
                    return False

                field_schema = self.RESPONSE_SCHEMA["properties"][field]

                if field_schema["type"] == "integer" and isinstance(value, str):
                    try:
                        # Works for "45", "045", "45.0"; leaves e.g. "12 deg" unchanged.
                        data[field] = int(float(value.strip()))
                        value = data[field]
                    except ValueError:
                        pass

                # Special handling for 'decision' field - should now be string
                if field == "decision":
                    if not isinstance(value, str):
                        self.logger.warning(f"Field '{field}' should be string (unique ID), got {type(value)}")
                        return False
                    # No min/max check needed for string ID, continue to next field
                    continue
                # heading fields not supported anymore
                # --- Original validation logic for other fields (analysis, memory) ---
                if field_schema["type"] == "string":
                    if not isinstance(value, str):
                        self.logger.warning(f"Field {field} should be string, got {type(value)}")
                        return False
                # --- Keep integer validation logic in case schema changes ---
                elif field_schema["type"] == "integer":
                    if not isinstance(value, (int, float)) or isinstance(value, bool):
                        self.logger.warning(f"Field {field} should be integer, got {type(value)}")
                        return False
                    # Convert float to int if needed
                    if isinstance(value, float):
                        # Note: modifying data dict here might be unexpected side effect,
                        # but kept from original code. Consider just checking if int(value) works.
                        data[field] = int(value)
                    # Check range constraints for integers (won't apply to 'decision')
                    if "minimum" in field_schema and value < field_schema["minimum"]:
                        self.logger.warning(f"Field {field} below minimum: {value} < {field_schema['minimum']}")
                        return False
                    if "maximum" in field_schema and value > field_schema["maximum"]:
                        self.logger.warning(f"Field {field} above maximum: {value} > {field_schema['maximum']}")
                        return False
                elif field_schema["type"] == "number":
                    if not isinstance(value, (int, float)) or isinstance(value, bool):
                        self.logger.warning(f"Field {field} should be number, got {type(value)}")
                        return False    

            return True
        except Exception as e:
            self.logger.error(f"Schema validation error: {e}")
            return False

    def _process_api_response(self, response_text: str, content: List, is_second_call: bool = False) -> Dict:
        """
        Process and validate API response text.
        
        Args:
            response_text: Raw response text from API
            content: Original content sent to API (for retries)
            is_second_call: Whether this is processing a second API call
            
        Returns:
            Validated JSON dictionary
            
        Raises:
            ValueError if JSON is invalid and cannot be repaired/retried
        """
        try:
            # First try direct JSON parsing
            output = json.loads(response_text)
            if self._validate_json_schema(output):
                return output
                
            # If schema validation failed, try repair
            self.logger.warning("Initial JSON parse succeeded but schema validation failed")
            repaired = repair_json(response_text)
            if repaired:
                output = json.loads(repaired)
                if self._validate_json_schema(output):
                    return output
                    
            # If we get here, JSON is valid but schema is wrong
            # For second call, we should raise to trigger retry
            if is_second_call:
                raise ValueError("Second call produced invalid schema")
                
            # Otherwise retry the first call
            raise ValueError("Response invalid after repair attempts")
        
        except json.JSONDecodeError:
            self.logger.warning("JSON decode error, attempting repair...")
            repaired = repair_json(response_text)
            if repaired:
                try:
                    output = json.loads(repaired)
                    if self._validate_json_schema(output):
                        return output
                except:
                    pass
                    
            # JSON repair failed or produced invalid schema
            if is_second_call:
                raise ValueError("Second call produced invalid JSON")
                
            # Retry first call
            raise ValueError("Response invalid after repair attempts")

    # ----------------------------------------------------------------------
    # FULL LLM round‑trip with automatic retry (network/parsing errors)
    # ----------------------------------------------------------------------
    @retry(
        stop=stop_after_attempt(4),
        wait=wait_exponential(multiplier=2, min=15, max=180),
        retry=retry_if_exception_type((
            ValueError,
            json.JSONDecodeError,
            Exception,
        )),
        reraise=True,
    )
    def call_llm_api(self, agent: Any, prompt: str, images: List[Dict], image_urls: List[str]) -> Dict:
        """
        Calls the configured LLM API with the given prompt and images, handling
        retries and response parsing.
        """
        try:
            self.logger.info(f"Calling LLM API using {self.model_name}")
            start_time = time.time()
            
            messages = [
                {"role": "user", "content": [
                    {"type": "text", "text": prompt}
                ] + [
                    {"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{img['data']}"}} 
                    for img in images
                ]}
            ]
            
            # Use the new llm_call wrapper (merge defaults with user-provided llm_params)
            _base_kwargs = {
                "max_tokens": 8000,
                "temperature": 0.5,
                "response_format": {"type": "json_object"},
            }
            _merged_kwargs = {**_base_kwargs, **(self.llm_params or {})}
            raw_response = llm_call(
                model=self.model_name,
                messages=messages,
                **_merged_kwargs
            )
            elapsed_time = time.time() - start_time
            response_text = raw_response["choices"][0]["message"]["content"]

            self.logger.debug(f"LLM response received: {response_text[:100]}...")

            # Parse the response JSON.
            output = self._process_api_response(response_text, messages)
            try:
                self.logger.debug(f"Parsed decision JSON: {str(output)[:120]}")
            except Exception:
                pass

            # Save experiment data
            experiment_data = {
                # Request details
                'request': {
                    'prompt': prompt,
                    'image_count': len(images),
                    'image_details': [
                        {
                            'index': i,
                            'mime_type': img['mime_type'],
                            'data_length': len(img['data']),
                            'data_prefix': img['data'][:20] + '...' if len(img['data']) > 20 else img['data'],
                            'image_url': image_urls[i] if i < len(image_urls) else "unknown"
                        } for i, img in enumerate(images)
                    ],
                    'image_urls': image_urls,
                    'model': self.model_name,
                    'temperature': 0.5,
                    'response_format': {"type": "json_object"},
                    'llm_params': self.llm_params,
                    'timestamp': time.strftime('%Y-%m-%d %H:%M:%S')
                },
                # Response details
                'response': {
                    'raw_text': response_text,
                    'parsed_json': output,
                    'text_length': len(response_text)
                },
                # Performance metrics
                'performance': {
                    'elapsed_time_seconds': elapsed_time,
                    'api_latency': elapsed_time
                },
                # Agent context
                'agent_context': {
                    'step_counter': agent.step_counter,
                    'observation_pano_id': agent.observation.get('pano_id', 'unknown') if hasattr(agent, 'observation') else 'unknown',
                    'current_plan': getattr(agent, 'current_plan', 'No current plan')
                },
                # Environment data
                'environment': {
                    'links_count': len(agent.observation.get('links', [])) if hasattr(agent, 'observation') else 0,
                    'previous_pano_id': agent.observation.get('previous_pano_id', 'unknown') if hasattr(agent, 'observation') else 'unknown'
                },
                # Additional metadata
                'metadata': {
                    'class': self.__class__.__name__,
                    'call_folder': self.call_folder,
                    'python_timestamp': time.time()
                }
            }
            
            # Try to capture LiteLLM-specific response metadata
            try:
                if 'usage' in raw_response:
                    usage = raw_response['usage']
                    # Convert Pydantic model to dict if needed (LiteLLM/OpenAI >= 1.1)
                    if hasattr(usage, 'model_dump'):
                        experiment_data['response']['usage'] = usage.model_dump()
                    elif hasattr(usage, 'dict'):
                        experiment_data['response']['usage'] = usage.dict()
                    else:
                        experiment_data['response']['usage'] = usage
                
                if 'id' in raw_response:
                    experiment_data['response']['id'] = raw_response['id']
                
                if 'finish_reason' in raw_response['choices'][0]:
                    experiment_data['response']['finish_reason'] = raw_response['choices'][0]['finish_reason']
                    
            except Exception as metadata_error:
                experiment_data['response']['metadata_error'] = str(metadata_error)
            
            # Save to experiment folder
            folder = 'gemini_calls'
            detailed_filename = f"{self.call_folder}/{folder}/decision_{agent.decision_counter}.json"
            detailed_filename_legacy = f"{self.call_folder}/{folder}/call_{agent.step_counter}.json"

            print(f"legacy: {detailed_filename_legacy}")
            print(f"detailed: {detailed_filename}")
            self.logger.debug(f"Saving API call to {detailed_filename}")
            
            directory = os.path.dirname(detailed_filename) 
            os.makedirs(directory, exist_ok=True)
            
            # Save the original format (for compatibility)
            with open(detailed_filename, 'w') as f:
                json.dump(experiment_data, f, indent=4)

            directory = os.path.dirname(detailed_filename_legacy) 
            os.makedirs(directory, exist_ok=True)
            
            with open(detailed_filename_legacy, 'w') as f:
                json.dump(experiment_data, f, indent=4)
            return output
            
        except Exception as e:
            # Existing error handling code remains the same
            self.logger.error(f"LLM API error: {e}", exc_info=True)
            chosen_index = random.randint(0, len(images) - 1)
            chosen_url = image_urls[chosen_index] if chosen_index < len(image_urls) else "unknown"
            return {
                'decision': chosen_index,
                'analysis': f"LLM API failed: {str(e)}, defaulting to random choice.",
                'memory': "API failure occurred",
                'chosen_image_url': chosen_url
            }

    def heading_to_cardinal(self, heading: float) -> str:
        """
        Convert a heading in degrees to a cardinal direction.

        Args:
            heading: The heading in degrees (0-360)

        Returns:
            The cardinal direction as a string (e.g., "North", "South")
        """
        directions = ["North", "East", "South", "West"]
        idx = int((heading + 45) / 90) % 4  # +45 to center ranges around cardinal points
        return directions[idx]

    def heading_to_cardinal_legacy(self, heading: float) -> str:
        """
        Convert a heading in degrees to a cardinal direction.

        Args:
            heading: The heading in degrees (0-360)

        Returns:
            The cardinal direction as a string (e.g., "North", "North-East")
        """
        directions = ["North", "North-East", "East", "South-East", 
                    "South", "South-West", "West", "North-West"]
        idx = int((heading + 22.5) / 45) % 8  # +22.5 to center ranges around cardinal points
        return directions[idx]
