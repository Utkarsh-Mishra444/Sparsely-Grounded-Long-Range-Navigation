import os
from datetime import datetime
#import datetime  
import json  
import sys  
import signal  
import traceback  
from core.utils import haversine, dump_coordinates, dump_evaluations, save_checkpoint  
from core.environment import StreetViewEnvironment  
from core.agent import StreetViewAgent  

# --- Global flag for interrupt handling ---
interrupted = False
def signal_handler(sig, frame):
    global interrupted  
    if not interrupted:
        # Avoid printing here to prevent reentrancy with Tee
        # print('\nCtrl+C detected! Attempting to save checkpoint before exiting...', flush=True)
        interrupted = True
    else:
        # Still print force exit message, but use original stderr just in case
        print('Shutdown signal received again. Forcing exit.', file=sys.__stderr__, flush=True)
        sys.exit(1) # Force exit on second Ctrl+C
signal.signal(signal.SIGINT, signal_handler)

# --- Simulation Class ---
class Simulation:
    def __init__(self, agent, env, strategy, config, experiment_folder, checkpoint_file="checkpoint_latest.pkl", is_resuming=False, resume_action=None, initial_visited_coordinates=None):
        """Initializes the Simulation object.

        Sets up all the core components and initializes state variables based on
        the provided configuration. Determines the coordinate filename and loads
        existing data if resuming.

        Args:
            agent (StreetViewAgent): The agent instance.
            env (StreetViewEnvironment): The environment instance.
            strategy (AdvancedStreetViewStrategy): The strategy instance.
            config (dict): The simulation configuration dictionary.
            experiment_folder (str): Path to the log/checkpoint directory.
            checkpoint_file (str, optional): Name for the checkpoint file.
                                             Defaults to "checkpoint_latest.pkl".
            is_resuming (bool, optional): True if resuming from a checkpoint.
                                          Defaults to False.
            resume_action (str, optional): The action saved in the checkpoint, if any.
                                          Defaults to None.
            initial_visited_coordinates (list, optional): Coordinates loaded from checkpoint.
                                                        Defaults to None.
        """
        self.agent = agent
        self.env = env
        self.strategy = strategy
        self.config = config
        self.experiment_folder = experiment_folder
        # self.checkpoint_file = checkpoint_file # Removed: Filename is now dynamic per decision
        # self.checkpoint_path = os.path.join(experiment_folder, checkpoint_file) # Removed: Path is now dynamic
        
        # Configuration parameters
        self.max_steps = config.get('max_steps', 700)
        self.max_decision_points = config.get('max_decision_points', 50)
        self.destination_coords = config['destination_coords']
        self.destination_radius = config.get('destination_radius', 50)
        self.termination_criteria = config.get('termination_criteria', 'distance')

        # Validate termination criteria
        valid_criteria = ['distance', 'polygon', 'both']
        if self.termination_criteria not in valid_criteria:
            raise ValueError(f"Invalid termination_criteria '{self.termination_criteria}'. Must be one of: {valid_criteria}")
        
        # State variables
        self.visited_coordinates = initial_visited_coordinates if initial_visited_coordinates is not None else []
        self.resume_action = resume_action
        self.decision_evaluations = []
        self.pending_evaluations = []
        self.checkpoint_saved = False # Tracks if *any* checkpoint was saved during the run
        
        # Generate unique filenames for this run
        self.timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
        self.strategy_name = strategy.__class__.__name__
        self.env_str = f"{config['initial_coords'][0]}_{config['initial_coords'][1]}_to_{self.destination_coords[0]}_{self.destination_coords[1]}".replace('.', '_')
        
        # Use a constant name for the coordinate file, joining with the folder path
        coord_base_filename = "visited_coordinates.json"
        self.coord_filename = os.path.join(self.experiment_folder, coord_base_filename)
        
        # Store base filenames for passing to dump functions
        self.coord_base_filename = coord_base_filename
        self.eval_base_filename = f"decision_evaluations_{self.timestamp}_{self.strategy_name}_{self.env_str}.json"
        # Store just the base filename; dump_evaluations will prepend experiment_folder
        self.eval_filename = self.eval_base_filename
        # Initialize branch pruning threshold and history
        self.min_score = self.config.get('min_score')
        self.score_history = []

    def log_coordinates_and_evaluations(self):
        """Logs the agent's current coordinates and dumps completed evaluations to files.

        Retrieves the current coordinates from the environment. If valid, appends 
        them to the `self.visited_coordinates` list and saves the list to the 
        coordinate log file using `dump_coordinates`. It also saves the current list 
        of completed `self.decision_evaluations` to the evaluation log file using 
        `dump_evaluations`. Prints the current step and location to the console.

        Returns:
            dict or None: The current coordinate dictionary ({'lat': ..., 'lng': ..., 
                          'pano_id': ...}) if successfully retrieved, otherwise None.
        """
        coords = self.env.get_current_coordinates()
        if coords and coords.get('lat') is not None:
            # Only append if it's a genuinely new point (different from the last one)
            if not self.visited_coordinates or self.visited_coordinates[-1]['pano_id'] != coords.get('pano_id'):
                # Add the step counter to the coordinate data
                coords['step'] = self.agent.step_counter
                self.visited_coordinates.append(coords)
            # Always dump the potentially updated list (REMOVED - only save in checkpoint/final)
            # dump_coordinates(self.visited_coordinates, self.coord_base_filename, self.experiment_folder)
            # Restore continuous writing for live visualization
            dump_coordinates(self.visited_coordinates, self.coord_base_filename, self.experiment_folder)
            dump_evaluations(self.decision_evaluations, self.eval_filename, self.experiment_folder)
            # Also save detailed decision history including analysis and updated memory
            try:
                detailed_log_path = os.path.join(self.experiment_folder, f"detailed_decision_log_{self.timestamp}.json")
                with open(detailed_log_path, 'w', encoding='utf-8') as f_detail:
                    json.dump(self.agent.decision_history, f_detail, indent=2)
                # print(f"Detailed decision history saved to: {detailed_log_path}")
            except Exception as e:
                print(f"Error saving detailed decision history: {e}", file=sys.stderr)
            print(f"Step {self.agent.step_counter}: Current pano {coords.get('pano_id','N/A')} at ({coords.get('lat','N/A')}, {coords.get('lng','N/A')})")
            return coords
        return None
    
    def process_pending_evaluations(self, is_decision_point, coords):
        """Updates and finalizes pending decision evaluations.

        Iterates through the `self.pending_evaluations` list. For each pending 
        evaluation, increments its 'hops' counter.
        
        An evaluation is considered complete and ready to be finalized if:
        1. It has been pending for 3 or more steps (`hops >= 3`).
        2. The agent has reached a new decision point (`is_decision_point` is True).
        
        When an evaluation is finalized, it calculates the walking distance from the 
        current coordinates (`coords`) to the destination. It compares this to the 
        distance recorded *before* the decision was made. If the distance decreased, 
        the decision is marked "RIGHT"; otherwise, it's marked "WRONG". The finalized 
        evaluation dictionary is added to `self.decision_evaluations`.
        
        Evaluations that are not yet finalized remain in the `pending_evaluations` list.

        Args:
            is_decision_point (bool): True if the current location has >= 3 links.
            coords (dict): The current coordinates dictionary.
        """
        if coords is None or coords.get('lat') is None:
            return
            
        lat_current, lng_current = coords['lat'], coords['lng']
        new_pending_evaluations = []
        
        for pending in self.pending_evaluations:
            pending['hops'] += 1
            if pending['hops'] >= 3 or is_decision_point:
                # Time to evaluate this pending decision
                # TEMPORARILY DISABLED FOR TESTING - skip API call entirely
                # distance_after_result = self.env.calculate_walking_distance(
                #     lat_current, lng_current,
                #     self.destination_coords[0], self.destination_coords[1]
                # )
                # distance_after = distance_after_result.get('distance')
                distance_after = None  # Skip distance calculation for testing
                
                # Evaluate decision effectiveness
                decision_status = "UNKNOWN"
                if pending['distance_before'] is not None and distance_after is not None:
                    decision_status = "RIGHT" if distance_after < pending['distance_before'] else "WRONG"
                
                # Log decision evaluation
                decision_eval = {
                    'decision_number': pending['decision_number'],
                    'action': pending['action'],
                    'status': decision_status,
                    'correct_ids'     : pending.get('correct_ids', []),
                    'alias_to_id_map' : pending.get('alias_to_id_map', {}),
                }
                
                self.decision_evaluations.append(decision_eval)
                
                # Log evaluation to terminal
                print(f"Decision evaluation after {pending['hops']} hops: {decision_status} - Walking distance changed by "
                      f"{None if pending['distance_before'] is None or distance_after is None else distance_after - pending['distance_before']} meters")
            else:
                # Keep this evaluation pending
                new_pending_evaluations.append(pending)
                
        # Replace the pending evaluations list with the updated one
        self.pending_evaluations = new_pending_evaluations
    
    def start_new_evaluation(self, coords, num_links, action, step):
        """Initiates tracking for a new decision made at a decision point.

        This method is called when the agent encounters a location with 3 or more 
        links (a decision point) and makes a choice (`action`).
        
        It calculates the current walking distance to the destination *before* the 
        action is taken. It then creates a new dictionary containing details about 
        this decision point (step number, action taken, coordinates before, distance 
        before, number of links, current decision number) and appends it to the 
        `self.pending_evaluations` list. This evaluation will be processed in 
        subsequent steps by `process_pending_evaluations`.

        Args:
            coords (dict): The coordinates dictionary at the decision point.
            num_links (int): The number of links available at this point (>= 3).
            action (str): The action chosen by the agent.
            step (int): The current simulation step number.
        """
        if coords is None or coords.get('lat') is None:
            return
            
        lat_current, lng_current = coords['lat'], coords['lng']
        
        # Calculate walking distance before action
        # TEMPORARILY DISABLED FOR TESTING - skip API call entirely
        # distance_before_result = self.env.calculate_walking_distance(
        #     lat_current, lng_current,
        #     self.destination_coords[0], self.destination_coords[1]
        # )
        # distance_before = distance_before_result.get('distance')
        distance_before = None  # Skip distance calculation for testing
                # ── NEW: which alias(es) are truly RIGHT at this node ─────────────
        current_obs      = self.agent.observation or {}
        evals_by_alias   = current_obs.get("evaluations_alias", {})
        correct_aliases  = [a for a, st in evals_by_alias.items()
                            if st.get("label") == "RIGHT"]

        last_info        = getattr(self.agent, "last_decision_info", {})
        alias_to_id_map  = last_info.get("alias_to_id_map", {})
        correct_ids      = [alias_to_id_map.get(a, a) for a in correct_aliases]
        # Start a pending evaluation
        self.pending_evaluations.append({
            'step': step,
            'action': action,
            'coords_before': {'lat': lat_current, 'lng': lng_current},
            'distance_before': distance_before,
            'num_links': num_links,
            'hops': 0,  # Will be incremented in the next iteration
            'decision_number': self.agent.decision_counter,
            'correct_ids'     : correct_ids,     # NEW
            'alias_to_id_map' : alias_to_id_map  # NEW
        })
    
    def check_arrival_at_destination(self, coords):
        """Checks if the agent has reached the destination proximity threshold.

        Checks termination criteria based on the termination_criteria configuration.
        - "distance": only checks if within destination_radius (50m) of destination_coords
        - "polygon": only checks if current coordinates are inside destination polygon
        - "both": checks both conditions (terminates when either is met)

        If the configured termination condition(s) are met:
        1. Finalizes any remaining `self.pending_evaluations` based on the arrival coordinates.
        2. Dumps the final evaluation list to the log file.
        3. Prints an arrival message.
        4. Sets the global `interrupted` flag to True to signal the simulation loop to stop.
        5. Returns True.

        If the configured termination criteria are not met, it prints the current distance and
        returns False.

        Args:
            coords (dict): The agent's current coordinates dictionary.

        Returns:
            bool: True if the destination is reached, False otherwise.
        """
        global interrupted

        if coords is None or coords.get('lat') is None:
            return False

        lat, lng = coords['lat'], coords['lng']

        # Check termination conditions based on criteria setting
        polygon_reached = False
        distance_reached = False
        termination_reason = None

        # Check polygon-based termination if criteria includes polygon
        if self.termination_criteria in ['polygon', 'both']:
            if hasattr(self.env, 'is_point_in_destination_polygon') and self.env.destination_polygon_obj is not None:
                polygon_reached = self.env.is_point_in_destination_polygon(lat, lng)
                if polygon_reached:
                    print("Reached destination polygon!")
                    termination_reason = "polygon"

        # Check distance-based termination if criteria includes distance
        h_distance = haversine(lat, lng, self.destination_coords[0], self.destination_coords[1])
        print(f"Distance to destination: {h_distance:.2f} m")

        if self.termination_criteria in ['distance', 'both'] and h_distance < self.destination_radius:
            distance_reached = True
            if not polygon_reached:  # Don't override polygon message
                print(f"Reached within {self.destination_radius} meters of destination!")
            termination_reason = "distance"

        # If any enabled termination condition is met
        termination_met = (self.termination_criteria in ['polygon', 'both'] and polygon_reached) or \
                         (self.termination_criteria in ['distance', 'both'] and distance_reached)

        if termination_met:
            # Process any remaining pending evaluations
            for pending in self.pending_evaluations:
                # TEMPORARILY DISABLED FOR TESTING - skip API call entirely
                # distance_after_result = self.env.calculate_walking_distance(
                #     lat, lng,
                #     self.destination_coords[0], self.destination_coords[1]
                # )
                # distance_after = distance_after_result.get('distance')
                distance_after = None  # Skip distance calculation for testing

                decision_status = "UNKNOWN"
                if pending['distance_before'] is not None and distance_after is not None:
                    decision_status = "RIGHT" if distance_after < pending['distance_before'] else "WRONG"

                decision_eval = {
                    'decision_number': pending['decision_number'],
                    'action': pending['action'],
                    'status': decision_status,
                    'reached_destination': True,
                    'termination_reason': termination_reason,
                    'correct_ids'     : pending.get('correct_ids', []),
                    'alias_to_id_map' : pending.get('alias_to_id_map', {}),

                }
                self.decision_evaluations.append(decision_eval)

            # Save final evaluations
            # Use the base filename for dumping evaluations
            dump_evaluations(self.decision_evaluations, self.eval_filename, self.experiment_folder)
            print(f"Reached destination after {self.agent.step_counter} steps via {termination_reason}.")
            interrupted = True  # Trigger checkpoint save and exit
            return True
        return False
    
    def execute_step(self):
        """Performs all actions required for a single step of the simulation.

        This is the core logic executed within the main simulation loop. It:
        1. Gets the current observation and coordinates.
        2. Checks if coordinates are valid; exits if not.
        3. Determines if the current location is a decision point.
        4. Calls `process_pending_evaluations` to update tracking.
        5. Gets the next action from the agent.
        6. If it's a decision point, calls `start_new_evaluation`.
        7. Exits if no action is available.
        8. Applies the action to the environment.
        9. Gets the new observation and updates the agent.
        10. Calls `log_coordinates_and_evaluations`.
        11. Calls `check_arrival_at_destination`; exits loop if True.
        12. Performs periodic checkpointing every 20 steps by calling `save_checkpoint`.
        13. Calculates and prints the step duration.
        14. Handles any exceptions during the step, sets `interrupted` flag, and returns False.

        Returns:
            bool: True if the step executed successfully and the simulation should 
                  continue, False if an error occurred, the destination was reached, 
                  or no action was available (signaling the loop to stop).
        """
        global interrupted
        
        step = self.agent.step_counter
        try:
            current_step_start_time = datetime.now()
            print(f"\n--- Step {step} / Dec {self.agent.decision_counter} ---", flush=True)

            # Get current observation
            current_obs = self.agent.observation if self.agent.observation else self.env.get_observation()
            coords_before = self.env.get_current_coordinates()
            
            if coords_before.get('lat') is None:
                print("ERROR: Environment lost coordinates. Cannot continue.", flush=True)
                interrupted = True
                return False

            # Check if this is a decision point
            links = current_obs.get('links', [])
            num_links = len(links)
            is_decision_point = num_links >= 3
            
            # Process pending evaluations
            self.process_pending_evaluations(is_decision_point, coords_before)
            
            # Get action (different handling for decision points)
            # Check if resuming with a pre-determined action
            action = None
            if self.resume_action:
                action = self.resume_action
                print(f"Using resume action from checkpoint: {action}", flush=True)
                self.resume_action = None # Use it only once
                # Need to re-fetch observation components because select_action usually does this
                current_obs = self.env.get_observation() # Ensure obs is fresh for decision logic
                links = current_obs.get('links', [])
                num_links = len(links)
                is_decision_point = True # If we saved an action, it was a decision point
            else:
                # If not resuming, get action normally
                action = self.agent.select_action()
                # Relay any environment events emitted by the strategy (e.g., dead_end)
                events = getattr(self.agent, 'env_events', []) or []
                if events:
                    safe_events = []
                    for ev in events:
                        try:
                            if isinstance(ev, dict) and ev.get('type') == 'dead_end':
                                keys = set(ev.keys())
                                if keys == {'type', 'parent_pano_id', 'child_pano_id'}:
                                    self.env.register_dead_end_edge(str(ev['parent_pano_id']), str(ev['child_pano_id']))
                                    safe_events.append(ev)
                        except Exception:
                            pass
                    # Clear only processed events
                    try:
                        for _ in safe_events:
                            self.agent.env_events.remove(_)
                    except Exception:
                        self.agent.env_events = []
            
            if is_decision_point:
                decision_number = self.agent.decision_counter # Get current decision number
                print(f"Decision point {decision_number} with {num_links} links available - starting evaluation")
                self.start_new_evaluation(coords_before, num_links, action, step)
                print(f"Action Chosen: {action}")

                # --- Save unique checkpoint AFTER decision is made --- 
                chkpt_filename = f"checkpoint_decision_{decision_number}.pkl"
                chkpt_path = os.path.join(self.experiment_folder, chkpt_filename)
                print(f"\n--- Saving decision point checkpoint to {chkpt_filename} (Step: {step}, Action: {action}) ---")
                # Call global save_checkpoint directly with unique path
                self.checkpoint_saved = save_checkpoint( 
                    self.agent,
                    self.env,
                    self.strategy,
                    self.config,
                    chkpt_path, # Use unique path
                    action=action,
                    visited_coordinates=self.visited_coordinates
                )
                print(f"--- Decision checkpoint saved --- ")
            # else:
                # print(f"Action: {action} (deterministic choice with {num_links} links)")
            
            if action is None:
                print(f"Step {step}: No actions available. Stopping.")
                interrupted = True
                return False
                
            # Apply action and update agent
            # print(f"Attempting to apply action: {action}", flush=True)
            self.env.apply_action(action)
            # print(f"Action {action} applied successfully.", flush=True)
            new_obs = self.env.get_observation()
            # Record score and enforce minimum threshold
            if 'score' in new_obs:
                score_val = new_obs.get('score')
                self.score_history.append(score_val)
                if self.min_score is not None and score_val < self.min_score:
                    print(f"Score {score_val} fell below minimum {self.min_score}. Aborting simulation.", flush=True)
                    return False
            self.agent.update(action, new_obs)
            
            if 'score' in new_obs:
                print(f"Score: {new_obs['score']}")
            
            # Log coordinates and check destination arrival
            coords_after = self.log_coordinates_and_evaluations()
            if self.check_arrival_at_destination(coords_after):
                return False  # Destination reached
            
            step_duration = (datetime.now() - current_step_start_time).total_seconds()
            #print(f"--- Step {step} duration: {step_duration:.2f}s ---")
            
            return True  # Step completed successfully
            
        except Exception as loop_error:
            print(f"\n!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!")
            print(f"ERROR during simulation loop step {step}: {loop_error}")
            print(f"!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!")
            traceback.print_exc()
            interrupted = True
            return False
    
    def process_remaining_evaluations(self):
        """Finalizes any pending evaluations when the simulation ends normally.
        
        This is called if the simulation loop terminates because it reached the 
        maximum step count or maximum decision point count (i.e., not due to 
        arrival, error, or interrupt).
        
        It iterates through any evaluations still in `self.pending_evaluations`, 
        calculates their final status (RIGHT/WRONG) based on the agent's final 
        position, marks them with `'max_steps_reached': True`, adds them to 
        `self.decision_evaluations`, and dumps the complete list to the log file.
        """
        coords = self.env.get_current_coordinates()
        if coords.get('lat') is None:
            return
            
        for pending in self.pending_evaluations:
            # TEMPORARILY DISABLED FOR TESTING - skip API call entirely
            # distance_after_result = self.env.calculate_walking_distance(
            #     coords['lat'], coords['lng'],
            #     self.destination_coords[0], self.destination_coords[1]
            # )
            # distance_after = distance_after_result.get('distance')
            distance_after = None  # Skip distance calculation for testing
            
            decision_status = "UNKNOWN"
            if pending['distance_before'] is not None and distance_after is not None:
                decision_status = "RIGHT" if distance_after < pending['distance_before'] else "WRONG"
            
            decision_eval = {
                'decision_number': pending['decision_number'],
                'action': pending['action'],
                'status': decision_status,
                'max_steps_reached': True,
                'correct_ids'     : pending.get('correct_ids', []),
                'alias_to_id_map' : pending.get('alias_to_id_map', {}),
            }
            self.decision_evaluations.append(decision_eval)
        
        # Use the base filename for dumping evaluations
        dump_evaluations(self.decision_evaluations, self.eval_filename, self.experiment_folder)
    
    def initialize(self):
        """Performs setup actions before the main simulation loop starts.
        
        1. Logs the starting coordinates of the current run/segment.
        2. If starting a brand new simulation (agent step count is 0), it 
           retrieves the very first observation from the environment and 
           provides it to the agent.
        """
        # Log the starting coordinates
        start_coords = self.env.get_current_coordinates()
        if start_coords.get('lat') is not None:
            # Add the step counter to the starting coordinate data
            start_coords['step'] = self.agent.step_counter
            self.visited_coordinates.append(start_coords)
            # Restore writing of initial coordinate
            dump_coordinates(self.visited_coordinates, self.coord_base_filename, self.experiment_folder)
            print(f"Step {self.agent.step_counter}: Started at pano {start_coords.get('pano_id', 'N/A')} at ({start_coords.get('lat', 'N/A')}, {start_coords.get('lng', 'N/A')})")
        
        # Get initial observation if starting fresh
        if self.agent.step_counter == 0:
            initial_obs = self.env.get_observation()
            self.agent.update(None, initial_obs)
    
    def run(self):
        """Executes the main simulation loop.

        Manages the overall flow of the simulation:
        1. Calls `initialize()` for setup.
        2. Enters a `while` loop that continues as long as:
           - The step count is below `max_steps`.
           - The decision count is below `max_decision_points`.
           - The `interrupted` flag is False.
        3. Inside the loop, calls `execute_step()`.
        4. Breaks the loop if `execute_step()` returns False.
        5. After the loop, if not interrupted (i.e., max steps/decisions reached), 
           calls `process_remaining_evaluations()`.
        6. If interrupted and no checkpoint was saved during the loop, attempts 
           a final `save_checkpoint()`.
        7. Prints final status messages.
        8. Returns the status of the last checkpoint save attempt.

        Returns:
            bool: True if a final checkpoint was attempted and saved successfully, 
                  False otherwise (or if no final attempt was needed).
        """
        global interrupted
        
        print(f"\n--- Starting/Resuming simulation loop in {self.experiment_folder} ---")
        print(f"--- Starting at Step: {self.agent.step_counter}, Decision: {self.agent.decision_counter} ---")
        
        self.initialize()
        
        step = self.agent.step_counter
        while step < self.max_steps and self.agent.decision_counter < self.max_decision_points:
            if interrupted:
                print("\nCtrl+C detected. Breaking simulation loop gracefully...", flush=True)
                break
                
            if not self.execute_step():
                break
                
            step = self.agent.step_counter  # Update step counter from agent
        
        # Process remaining evaluations if we reached max steps/decisions
        if not interrupted:
            print(f"\nMax steps ({self.max_steps}) or max decisions ({self.max_decision_points}) reached.")
            self.process_remaining_evaluations()
        
        print(f"Simulation loop finished. Coordinates saved to '{self.coord_filename}'.")
        print(f"Decision evaluations saved to '{self.eval_filename}'.")

        # --- Save final coordinates --- 
        print(f"--- Saving final {len(self.visited_coordinates)} coordinates to {self.coord_filename} ---", flush=True)
        try:
            # Use the base filename for dumping coordinates
            dump_coordinates(self.visited_coordinates, self.coord_base_filename, self.experiment_folder)
            print("Final coordinates saved successfully.", flush=True)
        except Exception as e:
            print(f"ERROR saving final coordinates to {self.coord_filename}: {e}", flush=True)
            traceback.print_exc()
        # ---------------------------

        # Dump the evolving score history for analysis
        try:
            score_hist_path = os.path.join(self.experiment_folder, 'score_history.json')
            with open(score_hist_path, 'w', encoding='utf-8') as f:
                json.dump(self.score_history, f, indent=2)
            print(f"Saved score history to {score_hist_path}")
        except Exception as e:
            print(f"ERROR saving score history: {e}", flush=True)
            traceback.print_exc()
        return self.checkpoint_saved