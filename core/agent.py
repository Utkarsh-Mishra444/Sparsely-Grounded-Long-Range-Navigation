"""StreetView navigation agent."""

from typing import List, Dict, Any, Optional
from core.base import Agent
from agents.self_positioning import SelfPositioningAgent
class StreetViewAgent(Agent):
    """
    A simplified agent for navigating Street View environments.
    Manages basic state like observation, step count, and movement history.
    Relies entirely on the provided strategy instance for decision-making.
    """
    def __init__(self,
                 strategy_instance: Any,
                 destination: str,
                 memory: str = ""):
        """
        Initialize the StreetViewAgent.

        Args:
            strategy_instance: An instantiated strategy object (e.g., AdvancedStreetViewStrategy).
            destination: The name of the target destination (used by the strategy).
            memory: Initial memory string for the strategy.
        """
        super().__init__(strategy_instance)
        self.observation: Optional[Dict[str, Any]] = None
        self.destination: str = destination
        self.decision_history: List[Dict[str, Any]] = [] # Populated by the strategy
        self.step_counter: int = 0
        self.call_folder: str = strategy_instance.call_folder # Accessed by strategy for logging
        self.decision_counter: int = 0 # Incremented by the strategy
        self.movement_history: List[str] = [] # List of cardinal directions (e.g., "North", "East")
        self.memory: str = memory # Populated and read by the strategy
        self.last_decision_info: Optional[Dict[str, Any]] = None
         # ── self-positioning sub-component ───────────────────
        # Use the same model + logger as the main strategy (unified LiteLLM backend)
        model_name = getattr(strategy_instance, "model_name", "gemini/gemini-2.5-flash")
        logger = getattr(strategy_instance, "logger", None)
        
        # If the strategy exposes provider-specific LLM params, pass them through
        strategy_llm_params = getattr(strategy_instance, "llm_params", None)
        self.pos_agent = SelfPositioningAgent(
            api_key=getattr(strategy_instance, "maps_api_key", None),
            model=model_name,
            logger_instance=logger,
            log_dir=self.call_folder,
            streetview_signing_secret=getattr(strategy_instance, "streetview_signing_secret", None),
            use_signed_streetview=getattr(strategy_instance, "use_signed_streetview", False),
        )
        # Attach for downstream usage (threaded via llm_call at call sites)
        if strategy_llm_params is not None:
            setattr(self.pos_agent, "llm_params", strategy_llm_params)
        # Queue for environment events emitted by strategy (e.g., dead_end)
        self.env_events = []
    def update(self, action: Optional[str], observation: Dict[str, Any]) -> None:
        """
        Update the agent's state after an action is applied.

        Args:
            action: The action alias that was just taken (or None on first step).
            observation: The new observation dictionary from the environment.
        """
        self.observation = observation
        if action is not None:
            self.step_counter += 1

            # Record the actual direction traveled based on arrival heading
            if observation and 'arrival_heading' in observation and observation['arrival_heading'] is not None:
                # The arrival_heading is the heading of the link that was followed TO get here.
                actual_travel_heading = observation['arrival_heading']

                # Convert to cardinal direction using the strategy's helper method
                if hasattr(self.strategy, 'heading_to_cardinal'):
                    actual_cardinal = self.strategy.heading_to_cardinal(actual_travel_heading)
                    self.movement_history.append(actual_cardinal)
                    print(f"Agent Update: Recorded actual movement direction: {actual_cardinal} (heading {actual_travel_heading:.1f}°)")
                else:
                     print("Agent Update: Warning - Strategy lacks 'heading_to_cardinal' method. Cannot record movement direction.")

    def select_action(self) -> Optional[str]:
        """
        Delegate action selection entirely to the strategy instance.

        Returns:
            The action alias selected by the strategy, or None if no action is possible.
        """
        return self.strategy.select_action(self, self.step_counter)

# --- END OF FILE agent2.py ---