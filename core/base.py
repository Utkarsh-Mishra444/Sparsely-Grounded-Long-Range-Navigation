"""Base classes for agents and environments."""


class Agent:
    """Base agent class that delegates action selection to a strategy."""
    
    def __init__(self, strategy_instance, **kwargs):
        self.strategy = strategy_instance
        for key, value in kwargs.items():
            setattr(self, key, value)

    def select_action(self):
        return self.strategy.select_action(self)

    def update(self, action, observation):
        # Default update behavior, can be overridden in subclasses
        pass


class Environment:
    """Base environment class defining the interface for simulation environments."""
    
    def __init__(self, initial_state):
        self.state = initial_state

    def get_observation(self):
        return self.state

    def apply_action(self, action):
        raise NotImplementedError("Subclasses must implement this method.")
