class Agent:
    def __init__(self, strategy_instance, **kwargs):
        self.strategy = strategy_instance
        for key, value in kwargs.items():
            setattr(self, key, value)

    def select_action(self):
        return self.strategy.select_action(self)

    def update(self, action, observation):
        # Default update behavior, can be overridden in subclasses
        pass