class Environment:
    def __init__(self, initial_state):
        self.state = initial_state

    def get_observation(self):
        return self.state

    def apply_action(self, action):
        raise NotImplementedError("Subclasses must implement this method.")