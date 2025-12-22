class Strategy:
    def select_action(self, agent):
        raise NotImplementedError("Subclasses must implement select_action")