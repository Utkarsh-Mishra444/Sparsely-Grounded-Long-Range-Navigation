import json, os
from strategies.memory_strategy import AdvancedStreetViewStrategy


class PromptLearnerStrategy(AdvancedStreetViewStrategy):
    """
    Wraps AdvancedStreetViewStrategy but injects the offline‑learned prompt.
    """

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        path = os.path.join(self.call_folder, "best_prompt.json")
        if not os.path.isfile(path):
            raise FileNotFoundError(
                f"{path} not found. Run optimise_prompt.py first."
            )
        with open(path, encoding="utf-8") as fp:
            bundle = json.load(fp)
        self.learned = bundle["rendered"]

    # single override
    def prepare_prompt(
        self,
        agent,
        num_images,
        cardinal_directions,
        previous_visits=None,
        unique_option_ids=None,
    ):
        dynamic = super().prepare_prompt(
            agent,
            num_images,
            cardinal_directions,
            previous_visits,
            unique_option_ids,
        )
        # learned prompt must contain {DYNAMIC} token
        if "{DYNAMIC}" not in self.learned:
            raise ValueError(
                "The learned prompt must include the literal string "
                "{DYNAMIC} where runtime content should go."
            )
        return self.learned.replace("{DYNAMIC}", dynamic)
