from abc import ABC, abstractmethod

class BasePolicy(ABC):
    @abstractmethod
    def act(self, model_output: dict) -> dict:
        pass

class VanillaPolicy(BasePolicy):
    """Always returns the model's primary response."""
    def act(self, model_output: dict) -> dict:
        return {"action": "answer", "response": model_output['text']}

class ConfidenceAbstentionPolicy(BasePolicy):
    """Abstains if model confidence is below a threshold."""
    def __init__(self, threshold: float = 0.5):
        self.threshold = threshold

    def act(self, model_output: dict) -> dict:
        if model_output['confidence'] < self.threshold:
            return {"action": "abstain", "response": "I am not confident in this answer."}
        return {"action": "answer", "response": model_output['text']}

class MoDSTPolicy(BasePolicy):
    """Abstains or fallbacks based on Tripwire Score."""
    def __init__(self, threshold: float = 0.5):
        self.threshold = threshold

    def act(self, tripwire_score: float, passes: dict) -> dict:
        if tripwire_score > self.threshold:
            # Policy choice: Fallback to grounding description
            return {
                "action": "fallback", 
                "response": f"I can only reliably say: {passes['y_img']}"
            }
        return {"action": "answer", "response": passes['y_full']}
