import unittest
from unittest.mock import MagicMock, patch
import torch

# Mock modules
with patch.dict('sys.modules', {
    'sentence_transformers': MagicMock(),
    'sentence_transformers.CrossEncoder': MagicMock(),
}):
    from modst.tripwire.score import MoDSTScore
    from modst.tripwire.grounding import GroundingTripwire, SafetyTripwire

class TestTripwireLogic(unittest.TestCase):
    def setUp(self):
        # Mock CrossEncoder
        self.mock_model = MagicMock()
        # predict returns logits: [Contradiction, Neutral, Entailment]
        # We need to mock return values per call logic
        
        self.patcher = patch('modst.tripwire.grounding.CrossEncoder', return_value=self.mock_model)
        self.patcher.start()

    def tearDown(self):
        self.patcher.stop()

    def test_scores(self):
        modst = MoDSTScore()
        
        # Mock Predictions
        # 1. Grounding (y_img, y_full): High Entailment (Consistent) -> [0, 0, 10] -> Prob ~1
        # 2. Fusion (y_full, y_text): Low Entailment (Independent) -> [10, 0, 0] -> Prob ~0
        # 3. Safety (Refusals): Checks pairs against "Prototype"
        
        def side_effect(pairs):
            pair = pairs[0]
            # Safety Check Logic
            if "Refusal" in pair[1]: # Prototype
                if "sorry" in pair[0]: return torch.tensor([[-10.0, -10.0, 10.0]]) # Entailment
                return torch.tensor([[10.0, -10.0, -10.0]]) # Contradiction (Not refusal)
            
            # Grounding/Fusion Logic
            if "cat" in pair[0] and "cat" in pair[1]:
                return torch.tensor([[-10.0, -10.0, 10.0]]) # Entailment
            
            return torch.tensor([[10.0, -10.0, -10.0]]) # Contradiction

        self.mock_model.predict.side_effect = side_effect
        
        passes = {
            "y_full": "A cat on a mat",
            "y_img": "A cat",
            "y_text": "A dog"
        }
        
        # Grounding: img entails full? Yes -> Score ~ 0
        # Fusion: full entails text? No -> Score ~ 1 (Distance) -> s_f (Collapse) ~ 0
        # Safety: full/text match refusal? No/No -> Gap 0
        
        scores = modst.compute(passes)
        print("Scores:", scores)
        self.assertLess(scores['grounding_discrepancy'], 0.1)

if __name__ == '__main__':
    unittest.main()
