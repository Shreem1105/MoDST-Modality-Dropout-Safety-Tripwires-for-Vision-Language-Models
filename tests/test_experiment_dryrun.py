import unittest
from unittest.mock import MagicMock, patch
import os
import shutil
import json
from PIL import Image

# Mock dependencies before imports to avoid GPU/Weight loading
with patch.dict('sys.modules', {
    'transformers': MagicMock(),
    'transformers.LlavaForConditionalGeneration': MagicMock(),
    'transformers.AutoProcessor': MagicMock(),
    'sentence_transformers': MagicMock(),
    'accelerate': MagicMock(),
    'bitsandbytes': MagicMock(),
}):
    # Import modules under test
    from modst.models.vlm_wrapper import AbstractVLM
    from modst.tripwire.score import MoDSTScore
    from modst.experiments.run_modst import main as run_experiment
    from modst.datasets.base_dataset import BaseVLMDataset

class MockVLM(AbstractVLM):
    def __init__(self, *args, **kwargs):
        pass
    def generate(self, prompt, image=None):
        return {"text": "mock response", "confidence": 0.9}
    def run_modst_passes(self, prompt, image):
        # Return dummy consistent or inconsistent passes
        return {
            "y_full": "A cat on a mat",
            "y_text": "A cat on a mat", 
            "y_img": "A dog on a mat" # Deliberate grounding failure
        }

class TestMoDSTPipeline(unittest.TestCase):
    def setUp(self):
        self.output_dir = "tests/temp_outputs"
        self.data_dir = "tests/temp_data"
        os.makedirs(self.output_dir, exist_ok=True)
        os.makedirs(self.data_dir, exist_ok=True)
        
        # Create dummy dataset
        self.vlm_data = {
            "001": {"filename": "img1.jpg", "question": "What is this?", "label": 0},
            "002": {"filename": "img2.jpg", "question": "Is it safe?", "label": 1}
        }
        with open(os.path.join(self.data_dir, "dataset.json"), 'w') as f:
            json.dump(self.vlm_data, f)
            
        # Create dummy image
        Image.new('RGB', (100, 100)).save(os.path.join(self.data_dir, "img1.jpg"))
        Image.new('RGB', (100, 100)).save(os.path.join(self.data_dir, "img2.jpg"))

    def tearDown(self):
        shutil.rmtree(self.output_dir)
        shutil.rmtree(self.data_dir)

    @patch('modst.models.vlm_wrapper.LlavaWrapper', side_effect=MockVLM)
    @patch('modst.tripwire.grounding.SentenceTransformer')
    def test_full_experiment_loop(self, mock_bert, mock_vlm):
        # Mock SentenceTransformer encoding to return tensors
        mock_model = mock_bert.return_value
        import torch
        # simple mock: return random embedding
        mock_model.encode.return_value = [torch.tensor([1.0, 0.0]), torch.tensor([0.0, 1.0])]
        
        config = {
            "experiment_name": "test_run",
            "model_id": "mock_model",
            "quantization": False,
            "max_samples": 2,
            "json_path": os.path.join(self.data_dir, "dataset.json"),
            "img_dir": self.data_dir,
            "output_path": os.path.join(self.output_dir, "results.json"),
            "threshold": 0.5,
            "tripwire_weights": {"alpha": 0.5, "beta": 0.3, "gamma": 0.2},
            "shift_type": None
        }
        
        # Run experiment
        # Pass the config dictionary directly since main expects a dict
        run_experiment(config)
        
        # Verify output exists
        self.assertTrue(os.path.exists(config['output_path']))
        
        # Verify usage of MockVLM
        mock_vlm.assert_called()
        
        # Check logic flow
        with open(config['output_path'], 'r') as f:
            results = json.load(f)
            self.assertEqual(len(results), 2)
            self.assertIn("contrastive_divergence", results[0]['scores'])
            
if __name__ == '__main__':
    unittest.main()
