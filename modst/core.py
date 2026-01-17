import torch
import torch.nn.functional as F
from typing import List, Dict, Any, Optional, Tuple
from sentence_transformers import SentenceTransformer, util

class MoDSTCore:
    """
    Modality-Dropout Safety Tripwires (MoDST) Core Logic.
    Implements the three-pass inference and Tripwire Score calculation.
    """
    def __init__(self, sim_model_name: str = "all-MiniLM-L6-v2"):
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.sim_model = SentenceTransformer(sim_model_name).to(self.device)

    def calculate_similarity(self, text1: str, text2: str) -> float:
        """Calculates cosine similarity between two strings using embeddings."""
        if not text1.strip() or not text2.strip():
            return 0.0
        embeddings = self.sim_model.encode([text1, text2], convert_to_tensor=True)
        return util.cos_sim(embeddings[0], embeddings[1]).item()

    def compute_tripwire_score(self, y_full: str, y_text: str, y_img: str) -> Dict[str, float]:
        """
        Computes the Tripwire Score (S) based on semantic discrepancies.
        S = 1 - (sim(y_full, y_img) + sim(y_full, y_text)) / 2
        """
        sim_img = self.calculate_similarity(y_full, y_img)
        sim_text = self.calculate_similarity(y_full, y_text)
        
        # Grounding Score: How much y_full depends on the image vs text priors
        # Safety Score: Higher discrepancy (lower similarity) indicates potential fusion failure
        score = 1.0 - (sim_img + sim_text) / 2.0
        
        return {
            "score": score,
            "sim_img": sim_img,
            "sim_text": sim_text
        }

    def safety_policy(self, score: float, threshold: float = 0.5) -> bool:
        """Returns True if the response is deemed 'safe'/grounded."""
        return score < threshold
