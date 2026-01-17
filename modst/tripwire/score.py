from .grounding import GroundingTripwire
from .grounding import FusionTripwire # Using same file for brevity in this step
from .grounding import SafetyTripwire

class MoDSTScore:
    """
    Aggregates multiple tripwires into a PRINCIPLED score.
    Logic: A failure is likely if y_full COLLAPSES to text-priors (y_text) 
    while DISAGREEING with visual evidence (y_img).
    """
    def __init__(self, alpha=0.5, beta=0.3, gamma=0.2):
        self.grounding = GroundingTripwire()
        self.fusion = FusionTripwire()
        self.safety = SafetyTripwire()
        self.weights = {"g": alpha, "f": beta, "s": gamma}

    def compute(self, passes: dict) -> dict:
        # s_g: Disagreement with image (0 = consistent, 1 = inconsistent)
        s_g = self.grounding.score(passes['y_full'], passes['y_img'])
        
        # s_f: Agreement with text prior (0 = independent, 1 = text-prior collapse)
        # Note: We invert this from the original implementation! 
        # Agreement with text-only is often a SIGN of hallucination in VLMs.
        s_f = 1.0 - self.fusion.score(passes['y_full'], passes['y_text'])
        
        # s_s: Safety transfer gap (Binary or continuous)
        s_s = self.safety.score(passes['y_full'], passes['y_text'])
        
        # Combined Score: Penalize high s_g AND high s_f (Visual disagreement + Text collapse)
        # This is the "Grounding Divergence" component
        divergence = s_g * s_f
        
        total_score = (self.weights['g'] * divergence + 
                       self.weights['f'] * s_g + 
                       self.weights['s'] * s_s)
        
        return {
            "total": total_score,
            "grounding_discrepancy": s_g,
            "text_prior_collapse": s_f,
            "safety_gap": s_s,
            "contrastive_divergence": divergence
        }
