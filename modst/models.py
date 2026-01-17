import torch
from PIL import Image
from transformers import LlamaTokenizer, AutoProcessor, LlavaForConditionalGeneration
from typing import Optional, List, Union

class LlavaMoDSTWrapper:
    """
    Wrapper for LLaVA-1.5 to support MoDST inference passes.
    """
    def __init__(self, model_id: str = "llava-hf/llava-1.5-7b-hf"):
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.model = LlavaForConditionalGeneration.from_pretrained(
            model_id, 
            torch_dtype=torch.float16, 
            low_cpu_mem_usage=True,
            device_map="auto"
        )
        self.processor = AutoProcessor.from_pretrained(model_id)
        self.null_img = Image.new('RGB', (224, 224), color = (0, 0, 0)) # Placeholder black image
        self.grounding_prompt = "USER: <image>\nDescribe the main scene and objects in this image. ASSISTANT:"

    @torch.no_grad()
    def generate(self, prompt: str, image: Optional[Image.Image] = None) -> str:
        """Standard generation."""
        inputs = self.processor(text=prompt, images=image, return_tensors="pt").to(self.device, torch.float16)
        output = self.model.generate(**inputs, max_new_tokens=128, do_sample=False)
        return self.processor.decode(output[0], skip_special_tokens=True).split("ASSISTANT:")[-1].strip()

    def run_modst_passes(self, prompt: str, image: Image.Image) -> dict:
        """Executes the three MoDST passes."""
        # 1. Full Context Pass
        y_full = self.generate(prompt, image)
        
        # 2. Text-Only Pass (Modality Dropout: Image)
        # Note: We use the same prompt but a null image
        y_text = self.generate(prompt, self.null_img)
        
        # 3. Image-Only Pass (Modality Dropout: Text prompt)
        y_img = self.generate(self.grounding_prompt, image)
        
        return {
            "y_full": y_full,
            "y_text": y_text,
            "y_img": y_img
        }
