import torch
from abc import ABC, abstractmethod
from PIL import Image
from transformers import AutoProcessor, LlavaForConditionalGeneration
from typing import Optional, Dict, Any

class AbstractVLM(ABC):
    """Abstract base class for VLMs in the MoDST experiment."""
    @abstractmethod
    def generate(self, prompt: str, image: Optional[Image.Image] = None) -> Dict[str, Any]:
        """Returns decoded text and optional metadata (logits, etc.)"""
        pass

    @abstractmethod
    def run_modst_passes(self, prompt: str, image: Image.Image) -> Dict[str, str]:
        """Runs FULL, TEXT_ONLY, and IMG_ONLY passes."""
        pass

class LlavaWrapper(AbstractVLM):
    """LLaVA-1.5 implementation with support for 4-bit quantization."""
    def __init__(self, model_id: str = "llava-hf/llava-1.5-7b-hf", load_in_4bit: bool = True, grounding_prompt: str = None):
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.model = LlavaForConditionalGeneration.from_pretrained(
            model_id,
            torch_dtype=torch.float16,
            low_cpu_mem_usage=True,
            device_map="auto",
            load_in_4bit=load_in_4bit
        )
        self.processor = AutoProcessor.from_pretrained(model_id)
        self.null_img = Image.new('RGB', (224, 224), color=(0, 0, 0))
        # Default prompt if none provided
        self.grounding_prompt = grounding_prompt if grounding_prompt else "USER: <image>\nList all visible objects, their attributes, and spatial relationships. ASSISTANT:"

    def generate(self, prompt: str, image: Optional[Image.Image] = None) -> Dict[str, Any]:
        inputs = self.processor(text=prompt, images=image, return_tensors="pt").to(self.device, torch.float16)
        with torch.no_grad():
            output = self.model.generate(**inputs, max_new_tokens=128, do_sample=False, return_dict_in_generate=True, output_scores=True)
        
        decoded = self.processor.decode(output.sequences[0], skip_special_tokens=True).split("ASSISTANT:")[-1].strip()
        # Mock confidence as max logit average for the baseline
        confidence = torch.stack(output.scores).max(dim=-1).values.mean().item() if output.scores else 0.0
        
        return {"text": decoded, "confidence": confidence}

    def run_modst_passes(self, prompt: str, image: Image.Image) -> Dict[str, str]:
        # Implementation of the three passes
        y_full = self.generate(prompt, image)["text"]
        y_text = self.generate(prompt, self.null_img)["text"]
        y_img = self.generate(self.grounding_prompt, image)["text"]
        
        return {
            "y_full": y_full,
            "y_text": y_text,
            "y_img": y_img
        }

class InstructBlipWrapper(AbstractVLM):
    """InstructBLIP implementation for generalization checks."""
    def __init__(self, model_id: str = "Salesforce/instructblip-vicuna-7b", load_in_4bit: bool = True, grounding_prompt: str = None):
        from transformers import InstructBlipProcessor, InstructBlipForConditionalGeneration
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.model = InstructBlipForConditionalGeneration.from_pretrained(
            model_id,
            torch_dtype=torch.float16,
            low_cpu_mem_usage=True,
            device_map="auto",
            load_in_4bit=load_in_4bit
        )
        self.processor = InstructBlipProcessor.from_pretrained(model_id)
        self.null_img = Image.new('RGB', (224, 224), color=(0, 0, 0))
        self.grounding_prompt = grounding_prompt if grounding_prompt else "Describe the image in detail."

    def generate(self, prompt: str, image: Optional[Image.Image] = None) -> Dict[str, Any]:
        # InstructBLIP requires image input, use null_img if None
        img_input = image if image else self.null_img
        inputs = self.processor(images=img_input, text=prompt, return_tensors="pt").to(self.device, torch.float16)
        
        with torch.no_grad():
            output = self.model.generate(
                **inputs,
                max_new_tokens=128,
                do_sample=False,
                return_dict_in_generate=True, 
                output_scores=True
            )
        
        decoded = self.processor.batch_decode(output.sequences, skip_special_tokens=True)[0].strip()
        confidence = torch.stack(output.scores).max(dim=-1).values.mean().item() if output.scores else 0.0
        return {"text": decoded, "confidence": confidence}

    def run_modst_passes(self, prompt: str, image: Image.Image) -> Dict[str, str]:
        # 1. Full
        y_full = self.generate(prompt, image)["text"]
        # 2. Text-Only (Modality Dropout) - Passing null image
        y_text = self.generate(prompt, self.null_img)["text"]
        # 3. Image-Only (MoDST Grounding)
        y_img = self.generate(self.grounding_prompt, image)["text"]
        
        return {"y_full": y_full, "y_text": y_text, "y_img": y_img}
