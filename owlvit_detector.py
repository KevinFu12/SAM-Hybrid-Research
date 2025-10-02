import torch
from transformers import OwlViTProcessor, OwlViTForObjectDetection
from config import device, OWL_MODEL_NAME, OWL_CONFIDENCE_THRESHOLD

class OwlViTDetector:
    def __init__(self):
        self.processor = OwlViTProcessor.from_pretrained(OWL_MODEL_NAME)
        self.model = OwlViTForObjectDetection.from_pretrained(OWL_MODEL_NAME).to(device)
        self.model.eval()
    
    def detect_objects(self, image, prompts):
        inputs = self.processor(
            text=prompts, 
            images=image, 
            return_tensors="pt"
        ).to(device)
        
        with torch.no_grad():
            outputs = self.model(**inputs)
        
        # Ensure target_sizes is on correct device
        target_sizes = torch.tensor([image.size[::-1]]).to(device)
        results = self.processor.post_process_grounded_object_detection(
            outputs=outputs, 
            target_sizes=target_sizes, 
            threshold=OWL_CONFIDENCE_THRESHOLD
        )[0]
        
        return results
    
    def prepare_prompts(self, gpt_prompt, medical_prompts):
        """Prepare prompt list for object detection"""
        return [gpt_prompt] + medical_prompts