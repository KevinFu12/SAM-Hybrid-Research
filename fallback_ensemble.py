import numpy as np
from config import FALLBACK_PROMPTS, MEDICAL_PROMPTS

class FallbackEnsemble:
    """
    Ensemble refinement for segmentation.
    Tries fallback prompts to improve segmentation if initial Dice is low.
    """
    def __init__(self, owl_detector, sam_segmenter):
        self.owl_detector = owl_detector
        self.sam_segmenter = sam_segmenter

    def refine_segmentation(self, image_pil, image_np, initial_dice, gt_mask, evaluator):
        """
        Try fallback prompts to improve segmentation if Dice is low.
        
        Args:
            image_pil: PIL Image
            image_np: numpy array (normalized to [0, 1])
            initial_dice: Initial Dice score to beat
            gt_mask: Ground truth mask
            evaluator: Evaluator instance
            
        Returns:
            tuple: (best_mask, best_dice, best_iou, best_hausdorff) if improved, 
                   else (None, initial_dice, 0.0, np.inf)
        """
        best_mask = None
        best_dice = initial_dice
        best_iou = 0.0
        best_hausdorff = np.inf
        
        successful_prompts = 0
        failed_prompts = 0

        # Try each fallback prompt
        for prompt in FALLBACK_PROMPTS:
            try:
                # Prepare prompts for OWL-ViT
                prompts = self.owl_detector.prepare_prompts(prompt, MEDICAL_PROMPTS)
                
                # Detect objects
                results_owl = self.owl_detector.detect_objects(image_pil, prompts)
                
                # Check if detection returned valid results
                if results_owl is None or "boxes" not in results_owl:
                    failed_prompts += 1
                    continue
                
                # Set image for SAM
                self.sam_segmenter.set_image((image_np * 255).astype(np.uint8))
                
                # Generate mask
                if len(results_owl["boxes"]) > 0:
                    mask = self.sam_segmenter.predict_from_boxes(
                        results_owl["boxes"],
                        results_owl.get("scores", None),
                        plural_flag=False
                    )
                else:
                    mask = self.sam_segmenter.auto_mask()
                
                # Evaluate
                dice, iou, hausdorff = evaluator.evaluate_segmentation(mask, gt_mask)[:3]
                
                # Update best if improved
                if dice > best_dice:
                    best_dice = dice
                    best_iou = iou
                    best_hausdorff = hausdorff
                    best_mask = mask.copy()
                    successful_prompts += 1
                    
            except Exception as e:
                failed_prompts += 1
                continue

        # Return results
        if best_mask is not None and best_dice > initial_dice:
            return best_mask, best_dice, best_iou, best_hausdorff
        else:
            return None, initial_dice, 0.0, np.inf