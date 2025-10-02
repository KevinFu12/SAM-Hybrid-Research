import os
import cv2
import gc
import argparse
import numpy as np
import pandas as pd
from PIL import Image
from tqdm import tqdm

from config import *
from gpt_prompter import ask_gpt_prompt, is_plural_wound, validate_gpt_response
from owlvit_detector import OwlViTDetector
from sam_segmenter import SAMSegmenter
from fallback_ensemble import FallbackEnsemble
from evaluator import Evaluator
from utils import load_image_mask

def main():
    parser = argparse.ArgumentParser(description="Run Hybrid GPT-4o + OWL-ViT + SAM Evaluation")
    parser.add_argument("--image_dir", type=str, required=True, help="Path to input images")
    parser.add_argument("--gt_mask_dir", type=str, required=True, help="Path to ground truth masks")
    parser.add_argument("--output_dir", type=str, default="./results_hybrid", help="Path to save results")
    parser.add_argument("--max_attempts", type=int, default=3, help="Max attempts per image with GPT-4o")
    args = parser.parse_args()

    # Initialize components
    print("Initializing models...")
    owl_detector = OwlViTDetector()
    sam_segmenter = SAMSegmenter()
    fallback_ensemble = FallbackEnsemble(owl_detector, sam_segmenter)
    evaluator = Evaluator()

    os.makedirs(args.output_dir, exist_ok=True)
    results = []
    image_files = sorted([f for f in os.listdir(args.image_dir) if f.lower().endswith(('.jpg', '.png'))])

    if len(image_files) == 0:
        print(f"No images found in {args.image_dir}")
        return

    print(f"Processing {len(image_files)} images with up to {args.max_attempts} attempts each...\n")

    for img_file in tqdm(image_files, desc="Hybrid Segmentation"):
        best_dice, best_iou, best_hausdorff = 0.0, 0.0, np.inf
        best_status = "FN"
        best_prompt = ""
        best_plural_flag = "singular"
        best_num_boxes = 0
        best_mask = None
        used_ensemble = False

        for attempt in range(args.max_attempts):
            try:
                # Load image and mask
                image_path = os.path.join(args.image_dir, img_file)
                mask_path = os.path.join(args.gt_mask_dir, img_file)
                
                if not os.path.exists(mask_path):
                    print(f"Warning: Ground truth mask not found for {img_file}, skipping...")
                    break
                
                image_np, gt_mask = load_image_mask(image_path, mask_path)
                
                # GPT-4o prompt generation
                gpt_prompt = ask_gpt_prompt(image_np)
                gpt_prompt = validate_gpt_response(gpt_prompt)
                plural_flag = is_plural_wound(gpt_prompt)
                
                # Prepare prompts for OWL-ViT
                prompts = owl_detector.prepare_prompts(gpt_prompt, MEDICAL_PROMPTS)
                image_pil = Image.fromarray((image_np * 255).astype(np.uint8))
                
                # Object detection with OWL-ViT
                results_owl = owl_detector.detect_objects(image_pil, prompts)
                
                # SAM segmentation
                sam_segmenter.set_image((image_np * 255).astype(np.uint8))
                
                if len(results_owl["boxes"]) > 0:
                    # Pass both boxes and scores to SAM
                    final_mask = sam_segmenter.predict_from_boxes(
                        results_owl["boxes"],
                        results_owl["scores"],
                        plural_flag
                    )
                else:
                    final_mask = sam_segmenter.auto_mask()
                
                # Initial evaluation
                dice, iou, hausdorff, status = evaluator.evaluate_segmentation(final_mask, gt_mask)
                
                # Apply refinement if needed
                ensemble_used = False
                if dice < DICE_REFINEMENT_THRESHOLD:
                    ensemble_mask, ensemble_dice, ensemble_iou, ensemble_hausdorff = (
                        fallback_ensemble.refine_segmentation(
                            image_pil, image_np, dice, gt_mask, evaluator
                        )
                    )
                    
                    if ensemble_mask is not None and ensemble_dice > dice:
                        final_mask = ensemble_mask
                        dice, iou, hausdorff = ensemble_dice, ensemble_iou, ensemble_hausdorff
                        status = "OK" if dice >= DICE_SUCCESS_THRESHOLD else "FN"
                        ensemble_used = True

                # Update best results
                if dice > best_dice:
                    best_dice, best_iou, best_hausdorff = dice, iou, hausdorff
                    best_status = status
                    best_prompt = gpt_prompt
                    best_plural_flag = "plural" if plural_flag else "singular"
                    best_num_boxes = len(results_owl["boxes"])
                    best_mask = final_mask.copy()
                    used_ensemble = ensemble_used

            except Exception as e:
                print(f"Error in attempt {attempt + 1} for {img_file}: {e}")
                continue
            
            finally:
                gc.collect()

        # Save best mask
        if best_mask is not None:
            output_path = os.path.join(args.output_dir, img_file.replace(".jpg", "_mask.png").replace(".png", "_mask.png"))
            cv2.imwrite(output_path, best_mask * 255)
        else:
            print(f"Warning: No valid mask generated for {img_file}")
            best_dice, best_iou, best_hausdorff = 0.0, 0.0, np.inf
            best_status = "ERROR"

        results.append({
            "Image": img_file,
            "Prompt": best_prompt,
            "WoundPlurality": best_plural_flag,
            "NumBoxes": best_num_boxes,
            "Dice": best_dice,
            "IoU": best_iou,
            "Hausdorff": best_hausdorff,
            "Status": best_status,
            "UsedEnsemble": used_ensemble
        })

    # Save results
    df = pd.DataFrame(results)
    csv_path = os.path.join(args.output_dir, "hybrid_results.csv")
    df.to_csv(csv_path, index=False)

    print(f"\n{'='*60}")
    print(f"Hybrid Evaluation Complete")
    print(f"{'='*60}")
    print(f"Results saved to: {args.output_dir}")
    print(f"Total images processed: {len(df)}")
    print(f"Average Dice: {df['Dice'].mean():.3f}")
    print(f"Average IoU: {df['IoU'].mean():.3f}")
    print(f"Average Hausdorff: {df['Hausdorff'].replace([np.inf], np.nan).mean():.2f}")
    print(f"Successful segmentations (Dice >= {DICE_SUCCESS_THRESHOLD}): {len(df[df['Status'] == 'OK'])}/{len(df)}")
    print(f"Ensemble methods used: {df['UsedEnsemble'].sum()} images")
    print(f"{'='*60}\n")

if __name__ == "__main__":
    main()