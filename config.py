import os
import torch
import warnings

# Device configuration
device = "cuda" if torch.cuda.is_available() else "cpu"

# Model paths
SAM_CHECKPOINT = os.environ.get("SAM_CHECKPOINT", "/content/sam_vit_b.pth")
SAM_MODEL_TYPE = os.environ.get("SAM_MODEL_TYPE", "vit_b")
OWL_MODEL_NAME = "google/owlvit-base-patch32"

# Validate SAM checkpoint exists
if not os.path.exists(SAM_CHECKPOINT):
    warnings.warn(
        f"SAM checkpoint not found at {SAM_CHECKPOINT}. "
        f"Please download it from: https://github.com/facebookresearch/segment-anything#model-checkpoints"
    )

# Prompt sets
MEDICAL_PROMPTS = [
    "diabetic foot ulcer", "open wound", "skin lesion", "tissue damage",
    "necrosis", "red inflamed area", "yellow exudate", "black eschar",
    "skin breakdown", "moist wound", "dry gangrene", "wet gangrene",
    "callus with ulcer", "plantar ulcer", "heel pressure sore",
    "foot sore", "skin ulceration", "infected wound", "pus wound",
    "red swollen area", "skin opening", "flesh wound", "skin defect"
]

LOCATION_PROMPTS = [
    "ulcer on heel", "wound on sole", "lesion on toe", "sore on plantar surface",
    "wound on lateral foot", "ulcer on medial foot", "sore on forefoot"
]

FALLBACK_PROMPTS = MEDICAL_PROMPTS + LOCATION_PROMPTS

# Thresholds
DICE_REFINEMENT_THRESHOLD = 0.2  # Trigger ensemble refinement if Dice < this
DICE_SUCCESS_THRESHOLD = 0.1     # Consider segmentation successful if Dice >= this
OWL_CONFIDENCE_THRESHOLD = 0.01  # Minimum confidence for OWL-ViT detections

# Display configuration info
def print_config():
    """Print current configuration"""
    print("="*60)
    print("Configuration")
    print("="*60)
    print(f"Device: {device}")
    print(f"SAM Checkpoint: {SAM_CHECKPOINT}")
    print(f"SAM Model Type: {SAM_MODEL_TYPE}")
    print(f"OWL Model: {OWL_MODEL_NAME}")
    print(f"Dice Refinement Threshold: {DICE_REFINEMENT_THRESHOLD}")
    print(f"Dice Success Threshold: {DICE_SUCCESS_THRESHOLD}")
    print(f"OWL Confidence Threshold: {OWL_CONFIDENCE_THRESHOLD}")
    print(f"Medical Prompts: {len(MEDICAL_PROMPTS)}")
    print(f"Location Prompts: {len(LOCATION_PROMPTS)}")
    print(f"Total Fallback Prompts: {len(FALLBACK_PROMPTS)}")
    print("="*60)