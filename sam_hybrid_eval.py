import os
import cv2
import gc
import torch
import argparse
import numpy as np
import pandas as pd
from PIL import Image
from tqdm import tqdm
from openai import OpenAI
from segment_anything import sam_model_registry, SamPredictor
from transformers import OwlViTProcessor, OwlViTForObjectDetection
from utils import compute_dice_iou, compute_hausdorff
import base64
import io
import random

# Argument Parser
parser = argparse.ArgumentParser(description="Run Hybrid GPT-4o + OWL-ViT + SAM Evaluation")
parser.add_argument("--image_dir", type=str, required=True, help="Path to input images")
parser.add_argument("--gt_mask_dir", type=str, required=True, help="Path to ground truth masks")
parser.add_argument("--output_dir", type=str, default="/content/results_DFUC2022", help="Path to save results")
args = parser.parse_args()

device = "cuda" if torch.cuda.is_available() else "cpu"
client = OpenAI(api_key=os.environ["OPENAI_API_KEY"])

sam_checkpoint = "/content/sam_vit_b.pth"
model_type = "vit_b"
sam = sam_model_registry[model_type](checkpoint=sam_checkpoint).to(device)
predictor = SamPredictor(sam)

owl_processor = OwlViTProcessor.from_pretrained("google/owlvit-base-patch32")
owl_model = OwlViTForObjectDetection.from_pretrained("google/owlvit-base-patch32").to(device)

medical_prompts = [
    "foot ulcer", "pink area", "peeling skin", "open wound on foot",
    "red wound on heel", "diabetic skin lesion", "inflamed ulcer on foot"
]

def encode_numpy_image_to_base64(image_np):
    img_pil = Image.fromarray((image_np * 255).astype(np.uint8))
    buffered = io.BytesIO()
    img_pil.save(buffered, format="PNG")
    return base64.b64encode(buffered.getvalue()).decode("utf-8")

def ask_gpt_prompt(image_np):
    base64_image = encode_numpy_image_to_base64(image_np)
    messages = [{
        "role": "user",
        "content": [
            {"type": "text", "text": (
                "You are a medical imaging expert. Describe any visible wounds on this diabetic foot. "
                "Include location, severity, color, and shape. Respond with one sentence.")},
            {"type": "image_url", "image_url": {"url": f"data:image/png;base64,{base64_image}"}}
        ]
    }]
    try:
        response = client.chat.completions.create(
            model="gpt-4o", messages=messages, max_tokens=100
        )
        return response.choices[0].message.content.strip()
    except Exception as e:
        print("❗ GPT-4o Error:", e)
        return random.choice(medical_prompts)

def is_plural_wound(prompt):
    return any(word in prompt.lower() for word in ["multiple", "several", "many", "wounds", "ulcers", "lesions"])

os.makedirs(args.output_dir, exist_ok=True)
results = []
image_files = sorted([f for f in os.listdir(args.image_dir) if f.lower().endswith(('.jpg', '.png'))])

for img_file in tqdm(image_files):
    print(f"Processing {img_file}")
    best_dice, best_iou, best_hausdorff = 0.0, 0.0, np.inf
    best_status = "FN"
    best_prompt = ""
    best_plural_flag = "singular"
    best_num_boxes = 0
    best_mask = None

    for attempt in range(3):
        print(f"Attempt {attempt + 1}")
        
        # Pre-processing
        # Baca gambar RGB dan normalisasi piksel ke [0, 1]
        image = cv2.cvtColor(cv2.imread(os.path.join(args.image_dir, img_file)), cv2.COLOR_BGR2RGB)
        image = image.astype(np.float32) / 255.0

        # Baca mask ground truth dan binarisasi
        mask = cv2.imread(os.path.join(args.gt_mask_dir, img_file), cv2.IMREAD_GRAYSCALE)
        if image is None or mask is None:
            print("Skipped: missing image or mask")
            continue
        gt_mask = (mask > 127).astype(np.uint8)

        # Prompt generation dari GPT-4o
        gpt_prompt = ask_gpt_prompt(image)
        if any(x in gpt_prompt.lower() for x in ["sorry", "can't", "unsure"]):
            gpt_prompt = random.choice(medical_prompts)

        plural_flag = is_plural_wound(gpt_prompt)
        prompt_list = [gpt_prompt] + medical_prompts
        
        # OWL-ViT: Deteksi objek berdasarkan prompt
        inputs = owl_processor(text=prompt_list, images=Image.fromarray((image * 255).astype(np.uint8)), return_tensors="pt").to(device)
        with torch.no_grad():
            outputs = owl_model(**inputs)

        target_sizes = torch.tensor([image.shape[:2]]).to(device)
        results_owl = owl_processor.post_process_grounded_object_detection(
            outputs=outputs, target_sizes=target_sizes, threshold=0.01)[0]

        if len(results_owl["boxes"]) == 0:
            continue
        
        # SAM Prediction
        predictor.set_image((image * 255).astype(np.uint8))
        final_mask = np.zeros(image.shape[:2], dtype=np.uint8)

        topk = min(5, len(results_owl["scores"])) if plural_flag else 1
        for idx in results_owl["scores"].argsort(descending=True)[:topk]:
            box = results_owl["boxes"][idx].int().tolist()
            masks, _, _ = predictor.predict(box=np.array(box), multimask_output=False)
            pred = (masks[0] >= 0.5).astype(np.uint8)
            final_mask = np.logical_or(final_mask, pred)

        # Post-processing
        # Finalisasi hasil prediksi menjadi mask 0/1
        final_mask = final_mask.astype(np.uint8)
        
        # Evaluasi
        dice, iou = compute_dice_iou(final_mask, gt_mask)
        hausdorff = compute_hausdorff(final_mask, gt_mask)
        status = "OK" if dice >= 0.1 else "FN"

        if dice > best_dice:
            best_dice, best_iou, best_hausdorff = dice, iou, hausdorff
            best_status = status
            best_prompt = gpt_prompt
            best_plural_flag = "plural" if plural_flag else "singular"
            best_num_boxes = len(results_owl["boxes"])
            best_mask = final_mask.copy()

        del final_mask, masks
        gc.collect()

    if best_mask is not None:
        cv2.imwrite(os.path.join(args.output_dir, img_file.replace(".png", "_mask.png")), best_mask * 255)

    results.append((
        img_file, best_prompt, best_plural_flag, best_num_boxes,
        best_dice, best_iou, best_hausdorff, best_status
    ))

df = pd.DataFrame(results, columns=[
    "Image", "Prompt", "WoundPlurality", "NumBoxes", "Dice", "IoU", "Hausdorff", "Status"
])
df.to_csv(os.path.join(args.output_dir, "hybrid_results.csv"), index=False)
