# Diabetic Foot Ulcer Segmentation Pipeline

A comprehensive pipeline for automated segmentation of diabetic foot ulcers using SAM (Segment Anything Model), OWL-ViT object detection, and GPT-4o vision capabilities.

## Features

- **SAM Auto Mode**: Fully automated segmentation using SAM's automatic mask generation
- **Hybrid Mode**: GPT-4o + OWL-ViT + SAM pipeline with intelligent prompt generation
- **Fallback Ensemble**: Automatic refinement using multiple prompts when initial segmentation is poor
- **Comprehensive Evaluation**: Dice, IoU, and Hausdorff distance metrics
- **Visualization Tools**: Generate plots, comparisons, and HTML reports

## Project Structure

```
├── __init__.py                 # Package initializer
├── config.py                   # Configuration and constants
├── main.py                     # Main entry point
├── logger.py                   # Logging configuration
│
├── sam_segmenter.py            # SAM wrapper class
├── owlvit_detector.py          # OWL-ViT object detector
├── gpt_prompter.py             # GPT-4o prompt generation
├── fallback_ensemble.py        # Ensemble refinement
├── evaluator.py                # Evaluation metrics
├── utils.py                    # Utility functions
│
├── sam_auto_eval.py            # SAM auto evaluation script
├── sam_hybrid_eval.py          # Hybrid evaluation script
│
├── requirements.txt            # Python dependencies
└── README.md                   # This file
```

## Installation

### 1. Clone Repository

```bash
git clone <your-repo-url>
cd diabetic-foot-ulcer-segmentation
```

### 2. Install Dependencies

```bash
pip install -r requirements.txt
```

### 3. Download SAM Checkpoint

Download the SAM ViT-B checkpoint from the [official repository](https://github.com/facebookresearch/segment-anything#model-checkpoints):

```bash
wget https://dl.fbaipublicfiles.com/segment_anything/sam_vit_b_01ec64.pth -O /content/sam_vit_b.pth
```

Or set custom path via environment variable:

```bash
export SAM_CHECKPOINT=/path/to/sam_vit_b.pth
```

### 4. Set OpenAI API Key (for Hybrid mode)

```bash
export OPENAI_API_KEY=your-api-key-here
```

## Usage

### Check Configuration

```bash
python main.py config
```

### SAM Auto Mode

Run automatic segmentation without object detection:

```bash
python main.py sam_auto \
    --image_dir ./data/images \
    --mask_dir ./data/masks \
    --output_dir ./results_sam_auto
```

### Hybrid Mode

Run hybrid pipeline with GPT-4o + OWL-ViT + SAM:

```bash
python main.py hybrid \
    --image_dir ./data/images \
    --mask_dir ./data/masks \
    --output_dir ./results_hybrid \
    --max_attempts 3
```

### Compare Results

Compare SAM Auto vs Hybrid methods:

```bash
python main.py compare \
    --sam_results ./results_sam_auto/sam_auto_results.csv \
    --hybrid_results ./results_hybrid/hybrid_results.csv
```

## Configuration

Edit `config.py` to customize:

- **Model paths**: SAM checkpoint location
- **Prompts**: Medical terminology for object detection
- **Thresholds**: Dice refinement and success thresholds
- **Device**: CUDA/CPU selection

Key parameters:

```python
DICE_REFINEMENT_THRESHOLD = 0.2  # Trigger ensemble when Dice < 0.2
DICE_SUCCESS_THRESHOLD = 0.1     # Consider successful when Dice >= 0.1
OWL_CONFIDENCE_THRESHOLD = 0.01  # OWL-ViT detection threshold
```

## Output

### CSV Results

Both modes generate CSV files with:

- **Image**: Filename
- **Dice**: Dice coefficient score
- **IoU**: Intersection over Union
- **Hausdorff**: Hausdorff distance
- **Status**: OK (success) or FN (false negative)

Hybrid mode additionally includes:

- **Prompt**: GPT-4o generated description
- **WoundPlurality**: singular/plural detection
- **NumBoxes**: Number of detected bounding boxes
- **UsedEnsemble**: Whether fallback ensemble was used

### Predicted Masks

Predicted masks are saved as `*_mask.png` in the output directory.

## Pipeline Details

### SAM Auto Pipeline

1. Load image and ground truth mask
2. Run SAM automatic mask generation
3. Combine all generated masks
4. Evaluate against ground truth

### Hybrid Pipeline

1. Load image and ground truth mask
2. Generate wound description with GPT-4o
3. Detect wound region with OWL-ViT using generated prompt
4. Segment using SAM with detected bounding boxes
5. If Dice < threshold, run fallback ensemble with alternative prompts
6. Select best result across all attempts

### Fallback Ensemble

When initial segmentation quality is low (Dice < 0.2):

1. Try each fallback prompt from medical terminology list
2. Run full detection + segmentation pipeline for each
3. Keep best result if improvement achieved

## Evaluation Metrics

- **Dice Coefficient**: Measures overlap between prediction and ground truth
- **IoU (Intersection over Union)**: Alternative overlap metric
- **Hausdorff Distance**: Maximum distance between mask boundaries

## Troubleshooting

### CUDA Out of Memory

Reduce batch size or use CPU:

```bash
export CUDA_VISIBLE_DEVICES=""  # Force CPU
```

### SAM Checkpoint Not Found

Set correct path:

```bash
export SAM_CHECKPOINT=/correct/path/to/sam_vit_b.pth
```

### OpenAI API Errors

Check API key and rate limits:

```bash
echo $OPENAI_API_KEY
```

### Poor Segmentation Results

Try adjusting thresholds in `config.py`:

- Lower `OWL_CONFIDENCE_THRESHOLD` for more detections
- Adjust `DICE_REFINEMENT_THRESHOLD