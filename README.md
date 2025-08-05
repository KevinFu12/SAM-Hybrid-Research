# Segmentasi Luka Kaki Diabetik Multimodal Hybrid (Zero-Shot)

Repositori ini berisi implementasi dari kerangka kerja segmentasi luka kaki diabetik (DFU) menggunakan pendekatan zero-shot hybrid: **GPT-4o + OWL-ViT + SAM**.

## Modul
- `utils.py`: Fungsi utilitas umum (perhitungan Dice, IoU, Hausdorff)
- `sam_auto_eval.py`: Evaluasi segmentasi otomatis dengan Segment Anything Model (SAM)
- `sam_hybrid_eval.py`: Evaluasi segmentasi hybrid GPT-4o + OWL-ViT + SAM

## Dataset
Dataset: **DFUC2022 (Diabetic Foot Ulcer Challenge 2022)**  
Dataset digunakan dari HuggingFace (berlisensi non-komersial).  
**Catatan**: Dataset tidak disertakan dalam repositori ini. Silakan unduh langsung dari halaman resmi [DFUC2022].

## Instalasi

1. Clone repositori ini:

```bash
git clone https://github.com/nama-kamu/dfu-hybrid-segmentation.git
cd dfu-hybrid-segmentation
```

2. Buat environment dan install dependensi:

```bash
pip install -r requirements.txt
```

3. Siapkan file model SAM `(sam_vit_b.pth)` dan letakkan di folder kerja atau `/content/`

4. Set environment variable untuk OpenAI API:

```bash
export OPENAI_API_KEY=your_api_key_here
```

## Menjalankan Pipeline

Evaluasi SAM Otomatis
```bash
python sam_auto_eval.py --image_dir path/to/images --mask_dir path/to/ground_truths --output_csv hasil_sam.csv
```

Evaluasi Hybrid GPT-4o + OWL-ViT + SAM:
```bash
python sam_hybrid_eval.py --image_dir path/to/images --gt_mask_dir path/to/ground_truths --output_dir hasil_hybrid/
```

Jalankan dari Main Launcher:
```bash
python main.py sam_auto
# atau
python main.py hybrid
```

