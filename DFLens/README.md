# DFLens — Multimodal Deepfake Detection System

DFLens is a web-based deepfake detection tool that analyses images, videos and audio for AI manipulation. It uses CLIP ViT-B/32 fine-tuned on FaceForensics++ (C23) and CelebDF-v2 for visual detection, AudioCNN trained on the FoR dataset for voice deepfake detection, and ELA (Error Level Analysis) for image forgery detection.

## Features

- **Image detection**: CLIP ViT-B/32 with MTCNN face detection, dual-path consensus mechanism, Bayesian prior correction, Grad-CAM heatmap
- **Video detection**: 16-frame sampling with per-frame analysis and averaged verdict
- **Voice detection**: Three-tier fallback — AASIST3 (pretrained), AudioCNN, heuristic scoring
- **Image forgery detection**: ELA-based tampering analysis
- **PDF report generation**: Downloadable detection reports via ReportLab

## Project Structure

```
Deepfake_system/
├── app.py                                  # Flask application (all endpoints)
├── templates/
│   └── index.html                          # Web interface
├── static/
│   └── style.css
├── models/
│   ├── best_voice_model.pth                # AudioCNN weights (FoR dataset)
│   ├── clip_detector.pth                   # CLIP ViT-B/32 weights (FF++ + CelebDF-v2)
│   └── model_info.json                     # Training metadata (hyperparameters, AUC)
├── deepfake_combined_dataset_latest.ipynb   # Vision model training pipeline
├── deepfake_audio_training.ipynb            # Audio model training pipeline
├── requirements.txt
└── README.md
```

MTCNN and AASIST3 are downloaded automatically at runtime (via facenet-pytorch and HuggingFace respectively) and do not need to be placed manually.

## Setup

### 1. Install dependencies

Requires **Python 3.12** (tested on 3.12.11).

```bash
python -m venv venv
venv\Scripts\activate        # Windows
source venv/bin/activate     # macOS / Linux
pip install -r requirements.txt
```

FFmpeg is also required for video processing:
- Ubuntu/Debian: `sudo apt install ffmpeg`
- macOS: `brew install ffmpeg`
- Windows: download from https://ffmpeg.org/download.html

### 2. Place model weights

Put the following files in the `models/` directory:

| File | Description | Trained on |
|------|-------------|------------|
| `clip_detector.pth` | CLIP ViT-B/32 (87.5M params, 32.4% unfrozen) | FF++ C23 + CelebDF-v2 |
| `best_voice_model.pth` | AudioCNN | FoR for-2sec dataset |

### 3. Run the application

```bash
python app.py
```

Open http://127.0.0.1:5000 in a browser.

## Training

The training pipelines are documented in Jupyter notebooks:

- **Vision model**: `deepfake_combined_dataset_latest.ipynb` — CLIP ViT-B/32 fine-tuned for 8 epochs with AdamW (lr=2e-5), batch size 16. Best model saved at epoch 3 (val AUC 0.9219). Trained on Lightning AI with Tesla T4 GPU.
- **Audio model**: `deepfake_audio_training.ipynb` — AudioCNN trained for 10 epochs on FoR for-2sec dataset (13,956 train / 2,826 val / 1,088 test samples).

Random seeds are fixed for reproducibility.

## Datasets

Training datasets are not included due to size. Download from:

- **FaceForensics++ (C23):** https://www.kaggle.com/datasets/ahmedelbanby/faceforensicsplusplus-c23-deepfakebench-structure
- **CelebDF-v2:** https://www.kaggle.com/datasets/reubensuju/celeb-df-v2
- **FoR (Fake-or-Real):** https://www.kaggle.com/datasets/mohammedabdeldayem/the-fake-or-real-dataset

Note: Datasets are only required for retraining. The pretrained model weights (clip_detector.pth and best_voice_model.pth) are sufficient to run the detection application.

## Results

| Dataset | Accuracy | AUC-ROC |
|---------|----------|---------|
| FF++ validation | 85.68% | 0.9131 |
| CelebDF-v2 validation | 92.45% | 0.9820 |
| CelebDF-v2 full | 93.38% | 0.9865 |
| Balanced test set (400 images) | 93.5% | — |
| AudioCNN on FoR test set (1,088 samples) | 86.21% | 0.9380 |

## Technology Stack

- Python 3.12, PyTorch 2.6, TorchVision
- HuggingFace Transformers (CLIP ViT-B/32, AASIST3), timm
- Flask (web framework)
- NumPy, Pillow (image processing)
- OpenCV (video processing), MTCNN (face detection via facenet-pytorch)
- librosa, soundfile (audio loading and MFCC extraction)
- scikit-learn (evaluation metrics)
- pytorch-grad-cam (Grad-CAM explainability)
- ReportLab (PDF report generation)
- matplotlib, seaborn (training visualisation)

## Course Information

INT4203E Artificial Intelligence — Individual Assignment  
January 2026 Session