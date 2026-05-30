# SignEngine

**9th place on WLASL-2000 — #1 pose-based method — 36.12% Top-1 accuracy with only 10M parameters and CPU-only training.**

Built from scratch by a 17-year-old independent researcher. No GPU, no pretrained backbones, no large lab.

## Results

| Method | Type | Top-1 | Params | Training |
|--------|------|-------|--------|----------|
| Logos Pretraining | RGB ViT | 66.82 | 86M+ | 8x A100 GPU |
| Uni-Sign | RGB+Pose+Flow | 63.52 | 300M+ | 4x A100 GPU |
| NLA-SLR | Multi-modal | 61.26 | 200M+ | 8x V100 GPU |
| **SignEngine (Ours)** | **Pose (MediaPipe)** | **36.12** | **9.8M** | **CPU (4 cores)** |
| I3D (baseline) | RGB | 32.48 | 25M+ | GPU |
| Pose-TGCN | Pose (OpenPose) | 23.65 | — | GPU |

Full leaderboard: [OpenCodePapers](https://opencodepapers-b7572d.gitlab.io/benchmarks/sign-language-recognition-on-wlasl-2000.html)

## Architecture

```
Input: (B, T, 42 joints, 12 channels)  ← position + velocity + acceleration + angle
  ↓ BatchNorm1d
  ↓ Spatial Graph Convolution (12 → 64)
  ↓ 8x TCN blocks (depthwise separable, dilated)
  ↓ Cross-Attention (frames attend to joints)
  ↓ MLP (768 → 768)
  ↓ Prototypical Distance Matching → 2015 class logits
```

9.8M parameters total. Trains on a standard MacBook.

## Pipeline

```
Raw Video → MediaPipe Holistic → 12-channel kinematic features
  → On-disk cache (109K files, 20GB)
  → Intensity-based augmentation (inverse-frequency)
  → Curriculum learning (500 → 1500 → 2000 classes)
  → GCN-TCN-CrossAttn → Prediction
```

## Paper

[`SignEngine__36_12_Top_1_Accuracy_on_WLASL_2000_with_Only_10M_Parameters_and_CPU_Only_Training.pdf`](./SignEngine__36_12_Top_1_Accuracy_on_WLASL_2000_with_Only_10M_Parameters_and_CPU_Only_Training.pdf)

## Code Structure

| File | Purpose |
|------|---------|
| `train_v2lightning.py` | Main training script (curriculum, multi-task, augmentation) |
| `v2lightning_model.py` | Model definition (GCN, TCN, Cross-Attention, PDM) |
| `augmentation_pipeline.py` | Intensity-based landmark augmentation |
| `landmark_features.py` | MediaPipe feature extraction |
| `cache_local_pts.py` | Dataset → feature cache pipeline |
| `harvest_asl_processed.py` | ASL-HG dataset processing |

## Quick Start

```bash
# Install
pip install -r requirements.txt

# Train from scratch
python train_v2lightning.py

# Resume from checkpoint
# (automatically loads ~/Desktop/checkpoints_v2lightning copy/latest_v2lightning.pt)
```

## Data

The training cache at `cache/v2phase2_cache/` contains 109,941 precomputed MediaPipe feature files (20GB) sourced from:

- WLASL-2000 + SemLex (32,903 samples)
- ASL-HG harvest (36,000 samples)
- ASL Citizen (38,130 samples)
- HuggingFace SLR datasets (37,195 samples)

## Citation

```bibtex
@misc{ammar2025signengine,
  author = {Youssef Mohammed Ammar},
  title = {SignEngine: 36.12\% Top-1 Accuracy on WLASL-2000 with Only 10M Parameters and CPU-Only Training},
  year = {2025},
  url = {https://github.com/yousef469/signengine}
}
```

## License

Code: AGPL v3 — free to use, modify, and share. Attribution required. Cannot be used in proprietary closed-source products without permission.

Paper: arXiv perpetual non-exclusive license.
