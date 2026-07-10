# UMVP360

**Personalized Multi-Modal Viewport Prediction for 360° Video**

UMVP360 predicts where a viewer will look in a 360° video up to **5 seconds ahead**, using past head motion, dense visual saliency, and optional user/audio/face signals.
---

## Features


- **Dense saliency** — 16×32 SalViT360 token grid (512 tokens/frame)
- **Rich trajectory input** — head, eye gaze, offset, and fixation features
- **User personalization** — per-user embedding + FiLM modulation
- **Multi-modal gating** — optional audio (log-mel) and face engagement context
- **Behavioral prior** — cross-user view-frequency map as an extra saliency channel
- **Horizon-weighted loss** — emphasizes long-term prediction accuracy

---

## Project Structure

```
UMVP360/
├── train.py              # Training + ablation flags
├── evaluate.py           # OD & IoU metrics on test set
├── dataset.py            # Multi-modal sliding-window loader
├── models/               # MMVP2, STAR-VP, and sub-modules
├── preprocessing/        # Dense saliency, audio, behavioral prior
└── tests/
```

---

## Quick Start

### 1. Preprocess data

```bash
python -m MMVP2.preprocessing.prepare_mmvp2_data
```

Requires preprocessed trajectory data from MUSE-VP, SalViT360 saliency maps, and raw 360° videos (Wu MMSys 2017).

### 2. Train

```bash
python -m MMVP2.train \
    --use_aux --use_audio --use_face --use_user --use_user_film \
    --use_behav --horizon_weighted \
    --tag full --amp
```

### 3. Evaluate

```bash
python -m MMVP2.evaluate \
    --checkpoint runs/mmvp2_full_*/checkpoints/best.pt \
    --output eval.json
```

Evaluation metrics: **Orthodromic Distance** (geodesic error) and **IoU** (9×16 tile overlap, 100° FOV).

---

## Ablation Flags

| Flag | Enables |
|------|---------|
| `--use_aux` | 11D LSTM input (eye + offset + fixation) |
| `--use_behav` | Behavioral prior saliency channel |
| `--use_audio` | Audio context in gating |
| `--use_face` | Face engagement in gating |
| `--use_user` | User embedding for LSTM init |
| `--use_user_film` | FiLM modulation on spatial attention |
| `--horizon_weighted` | Long-horizon emphasis in loss |

---

## Requirements

- Python 3.10+, PyTorch 2.0+
- NumPy, tqdm
- ffmpeg (audio preprocessing)
- tensorboard (optional)

> **Note:** The Python package is imported as `MMVP2`. Place this folder inside the STAR-VP repo or adjust `PYTHONPATH` accordingly.

---

## Dataset

- **25 participants** (`P001`–`P025`), **14 videos**, **5 fps**
- Train/val/test split: **8:1:1 per participant** (same as STAR-VP / MUSE-VP)
- Default windows: `T_M=25` (past 5 s) → `T_H=25` (future 5 s)

---

## References

- STAR-VP — *Improving Long-term Viewport Prediction in 360° Videos* (ACM MM 2024)
- SalViT360 — Dense 360° visual saliency
- Wu MMSys 2017 — 360° viewport dataset
