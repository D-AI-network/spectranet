# SpecTra-Net

**SpecTra-Net** is a PyTorch framework for spatiotemporal forecasting in complex network. It supports both **grid** (`T,H,W,C`) and **node** (`T,N,C`) inputs through a unified pipeline, and combines spectral-temporal conditioning with optimal-transport-based spatial alignment.

---

## 1. Key Components

1. **SFE (Spectral Feature Extraction)**  
   FFT-based frequency processing to stabilize/retain informative high-frequency patterns.

2. **T-Gating (Temporal Gating)**  
   Learns temporal importance weights and adaptively emphasizes change-sensitive time steps.

3. **GOT (Geometric Optimal Transport)**  
   Learns global spatial alignment via prototypes and transport plans using geometry (and optional embedding distance).

4. **MGAT (Multi-Graph Attention)**  
   Refines spatial representations on a KNN graph with anisotropic, scale-aware attention.

---

## 2. Features

- **Unified Input Support:** Grid (`T,H,W,C`) and Node (`T,N,C` or `T,N`) formats.
- **Safe Cropping:** Cropping is applied only for grid inputs; node inputs are automatically preserved.
- **Residual Forecasting:** Optional residual-to-last prediction (`y = last_input + delta`).
- **Metrics:** RMSE, MAE, R2.
- **Artifacts:** Saves best checkpoint + test predictions + JSON summary.

---

## 3. Requirements

- Python 3.8+
- PyTorch >= 1.10
- NumPy

**Optional (recommended):**
- CUDA-capable GPU for faster training

---

## 4. Installation

```bash
pip install -r requirements.txt


## 5. Data Format

You provide two .npy arrays:

TRAIN_NPY: training period (chronologically split into train/val = 9:1)

TEST_NPY: test period (held-out)

Supported raw shapes

Grid input

(T, H, W, C)

Node input

(T, N, C) or (T, N)

Internally, node inputs are standardized to:

(T, 1, N, C) or (T, 1, N, 1).

Forecasting setup (one-step ahead)

Given P_IN = P, sliding windows are created as:

X[i] = data[i : i+P]

y[i] = data[i+P]

So the windowed tensors become:

X: (T-P, P, H, W, C) (grid) or (T-P, P, 1, N, C) (node-view)

y: (T-P, H, W, C) (grid) or (T-P, 1, N, C) (node-view)

##6. Quick Start

Set file paths in the script:

cfg.TRAIN_NPY = "path/to/train.npy"
cfg.TEST_NPY  = "path/to/test.npy"


Choose channels (examples):

cfg.CHANNELS = [0]       # single channel
# cfg.CHANNELS = [0,1,2] # multi-channel
cfg.C = len(cfg.CHANNELS)


Select view mode:

cfg.NODE_MODE = True   # node view (recommended for (T,N,C) data)
# cfg.NODE_MODE = False # grid view (requires grid-shaped input)


(Optional) Provide coordinates for node datasets:

cfg.COORDS_NPY = None  # or "path/to/coords.npy" with shape (N,2)


Run:

python train.py

## 7. Configuration Notes

Important options in CFG:

Temporal

P_IN: input window length

USE_TGATING, TGATING_GAIN

USE_SFE, NUM_FREQUENCIES

PERIODS: periodic embedding periods (default [48, 168])

Spatial

USE_GOT, N_PROTO, GOT_EPS, GOT_TAU, GOT_USE_EMB_DIST

USE_MGAT, DLAK_K, MGAT_R0

Training

BATCH, EPOCHS, LR, WD

SEED, SAVE_DIR

## 8. Outputs

During training, the script saves the best checkpoint (by validation global RMSE) into:

SAVE_DIR/spectranet_<crop>_<channel>_NODE.pt (node mode)

SAVE_DIR/spectranet_<crop>_<channel>.pt (grid mode)

It also saves:

*_y_pred_test.npy: test predictions

*_y_true_test.npy: test ground truth

*_results.json: full config + metrics summary

## 9. Metrics

The evaluation reports:

RMSE

MAE

R²

