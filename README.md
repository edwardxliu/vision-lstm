# VisionLSTM / ViT Ablation Training

This repository provides a unified training and evaluation harness for VisionLSTM (ViL) and ViT on ImageNet-1K, Tiny-ImageNet, and Tiny-ImageNet-C, including:

- A single DDP-capable training script
- ViL and ViT backbones with PSWF / wavelet components
- A set of ablations (A0/A1/A2/A3, W3, W3_POOL_ONLY, W3_TOKENONLY, W3_RESIDUALONLY, W3_IMPROVED_WARMUP, etc.)
- Robustness evaluation on Tiny-ImageNet-C / ImageNet-C

---

## 1. Installation

```bash
pip install -r requirements.txt
```

Install PyTorch + CUDA from the official website according to your hardware. The `requirements.txt` only pins minimal versions for Python-side libraries.

---

## 2. Code structure

- `model_vil.py`  
  VisionLSTM backbone (`VisionLSTM2`) and related components:
  - `ViLBlock`, `ViLBlockPair`, `MatrixLSTMCell`, `FeatureExtractor`
  - Wavelet / Haar DWT modules, wavelet residual gate
  - Patch embedding and positional embedding helpers

- `model_builder.py`  
  - `get_ablation_cfg(ablation_id)` – maps ablation IDs (A0/A1/A2/A3/W3/...) to model config toggles.
  - `build_model_from_env(num_classes, img_size)` – constructs a ViL or ViT model according to environment variables (e.g. `MODEL_KIND`, `ABLATION`, `DIM`, `DEPTH`).

- `env_ddp.py`  
  - Env helpers: `env_int`, `env_float`, `env_str`, `env_bool`, `env_list_int`
  - DDP utilities: `is_dist`, `get_rank`, `get_world_size`, `is_main_process`, `ddp_print`
  - Reproducibility: `set_global_seed`
  - YAML config loader: `load_yaml_config_if_present()`  
    Reads `CONFIG` or `CFG` (YAML file) and populates `os.environ` with defaults (without overriding explicitly set env vars).

- `data_loader.py`  
  - Datasets:
    - `TinyImageNetCDataset` – Tiny-ImageNet-C layout
    - `PathLabelDataset` – minimal (path, label) dataset
    - `load_tiny_imagenet(root, split, transform)`
  - Class subset / per-class caps:
    - `select_subset_classes`
    - `filter_samples`
  - High-level builder:
    - `build_datasets_and_loaders(dataset_name, data_root, img_size, per_gpu_batch, num_workers, data_seed, device)`  
      Returns `(train_loader, val_loader, train_sampler, num_classes, info_dict)`.

- `train_helpers.py`  
  - Mixup / CutMix:
    - `mixup_cutmix`
  - Soft label loss:
    - `soft_cross_entropy`
  - EMA:
    - `create_ema_model`, `update_ema`
  - Branch alpha schedule:
    - `get_branch_alpha`, `try_set_head_alpha`

- `eval_helpers.py`  
  - Standard validation: `evaluate(model, loader, device, amp_dtype)`
  - Corruption eval: `evaluate_imagenet_c(model, imagenetc_root, img_size, device, amp_dtype, batch_size, num_workers, dataset_name, data_root)`

- `logging_helpers.py`  
  - JSON / JSONL: `save_json`, `append_jsonl`
  - Curves: `plot_metrics(metrics_jsonl, out_dir)`

- `train_ablation_ddp.py`  
  Main entry point for training and evaluation. It:
  - Loads defaults from YAML via `CONFIG` / `CFG` (using `env_ddp.load_yaml_config_if_present`)
  - Reads training and model hyperparameters from env
  - Builds datasets and loaders via `build_datasets_and_loaders`
  - Builds ViL or ViT via `build_model_from_env`
  - Supports:
    - `MODE=train` – training with EMA, mixup, JSONL logging, and learning-rate scheduling
    - `MODE=eval` – single validation pass on the validation set
    - `MODE=eval_imagenetc` – evaluation on ImageNet-C / Tiny-ImageNet-C

- `configs/` (YAML hyperparameter presets)
  - Tiny-ImageNet (regularized):
    - `tiny_reg_vil.yaml` – ViL
    - `tiny_reg_vit.yaml` – ViT
  - Tiny-ImageNet (no regularization):
    - `tiny_noreg_vil.yaml` – ViL
    - `tiny_noreg_vit.yaml` – ViT
  - ImageNet-1K, 50 epochs:
    - `in1k_vil_50ep.yaml` – ViL
    - `in1k_vit_50ep.yaml` – ViT
  - Tiny-ImageNet-C evaluation:
    - `tinyc_vil_eval.yaml` – ViL
    - `tinyc_vit_eval.yaml` – ViT

  **Important:**  
  All YAML files use placeholder paths:
  ```yaml
  DATA_ROOT: /path/to/tiny-imagenet-200
  IMAGENETC_ROOT: /path/to/Tiny-ImageNet-C
  ```
  You must replace these with your actual dataset paths.

- `scripts/`
  - `train_tiny_reg.sh`  
    Tiny-ImageNet, regularized training for ViL and ViT across all specified ablations.
  - `train_tiny_noreg.sh`  
    Tiny-ImageNet, no-regularization training for ViL and ViT.
  - `train_in1k_50ep.sh`  
    ImageNet-1K, 50-epoch training for ViL and ViT ablations.
  - `eval_tinyc.sh`  
    Tiny-ImageNet-C evaluation (ViL + ViT) using the `tinyc_*_eval.yaml` configs.
  - `resume_in1k_50_more.sh`  
    Resume ImageNet-1K training from an existing checkpoint for additional epochs.

---

## 3. Basic usage

### 3.1 Tiny-ImageNet training (regularized)

1. Edit dataset path in the config:

```yaml
# configs/tiny_reg_vil.yaml
DATA_ROOT: /absolute/path/to/tiny-imagenet-200
```

2. Run all Tiny-ImageNet regularized experiments (ViL + ViT):

```bash
bash scripts/train_tiny_reg.sh
```

This script:
- Uses `configs/tiny_reg_vil.yaml` for ViL experiments.
- Uses `configs/tiny_reg_vit.yaml` for ViT experiments.
- Sets `MODEL_KIND`, `ABLATION`, `DWT_FUSE`, and `RUN_TAG` per run.
- Calls `train_ablation_ddp.py` via `torch.distributed.run`.

If you want to run only a single experiment (e.g. ViL A1 baseline):

```bash
export CONFIG=configs/tiny_reg_vil.yaml
export MODEL_KIND=vil
export ABLATION=A1
export DWT_FUSE=none
export RUN_TAG=tiny_vil_A1_ch32_patch8_reg

python -m torch.distributed.run --nproc_per_node=8 train_ablation_ddp.py
```

### 3.2 Tiny-ImageNet training (no regularization)

1. Edit:

```yaml
# configs/tiny_noreg_vil.yaml / configs/tiny_noreg_vit.yaml
DATA_ROOT: /absolute/path/to/tiny-imagenet-200
```

2. Run:

```bash
bash scripts/train_tiny_noreg.sh
```

The script mirrors `train_tiny_reg.sh` but with `LABEL_SMOOTH=0`, `MIXUP_* = 0`, etc.

### 3.3 ImageNet-1K training (50 epochs)

1. Edit:

```yaml
# configs/in1k_vil_50ep.yaml / configs/in1k_vit_50ep.yaml
DATA_ROOT: /absolute/path/to/imagenet
```

2. Run all ImageNet-1K ViL/ViT experiments:

```bash
bash scripts/train_in1k_50ep.sh
```

---

## 4. Tiny-ImageNet-C evaluation

1. Edit Tiny-ImageNet-C eval configs:

```yaml
# configs/tinyc_vil_eval.yaml
DATA_ROOT: /absolute/path/to/tiny-imagenet-200
IMAGENETC_ROOT: /absolute/path/to/Tiny-ImageNet-C

# configs/tinyc_vit_eval.yaml
DATA_ROOT: /absolute/path/to/tiny-imagenet-200
IMAGENETC_ROOT: /absolute/path/to/Tiny-ImageNet-C
```

2. Optionally set number of processes (for eval, 1 GPU is usually enough):

```bash
export NPROC=1   # or more if desired
```

3. Run the evaluation script:

```bash
bash scripts/eval_tinyc.sh
```

The script:
- For ViL runs, sets `CONFIG=configs/tinyc_vil_eval.yaml` (unless `CONFIG` is already set).
- For ViT runs, sets `CONFIG=configs/tinyc_vit_eval.yaml`.
- Sets `MODE=eval_imagenetc` and appropriate `MODEL_KIND`, `ABLATION`, `DWT_FUSE`, `CKPT`, and `RUN_TAG`.
- Invokes `train_ablation_ddp.py` to load the checkpoint and call `evaluate_imagenet_c`.

---

## 5. Resuming ImageNet-1K training from a checkpoint

To continue training an existing ImageNet-1K model for more epochs:

```bash
export RESUME_CKPT=/absolute/path/to/ema_best.pth
export EXTRA_EPOCHS=50         # total epochs you want to run from this point
export MODEL_KIND=vil          # or vit_tiny

bash scripts/resume_in1k_50_more.sh
```

This script:
- Uses `in1k_vil_50ep.yaml` or `in1k_vit_50ep.yaml` as base config.
- Sets `EPOCHS=EXTRA_EPOCHS`.
- Uses `RESUME_CKPT` as the initial weights.

---

## 6. Key environment variables

- **Data & mode**
  - `DATASET`: `imagenet` or `tiny_imagenet`
  - `DATA_ROOT`: dataset root directory
  - `IMAGENETC_ROOT`: root directory for ImageNet-C / Tiny-ImageNet-C
  - `OUT_DIR`: output directory (default `./outputs_pswf_paper`)
  - `MODE`: `train`, `eval`, or `eval_imagenetc`

- **Model**
  - `MODEL_KIND`: `vil`, `vit_tiny`, or `mambavision` (stub)
  - `ABLATION`: e.g. `A1`, `A3`, `W3`, `W3_POOL_ONLY`, `W3_TOKENONLY`, `W3_RESIDUALONLY`, `W3_IMPROVED_WARMUP`
  - `DWT_FUSE`: `add`, `gated`, `LL`, `concat`, `none`
  - `DIM`, `DEPTH`, `FEAT_CH`, `PATCH_SIZE`, `STRIDE`, `AUTO_PATCH_DWT`

- **Training hyperparameters**
  - `IMG_SIZE`, `EPOCHS`, `PER_GPU_BATCH`, `ACCUM_STEPS`
  - `BASE_LR`, `WARMUP_EPOCHS`, `WEIGHT_DECAY`, `EMA_DECAY`
  - `LABEL_SMOOTH`, `MIXUP_PROB`, `MIXUP_ALPHA`, `CUTMIX_ALPHA`, `SWITCH_PROB`
  - `AMP_DTYPE`: `bf16` or `fp16`

---

## 7. Notes

- No machine-specific absolute paths are hard-coded in the source code; dataset locations are always provided via YAML (`DATA_ROOT`, `IMAGENETC_ROOT`) and/or environment variables.
- For DDP training and evaluation, use:
  ```bash
  python -m torch.distributed.run --nproc_per_node=N train_ablation_ddp.py
  ```
  where `N` is the number of GPUs per node.
- For simple experiments or evaluation, running with `N=1` is often sufficient and easier to debug.

