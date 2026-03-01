# TCAV Auditing 2

This repository contains a controlled TCAV experiment for auditing spurious feature reliance.

The setup is simple:
- Dataset: CIFAR-10
- Backbone: ResNet-18
- Target class: `3` (cat)
- Spurious feature: red `8x8` patch at the bottom-right corner
- Injection rule: patch is added only to target-class images in the spurious training split

Two models are trained and compared:
- `clean_model`: trained on clean data
- `spurious_model`: trained on patched target-class data

The final trained models and experiment outputs are already included in this repo.

## What is in this repo

- `configs/`: training configs for clean and spurious models
- `scripts/data_processing/`: data download / patch injection / test-set export utilities
- `scripts/training/`: model training entrypoint
- `scripts/evaluation/`: behavioral dependency test
- `scripts/tcav/`: TCAV run + score summarization + detection metrics
- `models/`: final checkpoints
- `results/`: behavioral test output, TCAV raw outputs, and summary tables

## Data and experiment pipeline

### 1) Download CIFAR-10

```bash
cd scripts/data_processing
python download_data.py
```

### 2) Build clean/spurious train splits and concept/control sets

```bash
cd scripts/data_processing
python add_patch.py
cd ../..
python scripts/data_processing/generate_patch_set.py
```

### 3) Export test sets for behavioral checks

```bash
cd scripts/data_processing
python export_testset.py
```

This produces:
- `data/test_clean`
- `data/test_patched_target`
- `data/test_patched_then_removed`

### 4) Train both models

Run from project root:

```bash
python scripts/training/train_model.py --cfg configs/train_clean.yaml
python scripts/training/train_model.py --cfg configs/train_spurious.yaml
```

Checkpoints are saved to:
- `models/clean_model/model.pth`
- `models/spurious_model/model.pth`

### 5) Behavioral ground-truth verification

```bash
cd scripts/evaluation
python behavior_test.py
```

Outputs:
- `results/behavioral_tests/behavior_test_summary.csv`
- `results/behavioral_tests/behavior_test_summary.txt`

### 6) Run TCAV

```bash
cd scripts/tcav
python run_tcav.py
```

Outputs:
- `results/tcav_scores/clean_tcav_raw.json`
- `results/tcav_scores/spurious_tcav_raw.json`
- `results/tcav_scores/clean_tcav_flat.jsonl`
- `results/tcav_scores/spurious_tcav_flat.jsonl`

### 7) Summarize TCAV scores into final table

```bash
cd scripts/tcav
python summarize_tcacv_results.py
```

Outputs:
- `results/tables/tcav_scores_table.csv`
- `results/tables/tcav_scores_summary.txt`

## Final result files (already included)

Main files used in the report/reflection:
- `results/behavioral_tests/behavior_test_summary.txt`
- `results/tables/tcav_scores_table.csv`
- `results/tables/tcav_scores_summary.txt`
- `results/behavioral_tests/README.md` (records one TP/FP decision setup)

## Notes

- Random seeds are fixed in key scripts/configs (`seed=42`) for reproducibility.
- TCAV in `run_tcav.py` is forced to CPU by default.
- Path note: most data/evaluation/TCAV scripts use relative paths (for example `../../...`), so run them from the directories shown in each command block.
