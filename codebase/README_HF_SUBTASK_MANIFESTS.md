# HF Subtask Manifests

This file explains how to reproduce filtered HF subtask manifests from the refreshed `20260322` combined manifest.

## Inputs

Base manifest:

- `data/manifests/hf_manifest_combined_20260322_ascii.csv`

Doctor-returned Shengrenyi files used upstream:

- `data/省人医5317份xml检查信息_20260322更新.xlsx`
- `data/省人医5317例补充临床标签_20260322更新.xlsx`
- `data/总体情况说明_20260322更新.docx`
- `data/临床标签_20260322更新.docx`

Additional Taizhou surgery list:

- `data/泰州900例_手术名单.xlsx`

The Taizhou workbook has three required columns:

- `xml文件名`
- `是否手术`
- `手术类型`

The current local run found `64` surgery XML rows in the workbook and matched all `64` to the combined manifest.

## Why These Files Exist

The refreshed Shengrenyi update added:

- `is_problem` flags for difficult or potentially unreliable rows
- `L_*` clinical subgroup labels for known interference or confounding patterns

The later doctor note in `data/doc.md` added:

- class `0` should be considered cleaner only when `BNP < 300 pg/mL`
- very extreme BNP values can be excluded

The Taizhou surgery workbook now adds:

- doctor-designated surgery rows for Taizhou XMLs
- surgery-type text for audit and reporting

These filtered datasets are useful for debugging, cleaner-case analysis, and sensitivity experiments. They should not silently replace the full refreshed benchmark.

## Reproduction

Run from the repo root:

```bash
python3 -m py_compile codebase/prepare_hf_subtask_manifests.py
python3 codebase/prepare_hf_subtask_manifests.py
```

If the Taizhou surgery workbook is stored elsewhere:

```bash
python3 codebase/prepare_hf_subtask_manifests.py \
  --taizhou-surgery-xlsx /path/to/泰州900例_手术名单.xlsx
```

The script writes all subtask CSVs under:

- `data/manifests/`

It also writes:

- `data/manifests/hf_manifest_combined_20260322_subtasks_summary.json`

The generated subtask CSVs include two Taizhou audit columns:

- `TZ_手术名单`
- `TZ_手术类型`

## Main Subtask Families

### Shengrenyi Doctor-Rule Subsets

These reuse the `is_problem` and `L_*` columns.

- `no_problem`: exclude `is_problem = 1`
- `doc_error_reduced`: exclude rows enriched in doctor-reported error sources
- `low_interference`: remove obvious surgery/device/intervention and ECG interference
- `cardiac_clean`: further remove right-heart-load and right-sided confounders
- `left_ventricular_clean`: strictest left-ventricular-focused subset

### BNP Subsets

These apply the doctor BNP rule from `data/doc.md`.

- `doctor_bnp_negative_pure`: class `0` requires `bnp_value < 300`
- `doctor_bnp_negative_pure_extreme_30000`: above plus exclude `bnp_value > 30000`
- `doctor_bnp_rule_partial_extreme_30000`: partial BNP/LVEF rule using available structured fields
- `low_interference_doctor_bnp_rule_partial_extreme_30000`: low-interference plus the partial BNP/LVEF rule

Important limitation:

- the full HFpEF rule also needs structured echo morphology fields such as LAD and wall thickness
- those measurements are not present as clean numeric columns in the current manifest

### Taizhou Surgery Subsets

These add the latest doctor-designated Taizhou surgery list.

- `taizhou_surgery_clean`: exclude only Taizhou surgery-list rows
- `doc_error_reduced_taizhou_surgery_clean`: `doc_error_reduced` plus Taizhou surgery-list exclusion
- `low_interference_taizhou_surgery_clean`: `low_interference` plus Taizhou surgery-list exclusion
- `cardiac_clean_taizhou_surgery_clean`: `cardiac_clean` plus Taizhou surgery-list exclusion
- `left_ventricular_clean_taizhou_surgery_clean`: `left_ventricular_clean` plus Taizhou surgery-list exclusion

### Taizhou Surgery Plus BNP Subsets

These combine the practical class-0 BNP rule with the Taizhou surgery-list exclusion.

- `doc_error_reduced_taizhou_surgery_doctor_bnp_negative_pure_extreme_30000`
- `low_interference_taizhou_surgery_doctor_bnp_negative_pure_extreme_30000`
- `cardiac_clean_taizhou_surgery_doctor_bnp_negative_pure_extreme_30000`
- `left_ventricular_clean_taizhou_surgery_doctor_bnp_negative_pure_extreme_30000`

Rule:

- apply the named Shengrenyi-style filter
- exclude rows where `TZ_手术名单 = 1`
- keep class `0` only when `bnp_value < 300`
- exclude rows where `bnp_value > 30000`

## Generated CSVs

Core and Shengrenyi-filtered outputs:

- `data/manifests/hf_manifest_combined_20260322_subtask_no_problem.csv`
- `data/manifests/hf_manifest_combined_20260322_subtask_doc_error_reduced.csv`
- `data/manifests/hf_manifest_combined_20260322_subtask_low_interference.csv`
- `data/manifests/hf_manifest_combined_20260322_subtask_cardiac_clean.csv`
- `data/manifests/hf_manifest_combined_20260322_subtask_left_ventricular_clean.csv`

BNP outputs:

- `data/manifests/hf_manifest_combined_20260322_subtask_doctor_bnp_negative_pure.csv`
- `data/manifests/hf_manifest_combined_20260322_subtask_doctor_bnp_negative_pure_extreme_30000.csv`
- `data/manifests/hf_manifest_combined_20260322_subtask_doctor_bnp_rule_partial_extreme_30000.csv`
- `data/manifests/hf_manifest_combined_20260322_subtask_low_interference_doctor_bnp_rule_partial_extreme_30000.csv`

Taizhou surgery outputs:

- `data/manifests/hf_manifest_combined_20260322_subtask_taizhou_surgery_clean.csv`
- `data/manifests/hf_manifest_combined_20260322_subtask_doc_error_reduced_taizhou_surgery_clean.csv`
- `data/manifests/hf_manifest_combined_20260322_subtask_low_interference_taizhou_surgery_clean.csv`
- `data/manifests/hf_manifest_combined_20260322_subtask_cardiac_clean_taizhou_surgery_clean.csv`
- `data/manifests/hf_manifest_combined_20260322_subtask_left_ventricular_clean_taizhou_surgery_clean.csv`

BNP plus Taizhou surgery outputs:

- `data/manifests/hf_manifest_combined_20260322_subtask_doc_error_reduced_taizhou_surgery_doctor_bnp_negative_pure_extreme_30000.csv`
- `data/manifests/hf_manifest_combined_20260322_subtask_low_interference_taizhou_surgery_doctor_bnp_negative_pure_extreme_30000.csv`
- `data/manifests/hf_manifest_combined_20260322_subtask_cardiac_clean_taizhou_surgery_doctor_bnp_negative_pure_extreme_30000.csv`
- `data/manifests/hf_manifest_combined_20260322_subtask_left_ventricular_clean_taizhou_surgery_doctor_bnp_negative_pure_extreme_30000.csv`

Older BNP combinations without Taizhou surgery exclusion are still generated for backward compatibility:

- `data/manifests/hf_manifest_combined_20260322_subtask_doc_error_reduced_doctor_bnp_negative_pure_extreme_30000.csv`
- `data/manifests/hf_manifest_combined_20260322_subtask_low_interference_doctor_bnp_negative_pure_extreme_30000.csv`
- `data/manifests/hf_manifest_combined_20260322_subtask_cardiac_clean_doctor_bnp_negative_pure_extreme_30000.csv`
- `data/manifests/hf_manifest_combined_20260322_subtask_left_ventricular_clean_doctor_bnp_negative_pure_extreme_30000.csv`

## Local Reference Counts

Current local counts:

- base refreshed combined: `6078`
- `no_problem`: `5866`
- `doc_error_reduced`: `3010`
- `low_interference`: `3879`
- `cardiac_clean`: `1535`
- `left_ventricular_clean`: `1374`
- `doctor_bnp_negative_pure`: `4408`
- `doctor_bnp_negative_pure_extreme_30000`: `4390`
- `taizhou_surgery_clean`: `6014`
- `doc_error_reduced_taizhou_surgery_clean`: `2946`
- `low_interference_taizhou_surgery_clean`: `3815`
- `cardiac_clean_taizhou_surgery_clean`: `1471`
- `left_ventricular_clean_taizhou_surgery_clean`: `1310`
- `doc_error_reduced_doctor_bnp_negative_pure_extreme_30000`: `2058`
- `low_interference_doctor_bnp_negative_pure_extreme_30000`: `2738`
- `cardiac_clean_doctor_bnp_negative_pure_extreme_30000`: `1129`
- `left_ventricular_clean_doctor_bnp_negative_pure_extreme_30000`: `1021`
- `doc_error_reduced_taizhou_surgery_doctor_bnp_negative_pure_extreme_30000`: `2000`
- `low_interference_taizhou_surgery_doctor_bnp_negative_pure_extreme_30000`: `2680`
- `cardiac_clean_taizhou_surgery_doctor_bnp_negative_pure_extreme_30000`: `1071`
- `left_ventricular_clean_taizhou_surgery_doctor_bnp_negative_pure_extreme_30000`: `963`
- `doctor_bnp_rule_partial_extreme_30000`: `3672`
- `low_interference_doctor_bnp_rule_partial_extreme_30000`: `2117`

Current Taizhou surgery-list matched class distribution:

- class `0`: `6`
- class `1`: `33`
- class `2`: `3`
- class `3`: `22`

## Recommended Use

Main benchmark:

- `hf_manifest_combined_20260322_ascii.csv`

Cleaner pilot:

- `hf_manifest_combined_20260322_subtask_no_problem.csv`

Practical reduced-interference pilot:

- `hf_manifest_combined_20260322_subtask_low_interference_taizhou_surgery_clean.csv`

Cleaner BNP-aware pilot:

- `hf_manifest_combined_20260322_subtask_low_interference_taizhou_surgery_doctor_bnp_negative_pure_extreme_30000.csv`

Smallest easy exploratory subset:

- `hf_manifest_combined_20260322_subtask_left_ventricular_clean_taizhou_surgery_doctor_bnp_negative_pure_extreme_30000.csv`

## Create Splits For A Chosen Subtask

After selecting a subtask manifest, create patient-level splits with the existing split utility.

Example:

```bash
python3 codebase/prepare_hf_patient_splits.py \
  --manifest-csv data/manifests/hf_manifest_combined_20260322_subtask_low_interference_taizhou_surgery_clean.csv \
  --output-dir data/splits_20260322_low_interference_taizhou_surgery_clean \
  --seed 1234
```

Use the matching `train.csv`, `valid.csv`, and `test.csv` from the same output directory for a subtask-only experiment.

## Leakage Warning

Do not mix independently generated subtask splits with the full benchmark test split. For example, this is invalid:

```bash
# Invalid: independently re-split clean train with unrelated full test.
TRAIN_CSV=data/splits_20260322_doc_error_reduced/train.csv
TEST_CSV=data/splits_20260322/test.csv
```

If you want a clean-train/full-test experiment, derive the clean training and validation CSVs by filtering the original full split assignment, and keep the original full test unchanged.

## Reporting Warning

These filtered subtasks make the classification problem easier by construction. That is acceptable for debugging, pilot experiments, sensitivity analysis, and cleaner-case supplementary experiments.

Do not replace the full refreshed benchmark silently. If a paper table uses any subtask result, state the exclusion rule explicitly.

## One-Tap 12-Lead Quality-Attention Sweep

To create splits, train, and test every marked subtask in one command, use:

```bash
SUMMARY_JSON=data/manifests/hf_manifest_combined_20260322_subtasks_summary.json \
HF_ROOT=data \
ENCODER_CKPT=data/pretrained_ckpt/12_lead_ECGFounder.pth \
BATCH_SIZE=128 \
EVAL_BATCH_SIZE=128 \
EPOCHS=5 \
bash codebase/run_hf_subtask_quality_attention_sweep.sh
```

The script will:

- read the generated subtask list from `SUMMARY_JSON`
- create `train.csv`, `valid.csv`, and `test.csv` for each subtask
- train `quality_attention` MIL with `IN_CHANNELS=12` and `LEAD_MODE=12`
- run inference on each matching test split
- write a sweep summary under `outputs/hf_subtask_quality_attention_sweep/`

Useful overrides:

- `TASKS="low_interference_taizhou_surgery_clean,left_ventricular_clean_taizhou_surgery_doctor_bnp_negative_pure_extreme_30000"`
- `SKIP_EXISTING=0`
- `RUN_SPLITS=0`
- `RUN_TRAIN=0`
- `RUN_INFER=0`

The default output layout is:

- splits: `data/splits_20260322_subtasks/<subtask_name>/`
- training: `outputs/mil_12lead_quality_attention/<subtask_name>/`
- inference: `outputs/infer_12lead_quality_attention/<subtask_name>/`

The generated `outputs/hf_subtask_quality_attention_sweep/sweep_summary.tsv` is the easiest file to inspect after the run finishes.
