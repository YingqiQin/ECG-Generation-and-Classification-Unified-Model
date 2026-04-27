# HF Subtask Manifests

This file explains how to reproduce the filtered HF subtask manifests derived from the refreshed `20260322` combined manifest.

## Why These Files Exist

The doctor-returned `20260322` Shengrenyi update added two kinds of information:

- explicit `is_problem` flags for rows considered difficult or potentially unreliable
- structured clinical subgroup labels `L_*` that identify known interference or confounding patterns

These fields can be used to define easier auxiliary tasks or pilot subsets. They should not silently replace the main benchmark. Treat them as:

- pilot tasks for faster method debugging
- auxiliary analyses for cleaner-case performance
- robustness or sensitivity subsets

Do not present them as the only result table without also reporting the full refreshed setting.

## Base Manifest

All subtask manifests are generated from:

- `data/manifests/hf_manifest_combined_20260322_ascii.csv`

This manifest already:

- uses the refreshed Shengrenyi XML pairing
- uses the recovered HF class labels
- uses ASCII-safe XML paths

## Generated Subtasks

The script creates five filtered manifests.

### 1. `no_problem`

Output:

- `data/manifests/hf_manifest_combined_20260322_subtask_no_problem.csv`

Rule:

- exclude rows where `is_problem = 1`

Use case:

- simplest “doctor-cleaned” subset

### 2. `doc_error_reduced`

Output:

- `data/manifests/hf_manifest_combined_20260322_subtask_doc_error_reduced.csv`

Rule:

- exclude rows where any of the following are `1`:
  - `is_problem`
  - `L_一_1_大血管_瓣膜_冠脉手术`
  - `L_一_2_起搏大类`
  - `L_二_2_左心重度狭窄瓣膜病`
  - `L_三_1_梗死史`
  - `L_四_1_束支阻滞`
  - `L_二_6_心室肥厚`
  - `L_四_4_床旁重症`

Why:

- these are the main error-associated factors explicitly highlighted in `总体情况说明_20260322更新.docx`

### 3. `low_interference`

Output:

- `data/manifests/hf_manifest_combined_20260322_subtask_low_interference.csv`

Rule:

- exclude rows where any of the following are `1`:
  - `is_problem`
  - `L_一_1_大血管_瓣膜_冠脉手术`
  - `L_一_2_起搏大类`
  - `L_一_3_消融术后`
  - `L_一_4_左心耳干预`
  - `L_四_1_束支阻滞`
  - `L_四_3_大量心包积液`
  - `L_四_4_床旁重症`

Why:

- this targets obvious signal and intervention interference rather than all structural complexity

### 4. `cardiac_clean`

Output:

- `data/manifests/hf_manifest_combined_20260322_subtask_cardiac_clean.csv`

Rule:

- start from `low_interference`
- additionally exclude:
  - `L_二_3_右心大_肺动脉高压`
  - `L_二_4_右心重度瓣膜病`
  - `L_二_5_仅右房增大`

Why:

- this removes strong right-heart-load related patterns that may confound left-sided HF classification

### 5. `left_ventricular_clean`

Output:

- `data/manifests/hf_manifest_combined_20260322_subtask_left_ventricular_clean.csv`

Rule:

- start from `cardiac_clean`
- additionally exclude:
  - `L_三_1_梗死史`
  - `L_二_6_心室肥厚`

Why:

- this is the strictest pilot subset, intended for easier debugging and cleaner left-ventricular-focused classification

## Exact Reproduction Command

Run from the repo root:

```bash
python3 -m py_compile codebase/prepare_hf_subtask_manifests.py
python3 codebase/prepare_hf_subtask_manifests.py
```

This writes:

- `data/manifests/hf_manifest_combined_20260322_subtask_no_problem.csv`
- `data/manifests/hf_manifest_combined_20260322_subtask_doc_error_reduced.csv`
- `data/manifests/hf_manifest_combined_20260322_subtask_low_interference.csv`
- `data/manifests/hf_manifest_combined_20260322_subtask_cardiac_clean.csv`
- `data/manifests/hf_manifest_combined_20260322_subtask_left_ventricular_clean.csv`
- `data/manifests/hf_manifest_combined_20260322_subtasks_summary.json`

## Local Reference Counts

From the current local run:

- base refreshed combined: `6078`
- `no_problem`: `5866`
- `doc_error_reduced`: `3010`
- `low_interference`: `3879`
- `cardiac_clean`: `1535`
- `left_ventricular_clean`: `1374`

Class distributions:

- `no_problem`: `{0: 2015, 1: 2701, 2: 562, 3: 588}`
- `doc_error_reduced`: `{0: 1196, 1: 1274, 2: 258, 3: 282}`
- `low_interference`: `{0: 1414, 1: 1750, 2: 356, 3: 359}`
- `cardiac_clean`: `{0: 502, 1: 719, 2: 124, 3: 190}`
- `left_ventricular_clean`: `{0: 434, 1: 643, 2: 113, 3: 184}`

## Recommended Use

Use order:

1. Main paper and main training:
   - `hf_manifest_combined_20260322_ascii.csv`
2. Cleaner pilot:
   - `hf_manifest_combined_20260322_subtask_no_problem.csv`
3. Easier debugging subset:
   - `hf_manifest_combined_20260322_subtask_low_interference.csv`
4. Strict exploratory subset only:
   - `hf_manifest_combined_20260322_subtask_left_ventricular_clean.csv`

## Create Splits For A Chosen Subtask

After selecting a subtask manifest, create patient-level splits with the existing split utility.

Example for `low_interference`:

```bash
python3 codebase/prepare_hf_patient_splits.py \
  --manifest-csv data/manifests/hf_manifest_combined_20260322_subtask_low_interference.csv \
  --output-dir data/splits_20260322_low_interference \
  --seed 1234
```

You can replace the manifest path with any other generated subtask CSV.

## Important Warning

These filtered subtasks make the classification problem easier by construction. That is acceptable for:

- debugging
- pilot experiments
- sensitivity analysis
- cleaner-case supplementary experiments

It is not acceptable to replace the full refreshed benchmark silently. If the paper uses any subtask result in the main text, the exclusion rule must be stated explicitly.
