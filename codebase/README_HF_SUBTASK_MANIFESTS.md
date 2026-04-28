# HF Subtask Manifests

This file explains how to reproduce the filtered HF subtask manifests derived from the refreshed `20260322` combined manifest.

## Why These Files Exist

The doctor-returned `20260322` Shengrenyi update added two kinds of information:

- explicit `is_problem` flags for rows considered difficult or potentially unreliable
- structured clinical subgroup labels `L_*` that identify known interference or confounding patterns

The later doctor note in [doc.md](/Users/qyqsmacbookpro/Desktop/HF-ECG-Research/data/doc.md) added two more practical screening ideas:

- if you want purer negative samples, only treat non-HF as reliable when `BNP < 300 pg/mL`
- you may exclude patients with extremely high BNP values

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

The script now creates thirteen filtered manifests:

- five interference-driven subsets
- four BNP-driven subsets
- four combined interference-plus-BNP subsets

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

### 6. `doctor_bnp_negative_pure`

Output:

- `data/manifests/hf_manifest_combined_20260322_subtask_doctor_bnp_negative_pure.csv`

Rule:

- keep class `0` only when `bnp_value < 300`
- exclude class-`0` rows with `bnp_value >= 300`
- exclude class-`0` rows with missing `bnp_value`

Why:

- this is the cleanest direct operationalization of the doctor's “negative sample purity” rule

### 7. `doctor_bnp_negative_pure_extreme_30000`

Output:

- `data/manifests/hf_manifest_combined_20260322_subtask_doctor_bnp_negative_pure_extreme_30000.csv`

Rule:

- start from `doctor_bnp_negative_pure`
- additionally exclude rows where `bnp_value > 30000`

Why:

- this keeps the negative-purity rule and trims the very extreme BNP tail with a fixed, reproducible threshold

### 8. `doc_error_reduced_doctor_bnp_negative_pure_extreme_30000`

Output:

- `data/manifests/hf_manifest_combined_20260322_subtask_doc_error_reduced_doctor_bnp_negative_pure_extreme_30000.csv`

Rule:

- start from `doc_error_reduced`
- keep class `0` only when `bnp_value < 300`
- exclude rows where `bnp_value > 30000`

Why:

- this is the most direct implementation of your request to combine the doctor BNP rule with the earlier error-reduced filtering

### 9. `low_interference_doctor_bnp_negative_pure_extreme_30000`

Output:

- `data/manifests/hf_manifest_combined_20260322_subtask_low_interference_doctor_bnp_negative_pure_extreme_30000.csv`

Rule:

- start from `low_interference`
- keep class `0` only when `bnp_value < 300`
- exclude rows where `bnp_value > 30000`

Why:

- this is a practical cleaner-case subset for model debugging when you want both reduced interference and purer negatives

### 10. `cardiac_clean_doctor_bnp_negative_pure_extreme_30000`

Output:

- `data/manifests/hf_manifest_combined_20260322_subtask_cardiac_clean_doctor_bnp_negative_pure_extreme_30000.csv`

Rule:

- start from `cardiac_clean`
- keep class `0` only when `bnp_value < 300`
- exclude rows where `bnp_value > 30000`

Why:

- this is a stricter cardiac-clean subset with the practical BNP negative-purity rule layered on top

### 11. `left_ventricular_clean_doctor_bnp_negative_pure_extreme_30000`

Output:

- `data/manifests/hf_manifest_combined_20260322_subtask_left_ventricular_clean_doctor_bnp_negative_pure_extreme_30000.csv`

Rule:

- start from `left_ventricular_clean`
- keep class `0` only when `bnp_value < 300`
- exclude rows where `bnp_value > 30000`

Why:

- this is the cleanest and easiest left-ventricular-focused subset currently available from the doctor-returned rules

### 12. `doctor_bnp_rule_partial_extreme_30000`

Output:

- `data/manifests/hf_manifest_combined_20260322_subtask_doctor_bnp_rule_partial_extreme_30000.csv`

Rule:

- class `0` requires `bnp_value < 300`
- class `1` requires `bnp_value >= 900`
- class `2` requires `40 < LVEF < 50`
- class `3` requires `LVEF <= 40`
- exclude rows where `bnp_value > 30000`

Important limitation:

- this is only a **partial** implementation of the doctor rule for HFpEF
- the full HFpEF rule also needs structured echo morphology fields such as LAD and wall thickness
- those structured measurements are not present as clean numeric columns in the current manifest

### 13. `low_interference_doctor_bnp_rule_partial_extreme_30000`

Output:

- `data/manifests/hf_manifest_combined_20260322_subtask_low_interference_doctor_bnp_rule_partial_extreme_30000.csv`

Rule:

- start from `low_interference`
- then apply the same partial doctor BNP/LVEF rule as `doctor_bnp_rule_partial_extreme_30000`
- exclude rows where `bnp_value > 30000`

Why:

- this is the easiest BNP-aware reduced-difficulty subset
- use it for pilot experiments only, not as a silent replacement for the full benchmark

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
- `data/manifests/hf_manifest_combined_20260322_subtask_doctor_bnp_negative_pure.csv`
- `data/manifests/hf_manifest_combined_20260322_subtask_doctor_bnp_negative_pure_extreme_30000.csv`
- `data/manifests/hf_manifest_combined_20260322_subtask_doc_error_reduced_doctor_bnp_negative_pure_extreme_30000.csv`
- `data/manifests/hf_manifest_combined_20260322_subtask_low_interference_doctor_bnp_negative_pure_extreme_30000.csv`
- `data/manifests/hf_manifest_combined_20260322_subtask_cardiac_clean_doctor_bnp_negative_pure_extreme_30000.csv`
- `data/manifests/hf_manifest_combined_20260322_subtask_left_ventricular_clean_doctor_bnp_negative_pure_extreme_30000.csv`
- `data/manifests/hf_manifest_combined_20260322_subtask_doctor_bnp_rule_partial_extreme_30000.csv`
- `data/manifests/hf_manifest_combined_20260322_subtask_low_interference_doctor_bnp_rule_partial_extreme_30000.csv`
- `data/manifests/hf_manifest_combined_20260322_subtasks_summary.json`

## Local Reference Counts

From the current local run:

- base refreshed combined: `6078`
- `no_problem`: `5866`
- `doc_error_reduced`: `3010`
- `low_interference`: `3879`
- `cardiac_clean`: `1535`
- `left_ventricular_clean`: `1374`
- `doctor_bnp_negative_pure`: `4408`
- `doctor_bnp_negative_pure_extreme_30000`: `4390`
- `doc_error_reduced_doctor_bnp_negative_pure_extreme_30000`: `2058`
- `low_interference_doctor_bnp_negative_pure_extreme_30000`: `2738`
- `cardiac_clean_doctor_bnp_negative_pure_extreme_30000`: `1129`
- `left_ventricular_clean_doctor_bnp_negative_pure_extreme_30000`: `1021`
- `doctor_bnp_rule_partial_extreme_30000`: `3672`
- `low_interference_doctor_bnp_rule_partial_extreme_30000`: `2117`

Class distributions:

- `no_problem`: `{0: 2015, 1: 2701, 2: 562, 3: 588}`
- `doc_error_reduced`: `{0: 1196, 1: 1274, 2: 258, 3: 282}`
- `low_interference`: `{0: 1414, 1: 1750, 2: 356, 3: 359}`
- `cardiac_clean`: `{0: 502, 1: 719, 2: 124, 3: 190}`
- `left_ventricular_clean`: `{0: 434, 1: 643, 2: 113, 3: 184}`
- `doctor_bnp_negative_pure`: `{0: 384, 1: 2839, 2: 581, 3: 604}`
- `doctor_bnp_negative_pure_extreme_30000`: `{0: 384, 1: 2831, 2: 575, 3: 600}`
- `doc_error_reduced_doctor_bnp_negative_pure_extreme_30000`: `{0: 246, 1: 1273, 2: 258, 3: 281}`
- `low_interference_doctor_bnp_negative_pure_extreme_30000`: `{0: 282, 1: 1745, 2: 354, 3: 357}`
- `cardiac_clean_doctor_bnp_negative_pure_extreme_30000`: `{0: 96, 1: 719, 2: 124, 3: 190}`
- `left_ventricular_clean_doctor_bnp_negative_pure_extreme_30000`: `{0: 81, 1: 643, 2: 113, 3: 184}`
- `doctor_bnp_rule_partial_extreme_30000`: `{0: 384, 1: 2157, 2: 540, 3: 591}`
- `low_interference_doctor_bnp_rule_partial_extreme_30000`: `{0: 282, 1: 1146, 2: 339, 3: 350}`

## Recommended Use

Use order:

1. Main paper and main training:
   - `hf_manifest_combined_20260322_ascii.csv`
2. Cleaner pilot:
   - `hf_manifest_combined_20260322_subtask_no_problem.csv`
3. Easier debugging subset:
   - `hf_manifest_combined_20260322_subtask_low_interference.csv`
4. Cleaner debugging subset with doctor BNP screening:
   - `hf_manifest_combined_20260322_subtask_low_interference_doctor_bnp_negative_pure_extreme_30000.csv`
5. Smallest easy subset:
   - `hf_manifest_combined_20260322_subtask_left_ventricular_clean_doctor_bnp_negative_pure_extreme_30000.csv`

If you want to reproduce the doctor's practical screening logic on top of an older subset, use this mapping:

- `doc_error_reduced` + BNP:
  `hf_manifest_combined_20260322_subtask_doc_error_reduced_doctor_bnp_negative_pure_extreme_30000.csv`
- `low_interference` + BNP:
  `hf_manifest_combined_20260322_subtask_low_interference_doctor_bnp_negative_pure_extreme_30000.csv`
- `cardiac_clean` + BNP:
  `hf_manifest_combined_20260322_subtask_cardiac_clean_doctor_bnp_negative_pure_extreme_30000.csv`
- `left_ventricular_clean` + BNP:
  `hf_manifest_combined_20260322_subtask_left_ventricular_clean_doctor_bnp_negative_pure_extreme_30000.csv`
4. BNP-aware negative-purity subset:
   - `hf_manifest_combined_20260322_subtask_doctor_bnp_negative_pure.csv`
5. BNP-aware easier pilot subset:
   - `hf_manifest_combined_20260322_subtask_low_interference_doctor_bnp_rule_partial_extreme_30000.csv`
6. Strict exploratory subset only:
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
