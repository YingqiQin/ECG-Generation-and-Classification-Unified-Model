# HF Manifest Refresh Runbook

This file records the exact data-manifest work done locally so you can reproduce it on the server after copying the original `data/` files.

## Scope

The manifest pipeline now supports:

- Taizhou original XML ECGs plus `泰州心电标签942.xlsx`
- Shengrenyi original XML ECGs plus the old class workbook `省人医房颤心超数据（含分类）.xlsx`
- Shengrenyi doctor-updated March 22, 2026 files:
  - `省人医5317份xml检查信息_20260322更新.xlsx`
  - `省人医5317例补充临床标签_20260322更新.xlsx`
  - `总体情况说明_20260322更新.docx`
  - `临床标签_20260322更新.docx`

## What Changed In Code

The following scripts/files were updated or added:

- [ecg_dataset.py](/Users/qyqsmacbookpro/Desktop/HF-ECG-Research/codebase/ecg_dataset.py)
  - XML ECG loading is supported directly.
  - MIL and lead-transfer datasets can read XML, not just `.npy`.
- [prepare_local_hf_xml_manifest.py](/Users/qyqsmacbookpro/Desktop/HF-ECG-Research/codebase/prepare_local_hf_xml_manifest.py)
  - Builds Taizhou manifest from original XML + Excel.
  - Builds old Shengrenyi manifest from original XML + old class workbook.
  - Builds new versioned Shengrenyi `20260322` manifest from the cleaned XML list + supplemental doctor labels + old HF class workbook.
  - Emits audit CSVs and ASCII-safe CSVs.
- [materialize_ascii_xml_aliases.py](/Users/qyqsmacbookpro/Desktop/HF-ECG-Research/codebase/materialize_ascii_xml_aliases.py)
  - Creates ASCII-only alias paths so training on the server does not depend on Chinese file paths.
- [prepare_hf_patient_splits.py](/Users/qyqsmacbookpro/Desktop/HF-ECG-Research/codebase/prepare_hf_patient_splits.py)
  - Builds reproducible patient-level `train.csv`, `valid.csv`, and `test.csv` files from the refreshed HF manifest.
- [evaluate_hf_subgroups.py](/Users/qyqsmacbookpro/Desktop/HF-ECG-Research/codebase/evaluate_hf_subgroups.py)
  - Evaluates inference outputs on `is_problem` and all `L_*` doctor-returned clinical subgroup columns.

## Expected Original Data Layout

The server should contain the same relative structure under `data/`:

```text
data/
  泰州ECG及标签/
    泰州ECG/
      *.xml
    泰州心电标签942.xlsx
  省人医/
    省人医ECG/
      房颤数据/
        *.xml
    省人医房颤心超数据（含分类）.xlsx
  省人医5317份xml检查信息_20260322更新.xlsx
  省人医5317例补充临床标签_20260322更新.xlsx
  总体情况说明_20260322更新.docx
  临床标签_20260322更新.docx
```

## How The New Shengrenyi Path Works

The new `20260322` Shengrenyi files do not directly contain the final HF class `0-3`.

The implemented logic is:

1. Use `省人医5317份xml检查信息_20260322更新.xlsx` as the cleaned XML-level source of truth.
2. Resolve each updated `XML文件名` back to the actual XML filename under `data/省人医/省人医ECG/房颤数据/`.
   - The new sheet uses alias-style names like `026728_赵生荣_20210225100226_7222120926855192_1_84.xml`.
   - The actual XML on disk is often ordered like `7222120926855192_026728_00010101000000_赵生荣_1_84.xml`.
   - The script maps between these two forms.
3. Join `省人医5317例补充临床标签_20260322更新.xlsx` by `XML文件名`.
4. Recover HF class `0-3` from the old workbook `省人医房颤心超数据（含分类）.xlsx` using:
   - normalized `门诊/住院号`
   - nearest `心电图检查时间`
5. Keep only rows whose recovered class match is within `7` days for the strict training manifest.
6. Preserve all `5137` updated Shengrenyi rows in a separate audit CSV.

## Local Output Summary

Running the refreshed script locally produced:

- Taizhou rows: `969`
- Old Shengrenyi rows: `6040`
- Old combined rows: `7009`
- Updated Shengrenyi XML rows resolved: `5137`
- Updated Shengrenyi strict training rows: `5109`
- Updated combined rows: `6078`
- Updated Shengrenyi `is_problem=1` rows: `212`

Updated Shengrenyi strict class distribution:

- `0: 1878`
- `1: 2299`
- `2: 492`
- `3: 440`

Updated combined class distribution:

- `0: 2054`
- `1: 2839`
- `2: 581`
- `3: 604`

## Exact Reproduction Commands

Run from the repo root:

```bash
python3 -m py_compile codebase/prepare_local_hf_xml_manifest.py
python3 codebase/prepare_local_hf_xml_manifest.py
```

This writes:

- `data/manifests/hf_manifest_taizhou.csv`
- `data/manifests/hf_manifest_taizhou_ascii.csv`
- `data/manifests/hf_manifest_shengrenyi.csv`
- `data/manifests/hf_manifest_shengrenyi_ascii.csv`
- `data/manifests/hf_manifest_combined.csv`
- `data/manifests/hf_manifest_combined_ascii.csv`
- `data/manifests/hf_manifest_ascii_mapping.csv`
- `data/manifests/hf_manifest_summary.json`
- `data/manifests/hf_manifest_shengrenyi_20260322.csv`
- `data/manifests/hf_manifest_shengrenyi_20260322_ascii.csv`
- `data/manifests/hf_manifest_shengrenyi_20260322_audit.csv`
- `data/manifests/hf_manifest_shengrenyi_20260322_audit_ascii.csv`
- `data/manifests/hf_manifest_combined_20260322.csv`
- `data/manifests/hf_manifest_combined_20260322_ascii.csv`
- `data/manifests/hf_manifest_ascii_mapping_20260322.csv`
- `data/manifests/hf_manifest_summary_20260322.json`

Then materialize ASCII alias files:

```bash
python3 codebase/materialize_ascii_xml_aliases.py \
  --data-root data \
  --mapping-csv data/manifests/hf_manifest_ascii_mapping_20260322.csv \
  --mode symlink
```

If the server filesystem does not support symlinks cleanly, use:

```bash
python3 codebase/materialize_ascii_xml_aliases.py \
  --data-root data \
  --mapping-csv data/manifests/hf_manifest_ascii_mapping_20260322.csv \
  --mode copy
```

## Which Manifest To Use For Training

For the new training runs, use:

- `data/manifests/hf_manifest_combined_20260322_ascii.csv`

This is the recommended training manifest because:

- it uses the doctor-updated Shengrenyi cleaned XML list
- it includes the updated supplemental clinical tags
- it avoids Chinese path issues by using ASCII alias paths

Use the audit file when you want to inspect excluded or uncertain cases:

- `data/manifests/hf_manifest_shengrenyi_20260322_audit.csv`

## Create Patient-Level Splits

After regenerating the manifests, create the downstream train/valid/test split:

```bash
python3 codebase/prepare_hf_patient_splits.py \
  --manifest-csv data/manifests/hf_manifest_combined_20260322_ascii.csv \
  --output-dir data/splits_20260322 \
  --seed 1234
```

This writes:

- `data/splits_20260322/train.csv`
- `data/splits_20260322/valid.csv`
- `data/splits_20260322/test.csv`
- `data/splits_20260322/split_summary.json`

Local reference counts from the current run:

- `train`: `4867` rows, `3549` patients
- `valid`: `596` rows, `444` patients
- `test`: `615` rows, `443` patients

If you want a cleaner mainline run, you can also exclude flagged rows:

```bash
python3 codebase/prepare_hf_patient_splits.py \
  --manifest-csv data/manifests/hf_manifest_combined_20260322_ascii.csv \
  --output-dir data/splits_20260322_no_problem \
  --seed 1234 \
  --exclude-problem
```

## Important Columns In The Updated Manifest

The updated Shengrenyi manifest includes:

- `class`
- `class_recovered`
- `class_match_gap_hours`
- `class_match_within_7d`
- `xml_file`
- `xml_file_alias`
- `is_problem`
- `L_一_1_大血管_瓣膜_冠脉手术`
- `L_一_2_起搏大类`
- `L_一_3_消融术后`
- `L_一_4_左心耳干预`
- `L_二_1_左心重度返流瓣膜病`
- `L_二_2_左心重度狭窄瓣膜病`
- `L_二_3_右心大_肺动脉高压`
- `L_二_4_右心重度瓣膜病`
- `L_二_5_仅右房增大`
- `L_二_6_心室肥厚`
- `L_三_1_梗死史`
- `L_四_1_束支阻滞`
- `L_四_2_明确舒张功能下降`
- `L_四_3_大量心包积液`
- `L_四_4_床旁重症`

## Sanity Checks To Run On The Server

After generating the manifests, check:

```bash
python3 - <<'PY'
import csv
from collections import Counter
path = 'data/manifests/hf_manifest_combined_20260322_ascii.csv'
with open(path, newline='') as f:
    rows = list(csv.DictReader(f))
print('rows', len(rows))
print('class_dist', Counter(r['class'] for r in rows))
print('non_ascii_paths', sum(any(ord(ch) > 127 for ch in r['xml_path']) for r in rows))
PY
```

Expected:

- `rows` should be `6078`
- class distribution should be close to:
  - `1: 2839`
  - `0: 2054`
  - `3: 604`
  - `2: 581`
- `non_ascii_paths` should be `0`

## Training Recommendation

For HF optimization, point the training code to:

- manifest: `data/manifests/hf_manifest_combined_20260322_ascii.csv`
- signal root: `data`

The existing training scripts still use the old env var name `NPY_ROOT`, but the loader now handles XML. So on the server, `NPY_ROOT=data` is acceptable.

Example:

```bash
TRAIN_CSV=data/splits_20260322/train.csv \
VALID_CSV=data/splits_20260322/valid.csv \
NPY_ROOT=data \
POOL_TYPE=attention \
bash codebase/run_mil_train.sh
```

## Subgroup Evaluation After Inference

After inference finishes and writes `predictions.csv`, evaluate subgroup robustness:

```bash
python3 codebase/evaluate_hf_subgroups.py \
  --predictions-csv /path/to/inference_out/predictions.csv \
  --manifest-csv data/splits_20260322/test.csv \
  --out-dir /path/to/inference_out/subgroups
```

This writes:

- `subgroup_metrics.json`
- `subgroup_metrics.csv`

The default behavior scores:

- `is_problem`
- every manifest column that starts with `L_`

## Notes

- The updated Shengrenyi strict manifest excludes `28` rows from direct training use:
  - `12` had no recoverable old HF class
  - `16` recovered a class but exceeded the `7`-day class-match threshold
- Those rows are still preserved in the audit CSV.
- The old manifests are kept for comparison and rollback. The `20260322` files are the new preferred source.
