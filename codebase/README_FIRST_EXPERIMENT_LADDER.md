# First Experiment Ladder: ECGFounder + MIMIC-IV-ECG + HF

This README is the shortest server-side path to the first **four-run** ladder for the renewed thesis:

1. `pretrained without alignment + attention MIL`
2. `lead-transfer pretraining on MIMIC-IV-ECG`
3. `aligned + attention MIL`
4. `aligned + hybrid MIL`

This is intentionally compact. It gives you:

- one no-alignment baseline
- one Stage A transfer job
- one post-transfer baseline
- one post-transfer method run

If these four runs are healthy, the next immediate ablation is:

- `aligned + quality_attention MIL`

## Assumptions

This repository is copied to a server where you already have:

- the official `ECGFounder` checkpoint
- `MIMIC-IV-ECG` waveform files and metadata
- your private HF train/valid manifests and waveform root
- a working Python environment with `torch`, `numpy`, `pandas`, `sklearn`, `wfdb`, and optionally `scipy`

No download step is included here.

For the MIMIC lead-transfer stage, the codebase now supports single-node multi-GPU DDP via `torchrun`.
Both the transfer trainer and the downstream MIL trainer now write TensorBoard event files by default under each run's `tensorboard/` subdirectory.

## Expected File Inputs

You need four paths on the server:

- `ECGFOUNDER_CKPT=/path/to/ecgfounder_official_checkpoint.pt`
- `MIMIC_RECORD_LIST_CSV=/path/to/mimic-iv-ecg/1.0/record_list.csv`
- `MIMIC_SIGNAL_ROOT=/path/to/mimic-iv-ecg/1.0`
- `HF_NPY_ROOT=/path/to/hf_numpy_root`

And two HF manifests:

- `HF_TRAIN_CSV=/path/to/hf_train.csv`
- `HF_VALID_CSV=/path/to/hf_valid.csv`

## Recommended Directory Layout

```bash
/path/to/project/
  codebase/
  outputs/
    mimic_transfer_manifests/
    mil_pretrained_attention/
    lead_transfer_ii/
    mil_aligned_attention/
    mil_aligned_hybrid/
```

## Step 0: Prepare MIMIC Transfer Manifests

Run this once if you do not already have patient-split transfer manifests.

```bash
python codebase/prepare_mimic_iv_ecg_transfer_manifests.py \
  --input-csv "${MIMIC_RECORD_LIST_CSV}" \
  --output-dir outputs/mimic_transfer_manifests \
  --path-column path \
  --subject-column subject_id \
  --study-column study_id \
  --path-format wfdb \
  --default-fs 500 \
  --valid-ratio 0.1 \
  --seed 1234
```

If your MIMIC metadata uses different column names, adjust those flags. The official PhysioNet `record_list.csv` should work as written.

## Run 1: Pretrained Without Alignment + Attention MIL

This measures what plain ECGFounder initialization gives you before any lead-transfer.

```bash
TRAIN_CSV="${HF_TRAIN_CSV}" \
VALID_CSV="${HF_VALID_CSV}" \
NPY_ROOT="${HF_NPY_ROOT}" \
OUT_DIR=outputs/mil_pretrained_attention \
NUM_CLASSES=4 \
CLASS_NAMES="normal hfref hfmr hfpef" \
ENCODER_CKPT="${ECGFOUNDER_CKPT}" \
IN_CHANNELS=1 \
LEAD_MODE=II_1ch \
POOL_TYPE=attention \
SEED=1234 \
BASE_SEED=1234 \
bash codebase/run_mil_train.sh
```

Output to keep:

- `outputs/mil_pretrained_attention/best_model.pt`
- `outputs/mil_pretrained_attention/best_metrics.json`

## Run 2: Lead-Transfer Pretraining on MIMIC-IV-ECG

This is Stage A. It uses:

- frozen 12-lead ECGFounder teacher
- single-lead student
- MIMIC-IV-ECG as the transfer corpus
- student initialized from the same ECGFounder checkpoint

```bash
TRAIN_CSV=outputs/mimic_transfer_manifests/transfer_train.csv \
VALID_CSV=outputs/mimic_transfer_manifests/transfer_valid.csv \
SIGNAL_ROOT="${MIMIC_SIGNAL_ROOT}" \
TEACHER_CKPT="${ECGFOUNDER_CKPT}" \
STUDENT_LEAD_MODE=II_1ch \
INIT_STUDENT_FROM_TEACHER=1 \
USE_DDP=1 \
NPROC_PER_NODE=4 \
SYNC_BN=1 \
OUT_DIR=outputs/lead_transfer_ii \
SEED=1234 \
BASE_SEED=1234 \
bash codebase/run_lead_transfer.sh
```

Output to keep:

- `outputs/lead_transfer_ii/best_student_encoder.pt`
- `outputs/lead_transfer_ii/best_metrics.json`

This checkpoint is the aligned single-lead encoder used in Runs 3 and 4.

Notes:

- `BATCH_SIZE` and `EVAL_BATCH_SIZE` are per-process under DDP.
- Effective training batch size is `BATCH_SIZE * NPROC_PER_NODE`.
- If you only want single-GPU transfer, omit `USE_DDP=1`.
- To visualize curves: `tensorboard --logdir outputs`

## Run 3: Aligned + Attention MIL

This isolates the effect of transfer before introducing the quality-aware method.

```bash
TRAIN_CSV="${HF_TRAIN_CSV}" \
VALID_CSV="${HF_VALID_CSV}" \
NPY_ROOT="${HF_NPY_ROOT}" \
OUT_DIR=outputs/mil_aligned_attention \
NUM_CLASSES=4 \
CLASS_NAMES="normal hfref hfmr hfpef" \
ENCODER_CKPT=outputs/lead_transfer_ii/best_student_encoder.pt \
IN_CHANNELS=1 \
LEAD_MODE=II_1ch \
POOL_TYPE=attention \
SEED=1234 \
BASE_SEED=1234 \
bash codebase/run_mil_train.sh
```

Output to keep:

- `outputs/mil_aligned_attention/best_model.pt`
- `outputs/mil_aligned_attention/best_metrics.json`

## Run 4: Aligned + Hybrid MIL

This is the first full method run in the compressed ladder.

```bash
TRAIN_CSV="${HF_TRAIN_CSV}" \
VALID_CSV="${HF_VALID_CSV}" \
NPY_ROOT="${HF_NPY_ROOT}" \
OUT_DIR=outputs/mil_aligned_hybrid \
NUM_CLASSES=4 \
CLASS_NAMES="normal hfref hfmr hfpef" \
ENCODER_CKPT=outputs/lead_transfer_ii/best_student_encoder.pt \
IN_CHANNELS=1 \
LEAD_MODE=II_1ch \
POOL_TYPE=hybrid \
QUALITY_ALPHA=1.0 \
TOPK=4 \
MIX_BETA=0.5 \
SEED=1234 \
BASE_SEED=1234 \
bash codebase/run_mil_train.sh
```

Output to keep:

- `outputs/mil_aligned_hybrid/best_model.pt`
- `outputs/mil_aligned_hybrid/best_metrics.json`

## How To Read The Ladder

The minimum comparisons you want after these four runs are:

- `Run 1` vs `Run 3`
  - does explicit lead-transfer beat plain ECGFounder initialization?
- `Run 3` vs `Run 4`
  - does the task-specific quality-aware MIL still help after transfer?
- `Run 1` vs `Run 4`
  - does the full two-stage path beat the simple pretrained baseline?

If `Run 3` does not beat `Run 1`, the transfer stage is not yet justified.

If `Run 4` does not beat `Run 3`, the downstream pooling contribution is not yet justified.

## Immediate Next Run After The Ladder

If the four-run ladder is stable, run this next:

```bash
TRAIN_CSV="${HF_TRAIN_CSV}" \
VALID_CSV="${HF_VALID_CSV}" \
NPY_ROOT="${HF_NPY_ROOT}" \
OUT_DIR=outputs/mil_aligned_quality_attention \
NUM_CLASSES=4 \
CLASS_NAMES="normal hfref hfmr hfpef" \
ENCODER_CKPT=outputs/lead_transfer_ii/best_student_encoder.pt \
IN_CHANNELS=1 \
LEAD_MODE=II_1ch \
POOL_TYPE=quality_attention \
QUALITY_ALPHA=1.0 \
SEED=1234 \
BASE_SEED=1234 \
bash codebase/run_mil_train.sh
```

That converts the compressed four-run ladder into the fuller post-transfer pooling ablation ladder.

## Practical Notes

- Start with `II_1ch` unless you already have a reason to prefer another lead.
- Keep `SEED` and `BASE_SEED` fixed for the first pass.
- Do not add reliability weighting until this ladder is healthy.
- Do not run multi-seed confirmation yet. First establish whether the single-seed direction is positive.
- If your HF class names differ, replace `CLASS_NAMES` accordingly.

## Minimal Success Condition

After these four runs, the project is in a good state only if:

- `Run 3 > Run 1` on your main validation metric
- `Run 4 >= Run 3` on the same metric

If that does not happen, do not expand the experiment matrix yet. Fix the transfer stage or the lead choice first.
