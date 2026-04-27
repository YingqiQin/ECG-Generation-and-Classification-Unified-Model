# Lead Transfer: ECGFounder + MIMIC-IV-ECG

This codepath is designed for the server workflow you described:

- **Teacher / initialization checkpoint**: official `ECGFounder` checkpoint
- **Stage A transfer corpus**: `MIMIC-IV-ECG`
- **No local download assumption**: this folder is meant to be copied to a server that already has the checkpoint and data
- **Server dependency for original PhysioNet records**: install the Python package `wfdb`

## What Is Implemented

1. `train_lead_transfer.py`
   - frozen 12-lead teacher
   - true 1-channel student
   - latent alignment loss
   - student-view consistency loss
   - exports `best_student_encoder.pt` for downstream MIL

2. `prepare_mimic_iv_ecg_transfer_manifests.py`
   - converts the official `record_list.csv` from MIMIC-IV-ECG into patient-split `transfer_train.csv` and `transfer_valid.csv`

3. `run_lead_transfer.sh`
   - shell wrapper for the transfer stage
   - defaults to initializing the student from the same ECGFounder checkpoint as the teacher
   - supports single-node multi-GPU DDP via `torchrun`
   - writes TensorBoard event files by default

## Expected Transfer Manifest Columns

The transfer dataset accepts any of these path aliases:

- `npy_dst`
- `npy_path`
- `path`
- `signal_path`
- `waveform_path`
- `file_path`

And any of these ID aliases:

- patient: `subject_id` or `patient_id`
- record: `study_id`, `record_id`, `ecg_id`, or `xml_file`

Sampling-rate aliases:

- `sampling_rate_hz`
- `fs`
- `sample_rate`
- `sampling_frequency`

## Recommended Server Workflow

### 1. Prepare MIMIC transfer manifests

If you already have patient-split transfer manifests, skip this step.

```bash
python codebase/prepare_mimic_iv_ecg_transfer_manifests.py \
  --input-csv /path/to/mimic-iv-ecg/1.0/record_list.csv \
  --output-dir /path/to/mimic_transfer_manifests \
  --path-column path \
  --subject-column subject_id \
  --study-column study_id \
  --path-format wfdb \
  --default-fs 500 \
  --valid-ratio 0.1 \
  --seed 1234
```

### 2. Run Stage A transfer pretraining

```bash
TRAIN_CSV=/path/to/mimic_transfer_manifests/transfer_train.csv \
VALID_CSV=/path/to/mimic_transfer_manifests/transfer_valid.csv \
SIGNAL_ROOT=/path/to/mimic-iv-ecg/1.0 \
TEACHER_CKPT=/path/to/ecgfounder_official_checkpoint.pt \
STUDENT_LEAD_MODE=II_1ch \
INIT_STUDENT_FROM_TEACHER=1 \
USE_DDP=1 \
NPROC_PER_NODE=4 \
SYNC_BN=1 \
OUT_DIR=/path/to/outputs/lead_transfer_ii \
bash codebase/run_lead_transfer.sh
```

Notes:

- The official MIMIC-IV-ECG download stores WFDB records under paths like `files/p1000/p10001725/s102147240/102147240`.
- The generated transfer manifests now store `waveform_path` and `path_format=wfdb`, so no `.npy` conversion is required.
- `INIT_STUDENT_FROM_TEACHER=1` means the student is initialized from the same ECGFounder checkpoint using flexible first-conv channel adaptation.
- `STUDENT_LEAD_MODE=II_1ch` is the default recommended starting point for single-lead transfer.
- Other valid examples are `I_1ch`, `V2_1ch`, `V5_1ch`.
- `USE_DDP=1` switches the wrapper to `torchrun --standalone`.
- `BATCH_SIZE` is per process, so effective train batch is `BATCH_SIZE * NPROC_PER_NODE`.
- TensorBoard logs default to `<OUT_DIR>/tensorboard`.
- You can visualize them with `tensorboard --logdir /path/to/outputs`.

### 3. Run downstream HF MIL with the aligned student checkpoint

```bash
TRAIN_CSV=/path/to/hf_train.csv \
VALID_CSV=/path/to/hf_valid.csv \
NPY_ROOT=/path/to/hf_numpy_root \
ENCODER_CKPT=/path/to/outputs/lead_transfer_ii/best_student_encoder.pt \
IN_CHANNELS=1 \
LEAD_MODE=II_1ch \
POOL_TYPE=attention \
OUT_DIR=/path/to/outputs/hf_aligned_attention \
bash codebase/run_mil_train.sh
```

Then add:

- `POOL_TYPE=quality_attention`
- `POOL_TYPE=hybrid`

to run the post-transfer pooling ladder.

## First Runs That Matter

For the renewed thesis, the minimum useful server-side sequence is:

1. `pretrained without alignment + attention MIL`
2. `lead-transfer on MIMIC-IV-ECG -> best_student_encoder.pt`
3. `aligned + attention MIL`
4. `aligned + quality_attention MIL`
5. `aligned + hybrid MIL`

That is the first defensible experimental ladder for the paper.
