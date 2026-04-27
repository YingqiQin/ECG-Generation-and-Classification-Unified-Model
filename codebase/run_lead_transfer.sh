#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

TRAIN_CSV="${TRAIN_CSV:-}"
VALID_CSV="${VALID_CSV:-}"
SIGNAL_ROOT="${SIGNAL_ROOT:-${NPY_ROOT:-}}"
TEACHER_CKPT="${TEACHER_CKPT:-}"
OUT_DIR="${OUT_DIR:-${SCRIPT_DIR}/outputs/lead_transfer}"

TEACHER_PRESET="${TEACHER_PRESET:-ecgfounder_large}"
STUDENT_PRESET="${STUDENT_PRESET:-ecgfounder_large}"
EMBEDDING_DIM="${EMBEDDING_DIM:-512}"
TEACHER_IN_CHANNELS="${TEACHER_IN_CHANNELS:-12}"
STUDENT_IN_CHANNELS="${STUDENT_IN_CHANNELS:-1}"
TEACHER_LEAD_MODE="${TEACHER_LEAD_MODE:-12}"
STUDENT_LEAD_MODE="${STUDENT_LEAD_MODE:-I_1ch}"
STUDENT_INIT_CKPT="${STUDENT_INIT_CKPT:-}"
INIT_STUDENT_FROM_TEACHER="${INIT_STUDENT_FROM_TEACHER:-1}"

CLIP_SEC="${CLIP_SEC:-10}"
TARGET_FS="${TARGET_FS:-500}"
DEFAULT_FS="${DEFAULT_FS:-}"
NORM="${NORM:-zscore}"
NO_BANDPASS="${NO_BANDPASS:-0}"
BASE_SEED="${BASE_SEED:-1234}"
SEED="${SEED:-1234}"

ALIGN_HIDDEN="${ALIGN_HIDDEN:-256}"
ALIGN_DROPOUT="${ALIGN_DROPOUT:-0.1}"
ALIGN_WEIGHT="${ALIGN_WEIGHT:-1.0}"
CONSISTENCY_WEIGHT="${CONSISTENCY_WEIGHT:-0.2}"
STUDENT_NOISE_STD="${STUDENT_NOISE_STD:-0.01}"
STUDENT_TIME_MASK_RATIO="${STUDENT_TIME_MASK_RATIO:-0.05}"

EPOCHS="${EPOCHS:-20}"
BATCH_SIZE="${BATCH_SIZE:-64}"
EVAL_BATCH_SIZE="${EVAL_BATCH_SIZE:-64}"
NUM_WORKERS="${NUM_WORKERS:-4}"
LR="${LR:-1e-4}"
WEIGHT_DECAY="${WEIGHT_DECAY:-1e-4}"
GRAD_CLIP="${GRAD_CLIP:-1.0}"
DEVICE="${DEVICE:-auto}"
USE_DDP="${USE_DDP:-0}"
NPROC_PER_NODE="${NPROC_PER_NODE:-1}"
SYNC_BN="${SYNC_BN:-0}"
DIST_BACKEND="${DIST_BACKEND:-nccl}"
MASTER_PORT="${MASTER_PORT:-29500}"
MONITOR="${MONITOR:-val_cosine}"
MONITOR_MODE="${MONITOR_MODE:-max}"
USE_TQDM="${USE_TQDM:-1}"
TQDM_UPDATE_INTERVAL="${TQDM_UPDATE_INTERVAL:-10}"
USE_TENSORBOARD="${USE_TENSORBOARD:-1}"
TB_DIR="${TB_DIR:-}"
TB_FLUSH_SECS="${TB_FLUSH_SECS:-30}"

if [[ -z "${TRAIN_CSV}" || -z "${VALID_CSV}" || -z "${SIGNAL_ROOT}" || -z "${TEACHER_CKPT}" ]]; then
  cat <<'EOF'
Required environment variables:
  TRAIN_CSV     path to the transfer-train CSV manifest
  VALID_CSV     path to the transfer-valid CSV manifest
  SIGNAL_ROOT   root directory containing ECG signal files
               for original MIMIC-IV-ECG, point this to the dataset root
               that contains record_list.csv and files/
  TEACHER_CKPT  pretrained richer-lead teacher checkpoint

Optional environment variables:
  NPY_ROOT                  backward-compatible alias for SIGNAL_ROOT
  OUT_DIR                  output directory                        default: ./outputs/lead_transfer
  TEACHER_PRESET           teacher Net1D preset                   default: ecgfounder_large
  STUDENT_PRESET           student Net1D preset                   default: ecgfounder_large
  EMBEDDING_DIM            shared encoder embedding dim           default: 512
  TEACHER_IN_CHANNELS      teacher input channels                 default: 12
  STUDENT_IN_CHANNELS      student input channels                 default: 1
  TEACHER_LEAD_MODE        teacher lead mode                      default: 12
  STUDENT_LEAD_MODE        student lead mode                      default: I_1ch
  STUDENT_INIT_CKPT        optional student init checkpoint
  INIT_STUDENT_FROM_TEACHER 1 to init student from ECGFounder teacher ckpt
                          when STUDENT_INIT_CKPT is empty        default: 1
  CLIP_SEC                 clip length in seconds                 default: 10
  TARGET_FS                target sampling rate                   default: 500
  DEFAULT_FS               fallback fs if manifest lacks it
  NORM                     zscore | none                          default: zscore
  NO_BANDPASS              1 disables bandpass                    default: 0
  ALIGN_HIDDEN             alignment head hidden size             default: 256
  ALIGN_WEIGHT             teacher-student align loss weight      default: 1.0
  CONSISTENCY_WEIGHT       student-view consistency weight        default: 0.2
  STUDENT_NOISE_STD        student noise std                      default: 0.01
  STUDENT_TIME_MASK_RATIO  masked fraction of student clip        default: 0.05
  EPOCHS                   number of epochs                       default: 20
  BATCH_SIZE               training batch size                    default: 64
  EVAL_BATCH_SIZE          validation batch size                  default: 64
  LR                       AdamW learning rate                    default: 1e-4
  WEIGHT_DECAY             AdamW weight decay                     default: 1e-4
  DEVICE                   auto | cpu | cuda | cuda:0 ...        default: auto
  USE_DDP                  1 enables single-node multi-GPU DDP    default: 0
  NPROC_PER_NODE           processes / GPUs for torchrun          default: 1
  SYNC_BN                  1 enables SyncBatchNorm under DDP      default: 0
  DIST_BACKEND             nccl | gloo                            default: nccl
  MASTER_PORT              torchrun master port                   default: 29500
  MONITOR                  val_cosine | val_loss | ...           default: val_cosine
  MONITOR_MODE             max | min                              default: max
  USE_TQDM                 1 enables tqdm progress bars           default: 1
  TQDM_UPDATE_INTERVAL     tqdm postfix update interval           default: 10
  USE_TENSORBOARD          1 writes TensorBoard event files       default: 1
  TB_DIR                   optional TensorBoard log directory
  TB_FLUSH_SECS            TensorBoard flush interval             default: 30

Example:
  TRAIN_CSV=/path/transfer_train.csv \
  VALID_CSV=/path/transfer_valid.csv \
  SIGNAL_ROOT=/path/to/mimic-iv-ecg/1.0 \
  TEACHER_CKPT=/path/ecgfounder_12lead.pt \
  STUDENT_LEAD_MODE=II_1ch \
  USE_DDP=1 \
  NPROC_PER_NODE=4 \
  OUT_DIR=/path/out \
  bash run_lead_transfer.sh
EOF
  exit 1
fi

mkdir -p "${OUT_DIR}"

TRAINER="${SCRIPT_DIR}/train_lead_transfer.py"

ARGS=(
  --train-csv "${TRAIN_CSV}"
  --valid-csv "${VALID_CSV}"
  --root "${SIGNAL_ROOT}"
  --out-dir "${OUT_DIR}"
  --teacher-ckpt "${TEACHER_CKPT}"
  --teacher-preset "${TEACHER_PRESET}"
  --student-preset "${STUDENT_PRESET}"
  --embedding-dim "${EMBEDDING_DIM}"
  --teacher-in-channels "${TEACHER_IN_CHANNELS}"
  --student-in-channels "${STUDENT_IN_CHANNELS}"
  --teacher-lead-mode "${TEACHER_LEAD_MODE}"
  --student-lead-mode "${STUDENT_LEAD_MODE}"
  --clip-sec "${CLIP_SEC}"
  --target-fs "${TARGET_FS}"
  --norm "${NORM}"
  --base-seed "${BASE_SEED}"
  --seed "${SEED}"
  --align-hidden "${ALIGN_HIDDEN}"
  --align-dropout "${ALIGN_DROPOUT}"
  --align-weight "${ALIGN_WEIGHT}"
  --consistency-weight "${CONSISTENCY_WEIGHT}"
  --student-noise-std "${STUDENT_NOISE_STD}"
  --student-time-mask-ratio "${STUDENT_TIME_MASK_RATIO}"
  --epochs "${EPOCHS}"
  --batch-size "${BATCH_SIZE}"
  --eval-batch-size "${EVAL_BATCH_SIZE}"
  --num-workers "${NUM_WORKERS}"
  --lr "${LR}"
  --weight-decay "${WEIGHT_DECAY}"
  --grad-clip "${GRAD_CLIP}"
  --device "${DEVICE}"
  --tqdm-update-interval "${TQDM_UPDATE_INTERVAL}"
  --dist-backend "${DIST_BACKEND}"
  --tb-flush-secs "${TB_FLUSH_SECS}"
  --monitor "${MONITOR}"
  --monitor-mode "${MONITOR_MODE}"
)

if [[ -n "${STUDENT_INIT_CKPT}" ]]; then
  ARGS+=(--student-init-ckpt "${STUDENT_INIT_CKPT}")
fi

if [[ "${INIT_STUDENT_FROM_TEACHER}" == "1" ]]; then
  ARGS+=(--init-student-from-teacher)
fi

if [[ -n "${DEFAULT_FS}" ]]; then
  ARGS+=(--default-fs "${DEFAULT_FS}")
fi

if [[ "${NO_BANDPASS}" == "1" ]]; then
  ARGS+=(--no-bandpass)
fi

if [[ "${USE_DDP}" == "1" ]]; then
  ARGS+=(--ddp)
fi

if [[ "${SYNC_BN}" == "1" ]]; then
  ARGS+=(--sync-bn)
fi

if [[ "${USE_TQDM}" != "1" ]]; then
  ARGS+=(--no-tqdm)
fi

if [[ "${USE_TENSORBOARD}" != "1" ]]; then
  ARGS+=(--no-tensorboard)
fi

if [[ -n "${TB_DIR}" ]]; then
  ARGS+=(--tensorboard-dir "${TB_DIR}")
fi

if [[ "${USE_DDP}" == "1" ]]; then
  LAUNCH_CMD=(
    torchrun
    --standalone
    --nproc_per_node "${NPROC_PER_NODE}"
    --master_port "${MASTER_PORT}"
    --
    "${TRAINER}"
    "${ARGS[@]}"
  )
  printf 'Torchrun command:\n%s\n' "${LAUNCH_CMD[*]}"
  "${LAUNCH_CMD[@]}"
else
  CMD=(python "${TRAINER}" "${ARGS[@]}")
  printf 'Running command:\n%s\n' "${CMD[*]}"
  "${CMD[@]}"
fi
