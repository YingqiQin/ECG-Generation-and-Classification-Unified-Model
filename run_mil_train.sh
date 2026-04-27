#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

TRAIN_CSV="${TRAIN_CSV:-}"
VALID_CSV="${VALID_CSV:-}"
NPY_ROOT="${NPY_ROOT:-}"
OUT_DIR="${OUT_DIR:-${SCRIPT_DIR}/outputs/mil_train}"

NUM_CLASSES="${NUM_CLASSES:-4}"
CLASS_NAMES="${CLASS_NAMES:-}"
CLASS_WEIGHTS="${CLASS_WEIGHTS:-}"
IN_CHANNELS="${IN_CHANNELS:-12}"
EMBEDDING_DIM="${EMBEDDING_DIM:-1024}"
LEAD_MODE="${LEAD_MODE:-12}"
SEG_SEC="${SEG_SEC:-4}"
K_SEGMENTS="${K_SEGMENTS:-16}"
BASE_SEED="${BASE_SEED:-1234}"
SEED="${SEED:-1234}"

POOL_TYPE="${POOL_TYPE:-attention}"
POOL_HIDDEN="${POOL_HIDDEN:-128}"
DROPOUT="${DROPOUT:-0.1}"
QUALITY_DIM="${QUALITY_DIM:-4}"
QUALITY_HIDDEN="${QUALITY_HIDDEN:-32}"
QUALITY_ALPHA="${QUALITY_ALPHA:-1.0}"
TOPK="${TOPK:-4}"
MIX_BETA="${MIX_BETA:-0.5}"

EPOCHS="${EPOCHS:-30}"
BATCH_SIZE="${BATCH_SIZE:-16}"
EVAL_BATCH_SIZE="${EVAL_BATCH_SIZE:-16}"
NUM_WORKERS="${NUM_WORKERS:-4}"
LR="${LR:-1e-4}"
WEIGHT_DECAY="${WEIGHT_DECAY:-1e-4}"
GRAD_CLIP="${GRAD_CLIP:-1.0}"
DEVICE="${DEVICE:-auto}"
MONITOR="${MONITOR:-balanced_accuracy}"
USE_TQDM="${USE_TQDM:-1}"
TQDM_UPDATE_INTERVAL="${TQDM_UPDATE_INTERVAL:-10}"
USE_TENSORBOARD="${USE_TENSORBOARD:-1}"
TB_DIR="${TB_DIR:-}"
TB_FLUSH_SECS="${TB_FLUSH_SECS:-30}"

ENCODER_CKPT="${ENCODER_CKPT:-}"
INIT_CKPT="${INIT_CKPT:-}"
STRICT_CKPT="${STRICT_CKPT:-0}"
TRAIN_WEIGHT_CSV="${TRAIN_WEIGHT_CSV:-}"
TRAIN_WEIGHT_KEY="${TRAIN_WEIGHT_KEY:-xml_file}"
TRAIN_WEIGHT_COLUMN="${TRAIN_WEIGHT_COLUMN:-weight}"
MIN_TRAIN_WEIGHT="${MIN_TRAIN_WEIGHT:-0.1}"
MAX_TRAIN_WEIGHT="${MAX_TRAIN_WEIGHT:-1.0}"

if [[ -z "${TRAIN_CSV}" || -z "${VALID_CSV}" || -z "${NPY_ROOT}" ]]; then
  cat <<'EOF'
Required environment variables:
  TRAIN_CSV  path to the training CSV manifest
  VALID_CSV  path to the validation CSV manifest
  NPY_ROOT   root directory containing ECG signal files referenced by the manifest

Optional environment variables:
  OUT_DIR            output directory                      default: ./outputs/mil_train
  NUM_CLASSES        number of classes                    default: 4
  CLASS_NAMES        quoted, space-separated class names  default: empty
  CLASS_WEIGHTS      quoted, space-separated weights      default: empty
  IN_CHANNELS        encoder input channels               default: 12
                    use 1 when LEAD_MODE is <lead>_1ch
  EMBEDDING_DIM      encoder embedding dim                default: 1024
  LEAD_MODE          12 | I | II | <lead>_1ch            default: 12
  SEG_SEC            segment length in seconds            default: 4
  K_SEGMENTS         number of MIL segments               default: 16
  BASE_SEED          dataset sampling seed                default: 1234
  SEED               global random seed                   default: 1234
  POOL_TYPE          attention | quality_attention | hybrid
  QUALITY_ALPHA      scaling factor for quality logits
  TOPK               top-k size for hybrid pooling
  MIX_BETA           attention/top-k mixing ratio
  EPOCHS             number of epochs                     default: 30
  BATCH_SIZE         train batch size                     default: 16
  EVAL_BATCH_SIZE    eval batch size                      default: 16
  LR                 AdamW learning rate                  default: 1e-4
  WEIGHT_DECAY       AdamW weight decay                   default: 1e-4
  DEVICE             auto | cpu | cuda | cuda:0 ...      default: auto
  MONITOR            validation metric for best model     default: balanced_accuracy
  USE_TQDM           1 enables tqdm progress bars         default: 1
  TQDM_UPDATE_INTERVAL tqdm postfix update interval       default: 10
  USE_TENSORBOARD    1 writes TensorBoard event files     default: 1
  TB_DIR             optional TensorBoard log directory
  TB_FLUSH_SECS      TensorBoard flush interval           default: 30
  ENCODER_CKPT       optional encoder checkpoint
  INIT_CKPT          optional full model checkpoint
  TRAIN_WEIGHT_CSV   optional reliability-weight CSV

Example:
  TRAIN_CSV=/path/train.csv \
  VALID_CSV=/path/valid.csv \
  NPY_ROOT=/path/npy_root \
  OUT_DIR=/path/out \
  POOL_TYPE=hybrid \
  QUALITY_ALPHA=1.0 \
  TOPK=4 \
  MIX_BETA=0.5 \
  NUM_CLASSES=4 \
  CLASS_NAMES="normal hfref hfmr hfpef" \
  bash run_mil_train.sh
EOF
  exit 1
fi

mkdir -p "${OUT_DIR}"

CMD=(
  python "${SCRIPT_DIR}/train_mil_classification.py"
  --train-csv "${TRAIN_CSV}"
  --valid-csv "${VALID_CSV}"
  --root "${NPY_ROOT}"
  --out-dir "${OUT_DIR}"
  --num-classes "${NUM_CLASSES}"
  --in-channels "${IN_CHANNELS}"
  --embedding-dim "${EMBEDDING_DIM}"
  --lead-mode "${LEAD_MODE}"
  --seg-sec "${SEG_SEC}"
  --K "${K_SEGMENTS}"
  --base-seed "${BASE_SEED}"
  --seed "${SEED}"
  --pool-type "${POOL_TYPE}"
  --pool-hidden "${POOL_HIDDEN}"
  --dropout "${DROPOUT}"
  --quality-dim "${QUALITY_DIM}"
  --quality-hidden "${QUALITY_HIDDEN}"
  --quality-alpha "${QUALITY_ALPHA}"
  --topk "${TOPK}"
  --mix-beta "${MIX_BETA}"
  --epochs "${EPOCHS}"
  --batch-size "${BATCH_SIZE}"
  --eval-batch-size "${EVAL_BATCH_SIZE}"
  --num-workers "${NUM_WORKERS}"
  --lr "${LR}"
  --weight-decay "${WEIGHT_DECAY}"
  --grad-clip "${GRAD_CLIP}"
  --device "${DEVICE}"
  --monitor "${MONITOR}"
  --tqdm-update-interval "${TQDM_UPDATE_INTERVAL}"
  --tb-flush-secs "${TB_FLUSH_SECS}"
  --train-weight-key "${TRAIN_WEIGHT_KEY}"
  --train-weight-column "${TRAIN_WEIGHT_COLUMN}"
  --min-train-weight "${MIN_TRAIN_WEIGHT}"
  --max-train-weight "${MAX_TRAIN_WEIGHT}"
)

if [[ -n "${CLASS_NAMES}" ]]; then
  # shellcheck disable=SC2206
  CLASS_NAMES_ARR=(${CLASS_NAMES})
  CMD+=(--class-names "${CLASS_NAMES_ARR[@]}")
fi

if [[ -n "${CLASS_WEIGHTS}" ]]; then
  # shellcheck disable=SC2206
  CLASS_WEIGHTS_ARR=(${CLASS_WEIGHTS})
  CMD+=(--class-weights "${CLASS_WEIGHTS_ARR[@]}")
fi

if [[ -n "${ENCODER_CKPT}" ]]; then
  CMD+=(--encoder-ckpt "${ENCODER_CKPT}")
fi

if [[ -n "${INIT_CKPT}" ]]; then
  CMD+=(--init-ckpt "${INIT_CKPT}")
fi

if [[ "${STRICT_CKPT}" == "1" ]]; then
  CMD+=(--strict-ckpt)
fi

if [[ -n "${TRAIN_WEIGHT_CSV}" ]]; then
  CMD+=(--train-weight-csv "${TRAIN_WEIGHT_CSV}")
fi

if [[ "${USE_TQDM}" != "1" ]]; then
  CMD+=(--no-tqdm)
fi

if [[ "${USE_TENSORBOARD}" != "1" ]]; then
  CMD+=(--no-tensorboard)
fi

if [[ -n "${TB_DIR}" ]]; then
  CMD+=(--tensorboard-dir "${TB_DIR}")
fi

printf 'Running command:\n%s\n' "${CMD[*]}"
"${CMD[@]}"
