#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

CSV_PATH="${CSV_PATH:-}"
NPY_ROOT="${NPY_ROOT:-}"
CKPT_PATH="${CKPT_PATH:-}"
OUT_DIR="${OUT_DIR:-${SCRIPT_DIR}/outputs/mil_inference}"

NUM_CLASSES="${NUM_CLASSES:-4}"
CLASS_NAMES="${CLASS_NAMES:-}"
IN_CHANNELS="${IN_CHANNELS:-12}"
EMBEDDING_DIM="${EMBEDDING_DIM:-1024}"
LEAD_MODE="${LEAD_MODE:-12}"
SEG_SEC="${SEG_SEC:-4}"
K_SEGMENTS="${K_SEGMENTS:-16}"
BASE_SEED="${BASE_SEED:-1234}"
POOL_TYPE="${POOL_TYPE:-attention}"
QUALITY_DIM="${QUALITY_DIM:-4}"
QUALITY_ALPHA="${QUALITY_ALPHA:-1.0}"
TOPK="${TOPK:-4}"
MIX_BETA="${MIX_BETA:-0.5}"
SAVE_POOL_OUTPUTS="${SAVE_POOL_OUTPUTS:-0}"
BATCH_SIZE="${BATCH_SIZE:-16}"
NUM_WORKERS="${NUM_WORKERS:-4}"
DEVICE="${DEVICE:-auto}"
STRICT_CKPT="${STRICT_CKPT:-0}"

if [[ -z "${CSV_PATH}" || -z "${NPY_ROOT}" || -z "${CKPT_PATH}" ]]; then
  cat <<'EOF'
Required environment variables:
  CSV_PATH   path to the inference CSV manifest
  NPY_ROOT   root directory containing ECG signal files referenced by the manifest
  CKPT_PATH  path to the MIL model checkpoint

Optional environment variables:
  OUT_DIR        output directory                      default: ./outputs/mil_inference
  NUM_CLASSES    number of classes                    default: 4
  CLASS_NAMES    quoted, space-separated class names  default: empty
  IN_CHANNELS    encoder input channels               default: 12
                 use 1 when LEAD_MODE is <lead>_1ch
  EMBEDDING_DIM  encoder embedding dim                default: 1024
  LEAD_MODE      12 | I | II | <lead>_1ch            default: 12
  SEG_SEC        segment length in seconds            default: 4
  K_SEGMENTS     number of MIL segments               default: 16
  BASE_SEED      deterministic sampling seed          default: 1234
  POOL_TYPE      attention | quality_attention | hybrid
  QUALITY_DIM    segment-quality feature count        default: 4
  QUALITY_ALPHA  scaling factor for quality logits    default: 1.0
  TOPK           top-k size for hybrid pooling        default: 4
  MIX_BETA       attention/top-k mixing ratio         default: 0.5
  SAVE_POOL_OUTPUTS 1 to save pooling diagnostics     default: 0
  BATCH_SIZE     inference batch size                 default: 16
  NUM_WORKERS    dataloader workers                   default: 4
  DEVICE         auto | cpu | cuda | cuda:0 ...      default: auto
  STRICT_CKPT    1 to require exact key match         default: 0

Example:
  CSV_PATH=/path/test.csv \
  NPY_ROOT=/path/npy_root \
  CKPT_PATH=/path/model.pt \
  OUT_DIR=/path/out \
  NUM_CLASSES=4 \
  CLASS_NAMES="normal hfref hfmr hfpef" \
  bash run_mil_inference.sh
EOF
  exit 1
fi

mkdir -p "${OUT_DIR}"

CMD=(
  python "${SCRIPT_DIR}/MIL_classification_inference.py"
  --csv-path "${CSV_PATH}"
  --root "${NPY_ROOT}"
  --ckpt "${CKPT_PATH}"
  --out-dir "${OUT_DIR}"
  --num-classes "${NUM_CLASSES}"
  --in-channels "${IN_CHANNELS}"
  --embedding-dim "${EMBEDDING_DIM}"
  --lead-mode "${LEAD_MODE}"
  --seg-sec "${SEG_SEC}"
  --K "${K_SEGMENTS}"
  --base-seed "${BASE_SEED}"
  --pool-type "${POOL_TYPE}"
  --quality-dim "${QUALITY_DIM}"
  --quality-alpha "${QUALITY_ALPHA}"
  --topk "${TOPK}"
  --mix-beta "${MIX_BETA}"
  --batch-size "${BATCH_SIZE}"
  --num-workers "${NUM_WORKERS}"
  --device "${DEVICE}"
)

if [[ -n "${CLASS_NAMES}" ]]; then
  # shellcheck disable=SC2206
  CLASS_NAMES_ARR=(${CLASS_NAMES})
  CMD+=(--class-names "${CLASS_NAMES_ARR[@]}")
fi

if [[ "${STRICT_CKPT}" == "1" ]]; then
  CMD+=(--strict-ckpt)
fi

if [[ "${SAVE_POOL_OUTPUTS}" == "1" ]]; then
  CMD+=(--save-pool-outputs)
fi

printf 'Running command:\n%s\n' "${CMD[*]}"
"${CMD[@]}"
