#!/usr/bin/env bash
set -uo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"

SUMMARY_JSON="${SUMMARY_JSON:-${REPO_ROOT}/data/manifests/hf_manifest_combined_20260322_subtasks_summary.json}"
SWEEP_ROOT="${SWEEP_ROOT:-${REPO_ROOT}/outputs/hf_subtask_quality_attention_sweep}"
SPLIT_BASE_DIR="${SPLIT_BASE_DIR:-${REPO_ROOT}/data/splits_20260322_subtasks}"
TRAIN_BASE_DIR="${TRAIN_BASE_DIR:-${REPO_ROOT}/outputs/mil_12lead_quality_attention}"
INFER_BASE_DIR="${INFER_BASE_DIR:-${REPO_ROOT}/outputs/infer_12lead_quality_attention}"
LOG_DIR="${LOG_DIR:-${SWEEP_ROOT}/logs}"

HF_ROOT="${HF_ROOT:-${REPO_ROOT}/data}"
ENCODER_CKPT="${ENCODER_CKPT:-${REPO_ROOT}/data/pretrained_ckpt/12_lead_ECGFounder.pth}"

NUM_CLASSES="${NUM_CLASSES:-4}"
CLASS_NAMES="${CLASS_NAMES:-normal hfref hfmr hfpef}"
IN_CHANNELS="${IN_CHANNELS:-12}"
EMBEDDING_DIM="${EMBEDDING_DIM:-1024}"
LEAD_MODE="${LEAD_MODE:-12}"
POOL_TYPE="${POOL_TYPE:-quality_attention}"
QUALITY_ALPHA="${QUALITY_ALPHA:-1.0}"
TOPK="${TOPK:-4}"
MIX_BETA="${MIX_BETA:-0.5}"
BATCH_SIZE="${BATCH_SIZE:-128}"
EVAL_BATCH_SIZE="${EVAL_BATCH_SIZE:-${BATCH_SIZE}}"
EPOCHS="${EPOCHS:-5}"
SEED="${SEED:-1234}"
BASE_SEED="${BASE_SEED:-1234}"
NUM_WORKERS="${NUM_WORKERS:-4}"
USE_TQDM="${USE_TQDM:-1}"
USE_TENSORBOARD="${USE_TENSORBOARD:-1}"

RUN_SPLITS="${RUN_SPLITS:-1}"
RUN_TRAIN="${RUN_TRAIN:-1}"
RUN_INFER="${RUN_INFER:-1}"
SKIP_EXISTING="${SKIP_EXISTING:-1}"
TASKS="${TASKS:-}"

log() {
  local message
  message="[$(date '+%F %T')] $*"
  printf '%s\n' "${message}"
  if [[ -n "${SWEEP_LOG:-}" ]]; then
    printf '%s\n' "${message}" >> "${SWEEP_LOG}"
  fi
}

fail() {
  log "ERROR: $*"
  exit 1
}

trim_leading() {
  local value="$1"
  value="${value#"${value%%[![:space:]]*}"}"
  printf '%s' "$value"
}

trim_trailing() {
  local value="$1"
  value="${value%"${value##*[![:space:]]}"}"
  printf '%s' "$value"
}

task_selected() {
  local name="$1"
  if [[ -z "${TASKS}" ]]; then
    return 0
  fi
  local IFS=','
  read -r -a requested_tasks <<< "${TASKS}"
  local wanted
  for wanted in "${requested_tasks[@]}"; do
    wanted="$(trim_trailing "$(trim_leading "${wanted}")")"
    if [[ -n "${wanted}" && "${wanted}" == "${name}" ]]; then
      return 0
    fi
  done
  return 1
}

[[ -f "${SUMMARY_JSON}" ]] || fail "Missing summary JSON: ${SUMMARY_JSON}. Run codebase/prepare_hf_subtask_manifests.py first."
[[ -f "${ENCODER_CKPT}" ]] || fail "Missing ECGFounder checkpoint: ${ENCODER_CKPT}."

mkdir -p "${SWEEP_ROOT}" "${SPLIT_BASE_DIR}" "${TRAIN_BASE_DIR}" "${INFER_BASE_DIR}" "${LOG_DIR}"
SWEEP_LOG="${SWEEP_ROOT}/sweep.log"
: > "${SWEEP_LOG}"

summary_tsv="${SWEEP_ROOT}/sweep_summary.tsv"
printf 'subtask\tmanifest\trows\tsplit_status\ttrain_status\tinfer_status\tsplit_dir\ttrain_dir\tinfer_dir\n' > "${summary_tsv}"

log "Starting HF 12-lead quality-attention sweep"
log "Summary JSON: ${SUMMARY_JSON}"
log "HF root: ${HF_ROOT}"
log "Encoder checkpoint: ${ENCODER_CKPT}"
log "Sweep root: ${SWEEP_ROOT}"

task_list_tsv="${SWEEP_ROOT}/subtasks.tsv"
if ! python3 - "${SUMMARY_JSON}" "${REPO_ROOT}" > "${task_list_tsv}" <<'PY'
import json
import sys
from pathlib import Path

summary_path = Path(sys.argv[1])
repo_root = Path(sys.argv[2])
summary = json.loads(summary_path.read_text(encoding="utf-8"))
for subtask_name, spec in summary["subtasks"].items():
    manifest = Path(spec["output_csv"])
    if not manifest.is_absolute():
        manifest = (repo_root / manifest).resolve()
    rows = spec.get("rows", "")
    print(f"{subtask_name}\t{manifest}\t{rows}")
PY
then
  fail "Could not parse subtask list from ${SUMMARY_JSON}."
fi

if [[ ! -s "${task_list_tsv}" ]]; then
  fail "No subtask manifests found in ${SUMMARY_JSON}."
fi

processed=0
failed=0

while IFS= read -r line; do
  IFS=$'\t' read -r subtask_name manifest_csv row_count <<< "${line}"
  [[ -n "${subtask_name}" ]] || continue
  if ! task_selected "${subtask_name}"; then
    continue
  fi

  processed=$((processed + 1))
  split_dir="${SPLIT_BASE_DIR}/${subtask_name}"
  train_dir="${TRAIN_BASE_DIR}/${subtask_name}"
  infer_dir="${INFER_BASE_DIR}/${subtask_name}"
  split_log="${LOG_DIR}/${subtask_name}_split.log"
  train_log="${LOG_DIR}/${subtask_name}_train.log"
  infer_log="${LOG_DIR}/${subtask_name}_infer.log"

  split_status="skipped"
  train_status="skipped"
  infer_status="skipped"

  log "[$processed] ${subtask_name} (rows=${row_count})"
  mkdir -p "${split_dir}" "${train_dir}" "${infer_dir}"

  if [[ "${RUN_SPLITS}" == "1" ]]; then
    if [[ "${SKIP_EXISTING}" == "1" && -f "${split_dir}/train.csv" && -f "${split_dir}/valid.csv" && -f "${split_dir}/test.csv" ]]; then
      split_status="existing"
      log "  split: existing"
    else
      log "  split: creating"
      if python3 "${SCRIPT_DIR}/prepare_hf_patient_splits.py" \
        --manifest-csv "${manifest_csv}" \
        --output-dir "${split_dir}" \
        --seed "${BASE_SEED}" \
        >"${split_log}" 2>&1; then
        split_status="ok"
      else
        split_status="failed"
        failed=1
        log "  split failed, see ${split_log}"
      fi
    fi
  else
    split_status="disabled"
  fi

  if [[ "${RUN_TRAIN}" == "1" ]]; then
    if [[ "${split_status}" == failed* ]]; then
      train_status="skipped:no_split"
      log "  train: skipped because split failed"
    elif [[ "${SKIP_EXISTING}" == "1" && -f "${train_dir}/best_model.pt" ]]; then
      train_status="existing"
      log "  train: existing"
    else
      log "  train: running"
      if TRAIN_CSV="${split_dir}/train.csv" \
         VALID_CSV="${split_dir}/valid.csv" \
         NPY_ROOT="${HF_ROOT}" \
         OUT_DIR="${train_dir}" \
         NUM_CLASSES="${NUM_CLASSES}" \
         CLASS_NAMES="${CLASS_NAMES}" \
         ENCODER_CKPT="${ENCODER_CKPT}" \
         EMBEDDING_DIM="${EMBEDDING_DIM}" \
         IN_CHANNELS="${IN_CHANNELS}" \
         LEAD_MODE="${LEAD_MODE}" \
         POOL_TYPE="${POOL_TYPE}" \
         QUALITY_ALPHA="${QUALITY_ALPHA}" \
         TOPK="${TOPK}" \
         MIX_BETA="${MIX_BETA}" \
         BATCH_SIZE="${BATCH_SIZE}" \
         EVAL_BATCH_SIZE="${EVAL_BATCH_SIZE}" \
         EPOCHS="${EPOCHS}" \
         SEED="${SEED}" \
         BASE_SEED="${BASE_SEED}" \
         NUM_WORKERS="${NUM_WORKERS}" \
         USE_TQDM="${USE_TQDM}" \
         USE_TENSORBOARD="${USE_TENSORBOARD}" \
         bash "${SCRIPT_DIR}/run_mil_train.sh" \
         >"${train_log}" 2>&1; then
        train_status="ok"
      else
        train_status="failed"
        failed=1
        log "  train failed, see ${train_log}"
      fi
    fi
  else
    train_status="disabled"
  fi

  if [[ "${RUN_INFER}" == "1" ]]; then
    if [[ "${train_status}" == failed* || "${train_status}" == "skipped:no_split" || "${train_status}" == "disabled" ]]; then
      infer_status="skipped:no_model"
      log "  infer: skipped because the checkpoint is unavailable"
    elif [[ "${SKIP_EXISTING}" == "1" && -f "${infer_dir}/predictions.csv" ]]; then
      infer_status="existing"
      log "  infer: existing"
    elif [[ ! -f "${train_dir}/best_model.pt" ]]; then
      infer_status="skipped:no_model"
      log "  infer: skipped because best_model.pt is missing"
    else
      log "  infer: running"
      if CSV_PATH="${split_dir}/test.csv" \
         NPY_ROOT="${HF_ROOT}" \
         CKPT_PATH="${train_dir}/best_model.pt" \
         OUT_DIR="${infer_dir}" \
         NUM_CLASSES="${NUM_CLASSES}" \
         CLASS_NAMES="${CLASS_NAMES}" \
         IN_CHANNELS="${IN_CHANNELS}" \
         LEAD_MODE="${LEAD_MODE}" \
         POOL_TYPE="${POOL_TYPE}" \
         QUALITY_ALPHA="${QUALITY_ALPHA}" \
         TOPK="${TOPK}" \
         MIX_BETA="${MIX_BETA}" \
         BATCH_SIZE="${EVAL_BATCH_SIZE}" \
         SEED="${SEED}" \
         BASE_SEED="${BASE_SEED}" \
         bash "${SCRIPT_DIR}/run_mil_inference.sh" \
         >"${infer_log}" 2>&1; then
        infer_status="ok"
      else
        infer_status="failed"
        failed=1
        log "  infer failed, see ${infer_log}"
      fi
    fi
  else
    infer_status="disabled"
  fi

  printf '%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\n' \
    "${subtask_name}" \
    "${manifest_csv}" \
    "${row_count}" \
    "${split_status}" \
    "${train_status}" \
    "${infer_status}" \
    "${split_dir}" \
    "${train_dir}" \
    "${infer_dir}" >> "${summary_tsv}"

  log "  done: split=${split_status} train=${train_status} infer=${infer_status}"
done < "${task_list_tsv}"

log "Sweep completed for ${processed} subtasks."
log "Sweep summary: ${summary_tsv}"
log "Top log: ${SWEEP_LOG}"
exit "${failed}"
