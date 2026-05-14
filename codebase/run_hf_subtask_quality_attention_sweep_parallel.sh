#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"

SUMMARY_JSON="${SUMMARY_JSON:-${REPO_ROOT}/data/manifests/hf_manifest_combined_20260322_subtasks_summary.json}"
PARALLEL_SWEEP_ROOT="${PARALLEL_SWEEP_ROOT:-${REPO_ROOT}/outputs/hf_subtask_quality_attention_sweep_parallel}"
SPLIT_BASE_DIR="${SPLIT_BASE_DIR:-${REPO_ROOT}/data/splits_20260322_subtasks}"
TRAIN_BASE_DIR="${TRAIN_BASE_DIR:-${REPO_ROOT}/outputs/mil_12lead_quality_attention}"
INFER_BASE_DIR="${INFER_BASE_DIR:-${REPO_ROOT}/outputs/infer_12lead_quality_attention}"

GPUS="${GPUS:-0,1,2,3,4,5,6,7}"
MAX_JOBS="${MAX_JOBS:-}"
TASKS="${TASKS:-}"

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
USE_TQDM="${USE_TQDM:-0}"
USE_TENSORBOARD="${USE_TENSORBOARD:-1}"
RUN_SPLITS="${RUN_SPLITS:-1}"
RUN_TRAIN="${RUN_TRAIN:-1}"
RUN_INFER="${RUN_INFER:-1}"
SKIP_EXISTING="${SKIP_EXISTING:-1}"

log() {
  printf '[%s] %s\n' "$(date '+%F %T')" "$*"
}

fail() {
  log "ERROR: $*"
  exit 1
}

IFS=',' read -r -a GPU_LIST <<< "${GPUS}"
if [[ "${#GPU_LIST[@]}" -eq 0 ]]; then
  fail "GPUS is empty."
fi

if [[ -z "${MAX_JOBS}" ]]; then
  MAX_JOBS="${#GPU_LIST[@]}"
fi
if (( MAX_JOBS < 1 )); then
  fail "MAX_JOBS must be >= 1."
fi
if (( MAX_JOBS > ${#GPU_LIST[@]} )); then
  MAX_JOBS="${#GPU_LIST[@]}"
fi

[[ -f "${SUMMARY_JSON}" ]] || fail "Missing summary JSON: ${SUMMARY_JSON}. Run codebase/prepare_hf_subtask_manifests.py first."
[[ -f "${ENCODER_CKPT}" ]] || fail "Missing ECGFounder checkpoint: ${ENCODER_CKPT}."

mkdir -p "${PARALLEL_SWEEP_ROOT}/logs" "${SPLIT_BASE_DIR}" "${TRAIN_BASE_DIR}" "${INFER_BASE_DIR}"
TASK_LIST="${PARALLEL_SWEEP_ROOT}/selected_subtasks.txt"
SUMMARY_TSV="${PARALLEL_SWEEP_ROOT}/parallel_summary.tsv"
printf 'subtask\tgpu\tstatus\tjob_root\tlog_file\n' > "${SUMMARY_TSV}"

if ! TASKS="${TASKS}" python3 - "${SUMMARY_JSON}" > "${TASK_LIST}" <<'PY'
import json
import os
import sys
from pathlib import Path

summary = json.loads(Path(sys.argv[1]).read_text(encoding="utf-8"))
wanted_raw = os.environ.get("TASKS", "").strip()
wanted = None
if wanted_raw:
    wanted = {item.strip() for item in wanted_raw.split(",") if item.strip()}
for subtask_name in summary["subtasks"]:
    if wanted is None or subtask_name in wanted:
        print(subtask_name)
PY
then
  fail "Could not parse subtask list from ${SUMMARY_JSON}."
fi

if [[ ! -s "${TASK_LIST}" ]]; then
  fail "No selected subtasks. Check TASKS='${TASKS}'."
fi

log "Parallel quality-attention sweep"
log "GPUs: ${GPUS}; MAX_JOBS=${MAX_JOBS}"
log "Task list: ${TASK_LIST}"
log "Summary: ${SUMMARY_TSV}"

declare -a PIDS=()
declare -a PID_TASKS=()
declare -a PID_GPUS=()
declare -a PID_LOGS=()
declare -a PID_ROOTS=()

wait_batch() {
  local count="$1"
  local idx pid task gpu log_file job_root status
  for ((idx = 0; idx < count; idx++)); do
    pid="${PIDS[$idx]}"
    task="${PID_TASKS[$idx]}"
    gpu="${PID_GPUS[$idx]}"
    log_file="${PID_LOGS[$idx]}"
    job_root="${PID_ROOTS[$idx]}"
    if wait "${pid}"; then
      status="ok"
    else
      status="failed"
      FAILED=1
    fi
    printf '%s\t%s\t%s\t%s\t%s\n' "${task}" "${gpu}" "${status}" "${job_root}" "${log_file}" >> "${SUMMARY_TSV}"
    log "finished task=${task} gpu=${gpu} status=${status}"
  done
  PIDS=()
  PID_TASKS=()
  PID_GPUS=()
  PID_LOGS=()
  PID_ROOTS=()
}

FAILED=0
slot=0
launched=0

while IFS= read -r subtask_name; do
  [[ -n "${subtask_name}" ]] || continue
  gpu="${GPU_LIST[$slot]}"
  job_root="${PARALLEL_SWEEP_ROOT}/jobs/${subtask_name}"
  log_file="${PARALLEL_SWEEP_ROOT}/logs/${subtask_name}.log"
  mkdir -p "${job_root}"

  log "launching task=${subtask_name} gpu=${gpu}"
  (
    CUDA_VISIBLE_DEVICES="${gpu}" \
    SUMMARY_JSON="${SUMMARY_JSON}" \
    SWEEP_ROOT="${job_root}" \
    SPLIT_BASE_DIR="${SPLIT_BASE_DIR}" \
    TRAIN_BASE_DIR="${TRAIN_BASE_DIR}" \
    INFER_BASE_DIR="${INFER_BASE_DIR}" \
    TASKS="${subtask_name}" \
    HF_ROOT="${HF_ROOT}" \
    ENCODER_CKPT="${ENCODER_CKPT}" \
    NUM_CLASSES="${NUM_CLASSES}" \
    CLASS_NAMES="${CLASS_NAMES}" \
    IN_CHANNELS="${IN_CHANNELS}" \
    EMBEDDING_DIM="${EMBEDDING_DIM}" \
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
    RUN_SPLITS="${RUN_SPLITS}" \
    RUN_TRAIN="${RUN_TRAIN}" \
    RUN_INFER="${RUN_INFER}" \
    SKIP_EXISTING="${SKIP_EXISTING}" \
    bash "${SCRIPT_DIR}/run_hf_subtask_quality_attention_sweep.sh"
  ) > "${log_file}" 2>&1 &

  PIDS+=("$!")
  PID_TASKS+=("${subtask_name}")
  PID_GPUS+=("${gpu}")
  PID_LOGS+=("${log_file}")
  PID_ROOTS+=("${job_root}")

  launched=$((launched + 1))
  slot=$((slot + 1))
  if (( slot >= MAX_JOBS )); then
    wait_batch "${#PIDS[@]}"
    slot=0
  fi
done < "${TASK_LIST}"

if (( ${#PIDS[@]} > 0 )); then
  wait_batch "${#PIDS[@]}"
fi

log "Launched ${launched} selected subtasks."
log "Parallel summary: ${SUMMARY_TSV}"
exit "${FAILED}"
