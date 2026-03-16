#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
LONG_BENCH_DIR="$(cd "${SCRIPT_DIR}/.." && pwd)"

MODEL_NAME="${MODEL_NAME:-mistral-7B-instruct-v0.2}"
MODEL_PATH="${MODEL_PATH:-/userhome/models/Mistral-7B-Instruct-v0.2}"
DATA_ROOT="${DATA_ROOT:-/userhome/datasets/LongBench}"
COMPRESS_ARGS_PATH="${COMPRESS_ARGS_PATH:-ablation_c4096_w32_k7_maxpool.json}"
DATASET_NAME="${DATASET_NAME:-qasper}"
CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}"

cd "${LONG_BENCH_DIR}"

export CUDA_VISIBLE_DEVICES

python pred_snap.py \
  --model "${MODEL_NAME}" \
  --model-path-override "${MODEL_PATH}" \
  --data-root "${DATA_ROOT}" \
  --local-files-only \
  --dataset "${DATASET_NAME}" \
  --compress_args_path "${COMPRESS_ARGS_PATH}"
