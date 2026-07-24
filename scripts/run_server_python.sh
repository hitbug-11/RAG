#!/usr/bin/env bash
set -euo pipefail

readonly RAG_SERVER_ROOT="/data/haojiachen/rag"

mkdir -p \
  "${RAG_SERVER_ROOT}/models/huggingface/hub" \
  "${RAG_SERVER_ROOT}/models/huggingface/xet" \
  "${RAG_SERVER_ROOT}/models/torch" \
  "${RAG_SERVER_ROOT}/models/cache" \
  "${RAG_SERVER_ROOT}/tmp"

export HF_HOME="${RAG_SERVER_ROOT}/models/huggingface"
export HF_HUB_CACHE="${RAG_SERVER_ROOT}/models/huggingface/hub"
export HF_XET_CACHE="${RAG_SERVER_ROOT}/models/huggingface/xet"
export HF_HUB_DISABLE_XET="1"
export HF_HUB_DOWNLOAD_TIMEOUT="120"
export TORCH_HOME="${RAG_SERVER_ROOT}/models/torch"
export XDG_CACHE_HOME="${RAG_SERVER_ROOT}/models/cache"
export TMPDIR="${RAG_SERVER_ROOT}/tmp"
export CUDA_VISIBLE_DEVICES="0"
export PYTHONUNBUFFERED="1"

source /data/anaconda3/etc/profile.d/conda.sh
conda activate ibqw
cd "${RAG_SERVER_ROOT}"

exec python -u "$@"
