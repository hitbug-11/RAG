#!/usr/bin/env bash
set -euo pipefail

readonly RAG_ROOT="/data/haojiachen/rag"
readonly INPUT="${RAG_ROOT}/results/ragc_reproduction/nq_paper_generation_inputs.json"
readonly GENERATIONS="${RAG_ROOT}/results/ragc_reproduction/nq_qwen3_8b_generations.jsonl"
readonly JUDGMENTS="${RAG_ROOT}/results/ragc_reproduction/nq_qwen3_8b_judgments.jsonl"
readonly METRICS="${RAG_ROOT}/results/ragc_reproduction/nq_qwen3_8b_metrics.json"
readonly RUNNER="${RAG_ROOT}/scripts/run_server_python.sh"
readonly SCRIPT="${RAG_ROOT}/scripts/RAG_C/reproduce_end_to_end.py"

export HF_HUB_OFFLINE="1"

cd "${RAG_ROOT}"

bash "${RUNNER}" "${SCRIPT}" generate-qwen \
  --input "${INPUT}" \
  --output "${GENERATIONS}" \
  --temperature 0.1 \
  --max-tokens 512 \
  --seed 100

bash "${RUNNER}" "${SCRIPT}" judge-qwen \
  --input "${INPUT}" \
  --generations "${GENERATIONS}" \
  --output "${JUDGMENTS}" \
  --temperature 0.1 \
  --max-tokens 10 \
  --seed 100 \
  --repeats 3

bash "${RUNNER}" "${SCRIPT}" evaluate \
  --input "${INPUT}" \
  --generations "${GENERATIONS}" \
  --judgments "${JUDGMENTS}" \
  --output "${METRICS}"
