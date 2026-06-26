#!/bin/bash
# Gemma-4-31B dense
# GPUs: 1node x 8 = 8  (optimizer/params on CPU via --fsdp-cpu-offload)
export RUN_ID=gemma-4-31b
export MODEL=gemma-4-31b-it
export NNODES=1 GPUS_PER_NODE=8 CPU_OFFLOAD=1 MAX_TOKENS_PER_GPU=12288 SGLANG_MEM=0.5
source "$(dirname "$0")/common.sh"
