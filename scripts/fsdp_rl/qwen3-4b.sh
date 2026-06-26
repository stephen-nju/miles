#!/bin/bash
# Qwen3-4B dense (validated)
# GPUs: 1node x 4 = 4
export RUN_ID=qwen3-4b
export MODEL=Qwen3-4B
export NNODES=1 GPUS_PER_NODE=4 CPU_OFFLOAD=0 MAX_TOKENS_PER_GPU=16384 SGLANG_MEM=0.6
source "$(dirname "$0")/common.sh"
