#!/bin/bash
# Gemma-4-26B-A4B MoE
# GPUs: 1node x 8 = 8  (optimizer/params on CPU via --fsdp-cpu-offload)
export RUN_ID=gemma-4-26b-a4b
export MODEL=gemma-4-26b-a4b-it
export NNODES=1 GPUS_PER_NODE=8 CPU_OFFLOAD=1 MAX_TOKENS_PER_GPU=10240 SGLANG_MEM=0.5
source "$(dirname "$0")/common.sh"
