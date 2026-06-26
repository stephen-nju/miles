#!/bin/bash
# Qwen3-Next-80B-A3B GatedDeltaNet MoE (single node + CPU offload)
# GPUs: 1node x 8 = 8  (optimizer/params on CPU via --fsdp-cpu-offload)
export RUN_ID=qwen3-next-80b-a3b
export MODEL=Qwen3-Next-80B-A3B
export NNODES=1 GPUS_PER_NODE=8 CPU_OFFLOAD=1 MAX_TOKENS_PER_GPU=8192 SGLANG_MEM=0.45
source "$(dirname "$0")/common.sh"
