#!/bin/bash
# Qwen3.5-35B-A3B GatedDeltaNet MoE (qwen3_5_moe)
# GPUs: 1node x 8 = 8  (optimizer/params on CPU via --fsdp-cpu-offload)
export RUN_ID=qwen3.5-35b-a3b
export MODEL=Qwen3.5-35B-A3B
export NNODES=1 GPUS_PER_NODE=8 CPU_OFFLOAD=1 MAX_TOKENS_PER_GPU=10240 SGLANG_MEM=0.5
source "$(dirname "$0")/common.sh"
