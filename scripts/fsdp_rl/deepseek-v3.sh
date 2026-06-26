#!/bin/bash
# DeepSeek-V3 671B MoE -- HUGE, multi-node (8x8=64); pure-FSDP is aggressive here (no EP/PP), sized for weights+grads+sglang with optimizer on CPU
# GPUs: 8node x 8 = 64  (optimizer/params on CPU via --fsdp-cpu-offload)
export RUN_ID=deepseek-v3
export MODEL=DeepSeek-V3
export NNODES=8 GPUS_PER_NODE=8 CPU_OFFLOAD=1 MAX_TOKENS_PER_GPU=8192 SGLANG_MEM=0.4
source "$(dirname "$0")/common.sh"
