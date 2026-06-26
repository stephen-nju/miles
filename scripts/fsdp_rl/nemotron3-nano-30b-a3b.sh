#!/bin/bash
# Nemotron-3-Nano-30B-A3B hybrid MoE (nemotron_h)
# GPUs: 1node x 8 = 8  (optimizer/params on CPU via --fsdp-cpu-offload)
export RUN_ID=nemotron3-nano-30b-a3b
export MODEL=NVIDIA-Nemotron-3-Nano-30B-A3B-BF16
export NNODES=1 GPUS_PER_NODE=8 CPU_OFFLOAD=1 MAX_TOKENS_PER_GPU=10240 SGLANG_MEM=0.5
source "$(dirname "$0")/common.sh"
