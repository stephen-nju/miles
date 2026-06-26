#!/bin/bash
# GLM-4.7-Flash MoE (glm4_moe_lite; fp32-master)
# GPUs: 1node x 8 = 8  (optimizer/params on CPU via --fsdp-cpu-offload)
export RUN_ID=glm4.7-flash
export MODEL=GLM-4.7-Flash
export NNODES=1 GPUS_PER_NODE=8 CPU_OFFLOAD=1 MAX_TOKENS_PER_GPU=10240 SGLANG_MEM=0.5
source "$(dirname "$0")/common.sh"
