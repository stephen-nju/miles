#!/bin/bash
# gpt-oss-20B MoE
# GPUs: 1node x 8 = 8
export RUN_ID=gpt-oss-20b
export MODEL=gpt-oss-20b-bf16
export NNODES=1 GPUS_PER_NODE=8 CPU_OFFLOAD=0 MAX_TOKENS_PER_GPU=12288 SGLANG_MEM=0.5
source "$(dirname "$0")/common.sh"
