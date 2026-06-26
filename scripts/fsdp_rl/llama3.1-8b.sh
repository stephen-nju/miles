#!/bin/bash
# Llama-3.1-8B dense
# GPUs: 1node x 4 = 4
export RUN_ID=llama3.1-8b
export MODEL=Llama-3.1-8B-Instruct
export NNODES=1 GPUS_PER_NODE=4 CPU_OFFLOAD=0 MAX_TOKENS_PER_GPU=16384 SGLANG_MEM=0.6
source "$(dirname "$0")/common.sh"
