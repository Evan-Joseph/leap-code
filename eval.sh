#!/bin/bash

# Leap-Code Evaluation Master Launcher
# Usage: bash eval.sh <model_key>

MODEL_KEY=$1

case $MODEL_KEY in
    "qwen3_2b_0shot")
        bash scripts/shell/eval_qwen3_2b_0shot_batch.sh
        ;;
    "qwen3_2b_1shot")
        bash scripts/shell/eval_qwen3_2b_1shot_batch.sh
        ;;
    "qwen3_4b")
        bash scripts/shell/eval_qwen3_4b_0shot_single.sh
        ;;
    "internvl")
        bash scripts/shell/eval_internvl2.5_2b_0shot_parallel.sh
        ;;
    "minicpm")
        bash scripts/shell/eval_minicpm_v2.6_0shot_4bit_parallel.sh
        ;;
    "qwen3_2b")
        bash scripts/shell/eval_qwen3_2b_0shot_parallel.sh
        ;;
    "qwen2_2b")
        bash scripts/shell/eval_qwen2_2b_0shot_parallel.sh
        ;;
    "qwen2.5_3b")
        bash scripts/shell/eval_qwen2.5_3b_0shot_parallel.sh
        ;;
    "deepseek")
        bash scripts/shell/eval_deepseek_vl2_small_0shot_parallel.sh
        ;;
    "qwen3_2b_batch")
        bash scripts/shell/eval_qwen3_2b_0shot_batch.sh
        ;;
    "qwen3_2b_1shot")
        bash scripts/shell/eval_qwen3_2b_1shot_batch.sh
        ;;
    "monitor")
        python scripts/utils/monitor_evaluation.py
        ;;
    *)
        echo "Usage: bash eval.sh {internvl|minicpm|qwen3_2b|qwen2_2b|qwen2.5_3b|deepseek|qwen3_2b_batch|qwen3_2b_1shot|monitor}"
        exit 1
        ;;
esac
