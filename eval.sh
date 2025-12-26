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
    "monitor")
        python scripts/utils/monitor_evaluation.py
        ;;
    *)
        echo "Usage: bash eval.sh {qwen3_2b_0shot|qwen3_2b_1shot|qwen3_4b|internvl|minicpm|monitor}"
        exit 1
        ;;
esac
