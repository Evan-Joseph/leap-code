#!/bin/bash

# MiniCPM-V-2.6 4-bit 并行评估脚本
# 针对 32GB 显存优化，开启 3 维度并行以实现 3 倍提速

PYTHON_PATH="/root/miniconda3/envs/qwen-ft-env/bin/python3"
MODEL_PATH="/root/autodl-tmp/leap-code/models/MiniCPM-V-2_6"
DATA_PATH="/root/autodl-tmp/leap-code/dataset/vlm_evaluation_v1.0"
LOG_DIR="/root/autodl-tmp/leap-code/logs/minicpm_parallel"

mkdir -p "$LOG_DIR"

echo "开始 MiniCPM-V-2.6 4-bit 并行评估 (3路并发)..."

# 第一组: M&T, CommenSence, Semantic
echo "启动第一组 (M&T, CommenSence, Semantic)..."
$PYTHON_PATH scripts/evaluation/run_vlm_evaluation.py --model_path $MODEL_PATH --dimension 'M&T' --num_episodes 5 --max_tasks 20 --data_path $DATA_PATH --quantization 4bit > "$LOG_DIR/mt.log" 2>&1 &
$PYTHON_PATH scripts/evaluation/run_vlm_evaluation.py --model_path $MODEL_PATH --dimension 'CommenSence' --num_episodes 5 --max_tasks 20 --data_path $DATA_PATH --quantization 4bit > "$LOG_DIR/commonsense.log" 2>&1 &
$PYTHON_PATH scripts/evaluation/run_vlm_evaluation.py --model_path $MODEL_PATH --dimension 'Semantic' --num_episodes 5 --max_tasks 20 --data_path $DATA_PATH --quantization 4bit > "$LOG_DIR/semantic.log" 2>&1 &

wait

# 第二组: Spatial, PhysicsLaw, Complex
echo "启动第二组 (Spatial, PhysicsLaw, Complex)..."
$PYTHON_PATH scripts/evaluation/run_vlm_evaluation.py --model_path $MODEL_PATH --dimension 'Spatial' --num_episodes 5 --max_tasks 20 --data_path $DATA_PATH --quantization 4bit > "$LOG_DIR/spatial.log" 2>&1 &
$PYTHON_PATH scripts/evaluation/run_vlm_evaluation.py --model_path $MODEL_PATH --dimension 'PhysicsLaw' --num_episodes 5 --max_tasks 20 --data_path $DATA_PATH --quantization 4bit > "$LOG_DIR/physics.log" 2>&1 &
$PYTHON_PATH scripts/evaluation/run_vlm_evaluation.py --model_path $MODEL_PATH --dimension 'Complex' --num_episodes 5 --max_tasks 20 --data_path $DATA_PATH --quantization 4bit > "$LOG_DIR/complex.log" 2>&1 &

wait

echo "所有维度评估完成！"
