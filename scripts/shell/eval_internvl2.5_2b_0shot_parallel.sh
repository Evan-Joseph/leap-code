#!/bin/bash

# InternVL2.5-2B 并行评估脚本
# 针对 32GB 显存机器优化，开启 2 维度并行

PYTHON_PATH="/root/miniconda3/envs/qwen-ft-env/bin/python3"
MODEL_PATH="/root/autodl-tmp/leap-code/models/InternVL2_5-2B"
DATA_PATH="/root/autodl-tmp/leap-code/dataset/vlm_evaluation_v1.0"

# 创建日志目录
mkdir -p logs_parallel

echo "开始 InternVL2.5-2B 并行评估..."

# 第一组维度: M&T, CommenSence, Semantic
echo "启动第一组维度 (M&T, CommenSence, Semantic)..."
nohup bash -c "
$PYTHON_PATH scripts/evaluation/run_vlm_evaluation.py --model_path $MODEL_PATH --dimension 'M&T' --num_episodes 5 --max_tasks 20 --data_path $DATA_PATH;
$PYTHON_PATH scripts/evaluation/run_vlm_evaluation.py --model_path $MODEL_PATH --dimension 'CommenSence' --num_episodes 5 --max_tasks 20 --data_path $DATA_PATH;
$PYTHON_PATH scripts/evaluation/run_vlm_evaluation.py --model_path $MODEL_PATH --dimension 'Semantic' --num_episodes 5 --max_tasks 20 --data_path $DATA_PATH;
" > logs_parallel/group1.log 2>&1 &

# 第二组维度: Spatial, PhysicsLaw, Complex
echo "启动第二组维度 (Spatial, PhysicsLaw, Complex)..."
nohup bash -c "
$PYTHON_PATH scripts/evaluation/run_vlm_evaluation.py --model_path $MODEL_PATH --dimension 'Spatial' --num_episodes 5 --max_tasks 20 --data_path $DATA_PATH;
$PYTHON_PATH scripts/evaluation/run_vlm_evaluation.py --model_path $MODEL_PATH --dimension 'PhysicsLaw' --num_episodes 5 --max_tasks 20 --data_path $DATA_PATH;
$PYTHON_PATH scripts/evaluation/run_vlm_evaluation.py --model_path $MODEL_PATH --dimension 'Complex' --num_episodes 5 --max_tasks 20 --data_path $DATA_PATH;
" > logs_parallel/group2.log 2>&1 &

echo "并行任务已启动。请使用以下命令监控进度："
echo "tail -f logs_parallel/group1.log"
echo "tail -f logs_parallel/group2.log"
