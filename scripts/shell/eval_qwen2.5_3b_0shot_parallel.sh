#!/bin/bash

# Qwen2.5-VL-3B-Instruct 0-shot 并行评估脚本
# 针对 32GB 显存优化，开启 3 维度并行

# 解析仓库根目录
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
WORK_DIR="$(cd "${SCRIPT_DIR}/../.." && pwd)"
cd "$WORK_DIR"

PYTHON_PATH="/root/miniconda3/envs/qwen-ft-env/bin/python3"
MODEL_PATH="$WORK_DIR/models/Qwen2.5-VL-3B-Instruct"
DATA_PATH="$WORK_DIR/dataset/vlm_evaluation_v1.0"

# 规范化日志和结果目录
CONFIG_NAME="qwen2.5_3b_0shot"
LOG_DIR="$WORK_DIR/logs/$CONFIG_NAME"
RESULT_DIR="$WORK_DIR/results/$CONFIG_NAME"

mkdir -p "$LOG_DIR"

echo "开始 Qwen2.5-VL-3B-Instruct 0-shot 并行评估 (3路并发)..."
echo "结果将保存至: $RESULT_DIR"

# 定义所有维度
DIMENSIONS=("M&T" "CommenSence" "Semantic" "Spatial" "PhysicsLaw" "Complex")
LOG_NAMES=("mt" "commonsense" "semantic" "spatial" "physics" "complex")

COMMANDS=()
LOGS=()

for i in "${!DIMENSIONS[@]}"; do
    dim="${DIMENSIONS[$i]}"
    log_name="${LOG_NAMES[$i]}"
    COMMANDS+=("$PYTHON_PATH scripts/evaluation/run_vlm_evaluation.py --model_path $MODEL_PATH --dimension '$dim' --num_episodes 5 --max_tasks 20 --data_path $DATA_PATH --output_dir '$RESULT_DIR'")
    LOGS+=("$LOG_DIR/$log_name.log")
done

echo "启动并行评估任务队列 (并发数: 3)..."
$PYTHON_PATH scripts/utils/task_runner.py \
    --commands "${COMMANDS[@]}" \
    --logs "${LOGS[@]}" \
    --concurrency 3

echo "所有维度评估完成！"
