#!/bin/bash

# Qwen3-VL-4B-Instruct 0-shot 评估脚本
# 作为一个强有力的对比锚点，分批次（每次2个维度）并行执行以节省显存

# 解析仓库根目录
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
WORK_DIR="$(cd "${SCRIPT_DIR}/../.." && pwd)"
cd "$WORK_DIR"

PYTHON_PATH="/root/miniconda3/envs/qwen-ft-env/bin/python3"

# 设置参数
: "${MODEL_PATH:="$WORK_DIR/models/Qwen3-VL-4B-Instruct"}"
: "${DATA_PATH:="$WORK_DIR/dataset/vlm_evaluation_v1.0"}"
: "${CONFIG_NAME:="qwen3_4b_0shot"}"
: "${LOG_DIR:="$WORK_DIR/logs/$CONFIG_NAME"}"
: "${RESULT_DIR:="$WORK_DIR/results/$CONFIG_NAME"}"
: "${DEVICE:="cuda:0"}"
: "${MAX_TASKS:=10}"
: "${NUM_EPISODES:=10}"
: "${FEW_SHOT:=0}"

# 创建日志目录
mkdir -p "$LOG_DIR"

echo "开始执行 Qwen3-VL-4B-Instruct 0-shot 评估..."
echo "配置: MAX_TASKS=$MAX_TASKS, NUM_EPISODES=$NUM_EPISODES"
echo "结果将保存至: $RESULT_DIR"

# 定义所有维度
DIMENSIONS=("M&T" "CommenSence" "Semantic" "Spatial" "PhysicsLaw" "Complex")
LOG_NAMES=("mt" "commonsense" "semantic" "spatial" "physics" "complex")

COMMANDS=()
LOGS=()

for i in "${!DIMENSIONS[@]}"; do
    dim="${DIMENSIONS[$i]}"
    log_name="${LOG_NAMES[$i]}"
    COMMANDS+=("$PYTHON_PATH scripts/evaluation/run_vlm_evaluation.py --dimension '$dim' --model_path '$MODEL_PATH' --max_tasks $MAX_TASKS --num_episodes $NUM_EPISODES --data_path '$DATA_PATH' --device '$DEVICE' --few-shot-num $FEW_SHOT --output_dir '$RESULT_DIR'")
    LOGS+=("$LOG_DIR/log_$log_name.log")
done

echo "启动 Qwen3-4B 0-shot 评估任务队列 (并发数: 2)..."
$PYTHON_PATH scripts/utils/task_runner.py \
    --commands "${COMMANDS[@]}" \
    --logs "${LOGS[@]}" \
    --concurrency 2

echo "所有维度评估已完成!"
echo "日志保存在: $LOG_DIR"
