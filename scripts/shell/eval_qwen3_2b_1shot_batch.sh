#!/bin/bash

# VLM模型评估脚本 (1-shot 并行版)
# 针对不同维度进行批量评估

# 解析仓库根目录
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
WORK_DIR="$(cd "${SCRIPT_DIR}/../.." && pwd)"
cd "$WORK_DIR"

PYTHON_PATH="/root/miniconda3/envs/qwen-ft-env/bin/python3"

# 设置参数（支持外部覆写）
: "${BASELINE_MODEL:="$WORK_DIR/models/Qwen3-VL-2B-Instruct"}"
: "${CHECKPOINTS_DIR:="$WORK_DIR/output"}"
: "${DATA_PATH:="$WORK_DIR/dataset/vlm_evaluation_v1.0"}"
: "${CONFIG_NAME:="qwen3_2b_1shot"}"
: "${LOG_DIR:="$WORK_DIR/logs/$CONFIG_NAME"}"
: "${RESULT_DIR:="$WORK_DIR/results/$CONFIG_NAME"}"
: "${DEVICE:="cuda:0"}"
: "${MAX_CHECKPOINTS:=35}" # 200-7000 步，每 200 步一个，共 35 个
: "${MAX_TASKS:=10}"
: "${NUM_EPISODES:=10}"
: "${FEW_SHOT:=1}" # 设置为 1-shot
: "${SKIP_BASELINE:=false}" # 是否跳过基线模型

# 创建日志目录
mkdir -p "$LOG_DIR"

# 构造额外参数
EXTRA_ARGS=""
if [ "$SKIP_BASELINE" = "true" ]; then
    EXTRA_ARGS="--skip-baseline"
fi

# 执行评估命令
echo "开始执行 VLM 模型 1-shot 评估 (Checkpoints 200-7000)..."
echo "结果将保存至: $RESULT_DIR"

# 定义所有维度
DIMENSIONS=("M&T" "CommenSence" "Semantic" "Spatial" "PhysicsLaw" "Complex")
LOG_NAMES=("mt" "commonsense" "semantic" "spatial" "physics" "complex")

COMMANDS=()
LOGS=()

for i in "${!DIMENSIONS[@]}"; do
    dim="${DIMENSIONS[$i]}"
    log_name="${LOG_NAMES[$i]}"
    COMMANDS+=("$PYTHON_PATH scripts/evaluation/run_vlm_evaluation.py --batch_mode --dimension '$dim' --baseline_model '$BASELINE_MODEL' --checkpoints_dir '$CHECKPOINTS_DIR' --max_checkpoints $MAX_CHECKPOINTS --max_tasks $MAX_TASKS --num_episodes $NUM_EPISODES --data_path '$DATA_PATH' --device '$DEVICE' --few-shot-num $FEW_SHOT $EXTRA_ARGS --output_dir '$RESULT_DIR'")
    LOGS+=("$LOG_DIR/log_$log_name.log")
done

echo "启动 Qwen3-2B 1-shot 批量评估任务队列 (并发数: 3)..."
$PYTHON_PATH scripts/utils/task_runner.py \
    --commands "${COMMANDS[@]}" \
    --logs "${LOGS[@]}" \
    --concurrency 3

echo "所有评估任务已完成!"
