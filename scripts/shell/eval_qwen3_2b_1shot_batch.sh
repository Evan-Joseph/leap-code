#!/bin/bash

# VLM模型评估脚本 (1-shot 并行版)
# 针对不同维度进行批量评估

# 解析仓库根目录（脚本所在目录）
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
WORK_DIR="${SCRIPT_DIR}"
cd "$WORK_DIR"

# 设置参数（支持外部覆写）
: "${BASELINE_MODEL:="$WORK_DIR/models/Qwen3-VL-2B-Instruct"}"
: "${CHECKPOINTS_DIR:="$WORK_DIR/output"}"
: "${DATA_PATH:="$WORK_DIR/dataset/vlm_evaluation_v1.0"}"
: "${LOG_DIR:="$WORK_DIR/logs_1shot"}"
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
echo "注意：为了防止显存溢出 (OOM)，将分批次并行执行 (每次最多 4 个维度)。"

# 第一批次：M&T, CommenSence, Semantic, Spatial
echo "--- 启动第一批次 (M&T, CommenSence, Semantic, Spatial) ---"
python scripts/evaluation/run_vlm_evaluation.py --batch_mode --dimension "M&T" --baseline_model "$BASELINE_MODEL" --checkpoints_dir "$CHECKPOINTS_DIR" --max_checkpoints $MAX_CHECKPOINTS --max_tasks $MAX_TASKS --num_episodes $NUM_EPISODES --data_path "$DATA_PATH" --device "$DEVICE" --few-shot-num $FEW_SHOT $EXTRA_ARGS > "$LOG_DIR/log_mt.log" 2>&1 &
PID_MT=$!
python scripts/evaluation/run_vlm_evaluation.py --batch_mode --dimension "CommenSence" --baseline_model "$BASELINE_MODEL" --checkpoints_dir "$CHECKPOINTS_DIR" --max_checkpoints $MAX_CHECKPOINTS --max_tasks $MAX_TASKS --num_episodes $NUM_EPISODES --data_path "$DATA_PATH" --device "$DEVICE" --few-shot-num $FEW_SHOT $EXTRA_ARGS > "$LOG_DIR/log_commonsense.log" 2>&1 &
PID_CommenSence=$!
python scripts/evaluation/run_vlm_evaluation.py --batch_mode --dimension "Semantic" --baseline_model "$BASELINE_MODEL" --checkpoints_dir "$CHECKPOINTS_DIR" --max_checkpoints $MAX_CHECKPOINTS --max_tasks $MAX_TASKS --num_episodes $NUM_EPISODES --data_path "$DATA_PATH" --device "$DEVICE" --few-shot-num $FEW_SHOT $EXTRA_ARGS > "$LOG_DIR/log_semantic.log" 2>&1 &
PID_Semantic=$!
python scripts/evaluation/run_vlm_evaluation.py --batch_mode --dimension "Spatial" --baseline_model "$BASELINE_MODEL" --checkpoints_dir "$CHECKPOINTS_DIR" --max_checkpoints $MAX_CHECKPOINTS --max_tasks $MAX_TASKS --num_episodes $NUM_EPISODES --data_path "$DATA_PATH" --device "$DEVICE" --few-shot-num $FEW_SHOT $EXTRA_ARGS > "$LOG_DIR/log_spatial.log" 2>&1 &
PID_Spatial=$!
wait $PID_MT $PID_CommenSence $PID_Semantic $PID_Spatial

# 第二批次：PhysicsLaw, Complex
echo "--- 启动第二批次 (PhysicsLaw, Complex) ---"
python scripts/evaluation/run_vlm_evaluation.py --batch_mode --dimension "PhysicsLaw" --baseline_model "$BASELINE_MODEL" --checkpoints_dir "$CHECKPOINTS_DIR" --max_checkpoints $MAX_CHECKPOINTS --max_tasks $MAX_TASKS --num_episodes $NUM_EPISODES --data_path "$DATA_PATH" --device "$DEVICE" --few-shot-num $FEW_SHOT $EXTRA_ARGS > "$LOG_DIR/log_physics.log" 2>&1 &
PID_PhysicsLaw=$!
python scripts/evaluation/run_vlm_evaluation.py --batch_mode --dimension "Complex" --baseline_model "$BASELINE_MODEL" --checkpoints_dir "$CHECKPOINTS_DIR" --max_checkpoints $MAX_CHECKPOINTS --max_tasks $MAX_TASKS --num_episodes $NUM_EPISODES --data_path "$DATA_PATH" --device "$DEVICE" --few-shot-num $FEW_SHOT $EXTRA_ARGS > "$LOG_DIR/log_complex.log" 2>&1 &
PID_Complex=$!
wait $PID_PhysicsLaw $PID_Complex

# 保存所有进程ID到文件 (虽然已经结束，但保留逻辑一致性)
echo "$PID_MT" > "$LOG_DIR/pid_mt.txt"
echo "$PID_CommenSence" > "$LOG_DIR/pid_commonsense.txt"
echo "$PID_Semantic" > "$LOG_DIR/pid_semantic.txt"
echo "$PID_Spatial" > "$LOG_DIR/pid_spatial.txt"
echo "$PID_PhysicsLaw" > "$LOG_DIR/pid_physics.txt"
echo "$PID_Complex" > "$LOG_DIR/pid_complex.txt"

echo "所有评估任务已完成!"
