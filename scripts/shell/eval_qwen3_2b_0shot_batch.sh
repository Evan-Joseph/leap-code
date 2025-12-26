#!/bin/bash

# VLM模型评估脚本
# 针对不同维度进行批量评估

# 解析仓库根目录（脚本所在目录）
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
WORK_DIR="${SCRIPT_DIR}"
cd "$WORK_DIR"

# 设置参数（支持外部覆写）
: "${BASELINE_MODEL:="$WORK_DIR/models/Qwen3-VL-2B-Instruct"}"
: "${CHECKPOINTS_DIR:="$WORK_DIR/output"}"
: "${DATA_PATH:="$WORK_DIR/dataset/vlm_evaluation_v1.0"}"
: "${LOG_DIR:="$WORK_DIR/logs"}"
: "${DEVICE:="cuda:0"}"
: "${MAX_CHECKPOINTS:=100}"
: "${MAX_TASKS:=10}"
: "${NUM_EPISODES:=10}"

# 创建日志目录
mkdir -p "$LOG_DIR"

# 执行评估命令
echo "开始执行VLM模型评估..."

# M&T维度评估
echo "启动M&T维度评估..."
nohup python scripts/evaluation/run_vlm_evaluation.py --batch_mode --dimension "M&T" --baseline_model "$BASELINE_MODEL" --checkpoints_dir "$CHECKPOINTS_DIR" --max_checkpoints $MAX_CHECKPOINTS --max_tasks $MAX_TASKS --num_episodes $NUM_EPISODES --data_path "$DATA_PATH" --device "$DEVICE" > "$LOG_DIR/log_mt.log" 2>&1 &
PID_MT=$!
echo "M&T维度评估进程ID: $PID_MT"

# CommenSence维度评估
echo "启动CommenSence维度评估..."
nohup python scripts/evaluation/run_vlm_evaluation.py --batch_mode --dimension "CommenSence" --baseline_model "$BASELINE_MODEL" --checkpoints_dir "$CHECKPOINTS_DIR" --max_checkpoints $MAX_CHECKPOINTS --max_tasks $MAX_TASKS --num_episodes $NUM_EPISODES --data_path "$DATA_PATH" --device "$DEVICE" > "$LOG_DIR/log_commonsense.log" 2>&1 &
PID_CommenSence=$!
echo "CommenSence维度评估进程ID: $PID_CommenSence"

# Semantic维度评估
echo "启动Semantic维度评估..."
nohup python scripts/evaluation/run_vlm_evaluation.py --batch_mode --dimension "Semantic" --baseline_model "$BASELINE_MODEL" --checkpoints_dir "$CHECKPOINTS_DIR" --max_checkpoints $MAX_CHECKPOINTS --max_tasks $MAX_TASKS --num_episodes $NUM_EPISODES --data_path "$DATA_PATH" --device "$DEVICE" > "$LOG_DIR/log_semantic.log" 2>&1 &
PID_Semantic=$!
echo "Semantic维度评估进程ID: $PID_Semantic"

# Spatial维度评估
echo "启动Spatial维度评估..."
nohup python scripts/evaluation/run_vlm_evaluation.py --batch_mode --dimension "Spatial" --baseline_model "$BASELINE_MODEL" --checkpoints_dir "$CHECKPOINTS_DIR" --max_checkpoints $MAX_CHECKPOINTS --max_tasks $MAX_TASKS --num_episodes $NUM_EPISODES --data_path "$DATA_PATH" --device "$DEVICE" > "$LOG_DIR/log_spatial.log" 2>&1 &
PID_Spatial=$!
echo "Spatial维度评估进程ID: $PID_Spatial"

# PhysicsLaw维度评估
echo "启动PhysicsLaw维度评估..."
nohup python scripts/evaluation/run_vlm_evaluation.py --batch_mode --dimension "PhysicsLaw" --baseline_model "$BASELINE_MODEL" --checkpoints_dir "$CHECKPOINTS_DIR" --max_checkpoints $MAX_CHECKPOINTS --max_tasks $MAX_TASKS --num_episodes $NUM_EPISODES --data_path "$DATA_PATH" --device "$DEVICE" > "$LOG_DIR/log_physics.log" 2>&1 &
PID_PhysicsLaw=$!
echo "PhysicsLaw维度评估进程ID: $PID_PhysicsLaw"

# Complex维度评估
echo "启动Complex维度评估..."
nohup python scripts/evaluation/run_vlm_evaluation.py --batch_mode --dimension "Complex" --baseline_model "$BASELINE_MODEL" --checkpoints_dir "$CHECKPOINTS_DIR" --max_checkpoints $MAX_CHECKPOINTS --max_tasks $MAX_TASKS --num_episodes $NUM_EPISODES --data_path "$DATA_PATH" --device "$DEVICE" > "$LOG_DIR/log_complex.log" 2>&1 &
PID_Complex=$!
echo "Complex维度评估进程ID: $PID_Complex"

# 保存所有进程ID到文件
echo "$PID_MT" > "$LOG_DIR/pid_mt.txt"
echo "$PID_CommenSence" > "$LOG_DIR/pid_commonsense.txt"
echo "$PID_Semantic" > "$LOG_DIR/pid_semantic.txt"
echo "$PID_Spatial" > "$LOG_DIR/pid_spatial.txt"
echo "$PID_PhysicsLaw" > "$LOG_DIR/pid_physics.txt"
echo "$PID_Complex" > "$LOG_DIR/pid_complex.txt"

echo "所有评估进程已启动!"
echo "日志文件保存在: $LOG_DIR"
echo ""
echo "查看进程状态命令:"
echo "ps aux | grep run_vlm_evaluation"
echo ""
echo "查看日志命令:"
echo "tail -f $LOG_DIR/log_mt.log"
echo "tail -f $LOG_DIR/log_commonsense.log"
echo "tail -f $LOG_DIR/log_semantic.log"
echo "tail -f $LOG_DIR/log_spatial.log"
echo "tail -f $LOG_DIR/log_physics.log"
echo "tail -f $LOG_DIR/log_complex.log"
echo ""
echo "停止所有评估进程命令:"
echo "kill $PID_MT $PID_CommonSense $PID_Semantic $PID_Spatial $PID_PhysicalLaw $PID_Complex"