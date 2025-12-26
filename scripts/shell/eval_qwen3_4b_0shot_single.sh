#!/bin/bash

# Qwen3-VL-4B-Instruct 0-shot 评估脚本
# 作为一个强有力的对比锚点，分批次（每次2个维度）并行执行以节省显存

# 解析仓库根目录
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
WORK_DIR="${SCRIPT_DIR}"
cd "$WORK_DIR"

# 设置参数
: "${MODEL_PATH:="$WORK_DIR/models/Qwen3-VL-4B-Instruct"}"
: "${DATA_PATH:="$WORK_DIR/dataset/vlm_evaluation_v1.0"}"
: "${LOG_DIR:="$WORK_DIR/logs_4b_0shot"}"
: "${DEVICE:="cuda:0"}"
: "${MAX_TASKS:=10}"
: "${NUM_EPISODES:=10}"
: "${FEW_SHOT:=0}"

# 创建日志目录
mkdir -p "$LOG_DIR"

echo "开始执行 Qwen3-VL-4B-Instruct 0-shot 评估..."
echo "配置: MAX_TASKS=$MAX_TASKS, NUM_EPISODES=$NUM_EPISODES"
echo "采用 2 维并发队列模式执行"

# 第一批次：M&T, CommenSence
echo "--- 启动第一批次 (M&T, CommenSence) ---"
python scripts/evaluation/run_vlm_evaluation.py --dimension "M&T" --model_path "$MODEL_PATH" --max_tasks $MAX_TASKS --num_episodes $NUM_EPISODES --data_path "$DATA_PATH" --device "$DEVICE" --few-shot-num $FEW_SHOT > "$LOG_DIR/log_mt.log" 2>&1 &
PID1=$!
python scripts/evaluation/run_vlm_evaluation.py --dimension "CommenSence" --model_path "$MODEL_PATH" --max_tasks $MAX_TASKS --num_episodes $NUM_EPISODES --data_path "$DATA_PATH" --device "$DEVICE" --few-shot-num $FEW_SHOT > "$LOG_DIR/log_commonsense.log" 2>&1 &
PID2=$!
wait $PID1 $PID2
echo "第一批次完成。"

# 第二批次：Semantic, Spatial
echo "--- 启动第二批次 (Semantic, Spatial) ---"
python scripts/evaluation/run_vlm_evaluation.py --dimension "Semantic" --model_path "$MODEL_PATH" --max_tasks $MAX_TASKS --num_episodes $NUM_EPISODES --data_path "$DATA_PATH" --device "$DEVICE" --few-shot-num $FEW_SHOT > "$LOG_DIR/log_semantic.log" 2>&1 &
PID1=$!
python scripts/evaluation/run_vlm_evaluation.py --dimension "Spatial" --model_path "$MODEL_PATH" --max_tasks $MAX_TASKS --num_episodes $NUM_EPISODES --data_path "$DATA_PATH" --device "$DEVICE" --few-shot-num $FEW_SHOT > "$LOG_DIR/log_spatial.log" 2>&1 &
PID2=$!
wait $PID1 $PID2
echo "第二批次完成。"

# 第三批次：PhysicsLaw, Complex
echo "--- 启动第三批次 (PhysicsLaw, Complex) ---"
python scripts/evaluation/run_vlm_evaluation.py --dimension "PhysicsLaw" --model_path "$MODEL_PATH" --max_tasks $MAX_TASKS --num_episodes $NUM_EPISODES --data_path "$DATA_PATH" --device "$DEVICE" --few-shot-num $FEW_SHOT > "$LOG_DIR/log_physics.log" 2>&1 &
PID1=$!
python scripts/evaluation/run_vlm_evaluation.py --dimension "Complex" --model_path "$MODEL_PATH" --max_tasks $MAX_TASKS --num_episodes $NUM_EPISODES --data_path "$DATA_PATH" --device "$DEVICE" --few-shot-num $FEW_SHOT > "$LOG_DIR/log_complex.log" 2>&1 &
PID2=$!
wait $PID1 $PID2
echo "第三批次完成。"

echo "所有维度评估已完成!"
echo "日志保存在: $LOG_DIR"
