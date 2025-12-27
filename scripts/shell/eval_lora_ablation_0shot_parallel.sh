#!/bin/bash

# LoRA 消融实验并行评估脚本
# 覆盖: 4个Preset x 6个Checkpoints x 6个维度
# 统一参数: num_episodes=5, max_tasks=20

set -e

# 解析仓库根目录
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
WORK_DIR="$(cd "${SCRIPT_DIR}/../.." && pwd)"
cd "$WORK_DIR"

# ===================== 环境配置 =====================
# 自动检测 python 路径
if [ -f "/root/miniconda3/envs/qwen-ft-env/bin/python" ]; then
    PYTHON_PATH="/root/miniconda3/envs/qwen-ft-env/bin/python"
elif [ -f "/root/autodl-tmp/envs/qwen-ft-env/bin/python" ]; then
    PYTHON_PATH="/root/autodl-tmp/envs/qwen-ft-env/bin/python"
else
    PYTHON_PATH="python3"
fi

# ===================== 实验参数 =====================
BASELINE_MODEL="$WORK_DIR/models/Qwen3-VL-2B-Instruct"
LORA_ROOT="$WORK_DIR/models/lora_models"
DATA_PATH="$WORK_DIR/dataset/vlm_evaluation_v1.0"
DEVICE="cuda:0"

# LoRA 配置
PRESETS=("light" "standard" "full" "aggressive")
STEPS=(2000 3000 4000 5000 6000 7000)
DIMENSIONS=("M&T" "CommenSence" "Semantic" "Spatial" "PhysicsLaw" "Complex")

# 评估参数 (与 0-shot parallel 保持一致)
NUM_EPISODES=5
MAX_TASKS=20

# 输出目录
LOG_ROOT="$WORK_DIR/logs/lora_ablation"
RESULT_ROOT="$WORK_DIR/results/lora_ablation"

mkdir -p "$LOG_ROOT"
mkdir -p "$RESULT_ROOT"

echo "=============================================="
echo "  LoRA 消融实验并行评估"
echo "=============================================="
echo "基座模型: $BASELINE_MODEL"
echo "LoRA 根目录: $LORA_ROOT"
echo "Presets: ${PRESETS[*]}"
echo "Steps: ${STEPS[*]}"
echo "参数: episodes=$NUM_EPISODES, max_tasks=$MAX_TASKS"
echo "日志: $LOG_ROOT"
echo "结果: $RESULT_ROOT"
echo "=============================================="

# ===================== 生成任务队列 =====================
COMMANDS=()
LOGS=()

for preset in "${PRESETS[@]}"; do
    LORA_DIR="$LORA_ROOT/$preset"
    
    # 检查 Preset 目录是否存在
    if [ ! -d "$LORA_DIR" ]; then
        echo "⚠️  [跳过] Preset 不存在: $preset ($LORA_DIR)"
        continue
    fi

    for step in "${STEPS[@]}"; do
        CHECKPOINT_DIR="$LORA_DIR/checkpoint-$step"
        
        # 检查 Checkpoint 是否存在
        if [ ! -d "$CHECKPOINT_DIR" ]; then
            continue
        fi
        
        # 检查 adapter_config.json 确保 checkpoint 完整
        if [ ! -f "$CHECKPOINT_DIR/adapter_config.json" ]; then
            continue
        fi

        for dim in "${DIMENSIONS[@]}"; do
            MODEL_ID="lora_${preset}_step${step}"
            OUTPUT_DIR="$RESULT_ROOT/$preset/step$step"  # run_vlm_evaluation_lora.py 会自动添加维度子目录吗？
            # 注意: run_vlm_evaluation_lora.py 的 output_dir 行为通常是直接使用传入的路径
            # 为了与其他脚本一致，这里的 output_dir 应该是维度的父级或者包含维度？
            # 查看 run_vlm_evaluation_lora.py 逻辑，它会把结果存到 output_dir 下
            # 如果我们并行跑，为了避免冲突，最好指定明确的子目录
            
            TASK_OUTPUT_DIR="$RESULT_ROOT/$preset/step$step"
            LOG_FILE="$LOG_ROOT/${preset}_step${step}_${dim//&/and}.log" # 处理 M&T 中的 & 符号
            
            # 构建命令
            # 使用 scripts/ablation/run_vlm_evaluation_lora.py
            CMD="$PYTHON_PATH scripts/ablation/run_vlm_evaluation_lora.py \
                --dimension '$dim' \
                --base_model_path '$BASELINE_MODEL' \
                --lora_adapter_path '$CHECKPOINT_DIR' \
                --model_name '$MODEL_ID' \
                --data_path '$DATA_PATH' \
                --output_dir '$TASK_OUTPUT_DIR' \
            # 使用占位符 __DEVICE__，由 task_runner 动态替换为 cuda:0 或 cuda:1
            CMD="$PYTHON_PATH scripts/ablation/run_vlm_evaluation_lora.py \
                --dimension '$dim' \
                --base_model_path '$BASELINE_MODEL' \
                --lora_adapter_path '$CHECKPOINT_DIR' \
                --model_name '$MODEL_ID' \
                --data_path '$DATA_PATH' \
                --output_dir '$TASK_OUTPUT_DIR' \
                --device '__DEVICE__' \
                --max_tasks $MAX_TASKS \
                --num_episodes $NUM_EPISODES"
            
            COMMANDS+=("$CMD")
            LOGS+=("$LOG_FILE")
        done
    done
done

TOTAL_TASKS=${#COMMANDS[@]}
if [ "$TOTAL_TASKS" -eq 0 ]; then
    echo "❌ 没有发现有效的 LoRA Checkpoint 任务。请检查路径。"
    exit 1
fi

echo "准备就绪: 共 $TOTAL_TASKS 个评估任务"
echo "启动双卡并行评估 (GPU 0,1 | 每卡3路 | 总并发6)..."

# 调用任务执行器 (启用 GPU 调度)
$PYTHON_PATH scripts/utils/task_runner.py \
    --commands "${COMMANDS[@]}" \
    --logs "${LOGS[@]}" \
    --gpus "0,1" \
    --tasks-per-gpu 3

echo ""
echo "=============================================="
echo "✅ LoRA 消融评估完成!"
echo "=============================================="
