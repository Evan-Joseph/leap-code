#!/bin/bash
# ================================================
# LoRA 消融实验评估脚本
# 不同预设串行，6 个维度并行
# 每 1000 步评估一次（1000, 2000, 3000, 4000, 5000）
# ================================================
set -euo pipefail

# 解析仓库根目录
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
WORK_DIR="$(cd "${SCRIPT_DIR}/../.." && pwd)"
cd "$WORK_DIR"

# ===================== 配置参数 =====================
: "${PYTHON_EXE:=/root/miniconda3/envs/qwen-ft-env/bin/python}"
: "${BASELINE_MODEL:=$WORK_DIR/models/Qwen3-VL-2B-Instruct}"
: "${LORA_PRESETS:=light standard full}"
: "${EVAL_STEPS:=1000 2000 3000 4000 5000}"
: "${DATA_PATH:=$WORK_DIR/dataset/vlm_evaluation_v1.0}"
: "${EVA_RESULTS_ROOT:=$WORK_DIR/eva_results_lora}"
: "${LOG_DIR:=$WORK_DIR/logs_lora}"
: "${DEVICE:=cuda:0}"
: "${MAX_TASKS:=10}"                   # 与全量微调评估一致
: "${NUM_EPISODES:=10}"                # 与全量微调评估一致

# 转换为数组
read -ra PRESET_ARRAY <<< "$LORA_PRESETS"
read -ra STEPS_ARRAY <<< "$EVAL_STEPS"

mkdir -p "$LOG_DIR"
mkdir -p "$EVA_RESULTS_ROOT"

echo "================================================"
echo "LoRA 消融实验评估（预设串行，维度并行）"
echo "================================================"
echo "基座模型: $BASELINE_MODEL"
echo "LoRA 预设: $LORA_PRESETS"
echo "评估步数: $EVAL_STEPS"
echo "结果目录: $EVA_RESULTS_ROOT"
echo "================================================"

# ===================== 动态任务池逻辑 =====================
MAX_PARALLEL=4  # 最大并行进程数

# 收集所有待评估的任务
tasks=()
for preset in "${PRESET_ARRAY[@]}"; do
    LORA_DIR="$WORK_DIR/models/lora_${preset}"
    [ ! -d "$LORA_DIR" ] && continue
    
    for step in "${STEPS_ARRAY[@]}"; do
        CHECKPOINT_DIR="${LORA_DIR}/checkpoint-${step}"
        [ ! -d "$CHECKPOINT_DIR" ] && continue
        [ ! -f "$CHECKPOINT_DIR/adapter_config.json" ] && continue
        
        for dim in "M&T" "CommenSence" "Semantic" "Spatial" "PhysicsLaw" "Complex"; do
            tasks+=("$preset|$step|$dim|$CHECKPOINT_DIR")
        done
    done
done

echo "总计待评估任务数: ${#tasks[@]}"

# 动态调度循环
current_task=0
total_tasks=${#tasks[@]}

while [ $current_task -lt $total_tasks ] || [ $(jobs -r | wc -l) -gt 0 ]; do
    # 当运行中的任务少于最大并行数，且还有待处理任务时
    while [ $(jobs -r | wc -l) -lt $MAX_PARALLEL ] && [ $current_task -lt $total_tasks ]; do
        # 解析任务参数
        IFS='|' read -r preset step dim ckpt_dir <<< "${tasks[$current_task]}"
        
        model_name="lora_${preset}_step${step}"
        output_dir="$EVA_RESULTS_ROOT/${preset}/step${step}/$dim"
        log_file="$LOG_DIR/${model_name}_${dim}.log"
        mkdir -p "$output_dir"
        
        echo "[启动] $model_name | 维度: $dim"
        
        "$PYTHON_EXE" "$SCRIPT_DIR/run_vlm_evaluation_lora.py" \
            --dimension "$dim" \
            --base_model_path "$BASELINE_MODEL" \
            --lora_adapter_path "$ckpt_dir" \
            --model_name "$model_name" \
            --data_path "$DATA_PATH" \
            --output_dir "$output_dir" \
            --device "$DEVICE" \
            --max_tasks "$MAX_TASKS" \
            --num_episodes "$NUM_EPISODES" > "$log_file" 2>&1 &
        
        current_task=$((current_task + 1))
        sleep 2 # 稍微错开启动时间，避免瞬间显存峰值
    done
    
    # 等待一段时间再检查
    sleep 5
done

echo ""
echo "================================================"
echo "所有 LoRA 消融评估完成！"
echo "================================================"
echo ""
echo "结果目录: $EVA_RESULTS_ROOT"
echo ""
echo "目录结构:"
for preset in "${PRESET_ARRAY[@]}"; do
    if [ -d "$EVA_RESULTS_ROOT/$preset" ]; then
        echo "  $preset/"
        for step in "${STEPS_ARRAY[@]}"; do
            if [ -d "$EVA_RESULTS_ROOT/$preset/step$step" ]; then
                echo "    step$step/"
            fi
        done
    fi
done
