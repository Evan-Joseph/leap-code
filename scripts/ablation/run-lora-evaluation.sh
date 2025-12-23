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
: "${BASELINE_MODEL:=$WORK_DIR/models/Qwen3-VL-2B-Instruct}"
: "${LORA_PRESETS:=light standard full}"
: "${EVAL_STEPS:=1000 2000 3000 4000 5000}"
: "${DATA_PATH:=$WORK_DIR/dataset/vlm_evaluation_v1.0}"
: "${EVA_RESULTS_ROOT:=$WORK_DIR/eva_results_lora}"
: "${LOG_DIR:=$WORK_DIR/logs_lora}"
: "${DEVICE:=cuda:0}"
: "${MAX_TASKS:=10}"                   # 与全量微调评估一致
: "${NUM_EPISODES:=10}"                # 与全量微调评估一致

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

# ===================== 评估单个 checkpoint 的所有维度（并行） =====================
evaluate_checkpoint_parallel() {
    local preset=$1
    local step=$2
    local checkpoint_dir=$3
    local model_name="lora_${preset}_step${step}"
    local output_base="$EVA_RESULTS_ROOT/${preset}/step${step}"
    
    echo ""
    echo "========================================"
    echo "开始评估: $model_name (6 维度并行)"
    echo "========================================"
    
    mkdir -p "$output_base"
    
    # 启动 6 个维度的并行评估
    local pids=()
    
    for dim in "M&T" "CommonSense" "Semantic" "Spatial" "PhysicalLaw" "Complex"; do
        local log_file="$LOG_DIR/${model_name}_${dim}.log"
        local output_dir="$output_base/$dim"
        mkdir -p "$output_dir"
        
        echo "  启动 $dim 维度评估..."
        
        nohup python "$SCRIPT_DIR/run_vlm_evaluation_lora.py" \
            --dimension "$dim" \
            --base_model_path "$BASELINE_MODEL" \
            --lora_adapter_path "$checkpoint_dir" \
            --model_name "$model_name" \
            --data_path "$DATA_PATH" \
            --output_dir "$output_dir" \
            --device "$DEVICE" \
            --max_tasks "$MAX_TASKS" \
            --num_episodes "$NUM_EPISODES" > "$log_file" 2>&1 &
        
        pids+=($!)
    done
    
    echo "  进程 IDs: ${pids[*]}"
    echo "  等待所有维度评估完成..."
    
    # 等待所有维度完成
    for pid in "${pids[@]}"; do
        wait $pid || echo "  [警告] 进程 $pid 退出异常"
    done
    
    echo "  [完成] $model_name 所有维度评估完成"
}

# ===================== 主循环：预设串行，checkpoint 串行，维度并行 =====================
PRESET_ARRAY=($LORA_PRESETS)
STEPS_ARRAY=($EVAL_STEPS)

for preset in "${PRESET_ARRAY[@]}"; do
    echo ""
    echo "================================================"
    echo "开始评估 LoRA 预设: $preset"
    echo "================================================"
    
    LORA_DIR="$WORK_DIR/output_lora_${preset}"
    
    if [ ! -d "$LORA_DIR" ]; then
        echo "[跳过] LoRA 目录不存在: $LORA_DIR"
        continue
    fi
    
    for step in "${STEPS_ARRAY[@]}"; do
        CHECKPOINT_DIR="${LORA_DIR}/checkpoint-${step}"
        
        if [ ! -d "$CHECKPOINT_DIR" ]; then
            echo "[跳过] checkpoint 不存在: $CHECKPOINT_DIR"
            continue
        fi
        
        if [ ! -f "$CHECKPOINT_DIR/adapter_config.json" ]; then
            echo "[跳过] 不是有效的 LoRA checkpoint: $CHECKPOINT_DIR"
            continue
        fi
        
        evaluate_checkpoint_parallel "$preset" "$step" "$CHECKPOINT_DIR"
    done
    
    echo ""
    echo "[完成] $preset 预设所有 checkpoint 评估完成"
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
