#!/usr/bin/env bash
# ================================================
# LoRA 消融实验批量运行脚本
# 自动运行多种 LoRA 配置进行对比实验
# 所有实验统一在 5000 步停止，确保公平对比
# ================================================
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

# ===================== 实验配置 =====================
# 统一的最大训练步数（确保公平对比）
MAX_STEPS="${MAX_STEPS:-5000}"

# 定义要测试的预设列表
PRESETS=("light" "standard" "full")

echo "================================================"
echo "LoRA 消融实验批量运行"
echo "================================================"
echo "统一最大步数: ${MAX_STEPS}"
echo "预设配置: ${PRESETS[*]}"
echo "================================================"

for preset in "${PRESETS[@]}"; do
    echo ""
    echo "========================================"
    echo "开始运行: ${preset} 配置"
    echo "========================================"
    
    bash "${SCRIPT_DIR}/run-lora-training.sh" --preset "${preset}" --max-steps "${MAX_STEPS}"
    
    echo ""
    echo "[完成] ${preset} 配置训练完成"
    echo ""
done

echo "================================================"
echo "所有消融实验完成！"
echo "================================================"
echo ""
echo "实验配置:"
echo "  - 最大步数: ${MAX_STEPS}"
echo "  - 预设配置: ${PRESETS[*]}"
echo ""
echo "结果目录:"
for preset in "${PRESETS[@]}"; do
    echo "  - output_lora_${preset}/"
done
echo ""
echo "评测命令示例:"
echo "  python scripts/evaluation/run_vlm_evaluation.py --model_path output_lora_standard/checkpoint-5000 --dimension all"
