#!/usr/bin/env bash
# ================================================
# LoRA 消融实验批量运行脚本
# 自动运行多种 LoRA 配置进行对比实验
# ================================================
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

echo "================================================"
echo "LoRA 消融实验批量运行"
echo "================================================"

# 定义要测试的预设列表
PRESETS=("light" "standard" "full")

for preset in "${PRESETS[@]}"; do
    echo ""
    echo "========================================"
    echo "开始运行: ${preset} 配置"
    echo "========================================"
    
    bash "${SCRIPT_DIR}/run-lora-training.sh" --preset "${preset}"
    
    echo ""
    echo "[完成] ${preset} 配置训练完成"
    echo ""
done

echo "================================================"
echo "所有消融实验完成！"
echo "================================================"
echo ""
echo "结果目录:"
for preset in "${PRESETS[@]}"; do
    echo "  - output_lora_${preset}/"
done
