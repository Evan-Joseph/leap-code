#!/usr/bin/env bash
# ================================================
# LoRA 消融实验启动脚本  run-lora-training.sh
# ================================================
# 核心策略说明：
# 1) 使用 PEFT 库的 LoRA 适配器进行参数高效微调。
# 2) 支持多种预设配置：standard, light, full, aggressive。
# 3) 相比全量微调，LoRA 训练速度更快、显存占用更低。
# 4) 适用于消融实验：对比不同 LoRA 配置的效果。
#
# 使用前请确保已激活 Conda 环境: conda activate qwen-ft-env
# ================================================
set -euo pipefail

# 获取脚本所在目录（绝对路径）
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
# 获取工作区根目录（scripts/ablation 的上两级）
WORK_DIR="$(cd "${SCRIPT_DIR}/../.." && pwd)"

# ===================== 基础配置 =====================
MODEL_DIR="${WORK_DIR}/models/Qwen3-VL-2B-Instruct"
TRAIN_FILE="${WORK_DIR}/data/train_151230.jsonl"
IMAGE_ROOT="${WORK_DIR}/"

# ===================== LoRA 配置 =====================
# 预设选项: standard, light, full, aggressive
LORA_PRESET="${LORA_PRESET:-standard}"

# 输出目录（根据预设自动命名）
OUTPUT_DIR="${WORK_DIR}/output_lora_${LORA_PRESET}"

# ===================== 训练超参数 =====================
MAX_STEPS=5000                                    # 最大训练步数（统一消融实验终止点）
EPOCHS=2                                          # 训练轮数（当 MAX_STEPS 设置时会被忽略）
BATCH_SIZE=4                                      # 每卡批大小（LoRA 可以用更大的批次）
GRAD_ACC=4                                        # 梯度累积，等效全局批 = 16
LR=2e-4                                          # 初始学习率（LoRA 通常使用更高的学习率）
WARMUP_STEPS=100                                 # 预热步数
WEIGHT_DECAY=0.01                                 # 权重衰减
LOGGING_STEPS=10                                  # 日志打印频率
SAVE_STEPS=500                                    # checkpoint 保存频率（5000步保存10个）
SAVE_TOTAL_LIMIT=15                               # 最多保留的 checkpoint 数量
SEED=42                                           # 随机种子
ATTN_IMPL="sdpa"                                 # 使用 PyTorch SDPA
BF16_FLAG="--bf16"                               # 混合精度训练
GRAD_CP_FLAG="--gradient_checkpointing"          # 开启梯度检查点

# ===================== 参数解析 =====================
# 支持通过命令行参数覆盖配置
while [[ $# -gt 0 ]]; do
    case $1 in
        --preset)
            LORA_PRESET="$2"
            OUTPUT_DIR="${WORK_DIR}/output_lora_${LORA_PRESET}"
            shift 2
            ;;
        --lr)
            LR="$2"
            shift 2
            ;;
        --epochs)
            EPOCHS="$2"
            shift 2
            ;;
        --batch-size)
            BATCH_SIZE="$2"
            shift 2
            ;;
        --output-dir)
            OUTPUT_DIR="$2"
            shift 2
            ;;
        --max-steps)
            MAX_STEPS="$2"
            shift 2
            ;;
        *)
            echo "未知参数: $1"
            echo "可用参数: --preset, --lr, --epochs, --batch-size, --output-dir, --max-steps"
            exit 1
            ;;
    esac
done

# ===================== 路径转换 =====================
MODEL_DIR="$(cd "${MODEL_DIR}" 2>/dev/null && pwd || echo "${MODEL_DIR}")"
TRAIN_FILE="$(cd "$(dirname "${TRAIN_FILE}")" 2>/dev/null && echo "$(pwd)/$(basename "${TRAIN_FILE}")" || echo "${TRAIN_FILE}")"
IMAGE_ROOT="$(cd "${IMAGE_ROOT}" 2>/dev/null && pwd || echo "${IMAGE_ROOT}")"
OUTPUT_DIR="$(mkdir -p "${OUTPUT_DIR}" && cd "${OUTPUT_DIR}" && pwd)"

# ===================== 环境检查 =====================
if [ ! -d "${MODEL_DIR}" ]; then
    echo "错误: 模型目录 ${MODEL_DIR} 不存在!"
    echo "请先运行 scripts/download/download_model.sh 脚本下载预训练模型"
    exit 1
fi

# ===================== 打印配置 =====================
echo "================================================"
echo "LoRA 消融实验训练配置"
echo "================================================"
echo "LoRA 预设: ${LORA_PRESET}"
echo "模型目录: ${MODEL_DIR}"
echo "训练文件: ${TRAIN_FILE}"
echo "输出目录: ${OUTPUT_DIR}"
echo "学习率: ${LR}"
echo "批次大小: ${BATCH_SIZE}"
echo "梯度累积: ${GRAD_ACC}"
echo "最大步数: ${MAX_STEPS}"
echo "================================================"

mkdir -p "${OUTPUT_DIR}" || true

# ===================== 启动训练 =====================
python "${SCRIPT_DIR}/run-lora-finetuning.py" \
  --model_name_or_path "${MODEL_DIR}" \
  --train_file "${TRAIN_FILE}" \
  --image_root "${IMAGE_ROOT}" \
  --output_dir "${OUTPUT_DIR}" \
  --lora_preset "${LORA_PRESET}" \
  --max_steps "${MAX_STEPS}" \
  --per_device_train_batch_size "${BATCH_SIZE}" \
  --gradient_accumulation_steps "${GRAD_ACC}" \
  --learning_rate "${LR}" \
  --warmup_steps "${WARMUP_STEPS}" \
  --weight_decay "${WEIGHT_DECAY}" \
  --logging_steps "${LOGGING_STEPS}" \
  --save_steps "${SAVE_STEPS}" \
  --save_total_limit "${SAVE_TOTAL_LIMIT}" \
  --model_max_length 4096 \
  --seed "${SEED}" \
  --attn_implementation "${ATTN_IMPL}" \
  --image_tokens_max 768 \
  ${BF16_FLAG} \
  ${GRAD_CP_FLAG}
