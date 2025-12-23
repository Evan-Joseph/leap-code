#!/usr/bin/env bash
# ================================================
# 正式全量训练启动脚本  run-training.sh
# ================================================
# 核心策略说明：
# 1) 不编译 flash-attn：避免在付费机器上耗费额外编译时间与潜在的CUDA编译不稳定。
# 2) 使用 PyTorch 原生 Scaled Dot Product Attention (SDPA) (--attn_implementation=sdpa)：
#    - SDPA 在 PyTorch 2.x 中已高度优化，支持多种后端（FlashAttention / Math / MemEfficient）自动选择。
#    - 免编译，直接利用现有 GPU 能力；在多模态场景中稳定性更佳。
# 3) 全量训练：不启用 LoRA；保持 run-finetuning.py 默认全参可训练。
# 4) 训练时长较长：请结合 logging_steps 与 save_steps 监控进度与中途断点。
#
# 使用前请确保已激活 Conda 环境: conda activate qwen-ft-env
# 并已通过 pip 安装对齐的依赖版本（transformers Git 提交、torch 等）。
#
# 注意：在运行此脚本之前，请先运行 ./download_model.sh 脚本下载预训练模型
# ================================================
set -euo pipefail

# 获取脚本所在目录（绝对路径）
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
# 获取工作区根目录（scripts 的父目录）
WORK_DIR="$(cd "${SCRIPT_DIR}/.." && pwd)"

MODEL_DIR="${WORK_DIR}/models/Qwen3-VL-2B-Instruct"          # 预训练模型本地目录
TRAIN_FILE="${WORK_DIR}/data/train_151230.jsonl"             # 训练集 (151k数据) 
IMAGE_ROOT="${WORK_DIR}/"                         # 图像根路径 
OUTPUT_DIR="${WORK_DIR}/output" # 正式输出目录

# 转换为绝对路径（确保从任何目录执行都能正确找到文件）
MODEL_DIR="$(cd "${MODEL_DIR}" 2>/dev/null && pwd || echo "${MODEL_DIR}")"
TRAIN_FILE="$(cd "$(dirname "${TRAIN_FILE}")" 2>/dev/null && echo "$(pwd)/$(basename "${TRAIN_FILE}")" || echo "${TRAIN_FILE}")"
IMAGE_ROOT="$(cd "${IMAGE_ROOT}" 2>/dev/null && pwd || echo "${IMAGE_ROOT}")"
OUTPUT_DIR="$(mkdir -p "${OUTPUT_DIR}" && cd "${OUTPUT_DIR}" && pwd)"
EPOCHS=2                                          # 训练轮数 (151k数据集建议1轮)
BATCH_SIZE=6                                      # 每卡批大小（vGPU 32GB 可提升到6-8）
GRAD_ACC=6                                        # 梯度累积，等效全局批 = 36
LR=1.5e-5                                         # 初始学习率 (大数据集适当降低)
WARMUP_STEPS=800                                 # 预热步数（根据151k数据规模调整）
WEIGHT_DECAY=0.01                                 # 权重衰减
LOGGING_STEPS=10                                  # 日志打印频率
SAVE_STEPS=200                                    # checkpoint 保存频率
SAVE_TOTAL_LIMIT=100                               # 最多保留的 checkpoint 数量
SEED=42                                           # 随机种子
ATTN_IMPL="sdpa"                                 # 使用 PyTorch SDPA，避免 flash-attn 编译
BF16_FLAG="--bf16"                               # vGPU 支持混合精度训练，启用以节省显存提升稳定性
FP16_FLAG=""                                     # 若需改为 FP16 可设置为 "--fp16" 并移除 BF16_FLAG
GRAD_CP_FLAG="--gradient_checkpointing"          # 开启梯度检查点节省显存
DEEPSPEED_CONFIG=""                              # 可选: 指向 DeepSpeed zero 配置文件，例如 scripts/zero3.json

# 若需要 DeepSpeed：取消下一行注释并设置配置文件路径
# DEEPSPEED_CONFIG="--deepspeed qwen3-official/qwen-vl-finetune/scripts/zero3.json"

# 检查模型目录是否存在
if [ ! -d "${MODEL_DIR}" ]; then
    echo "错误: 模型目录 ${MODEL_DIR} 不存在!"
    echo "请先运行 ../scripts/download/download_model.sh 脚本下载预训练模型"
    exit 1
fi

mkdir -p "${OUTPUT_DIR}" || true

# ---------------------- 启动训练 ----------------------
python "${SCRIPT_DIR}/run-finetuning.py" \
  --model_name_or_path "${MODEL_DIR}" \
  --train_file "${TRAIN_FILE}" \
  --image_root "${IMAGE_ROOT}" \
  --output_dir "${OUTPUT_DIR}" \
  --num_train_epochs "${EPOCHS}" \
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
  ${FP16_FLAG} \
  ${GRAD_CP_FLAG} \
  ${DEEPSPEED_CONFIG}