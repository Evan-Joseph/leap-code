#!/bin/bash

# DeepSeek-VL2-Tiny 0-shot 并行评估脚本
# 显存需求: 6-9 GB
# 注意: 需要在 deepseek-vl2-env 环境中运行 (transformers==4.38.2)

# 解析仓库根目录
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
WORK_DIR="$(cd "${SCRIPT_DIR}/../.." && pwd)"
cd "$WORK_DIR"

# 使用 DeepSeek-VL2 专用环境
# 如果使用 conda prefix 形式的环境
if [ -d "/root/autodl-tmp/envs/deepseek-vl2-env" ]; then
    PYTHON_PATH="/root/autodl-tmp/envs/deepseek-vl2-env/bin/python3"
else
    # 回退到默认环境
    PYTHON_PATH="python3"
fi

MODEL_PATH="$WORK_DIR/models/deepseek-vl2-tiny"
DATA_PATH="$WORK_DIR/dataset/vlm_evaluation_v1.0"

# 规范化日志和结果目录
CONFIG_NAME="deepseek_vl2_tiny_0shot"
LOG_DIR="$WORK_DIR/logs/$CONFIG_NAME"
RESULT_DIR="$WORK_DIR/results/$CONFIG_NAME"

mkdir -p "$LOG_DIR"

echo "=============================================="
echo "  DeepSeek-VL2-Tiny 0-shot 并行评估"
echo "=============================================="
echo "Python: $PYTHON_PATH"
echo "模型: $MODEL_PATH"
echo "结果目录: $RESULT_DIR"
echo ""

# 检查模型是否存在
if [ ! -d "$MODEL_PATH" ]; then
    echo "❌ 模型不存在: $MODEL_PATH"
    echo "请先下载模型:"
    echo "  export HF_ENDPOINT=https://hf-mirror.com"
    echo "  bash scripts/download/hfd.sh deepseek-ai/deepseek-vl2-tiny --local-dir $MODEL_PATH"
    exit 1
fi

# 检查 Python 环境
if ! $PYTHON_PATH -c "import transformers; assert transformers.__version__ == '4.38.2'" 2>/dev/null; then
    echo "⚠️  警告: transformers 版本可能不是 4.38.2"
    echo "推荐在 deepseek-vl2-env 环境中运行"
fi

# 定义所有维度
DIMENSIONS=("M&T" "CommenSence" "Semantic" "Spatial" "PhysicsLaw" "Complex")
LOG_NAMES=("mt" "commonsense" "semantic" "spatial" "physics" "complex")

COMMANDS=()
LOGS=()

for i in "${!DIMENSIONS[@]}"; do
    dim="${DIMENSIONS[$i]}"
    log_name="${LOG_NAMES[$i]}"
    COMMANDS+=("$PYTHON_PATH scripts/evaluation/run_vlm_evaluation.py --model_path $MODEL_PATH --dimension '$dim' --num_episodes 5 --max_tasks 20 --data_path $DATA_PATH --output_dir '$RESULT_DIR'")
    LOGS+=("$LOG_DIR/$log_name.log")
done

echo "启动并行评估任务队列 (并发数: 3)..."
$PYTHON_PATH scripts/utils/task_runner.py \
    --commands "${COMMANDS[@]}" \
    --logs "${LOGS[@]}" \
    --concurrency 3

echo ""
echo "=============================================="
echo "✅ 所有维度评估完成!"
echo "结果保存至: $RESULT_DIR"
echo "=============================================="
