#!/usr/bin/env bash
# ================================================
# 模型下载脚本  download_model.sh
# ================================================
# 用于从 Hugging Face 下载 Qwen3-VL-2B-Instruct 模型到本地目录
# 使用 hf-mirror 加速国内下载
# ================================================

set -euo pipefail

# 解析脚本与仓库根路径，使用绝对路径确保从任意 cwd 调用脚本都正确
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"  # scripts/download
REPO_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"                 # repo root (parent of scripts)

# 创建模型目录（仓库下的 models）
mkdir -p "${REPO_ROOT}/models"

# 设置 HF 镜像端点
export HF_ENDPOINT=${HF_ENDPOINT:-https://hf-mirror.com}

echo "==============================================="
echo "Qwen3-VL-2B-Instruct 模型下载脚本"
echo "==============================================="
echo "开始下载 Qwen3-VL-2B-Instruct 模型..."
echo "模型将被下载到: models/Qwen3-VL-2B-Instruct"
echo ""

# 检查 hfd.sh 脚本是否存在
HFD_SH="${SCRIPT_DIR}/hfd.sh"
if [ ! -f "${HFD_SH}" ]; then
    echo "错误: 未找到 hfd.sh 脚本 (${HFD_SH})"
    echo "请先下载 hfd.sh 脚本到 scripts/download/ 或将其放在同一目录下。"
    echo "  wget https://hf-mirror.com/hfd/hfd.sh -O ${HFD_SH}"
    echo "  chmod a+x ${HFD_SH}"
    exit 1
fi

# 检查是否已安装依赖工具
if ! command -v git &> /dev/null; then
    echo "错误: 未找到 git 命令"
    echo "请先安装 git:"
    echo "  sudo apt-get update && sudo apt-get install git"
    exit 1
fi

if ! command -v aria2c &> /dev/null; then
    echo "警告: 未找到 aria2c 命令，将使用 wget 下载"
    echo "建议安装 aria2c 以获得更好的下载性能:"
    echo "  sudo apt-get install aria2"
    DOWNLOAD_TOOL="wget"
else
    DOWNLOAD_TOOL="aria2c"
fi

# 使用 hfd.sh 下载模型
# hfd.sh 是基于 git + aria2 的专用下载工具，支持断点续传和多线程下载
echo "正在下载模型文件..."
echo "这可能需要一些时间，请耐心等待..."
echo "下载工具: $DOWNLOAD_TOOL"
echo ""

download_status=0
"${HFD_SH}" Qwen/Qwen3-VL-2B-Instruct \
    --tool ${DOWNLOAD_TOOL} \
    -x 4 \
    --local-dir "${REPO_ROOT}/models/Qwen3-VL-2B-Instruct" \
    || download_status=$?

if [ $download_status -ne 0 ]; then
    echo ""
    echo "警告: 模型下载过程中出现错误 (退出码: $download_status)"
    echo "这可能是由于网络问题或模型文件较大导致的中断"
    echo "您可以重新运行此脚本，它将从断点继续下载"
    exit $download_status
fi

echo ""
echo "==============================================="
echo "模型下载完成!"
echo "模型位置: ${REPO_ROOT}/models/Qwen3-VL-2B-Instruct"
echo ""
echo "现在您可以运行训练脚本:"
echo "  cd scripts && ./run-training.sh"
echo "==============================================="