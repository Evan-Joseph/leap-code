# DeepSeek-VL2 环境配置与评估指南

> **更新时间**: 2025-12-27
> **推荐模型**: DeepSeek-VL2-Tiny (6-9GB 显存)

---

## 📊 模型显存要求

| 模型 | 显存要求 | 说明 |
|------|----------|------|
| **DeepSeek-VL2-Tiny** | **6-9 GB** | ✅ 推荐，单卡 32GB 轻松运行 |
| DeepSeek-VL2-Small | 40-80 GB | ⚠️ 需要 A100 80GB 或多卡 |

---

## 🔧 环境配置

### 服务器端操作

```bash
# 1. 创建环境 (使用数据盘，不占系统盘)
conda create --prefix /root/autodl-tmp/envs/deepseek-vl2-env python=3.10 -y

# 2. 激活环境
conda activate /root/autodl-tmp/envs/deepseek-vl2-env

# 3. 安装依赖 (使用清华镜像)
pip install -r envs/deepseek-vl2-requirements.txt -i https://pypi.tuna.tsinghua.edu.cn/simple

# 4. 克隆 DeepSeek-VL2 仓库
git clone https://github.com/deepseek-ai/DeepSeek-VL2.git

# 5. 安装 DeepSeek-VL2
cd DeepSeek-VL2 && pip install -e . -i https://pypi.tuna.tsinghua.edu.cn/simple && cd ..
```

---

## 📥 下载模型

```bash
# 下载 DeepSeek-VL2-Tiny (推荐)
mkdir -p ./models/deepseek-vl2-tiny
huggingface-cli download deepseek-ai/deepseek-vl2-tiny --local-dir ./models/deepseek-vl2-tiny

# 如果没有 huggingface-cli，先安装
pip install huggingface_hub
```

---

## ✅ 验证安装

```bash
# 确保激活正确环境
conda activate /root/autodl-tmp/envs/deepseek-vl2-env

# 验证版本
python -c "import transformers; print(f'Transformers: {transformers.__version__}')"
# 应该输出: Transformers: 4.38.2

# 运行验证脚本
python test_deepseek_vl2_inference.py --model_path ./models/deepseek-vl2-tiny
```

---

## 🚀 运行评估

```bash
# 激活环境
conda activate /root/autodl-tmp/envs/deepseek-vl2-env

# 单维度评估
python scripts/evaluation/run_vlm_evaluation.py \
    --model_path ./models/deepseek-vl2-tiny \
    --dimension M\&T \
    --num_episodes 5

# 完整评估
bash scripts/shell/eval_deepseek_vl2_tiny_0shot_parallel.sh
```

---

## 🗂️ 多环境管理

| 环境 | 路径 | 用途 | Transformers |
|------|------|------|--------------|
| `qwen-ft-env` | 系统环境 | Qwen2.5-VL, InternVL, MiniCPM-V | 4.57.3 |
| `deepseek-vl2-env` | `/root/autodl-tmp/envs/` | DeepSeek-VL2 | 4.38.2 |

---

## 📚 相关资源

- [DeepSeek-VL2 官方仓库](https://github.com/deepseek-ai/DeepSeek-VL2)
- [DeepSeek-VL2-Tiny (HuggingFace)](https://huggingface.co/deepseek-ai/deepseek-vl2-tiny)
- [DeepSeek-VL2-Small (HuggingFace)](https://huggingface.co/deepseek-ai/deepseek-vl2-small)
