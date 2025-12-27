# DeepSeek-VL2 环境配置与评估指南

> **更新时间**: 2025-12-27
> **状态**: 📦 环境隔离方案

---

## 📋 背景

DeepSeek-VL2 官方代码基于 **Transformers 4.38.2**，与当前主环境使用的 **Transformers 4.57.3** 存在严重 API 不兼容问题：

| 问题 | 说明 |
|------|------|
| `LlamaFlashAttention2` | Transformers 4.48+ 移除了此类 |
| `GenerationMixin` | Transformers 4.50+ 中 PreTrainedModel 不再继承 |
| `DynamicCache.seen_tokens` | 新版本中 API 改变 |
| `generation_config` | 新版本中初始化逻辑变化 |

**结论**: 通过代码修改实现兼容的成本过高，推荐使用**独立 Conda 环境**。

---

## 🔧 环境配置

### 方案 A: 使用 YAML 文件创建 (推荐)

```bash
# 创建环境
conda env create -f envs/deepseek-vl2-env.yml

# 激活环境
conda activate deepseek-vl2-env
```

### 方案 B: 手动创建

```bash
# 1. 创建环境
conda create -n deepseek-vl2-env python=3.10 -y
conda activate deepseek-vl2-env

# 2. 安装 PyTorch (CUDA 12.1)
pip install torch==2.1.0 torchvision==0.16.0 --index-url https://download.pytorch.org/whl/cu121

# 3. 安装 Transformers (指定版本！)
pip install transformers==4.38.2

# 4. 安装其他依赖
pip install accelerate bitsandbytes sentencepiece
pip install attrdict einops timm xformers
pip install colorama tqdm pillow pyyaml
```

---

## 📁 克隆 DeepSeek-VL2 仓库

```bash
cd /root/autodl-tmp/leap-code

# 克隆官方仓库
git clone https://github.com/deepseek-ai/DeepSeek-VL2.git

# 可选: 安装为包
cd DeepSeek-VL2
pip install -e .
cd ..
```

---

## ✅ 验证安装

```bash
# 确保在正确的环境中
conda activate deepseek-vl2-env

# 验证版本
python -c "import transformers; print(f'Transformers: {transformers.__version__}')"
# 应该输出: Transformers: 4.38.2

# 运行官方推理示例
python DeepSeek-VL2/inference.py --model_path ./models/deepseek-vl2-small
```

---

## 🚀 运行 VLABench 评估

```bash
# 1. 激活 DeepSeek 环境
conda activate deepseek-vl2-env

# 2. 进入项目目录
cd /root/autodl-tmp/leap-code

# 3. 单维度评估
python scripts/evaluation/run_vlm_evaluation.py \
    --model_path ./models/deepseek-vl2-small \
    --dimension M\&T \
    --num_episodes 5

# 4. 完整并行评估
bash scripts/shell/eval_deepseek_vl2_small_0shot_parallel.sh
```

---

## 🗂️ 多环境管理

| 环境名 | 用途 | Transformers 版本 |
|--------|------|-------------------|
| `qwen-ft-env` | Qwen2.5-VL, InternVL, MiniCPM-V | 4.57.3 |
| `deepseek-vl2-env` | DeepSeek-VL2 专用 | 4.38.2 |

### 切换环境示例

```bash
# 评估 Qwen2.5-VL
conda activate qwen-ft-env
python scripts/evaluation/run_vlm_evaluation.py --model_path ./models/qwen2.5-vl-7b ...

# 评估 DeepSeek-VL2
conda activate deepseek-vl2-env
python scripts/evaluation/run_vlm_evaluation.py --model_path ./models/deepseek-vl2-small ...
```

---

## ⚠️ 注意事项

1. **不要混用环境**: DeepSeek-VL2 必须在 `deepseek-vl2-env` 中运行

2. **模型路径**: 确保模型已下载到 `./models/deepseek-vl2-small`

3. **显存需求**:
   - 不量化: ~70GB (需要 A100 80G)
   - 4-bit 量化: ~15GB (32GB GPU 可运行)

4. **如果使用量化**: 确保 `bitsandbytes` 已正确安装

---

## 📚 相关资源

- [DeepSeek-VL2 官方仓库](https://github.com/deepseek-ai/DeepSeek-VL2)
- [HuggingFace Model Card](https://huggingface.co/deepseek-ai/deepseek-vl2-small)
- [DeepSeek-VL2 论文](https://arxiv.org/abs/2412.10302)
