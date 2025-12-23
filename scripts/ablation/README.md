# 消融实验脚本

本目录包含用于 LoRA 参数高效微调的消融实验脚本。

## 文件说明

| 文件 | 说明 |
|------|------|
| `run-lora-finetuning.py` | LoRA 微调主脚本，支持多种预设配置 |
| `run-lora-training.sh` | 单次 LoRA 训练启动脚本 |
| `run-ablation-experiments.sh` | 批量消融实验脚本 |

## LoRA 预设配置

| 预设 | r | alpha | dropout | target_modules | 说明 |
|------|---|-------|---------|----------------|------|
| `light` | 8 | 16 | 0.1 | q_proj, v_proj | 最小化参数 |
| `standard` | 16 | 32 | 0.05 | q_proj, k_proj, v_proj, o_proj | 标准配置 |
| `full` | 32 | 64 | 0.05 | 所有投影层 + MLP | 覆盖更多层 |
| `aggressive` | 64 | 128 | 0.05 | 所有投影层 + MLP | 高秩 LoRA |

## 使用方法

### 1. 单次训练

```bash
# 使用默认配置（standard，5000步）
bash scripts/ablation/run-lora-training.sh

# 使用指定预设
bash scripts/ablation/run-lora-training.sh --preset light

# 自定义最大步数
bash scripts/ablation/run-lora-training.sh --preset full --max-steps 3000

# 自定义参数
bash scripts/ablation/run-lora-training.sh --preset full --lr 1e-4 --max-steps 5000
```

### 2. 批量消融实验

```bash
# 自动运行 light, standard, full 三种配置（默认 5000 步）
bash scripts/ablation/run-ablation-experiments.sh

# 自定义步数
MAX_STEPS=3000 bash scripts/ablation/run-ablation-experiments.sh
```

> **注意**: 所有消融实验默认统一在 **5000 步** 停止，以确保公平对比。

### 3. Python 脚本直接调用

```bash
python scripts/ablation/run-lora-finetuning.py \
    --model_name_or_path ./models/Qwen3-VL-2B-Instruct \
    --train_file ./data/train_151230.jsonl \
    --image_root ./ \
    --output_dir ./output_lora_custom \
    --lora_preset standard \
    --lora_r 32 \
    --lora_alpha 64 \
    --bf16
```

## 自定义 LoRA 参数

可以通过命令行参数覆盖预设配置：

```bash
python scripts/ablation/run-lora-finetuning.py \
    --lora_preset standard \
    --lora_r 24 \
    --lora_alpha 48 \
    --lora_dropout 0.1 \
    --lora_target_modules "q_proj,v_proj,o_proj"
```

## 输出目录结构

训练完成后，输出目录包含：

```
output_lora_standard/
├── checkpoint-200/
│   ├── adapter_config.json      # LoRA 配置
│   ├── adapter_model.safetensors # LoRA 权重
│   └── ...
├── adapter_config.json          # 最终 LoRA 配置
├── adapter_model.safetensors    # 最终 LoRA 权重
└── ...
```

## 加载训练好的 LoRA 模型

```python
from transformers import AutoModelForImageTextToText, AutoProcessor
from peft import PeftModel

# 加载基础模型
base_model = AutoModelForImageTextToText.from_pretrained(
    "models/Qwen3-VL-2B-Instruct",
    torch_dtype=torch.bfloat16,
    trust_remote_code=True,
)

# 加载 LoRA 适配器
model = PeftModel.from_pretrained(base_model, "output_lora_standard")

# 合并权重（可选，用于推理加速）
model = model.merge_and_unload()
```

## 全量微调 vs LoRA 对比

| 方面 | 全量微调 | LoRA |
|------|---------|------|
| 可训练参数 | ~2B (100%) | ~10M (~0.5%) |
| 显存占用 | 高 | 低 |
| 训练速度 | 慢 | 快 |
| 存储空间 | 大 (~4GB) | 小 (~40MB) |
| 性能 | 最佳 | 接近全量 |
