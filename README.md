<a name="leap"></a>

# LEAP: Logical Embodied Action Planning for Long-Horizon Robotic Tasks via Generative Vision-Language Alignment

<p align="center">
  <a href="https://huggingface.co/EvanSirius/leap-ckpts"><img src="https://img.shields.io/badge/🤗%20Model-leap--ckpts-blue" alt="Model"></a>
  <a href="https://huggingface.co/datasets/EvanSirius/leap-agibot-processed"><img src="https://img.shields.io/badge/🤗%20Dataset-leap--agibot--processed-green" alt="Dataset"></a>
  <a href="https://github.com/OpenMOSS/VLABench"><img src="https://img.shields.io/badge/Benchmark-VLABench-orange" alt="VLABench"></a>
</p>

## 1. 介绍 · Introduction

**LEAP** 聚焦多模态机器人任务规划，围绕 **Qwen3-VL-2B-Instruct** 构建全参微调、VLABench 六维度评测与 LLM-as-a-Judge 盲审流程。当前版本旨在提供可复现实验脚本，便于研究者验证和扩展以下能力：

- Memory & Tasks、CommonSense、Semantic、Spatial、PhysicsLaw、Complex 六个维度的统一评测。
- 训练、评测、盲审、可视化脚本的标准化组织，降低跨场景复现门槛。
- LOVE-Agibot-Beta 等公开数据的下载、清洗与首帧提取工具。

## 2. 仓库结构 · Repository Structure

```text
LEAP/
├── configs/        # 全局路径、训练/评测常量（WorkspaceConfig）
├── data/           # JSONL、LOVE-Agibot 图像
├── dataset/        # vlm_evaluation_v1.0（按 CommonSense/Complex/... 维度拆分）
├── logs/           # 各维度评测日志与 PID 记录
├── models/         # HuggingFace 本地缓存（如 Qwen3-VL-2B-Instruct）
├── output/         # 训练 checkpoint-* 与最终权重
├── scripts/
│   ├── download/   # 模型 & VLABench 下载脚本
│   ├── training/   # run-finetuning.py、run-training.sh 全量微调入口
│   ├── ablation/   # LoRA 消融实验脚本（参数高效微调）
│   ├── evaluation/ # VLABench/LLM-Judge/可视化脚本
│   └── utils/      # 数据清洗、LOVE-Agibot 处理
├── VLABench/       # 官方评测子模块（需执行 git submodule update --init）
├── eva_results/    # 最新评测结果，按维度/模型分层
├── qwen-ft-env.yml # Conda 环境定义
└── README.md
```

## 3. 快速开始 · Quick Start

### 3.1 克隆仓库

```bash
# 包含 VLABench 子模块
git clone --recursive https://github.com/Evan-Joseph/leap-code.git
cd leap-code

# 如果忘记 --recursive，可以后续执行：
git submodule update --init --recursive
```

### 3.2 环境准备

```bash
conda env create -f qwen-ft-env.yml
conda activate qwen-ft-env
```

### 3.3 下载模型与数据

```bash
# 1) 获取 Qwen3-VL-2B-Instruct（脚本内置 hf-mirror 加速，修复了绝对路径问题）
bash scripts/download/download_model.sh

# 2) 下载 VLABench 评测集（优化了限流重试逻辑，支持断点续传）
python scripts/download/download_vlabench_with_retry.py

# 3) 验证 VLABench 数据完整性（新增验证脚本，确保下载无损）
python scripts/utils/verify_dataset.py

# 4) （可选）下载 LEAP 预处理数据集
huggingface-cli download EvanSirius/leap-agibot-processed --local-dir data/
```

### 3.4 全参微调示例

```bash
bash scripts/training/run-training.sh
```

脚本默认执行以下步骤：

1. 根据 `scripts/` 上级目录定位仓库根路径。
2. 读取 `data/train_151230.jsonl` 与 `models/Qwen3-VL-2B-Instruct/`。
3. 以 BF16、SDPA 注意力和 6×6 的梯度累积开展训练，并将 checkpoint 写入 `output/`。

### 3.5 LoRA 参数高效微调（消融实验）

```bash
# 使用默认 standard 预设
bash scripts/ablation/run-lora-training.sh

# 使用指定预设（light/standard/full/aggressive）
bash scripts/ablation/run-lora-training.sh --preset full

# 批量运行多种配置进行消融实验
bash scripts/ablation/run-ablation-experiments.sh
```

LoRA 预设配置说明：
- `light`: r=8, 最小化参数，适合快速验证
- `standard`: r=16, 标准配置，平衡效果与效率
- `full`: r=32, 覆盖更多层，接近全量微调效果
- `aggressive`: r=64, 高秩 LoRA，最大化表达能力

### 3.6 VLABench 多维度评测

```bash
# 全维度（M&T/CommonSense/.../Complex）批量评测
python scripts/evaluation/run_vlm_evaluation.py \
	--checkpoint output/checkpoint-200 \
	--dimension all

# 单独跑 M&T
python scripts/evaluation/run_vlm_evaluation.py \
	--checkpoint output/checkpoint-200 \
	--dimension "M&T"

# Blind-10 LLM-as-a-Judge 流水线
python scripts/evaluation/run_vlm_output.py --baseline_model models/Qwen3-VL-2B-Instruct
python scripts/evaluation/analyze_output_results.py
```

## 4. 结果与可视化 · Results

| Model | M&T ↑ | CommonSense ↑ | Semantic ↑ | Spatial ↑ | PhysicalLaw ↑ | Complex ↑ | Avg ↑ |
| --- | --- | --- | --- | --- | --- | --- | --- |
| Qwen3-VL-2B-Baseline | 29.60 | 25.70 | 25.89 | 31.10 | 24.00 | 14.28 | 25.10 |
| **LEAP (checkpoint-5000)** | **32.96** | **27.27** | **28.49** | **31.62** | **30.27** | **19.67** | **28.38** |
| Improvement | +11.3% | +6.1% | +10.0% | +1.7% | +26.1% | +37.7% | **+13.1%** |

> 数据来源：`eva_results/<dimension>/<model>/final_score.json` 中 `total_score` 的均值；图像可通过 `scripts/evaluation/draw_*.py` 生成。

## 5. 数据与模型 · Datasets & Models

| 资源 | 链接 | 说明 |
| --- | --- | --- |
| **LEAP Checkpoints** | [🤗 EvanSirius/leap-ckpts](https://huggingface.co/EvanSirius/leap-ckpts) | 全量微调权重 (checkpoint-200 ~ checkpoint-7000) |
| **LEAP Dataset** | [🤗 EvanSirius/leap-agibot-processed](https://huggingface.co/datasets/EvanSirius/leap-agibot-processed) | 训练集、测试集、盲评集 |
| **VLABench** | [GitHub OpenMOSS/VLABench](https://github.com/OpenMOSS/VLABench) | 官方评测框架（Git Submodule） |
| **基座模型** | [🤗 Qwen/Qwen3-VL-2B-Instruct](https://huggingface.co/Qwen/Qwen3-VL-2B-Instruct) | Qwen3-VL 2B 指令微调版 |

## 6. Citation & Acknowledgement

```bibtex
@article{zhang2024vlabench,
	title={VLABench: A Large-Scale Benchmark for Language-Conditioned Robotics Manipulation with Long-Horizon Reasoning Tasks},
	author={Shiduo Zhang and Zhe Xu and Peiju Liu and others},
	journal={arXiv preprint arXiv:2412.18194},
	year={2024}
}

@article{bai2025qwen3vl,
	title={Qwen3-VL Technical Report},
	author={Shuai Bai and Yuxuan Cai and Ruizhe Chen and others},
	journal={arXiv preprint arXiv:2511.21631},
	year={2025}
}
```

- 感谢 Qwen 团队开放 Qwen3-VL 系列，使得本仓库可以在开源权重上构建。
- 致谢 VLABench、LOVE-Agibot 等项目提供数据与评测基础设施。

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.
