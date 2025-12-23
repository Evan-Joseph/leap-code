<a name="project-name"></a>

# [PROJECT_NAME]: Multimodal Task Planning with Qwen3-VL on VLABench

## 1. 介绍 · Introduction

`[PROJECT_NAME]` 聚焦多模态机器人任务规划，围绕 **Qwen3-VL-2B-Instruct** 构建全参微调、VLABench 六维度评测与 LLM-as-a-Judge 盲审流程。当前版本旨在提供可复现实验脚本，便于研究者验证和扩展以下能力：

- Memory & Tasks、CommonSense、Semantic、Spatial、PhysicsLaw、Complex 六个维度的统一评测。
- 训练、评测、盲审、可视化脚本的标准化组织，降低跨场景复现门槛。
- LOVE-Agibot-Beta 等公开数据的下载、清洗与首帧提取工具。

## 2. 仓库结构 · Repository Structure

```text
[PROJECT_NAME]/
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
├── VLABench/       # 官方评测子模块（仓库内已内置，如需同步再执行 submodule）
├── eva_results/    # 最新评测结果，按维度/模型分层
├── backup_eva_results/ # 归档历史评测结果
├── run_vlabench_evaluation_bingXing.sh # 并行批量评测脚本示例
├── qwen-ft-env.yml # Conda 环境定义
└── README.md
```

## 3. 快速开始 · Quick Start

### 3.1 环境准备

```bash
conda env create -f qwen-ft-env.yml
conda activate qwen-ft-env
```

### 3.2 下载模型与数据

```bash
# 1) 获取 Qwen3-VL-2B-Instruct（脚本内置 hf-mirror 加速）
bash scripts/download/download_model.sh

# 2) 下载 VLABench 评测集（智能限流重试）
python scripts/download/download_vlabench_with_retry.py

# 3) （可选）抓取 LOVE-Agibot-Beta 图像首帧
python scripts/utils/prepare_love_agibot.py --num-workers 4
```

### 3.3 全参微调示例

```bash
bash scripts/training/run-training.sh
```

脚本默认执行以下步骤：

1. 根据 `scripts/` 上级目录定位仓库根路径。
2. 读取 `data/train_151230.jsonl` 与 `models/Qwen3-VL-2B-Instruct/`。
3. 以 BF16、SDPA 注意力和 6×6 的梯度累积开展训练，并将 checkpoint 写入 `output/`。

### 3.4 LoRA 参数高效微调（消融实验）

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

### 3.5 VLABench 多维度评测

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

| Model | M&T ↑ | CommonSense ↑ | Complex ↑ | Avg Final Score ↑ |
| --- | --- | --- | --- | --- |
| Qwen3-VL-2B-Baseline | 29.60 | 25.70 | 14.28 | 23.19 |
| [PROJECT_NAME] Finetuned (checkpoint-5000) | 32.96 | 27.28 | 19.67 | 26.63 |

> 数据来源：`eva_results/<dimension>/<model>/final_score.json` 中 `total_score` 的均值；图像可通过 `scripts/evaluation/draw_*.py` 生成。

## 5. 数据与模型 · Datasets & Models

- **VLABench**：官方仓库 <https://github.com/VLABench/VLABench>，本项目直接嵌入子模块用于评测。
- **LOVE-Agibot-Beta**：Hugging Face `EvanSirius/LOVE-Agibot-Beta`，脚本默认只提取首帧以控制大小。
- **基座模型**：`Qwen/Qwen3-VL-2B-Instruct`（Hugging Face）。
- **自有 checkpoint**：`output/checkpoint-*` 为中间断点，可选择性同步到公开 Model Zoo（若发布的话在此添加链接）。

## 6. Citation & Acknowledgement

```
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

