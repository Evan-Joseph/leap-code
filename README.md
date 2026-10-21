<a name="leap"></a>

# LEAP: Logical Embodied Action Planning for Long-Horizon Robotic Tasks via Generative Vision-Language Alignment

<p align="center">
  <a href="README_en.md">English</a> · <b>简体中文</b>
</p>

<p align="center">
  <a href="https://huggingface.co/EvanSirius/leap-ckpts"><img src="https://img.shields.io/badge/HuggingFace-leap--ckpts-blue" alt="Model"></a>
  <a href="https://huggingface.co/datasets/EvanSirius/leap-agibot-processed"><img src="https://img.shields.io/badge/Dataset-leap--agibot--processed-green" alt="Dataset"></a>
  <a href="https://github.com/OpenMOSS/VLABench"><img src="https://img.shields.io/badge/Benchmark-VLABench-orange" alt="VLABench"></a>
  <a href="LICENSE"><img src="https://img.shields.io/badge/License-MIT-gray" alt="License"></a>
</p>

<p align="center">
  <img src="assets/leap_framework.png" width="96%" alt="LEAP Framework Overview" />
</p>

## 1. 简介

**LEAP** 面向长程多模态机器人任务规划，基于 **Qwen3-VL-2B-Instruct** 构建从动作蓝图（Action Blueprint）微调、VLABench 六维度评测到 LLM-as-a-Judge 盲审的完整实验流程。本仓库提供论文对应的训练、消融与评测代码：

- **六维度统一评测**：覆盖 Memory & Tasks (`M&T`)、`CommonSense`、`Semantic`、`Spatial`、`PhysicalLaw` 与 `Complex` 六类长程操作推理场景。
- **全参微调与 LoRA 消融**：提供 BF16 + SDPA 全参微调脚本及 4 档秩（`r=8/16/32/64`）的参数高效微调对比实验。
- **数据管线与盲审工具**：包含 `LOVE-Agibot-Beta` 关键帧抽取、指令清洗、VLABench 断点续传下载与双盲对比分析脚本。

<p align="center">
  <img src="assets/visual_execution.png" width="88%" alt="LEAP Visual Execution Comparison on VLABench" />
</p>

## 2. 仓库结构

```text
leap-code/
├── assets/               # 论文框架图与 VLABench 规划执行对比图
├── configs/              # 全局路径与训练/评测常量 (WorkspaceConfig)
├── scripts/
│   ├── download/         # 基座模型与 VLABench 评测集下载脚本
│   ├── training/         # 全量微调入口 (run-finetuning.py, run-training.sh)
│   ├── ablation/         # LoRA 消融实验脚本 (light / standard / full / aggressive)
│   ├── evaluation/       # VLABench 六维度评测、LLM-Judge 盲审与绘图脚本
│   ├── shell/            # 多模型批量评测启动脚本
│   └── utils/            # LOVE-Agibot 数据清洗与完整性校验
├── VLABench/             # 官方评测子模块 (git submodule)
├── eval.sh               # 统一评测入口脚本
├── qwen-ft-env.yml       # Conda 环境定义
├── README.md             # 中文说明
└── README_en.md          # 英文说明
```

## 3. 快速开始

### 3.1 克隆仓库与子模块

```bash
git clone --recursive https://github.com/Evan-Joseph/leap-code.git
cd leap-code

# 若克隆时未加 --recursive，可执行：
git submodule update --init --recursive
```

### 3.2 环境配置

```bash
conda env create -f qwen-ft-env.yml
conda activate qwen-ft-env
```

### 3.3 下载模型与数据

```bash
# 1) 下载 Qwen3-VL-2B-Instruct 基座权重（内置镜像加速）
bash scripts/download/download_model.sh

# 2) 下载 VLABench 评测集（支持限流重试与断点续传）
python scripts/download/download_vlabench_with_retry.py

# 3) 校验 VLABench 数据集完整性
python scripts/utils/verify_dataset.py

# 4)（可选）下载 LEAP 预处理数据集
huggingface-cli download EvanSirius/leap-agibot-processed --local-dir data/
```

### 3.4 全参微调

```bash
bash scripts/training/run-training.sh
```

脚本默认读取 `data/train_151230.jsonl` 与 `models/Qwen3-VL-2B-Instruct/`，启用 BF16 混合精度、SDPA 注意力与梯度累积（`6×6`），输出权重保存至 `output/`。

### 3.5 LoRA 参数高效微调（消融实验）

```bash
# 默认 standard 预设 (r=16)
bash scripts/ablation/run-lora-training.sh

# 指定预设档位 (light: r=8 | standard: r=16 | full: r=32 | aggressive: r=64)
bash scripts/ablation/run-lora-training.sh --preset full

# 批量运行多组消融实验
bash scripts/ablation/run-ablation-experiments.sh
```

### 3.6 VLABench 评测与盲审

```bash
# 全维度批量评测 (M&T / CommonSense / Semantic / Spatial / PhysicalLaw / Complex)
python scripts/evaluation/run_vlm_evaluation.py \
    --checkpoint output/checkpoint-5000 \
    --dimension all

# 单维度评测示例
python scripts/evaluation/run_vlm_evaluation.py \
    --checkpoint output/checkpoint-5000 \
    --dimension "M&T"

# Blind-10 LLM-as-a-Judge 盲审流水线
python scripts/evaluation/run_vlm_output.py --baseline_model models/Qwen3-VL-2B-Instruct
python scripts/evaluation/analyze_output_results.py
```

## 4. 实验结果 (VLABench)

| 模型 | M&T ↑ | CommonSense ↑ | Semantic ↑ | Spatial ↑ | PhysicalLaw ↑ | Complex ↑ | Avg ↑ |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| Qwen3-VL-2B-Baseline | 29.60 | 25.70 | 25.89 | 31.10 | 24.00 | 14.28 | 25.10 |
| **LEAP (checkpoint-5000)** | **32.96** | **27.27** | **28.49** | **31.62** | **30.27** | **19.67** | **28.38** |
| 相对提升 | +11.3% | +6.1% | +10.0% | +1.7% | +26.1% | +37.7% | **+13.1%** |

> 各维度得分为 `eva_results/<dimension>/<model>/final_score.json` 中 `total_score` 的算术均值，结果可视化脚本位于 `scripts/evaluation/draw_*.py`。

## 5. 模型权重与数据集

| 资源 | 链接 | 说明 |
| :--- | :--- | :--- |
| **LEAP Checkpoints** | [EvanSirius/leap-ckpts](https://huggingface.co/EvanSirius/leap-ckpts) | 全参微调权重 (`checkpoint-200` 至 `checkpoint-7000`) |
| **LEAP Dataset** | [EvanSirius/leap-agibot-processed](https://huggingface.co/datasets/EvanSirius/leap-agibot-processed) | 训练集、测试集与盲评子集 |
| **VLABench** | [OpenMOSS/VLABench](https://github.com/OpenMOSS/VLABench) | 官方长程操作评测基准（Git Submodule） |
| **Base Model** | [Qwen/Qwen3-VL-2B-Instruct](https://huggingface.co/Qwen/Qwen3-VL-2B-Instruct) | Qwen3-VL 2B 指令微调基座 |

## 6. 引用与致谢

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

感谢 Qwen 团队开源 Qwen3-VL 系列模型，以及 VLABench、LOVE-Agibot 项目提供的数据集与评测基础设施。

## 许可证

本项目采用 [MIT License](LICENSE) 开源。
