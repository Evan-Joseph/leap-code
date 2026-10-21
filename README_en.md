<a name="leap-en"></a>

# LEAP: Logical Embodied Action Planning for Long-Horizon Robotic Tasks via Generative Vision-Language Alignment

<p align="center">
  <b>English</b> · <a href="README.md">简体中文</a>
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

## 1. Overview

**LEAP** targets multimodal robotic task planning built on **Qwen3-VL-2B-Instruct**, providing an end-to-end workflow covering Action Blueprint fine-tuning, the six-dimensional VLABench evaluation suite, and an LLM-as-a-Judge blind review pipeline:

- **Six-Dimensional Evaluation**: Unified evaluation across Memory & Tasks (`M&T`), `CommonSense`, `Semantic`, `Spatial`, `PhysicalLaw`, and `Complex` manipulation splits.
- **Full Fine-Tuning & LoRA Ablations**: Reproducible BF16 + SDPA full-parameter training scripts alongside four LoRA rank presets (`r=8/16/32/64`).
- **Data Processing & Blind Evaluation**: Utilities for downloading, cleaning, and extracting keyframes from `LOVE-Agibot-Beta` and running blind pairwise comparisons.

<p align="center">
  <img src="assets/visual_execution.png" width="88%" alt="LEAP Visual Execution Comparison on VLABench" />
</p>

## 2. Repository Structure

```text
leap-code/
├── assets/               # Framework diagram and VLABench execution traces
├── configs/              # WorkspaceConfig with global paths and train/eval constants
├── scripts/
│   ├── download/         # Base model & VLABench download helpers
│   ├── training/         # Full fine-tuning entrypoints (run-finetuning.py, run-training.sh)
│   ├── ablation/         # LoRA ablation experiment scripts
│   ├── evaluation/       # VLABench evaluation, LLM-Judge, and plotting scripts
│   ├── shell/            # Batch evaluation shell launchers
│   └── utils/            # Data cleaning and dataset verification utilities
├── VLABench/             # Official benchmark submodule (git submodule)
├── eval.sh               # Unified evaluation entrypoint
├── qwen-ft-env.yml       # Conda environment specification
├── README.md             # Chinese documentation
└── README_en.md          # English documentation
```

## 3. Quick Start

### 3.1 Clone the Repository

```bash
git clone --recursive https://github.com/Evan-Joseph/leap-code.git
cd leap-code

# If cloned without --recursive:
git submodule update --init --recursive
```

### 3.2 Environment Setup

```bash
conda env create -f qwen-ft-env.yml
conda activate qwen-ft-env
```

### 3.3 Download Models & Data

```bash
# 1) Pull Qwen3-VL-2B-Instruct
bash scripts/download/download_model.sh

# 2) Fetch VLABench evaluation set with retry support
python scripts/download/download_vlabench_with_retry.py

# 3) Verify VLABench dataset integrity
python scripts/utils/verify_dataset.py

# 4) (Optional) Download LEAP pre-processed dataset
huggingface-cli download EvanSirius/leap-agibot-processed --local-dir data/
```

### 3.4 Full-Parameter Fine-Tuning

```bash
bash scripts/training/run-training.sh
```

The script loads `data/train_151230.jsonl` and `models/Qwen3-VL-2B-Instruct/`, runs BF16 training with SDPA attention and `6×6` gradient accumulation, and writes checkpoints under `output/`.

### 3.5 LoRA Parameter-Efficient Fine-Tuning (Ablation)

```bash
# Default standard preset (r=16)
bash scripts/ablation/run-lora-training.sh

# Specific preset (light: r=8 | standard: r=16 | full: r=32 | aggressive: r=64)
bash scripts/ablation/run-lora-training.sh --preset full

# Run batch ablation experiments
bash scripts/ablation/run-ablation-experiments.sh
```

### 3.6 VLABench Evaluation & Blind Review

```bash
# Evaluate all dimensions (M&T / CommonSense / Semantic / Spatial / PhysicalLaw / Complex)
python scripts/evaluation/run_vlm_evaluation.py \
    --checkpoint output/checkpoint-5000 \
    --dimension all

# Single-dimension evaluation example
python scripts/evaluation/run_vlm_evaluation.py \
    --checkpoint output/checkpoint-5000 \
    --dimension "M&T"

# Blind-10 LLM-as-a-Judge pipeline
python scripts/evaluation/run_vlm_output.py --baseline_model models/Qwen3-VL-2B-Instruct
python scripts/evaluation/analyze_output_results.py
```

## 4. Results (VLABench)

| Model | M&T ↑ | CommonSense ↑ | Semantic ↑ | Spatial ↑ | PhysicalLaw ↑ | Complex ↑ | Avg ↑ |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| Qwen3-VL-2B-Baseline | 29.60 | 25.70 | 25.89 | 31.10 | 24.00 | 14.28 | 25.10 |
| **LEAP (checkpoint-5000)** | **32.96** | **27.27** | **28.49** | **31.62** | **30.27** | **19.67** | **28.38** |
| Relative Improvement | +11.3% | +6.1% | +10.0% | +1.7% | +26.1% | +37.7% | **+13.1%** |

> Scores are averaged from `eva_results/<dimension>/<model>/final_score.json` (`total_score`). Plotting scripts are available under `scripts/evaluation/draw_*.py`.

## 5. Datasets & Models

| Resource | Link | Description |
| :--- | :--- | :--- |
| **LEAP Checkpoints** | [EvanSirius/leap-ckpts](https://huggingface.co/EvanSirius/leap-ckpts) | Full fine-tuning weights (`checkpoint-200` to `checkpoint-7000`) |
| **LEAP Dataset** | [EvanSirius/leap-agibot-processed](https://huggingface.co/datasets/EvanSirius/leap-agibot-processed) | Training set, test set, and blind evaluation split |
| **VLABench** | [OpenMOSS/VLABench](https://github.com/OpenMOSS/VLABench) | Official evaluation framework (Git Submodule) |
| **Base Model** | [Qwen/Qwen3-VL-2B-Instruct](https://huggingface.co/Qwen/Qwen3-VL-2B-Instruct) | Qwen3-VL 2B instruction-tuned base model |

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

We thank the Qwen team for open-sourcing Qwen3-VL, as well as the VLABench and LOVE-Agibot teams for providing datasets and evaluation infrastructure.

## License

This project is licensed under the [MIT License](LICENSE).
