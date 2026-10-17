# LEAP: Action Blueprint Generation for Long-Horizon Embodied Planning

> Official PyTorch implementation of **LEAP** (Logical Embodied Action Planning for Long-Horizon Robotic Tasks via Vision-Language Alignment).

<p align="center">
  <img src="https://img.shields.io/badge/Task-Embodied_AI-blue.svg?style=flat-square" alt="Task" />
  <img src="https://img.shields.io/badge/Benchmark-VLABench-green.svg?style=flat-square" alt="Benchmark" />
  <img src="https://img.shields.io/badge/Framework-PyTorch_2.x-EE4C2C.svg?style=flat-square&logo=pytorch" alt="PyTorch" />
  <img src="https://img.shields.io/badge/Backbone-Qwen3--VL-orange.svg?style=flat-square" alt="Qwen-VL" />
  <img src="https://img.shields.io/badge/License-Apache_2.0-blue.svg?style=flat-square" alt="License" />
</p>

---

## 📢 News

- **[2026/04]** Released the core LEAP training pipeline, VLABench six-dimensional evaluation suite, and Love-Agibot dataset preprocessing scripts.

---

## 💡 Overview

Long-horizon robotic manipulation tasks pose significant challenges for autonomous agents due to compounding errors, lack of global task reasoning, and severe inference latency when calling large vision-language models at every control step. 

**LEAP** reformulates complex embodied manipulation into an **Action Blueprint Generation** paradigm. By aligning high-level task goals with multi-modal visual observations at initial frames, LEAP predicts structured, physically executable milestone sequences that guide low-level control policies while drastically reducing reactive re-planning overhead.

### Key Highlights
- **Action Blueprint Paradigm**: Decouples strategic long-horizon temporal decomposition from fast closed-loop sensorimotor execution.
- **Compact & Edge-Deployable**: Fine-tuned on compact vision-language backbones (`Qwen3-VL-2B-Instruct`) via Full-SFT and LoRA, achieving real-time inference on consumer-grade hardware.
- **Standardized Multi-Dimensional Evaluation**: Benchmarked across 6 core capabilities in VLABench: *Memory & Tasks*, *CommonSense*, *Semantic Understanding*, *Spatial Reasoning*, *Physical Laws*, and *Complex Compositions*.

---

## 📊 Benchmark Results on VLABench

| Model / Supervision Target | Overall Avg | CommonSense | Spatial | PhysicsLaw | Complex | Parse Success |
|---|:---:|:---:|:---:|:---:|:---:|:---:|
| Qwen3-VL-2B (Base Zero-shot) | 25.10 | 24.80 | 25.30 | 23.90 | 22.40 | 93.16% |
| High-level Paraphrase SFT | 24.26 | 23.50 | 24.10 | 24.80 | 22.10 | 94.20% |
| **LEAP (Full SFT)** | **28.38** | **29.10** | **28.70** | **27.60** | **26.90** | **96.00%** |
| **LEAP (LoRA)** | **28.22** | **28.80** | **28.40** | **27.50** | **26.50** | **95.60%** |

---

## 🛠️ Repository Structure

```text
LEAP/
├── configs/          # Workspace configurations, dataset paths, hyperparameter presets
├── data/             # Dataset JSONL metadata and Love-Agibot sample frame manifests
├── dataset/          # Multi-dimensional benchmark splits (CommonSense, Complex, Spatial, etc.)
├── evaluation/       # VLABench evaluation harness and LLM-as-a-Judge scoring pipelines
├── scripts/          # Shell helpers for downloading, preprocessing, training, and inference
├── training/         # SFT and LoRA fine-tuning code with response-only loss masking
└── visualizations/   # Trajectory rollout plots and attention alignment visualization tools
```

---

## 🚀 Quick Start

### 1. Environment Setup

```bash
conda create -n leap python=3.10 -y
conda activate leap

pip install torch torchvision --index-url https://download.pytorch.org/whl/cu121
pip install -r requirements.txt
```

### 2. Dataset Preparation

Download the benchmark dataset and extract initial observation keyframes:

```bash
bash scripts/download_data.sh
python scripts/extract_keyframes.py --dataset_dir ./data/love_agibot
```

### 3. Training & Fine-Tuning

Run Full SFT or LoRA fine-tuning on `Qwen3-VL-2B-Instruct`:

```bash
# LoRA Fine-tuning
bash scripts/train_lora.sh --config configs/train_lora.yaml

# Full Parameter Fine-tuning
bash scripts/train_full.sh --config configs/train_full.yaml
```

### 4. Evaluation on VLABench

Evaluate the trained policy against the 6 diagnostic benchmark dimensions:

```bash
python -m evaluation.run_eval \
    --model_path ./checkpoints/leap-lora \
    --benchmark_dir ./dataset/vlm_evaluation_v1.0 \
    --output_dir ./results/eval_run
```

---

## 📖 Citation

If you find LEAP useful for your research in embodied AI and long-horizon planning, please consider citing our work:

```bibtex
@article{gao2026leap,
  title={LEAP: Action Blueprint Generation for Long-Horizon Embodied Planning via Vision-Language Alignment},
  author={Gao, Shenzhou and collaborators},
  journal={arXiv preprint},
  year={2026}
}
```

---

## 📄 License

This codebase is released under the [Apache 2.0 License](./LICENSE).
