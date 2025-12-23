<a name="leap-en"></a>

# LEAP: Logical Embodied Action Planning for Long-Horizon Robotic Tasks via Generative Vision-Language Alignment

<p align="center">
  <a href="https://huggingface.co/EvanSirius/leap-ckpts"><img src="https://img.shields.io/badge/🤗%20Model-leap--ckpts-blue" alt="Model"></a>
  <a href="https://huggingface.co/datasets/EvanSirius/leap-agibot-processed"><img src="https://img.shields.io/badge/🤗%20Dataset-leap--agibot--processed-green" alt="Dataset"></a>
  <a href="https://github.com/OpenMOSS/VLABench"><img src="https://img.shields.io/badge/Benchmark-VLABench-orange" alt="VLABench"></a>
</p>

## 1. Overview

**LEAP** focuses on multimodal robotic task planning built on **Qwen3-VL-2B-Instruct**, offering a complete workflow that covers full-parameter fine-tuning, the six-dimensional VLABench evaluation suite, and an LLM-as-a-Judge blind review process. The repository provides reproducible scripts so researchers can quickly validate and extend the following capabilities:

- Unified evaluation for Memory & Tasks, CommonSense, Semantic, Spatial, PhysicsLaw, and Complex dimensions.
- Standardized organization for training, evaluation, blind review, and visualization scripts to reduce cross-scenario reproduction cost.
- Utilities for downloading, cleaning, and extracting key frames from datasets such as LOVE-Agibot-Beta.

## 2. Repository Structure

```text
LEAP/
├── configs/        # WorkspaceConfig with global paths plus train/eval constants
├── data/           # JSONL conversations and LOVE-Agibot images
├── dataset/        # vlm_evaluation_v1.0 split (CommonSense/Complex/... folders)
├── logs/           # Dimension-wise evaluation logs plus PID trackers
├── models/         # Local Hugging Face cache (e.g., Qwen3-VL-2B-Instruct)
├── output/         # Training checkpoints and final weights
├── scripts/
│   ├── download/   # Model & VLABench download helpers
│   ├── training/   # run-finetuning.py and run-training.sh entrypoints
│   ├── ablation/   # LoRA ablation experiment scripts (parameter-efficient fine-tuning)
│   ├── evaluation/ # VLABench, LLM-Judge, visualization scripts
│   └── utils/      # Data cleaning + LOVE-Agibot processing
├── VLABench/       # Official benchmark submodule (run git submodule update --init)
├── eva_results/    # Latest evaluation metrics organized by dimension/model
├── qwen-ft-env.yml # Conda environment specification
└── README*.md
```

## 3. Quick Start

### 3.1 Clone the Repository

```bash
# Clone with VLABench submodule
git clone --recursive https://github.com/Evan-Joseph/leap-code.git
cd leap-code

# If you forgot --recursive, run afterwards:
git submodule update --init --recursive
```

### 3.2 Environment Setup

```bash
conda env create -f qwen-ft-env.yml
conda activate qwen-ft-env
```

### 3.3 Download Models & Data

```bash
# 1) Pull Qwen3-VL-2B-Instruct (script ships with hf-mirror acceleration)
bash scripts/download/download_model.sh

# 2) Fetch VLABench evaluation set with smart retry logic
python scripts/download/download_vlabench_with_retry.py

# 3) (Optional) Download LEAP pre-processed dataset
huggingface-cli download EvanSirius/leap-agibot-processed --local-dir data/
```

### 3.4 Full-Parameter Fine-Tuning

```bash
bash scripts/training/run-training.sh
```

The script automatically:

1. Resolves the repository root relative to the `scripts/` directory.
2. Loads `data/train_151230.jsonl` together with `models/Qwen3-VL-2B-Instruct/`.
3. Launches BF16 training with SDPA attention, gradient accumulation (6×6), and stores checkpoints under `output/`.

### 3.5 LoRA Parameter-Efficient Fine-Tuning (Ablation)

```bash
# Use default standard preset
bash scripts/ablation/run-lora-training.sh

# Use specific preset (light/standard/full/aggressive)
bash scripts/ablation/run-lora-training.sh --preset full

# Run batch ablation experiments
bash scripts/ablation/run-ablation-experiments.sh
```

LoRA Preset Configurations:
- `light`: r=8, minimal parameters, good for quick validation
- `standard`: r=16, balanced between performance and efficiency
- `full`: r=32, covers more layers, close to full fine-tuning
- `aggressive`: r=64, high-rank LoRA for maximum expressiveness

### 3.6 VLABench Evaluation & Blind Review

```bash
# Evaluate all dimensions (M&T/CommonSense/.../Complex)
python scripts/evaluation/run_vlm_evaluation.py \
    --checkpoint output/checkpoint-200 \
    --dimension all

# Run single-dimension evaluation (e.g., M&T)
python scripts/evaluation/run_vlm_evaluation.py \
    --checkpoint output/checkpoint-200 \
    --dimension "M&T"

# Blind-10 LLM-as-a-Judge pipeline
python scripts/evaluation/run_vlm_output.py --baseline_model models/Qwen3-VL-2B-Instruct
python scripts/evaluation/analyze_output_results.py
```

## 4. Results & Visualization

| Model | M&T ↑ | CommonSense ↑ | Semantic ↑ | Spatial ↑ | PhysicalLaw ↑ | Complex ↑ | Avg ↑ |
| --- | --- | --- | --- | --- | --- | --- | --- |
| Qwen3-VL-2B-Baseline | 29.60 | 25.70 | 25.89 | 31.10 | 24.00 | 14.28 | 25.10 |
| **LEAP (checkpoint-5000)** | **32.96** | **27.27** | **28.49** | **31.62** | **30.27** | **19.67** | **28.38** |
| Improvement | +11.3% | +6.1% | +10.0% | +1.7% | +26.1% | +37.7% | **+13.1%** |

> Scores are averaged from `eva_results/<dimension>/<model>/final_score.json` (`total_score`). Visualization scripts live in `scripts/evaluation/draw_*.py`.

## 5. Datasets & Models

| Resource | Link | Description |
| --- | --- | --- |
| **LEAP Checkpoints** | [🤗 EvanSirius/leap-ckpts](https://huggingface.co/EvanSirius/leap-ckpts) | Full fine-tuning weights (checkpoint-200 ~ checkpoint-7000) |
| **LEAP Dataset** | [🤗 EvanSirius/leap-agibot-processed](https://huggingface.co/datasets/EvanSirius/leap-agibot-processed) | Training set, test set, blind evaluation set |
| **VLABench** | [GitHub OpenMOSS/VLABench](https://github.com/OpenMOSS/VLABench) | Official evaluation framework (Git Submodule) |
| **Base Model** | [🤗 Qwen/Qwen3-VL-2B-Instruct](https://huggingface.co/Qwen/Qwen3-VL-2B-Instruct) | Qwen3-VL 2B instruction-tuned version |

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

- Thanks to the Qwen team for open-sourcing Qwen3-VL, enabling this project to build on accessible weights.
- Appreciation to VLABench, LOVE-Agibot, and related projects for providing datasets and evaluation infrastructure.

## 7. Tips & Troubleshooting

- **Submodule**: Run `git submodule update --init --recursive` after cloning to populate `VLABench/`.
- **Monitoring**: `tmp_monitor_evaluation.py` and `tmp_analyze_data.py` offer lightweight hooks for tracking evaluation progress when running multiple rounds.
- **Storage hygiene**: Move finished evaluation JSON/plots into `backup_eva_results/` to keep `eva_results/` focused on current experiments.
- **Reproducibility**: Pin your CUDA/cuDNN version inside `qwen-ft-env.yml` before sharing checkpoints to avoid runtime drift.

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.
