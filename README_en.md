<a name="project-name-en"></a>

# [PROJECT_NAME]: Multimodal Task Planning with Qwen3-VL on VLABench

## 1. Overview

`[PROJECT_NAME]` focuses on multimodal robotic task planning built on **Qwen3-VL-2B-Instruct**, offering a complete workflow that covers full-parameter fine-tuning, the six-dimensional VLABench evaluation suite, and an LLM-as-a-Judge blind review process. The repository provides reproducible scripts so researchers can quickly validate and extend the following capabilities:

- Unified evaluation for Memory & Tasks, CommonSense, Semantic, Spatial, PhysicsLaw, and Complex dimensions.
- Standardized organization for training, evaluation, blind review, and visualization scripts to reduce cross-scenario reproduction cost.
- Utilities for downloading, cleaning, and extracting key frames from datasets such as LOVE-Agibot-Beta.

## 2. Repository Structure

```text
[PROJECT_NAME]/
├── configs/        # WorkspaceConfig with global paths plus train/eval constants
├── data/           # JSONL conversations and LOVE-Agibot images
├── dataset/        # vlm_evaluation_v1.0 split (CommonSense/Complex/... folders)
├── logs/           # Dimension-wise evaluation logs plus PID trackers
├── models/         # Local Hugging Face cache (e.g., Qwen3-VL-2B-Instruct)
├── output/         # Training checkpoints and final weights
├── scripts/
│   ├── download/   # Model & VLABench download helpers
│   ├── training/   # run-finetuning.py and run-training.sh entrypoints
│   ├── evaluation/ # VLABench, LLM-Judge, visualization scripts
│   └── utils/      # Data cleaning + LOVE-Agibot processing
├── VLABench/       # Official benchmark submodule (vendored; update via submodule when needed)
├── eva_results/    # Latest evaluation metrics organized by dimension/model
├── backup_eva_results/ # Archived evaluation snapshots for reproducibility
├── run_vlabench_evaluation_bingXing.sh # Example script for parallel evaluation batches
├── qwen-ft-env.yml # Conda environment specification
└── README*.md
```

## 3. Quick Start

### 3.1 Environment

```bash
conda env create -f qwen-ft-env.yml
conda activate qwen-ft-env
```

### 3.2 Download Models & Data

```bash
# 1) Pull Qwen3-VL-2B-Instruct (script ships with hf-mirror acceleration)
bash scripts/download/download_model.sh

# 2) Fetch VLABench evaluation set with smart retry logic
python scripts/download/download_vlabench_with_retry.py

# 3) (Optional) Extract the first frame of LOVE-Agibot-Beta videos
python scripts/utils/prepare_love_agibot.py --num-workers 4
```

### 3.3 Full-Parameter Fine-Tuning

```bash
bash scripts/training/run-training.sh
```

The script automatically:

1. Resolves the repository root relative to the `scripts/` directory.
2. Loads `data/train_151230.jsonl` together with `models/Qwen3-VL-2B-Instruct/`.
3. Launches BF16 training with SDPA attention, gradient accumulation (6×6), and stores checkpoints under `output/`.

### 3.4 VLABench Evaluation & Blind Review

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

| Model | M&T ↑ | CommonSense ↑ | Complex ↑ | Avg Final Score ↑ |
| --- | --- | --- | --- | --- |
| Qwen3-VL-2B-Baseline | 29.60 | 25.70 | 14.28 | 23.19 |
| [PROJECT_NAME] Finetuned (checkpoint-5000) | 32.96 | 27.28 | 19.67 | 26.63 |

> Scores are averaged from `eva_results/<dimension>/<model>/final_score.json` (`total_score`). Visualization scripts live in `scripts/evaluation/draw_*.py`.

## 5. Datasets & Models

- **VLABench** – official repository: <https://github.com/VLABench/VLABench>; shipped as a git submodule for evaluation.
- **LOVE-Agibot-Beta** – Hugging Face dataset `EvanSirius/LOVE-Agibot-Beta`; the provided utility defaults to first-frame extraction for size control.
- **Base model** – `Qwen/Qwen3-VL-2B-Instruct` from Hugging Face Hub.
- **Project checkpoints** – intermediates under `output/checkpoint-*`; publish them to your preferred Model Zoo when appropriate.

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

- Thanks to the Qwen team for open-sourcing Qwen3-VL, enabling this project to build on accessible weights.
- Appreciation to VLABench, LOVE-Agibot, and related projects for providing datasets and evaluation infrastructure.

## 7. Tips & Troubleshooting

- **Submodule**: Run `git submodule update --init --recursive` after cloning to populate `VLABench/`.
- **Monitoring**: `tmp_monitor_evaluation.py` and `tmp_analyze_data.py` offer lightweight hooks for tracking evaluation progress when running multiple rounds.
- **Storage hygiene**: Move finished evaluation JSON/plots into `backup_eva_results/` to keep `eva_results/` focused on current experiments.
- **Reproducibility**: Pin your CUDA/cuDNN version inside `qwen-ft-env.yml` before sharing checkpoints to avoid runtime drift.
