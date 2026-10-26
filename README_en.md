<a name="leap-en"></a>

# Structured Supervision Targets as a Knowledge Interface for Reliable Parser-Gated Plan Generation in Compact Vision-Language Models

<p align="center">
  <b>English</b> · <a href="README.md">简体中文</a>
</p>

<p align="center">
  <a href="https://huggingface.co/EvanSirius/leap-ckpts"><img src="https://img.shields.io/badge/HuggingFace-leap--ckpts-blue" alt="Model"></a>
  <a href="https://huggingface.co/datasets/EvanSirius/leap-agibot-processed"><img src="https://img.shields.io/badge/Dataset-leap--agibot--processed-green" alt="Dataset"></a>
  <a href="https://huggingface.co/datasets/EvanSirius/LOVE-Agibot-Beta"><img src="https://img.shields.io/badge/Source%20Corpus-LOVE--Agibot--Beta-teal" alt="Source Corpus"></a>
  <a href="https://github.com/OpenMOSS/VLABench"><img src="https://img.shields.io/badge/Benchmark-VLABench-orange" alt="VLABench"></a>
  <a href="LICENSE"><img src="https://img.shields.io/badge/License-MIT-gray" alt="License"></a>
</p>

<p align="center">
  <img src="assets/leap_framework.png" width="96%" alt="LEAP Knowledge Interface & Two-Stage Contract Framework" />
</p>

## 1. Overview

This repository provides the official PyTorch implementation for **"Structured Supervision Targets as a Knowledge Interface for Reliable Parser-Gated Plan Generation in Compact Vision-Language Models"** (project codename **LEAP**). Built around the compact **Qwen3-VL-2B-Instruct** backbone, this study formulates the supervision-target representation as an explicit **Knowledge Interface ($\mathcal{I}_{\mathcal{K}}$)** bridging continuous multimodal perception and discrete symbolic verifiers:

- **Controlled Isolation of the Supervision Target (151,230 Training Samples / 184 Manipulation Tasks)**: Under a fixed visual interface (single initial head-camera frame $I_1$, `head_color`), response-only loss masking, deterministic greedy decoding, and strict parser-gated `VLABench` evaluation ($\psi_{\mathrm{raw}}$), an ordered numbered-step target ($\mathcal{T}_{\mathrm{steps}}$) elevates the six-dimension macro average from **25.10** (Base) to **28.38** under Full SFT ($+3.29$, $p < 0.001$ over 10,000 paired bootstrap resamples) and **28.22** under rank-16 LoRA ($+3.13$, $p < 0.001$). By contrast, a strictly matched **Same-Domain Task-Restatement Control ($\mathcal{T}_{\mathrm{ctrl}}$)** trained on the exact same image-instruction pairs and schedule reaches only **25.15** ($+0.05$), demonstrating that planning gains originate from explicit action-decomposition structure rather than domain exposure.
- **Two-Stage Symbolic Contract Audit ($\psi_{\mathrm{raw}} \to \psi_{\mathrm{relax}} \to \psi_{\mathrm{norm}}$)**:
  - **Stage 1 (Syntax Gate Sensitivity)**: Relaxing Markdown code-fence extraction ($\psi_{\mathrm{relax}}$) across all 4,500 evaluation instances yields `0 / 4,500` salvageable failures, confirming that unparsed outputs reflect genuine structural truncation or schema drift; Full SFT maintains **96.00%** parse survival compared with **87.75%** for LoRA.
  - **Stage 2 (Entity Type-Normalization)**: Resolving the Python type mismatch between quoted string entity identifiers (`"4"`) induced by the prompt schema and unquoted integer identifiers (`4`) stored in `VLABench` ground truth lifts Full SFT (5k) from `28.38` to **38.90** (`+10.52`, with **70.68%** skill-sequence match), outperforming both Base (`35.66`) and LoRA (`33.58`) across all six capability dimensions.
- **Reproducible End-to-End Pipeline**: Includes `LOVE-Agibot-Beta` keyframe extraction and target curation, BF16 + SDPA full-parameter SFT, LoRA rank ablations ($r \in \{8, 16, 32, 64\}$), six-dimension `VLABench` evaluation, and deterministic Stage 1 / Stage 2 contract audit scripts.

<p align="center">
  <img src="assets/visual_execution.png" width="88%" alt="LEAP Visual Execution Comparison on VLABench" />
</p>

## 2. Repository Structure

```text
leap-code/
├── assets/               # Framework overview and VLABench execution comparison diagrams
├── configs/              # WorkspaceConfig with global paths and train/eval constants
├── scripts/
│   ├── download/         # Base model and VLABench download helpers
│   ├── training/         # Full fine-tuning entrypoints (run-finetuning.py, run-training.sh)
│   ├── ablation/         # LoRA ablation experiment scripts (light / standard / full / aggressive)
│   ├── evaluation/       # VLABench evaluation, LLM-Judge, and plotting scripts
│   ├── audit/            # Two-stage symbolic contract audit scripts (Stage 1 & Stage 2)
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
# 1) Pull Qwen3-VL-2B-Instruct base weights
bash scripts/download/download_model.sh

# 2) Fetch VLABench evaluation set with retry support
python scripts/download/download_vlabench_with_retry.py

# 3) Verify VLABench dataset integrity
python scripts/utils/verify_dataset.py

# 4) Download pre-processed supervision splits (151,230 train / 16,812 test)
hf download EvanSirius/leap-agibot-processed --repo-type dataset --local-dir data/
```

### 3.4 Full-Parameter Fine-Tuning (Full SFT)

```bash
bash scripts/training/run-training.sh
```

The script loads `data/train_151230.jsonl` and `models/Qwen3-VL-2B-Instruct/`, runs BF16 training with SDPA attention, cosine learning rate schedule (peak LR $1.5\times 10^{-5}$, 800-step linear warmup), and effective batch size `36` (`6×6` gradient accumulation), writing checkpoints under `output/`.

### 3.5 LoRA Parameter-Efficient Fine-Tuning (Ablation)

```bash
# Default standard preset (r=16, alpha=32)
bash scripts/ablation/run-lora-training.sh

# Specific preset (light: r=8 | standard: r=16 | full: r=32 | aggressive: r=64)
bash scripts/ablation/run-lora-training.sh --preset full

# Run batch ablation experiments
bash scripts/ablation/run-ablation-experiments.sh
```

### 3.6 VLABench Evaluation & Two-Stage Symbolic Contract Audit

```bash
# 1) Evaluate all six dimensions (M&T / CommonSense / Semantic / Spatial / PhysicalLaw / Complex)
python scripts/evaluation/run_vlm_evaluation.py \
    --checkpoint output/checkpoint-5000 \
    --dimension all

# 2) Stage 1 Syntax Gate Sensitivity Audit (\psi_raw vs. \psi_relax)
python scripts/audit/audit_parser_sensitivity.py

# 3) Stage 2 Entity Type-Normalization Contract Audit (\psi_raw vs. \psi_norm)
python scripts/audit/audit_stage2_entity_contract.py
```

## 4. Experimental Results

### 4.1 Strict Parser-Gated Evaluation ($\psi_{\mathrm{raw}}$, 10,000 Paired Bootstrap 95% CIs)

| Supervision Target & Adaptation | M&T ↑ | Semantic ↑ | Spatial ↑ | PhysicalLaw ↑ | Complex ↑ | CommonSense ↑ | Macro Avg ($\psi_{\mathrm{raw}}$) ↑ | 95% Bootstrap CI |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| `Qwen3-VL-2B` (Base) | 29.60 | 25.89 | 31.10 | 24.00 | 14.28 | 25.70 | 25.10 | `[24.71, 25.48]` |
| `Same-Domain Task-Restatement Control` ($\mathcal{T}_{\mathrm{ctrl}}$) | 28.89 | 24.89 | 30.00 | 26.67 | 11.13 | 24.00 | 25.15 | `[24.75, 25.55]` |
| `Structured Steps + LoRA` ($r=16$, 5k) | **33.18** | 26.73 | 30.70 | 29.60 | 19.02 | **30.10** | 28.22 | `[27.83, 28.61]` |
| **`Structured Steps + Full SFT` (5k)** | 32.96 | **28.49** | **31.62** | **30.27** | **19.67** | 27.27 | **28.38** | **`[28.01, 28.75]`** |

### 4.2 Two-Stage Symbolic Contract Audit (Stage 1 Syntax Gate & Stage 2 Entity Type-Normalization $\psi_{\mathrm{norm}}$)

| Model Variant (N = 4,500) | Parse Survival ↑ | $\psi_{\mathrm{raw}}$ ↑ | $\psi_{\mathrm{norm}}$ ↑ | $\Delta(\psi_{\mathrm{norm}}-\psi_{\mathrm{raw}})$ | M&T | Sem. | Spa. | Phy. | Cpx. | CS | Skill Match ↑ | Entity Match ↑ | Joint Step ↑ | Exact Plan ↑ |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| `Qwen3-VL-2B` (Base) | 99.18% | 25.10 | 35.66 | +10.56 | 45.33 | 40.13 | 39.92 | 24.00 | 24.62 | 39.92 | 62.52% | 26.62% | 18.13% | 4.33% |
| `Structured Steps + LoRA` (5k) | 87.75% | 28.22 | 33.58 | +5.36 | 42.18 | 32.60 | 36.85 | 29.73 | 23.47 | 36.65 | 60.12% | 23.78% | 18.31% | 2.41% |
| `Structured Steps + LoRA` (7k) | 88.80% | 29.16 | 34.80 | +5.64 | 42.16 | 33.89 | 36.77 | **31.40** | 26.27 | 38.30 | 62.24% | 24.75% | 18.91% | 2.49% |
| `Structured Steps + Full SFT` (7k) | **96.42%** | 28.37 | 38.84 | +10.47 | 46.91 | 40.84 | 40.90 | 30.80 | **32.89** | 40.67 | 70.58% | 26.49% | **22.35%** | **6.71%** |
| **`Structured Steps + Full SFT` (5k)** | 96.00% | **28.38** | **38.90** | **+10.52** | **47.09** | **41.51** | **40.95** | 30.33 | 32.49 | **41.02** | **70.68%** | **26.54%** | 22.05% | 6.60% |

## 5. Datasets & Models

| Resource | Link | Description |
| :--- | :--- | :--- |
| **LEAP Checkpoints** | [EvanSirius/leap-ckpts](https://huggingface.co/EvanSirius/leap-ckpts) | Full fine-tuning weights (`checkpoint-200` to `checkpoint-7000`) |
| **LEAP Processed Dataset** | [EvanSirius/leap-agibot-processed](https://huggingface.co/datasets/EvanSirius/leap-agibot-processed) | Structured-step training split (`train_151230.jsonl`), test split (`test_16812.jsonl`), and image archives |
| **LOVE-Agibot-Beta** | [EvanSirius/LOVE-Agibot-Beta](https://huggingface.co/datasets/EvanSirius/LOVE-Agibot-Beta) | Task-sharded keyframe archives extracted from AgiBot-Beta (434 task shards) |
| **VLABench** | [OpenMOSS/VLABench](https://github.com/OpenMOSS/VLABench) | Official six-dimension evaluation framework (Git Submodule) |
| **Base Model** | [Qwen/Qwen3-VL-2B-Instruct](https://huggingface.co/Qwen/Qwen3-VL-2B-Instruct) | Qwen3-VL 2B instruction-tuned base model |

## 6. Citation & Acknowledgement

```bibtex
@article{gao2026structured,
  title={Structured Supervision Targets as a Knowledge Interface for Reliable Parser-Gated Plan Generation in Compact Vision-Language Models},
  author={Shenzhou Gao and Peijian Zhang},
  year={2026}
}

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

We thank the Qwen team for open-sourcing Qwen3-VL, as well as the VLABench and AgiBot World teams for providing datasets and evaluation infrastructure.

## License

This project is licensed under the [MIT License](LICENSE).
