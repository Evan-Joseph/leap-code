# 🤖 LEAP: Action Blueprint Generation for Long-Horizon Embodied Planning via Vision-Language Alignment

<p align="center">
  <b>Official PyTorch Implementation & Evaluation Suite</b>
</p>

<p align="center">
  <img src="https://img.shields.io/badge/Python-3.10%2B-blue?logo=python" alt="Python 3.10+">
  <img src="https://img.shields.io/badge/PyTorch-2.2%2B-orange?logo=pytorch" alt="PyTorch 2.2+">
  <img src="https://img.shields.io/badge/Benchmark-VLABench-green" alt="VLABench">
  <img src="https://img.shields.io/badge/Embodied%20AI-Planning%20%26%20Robotics-red" alt="Embodied AI">
  <img src="https://img.shields.io/badge/License-Apache--2.0-yellow.svg" alt="License: Apache-2.0">
</p>

---

## 🏛️ 框架全景与算法架构 (Framework Architecture)

<p align="center">
  <img src="assets/leap_framework.png" width="100%" alt="LEAP System Pipeline & Alignment Dynamics" style="border-radius: 8px; box-shadow: 0 4px 20px rgba(0,0,0,0.12);" />
</p>

<p align="center">
  <i>LEAP 架构全景：通过分层视觉-语言表征对齐，将高维长程任务自适应解构为具备因果依赖的行动蓝图（Action Blueprints）</i>
</p>

---

## 🎬 可视化规划与实体轨迹推演 (Visual Execution & Trajectory)

<p align="center">
  <img src="assets/visual_execution.png" width="100%" alt="Visual Execution across Long-Horizon Embodied Tasks" style="border-radius: 8px; box-shadow: 0 4px 20px rgba(0,0,0,0.12);" />
</p>

<p align="center">
  <i>在多阶段长程复合机械臂操作与移动导航任务中的时空注意力对齐与动作执行跟踪</i>
</p>

---

## 📖 论文概述 (Overview)

长程具身智能规划（Long-Horizon Embodied Planning）要求智能体在多变的非结构化环境中，根据高层自然语言指令持续进行多阶段推理与物理交互。现有大视觉语言模型（VLM）直接端到端生成低层控制信号时，极易因时间累积误差与幻觉导致规划崩溃。

**LEAP** 提出了基于层次化视觉-语言对齐的**行动蓝图生成机制**：
1. **语义蓝图解耦**：将长时序指令映射为包含前提条件（Preconditions）与效应（Effects）的形式化因果拓扑图；
2. **多模态时空对齐**：通过对比学习与跨注意力机制，强化目标物体几何状态与当前执行阶段的特征绑定；
3. **闭环重规划防护**：在执行异常或环境阻断时触发局部拓扑重构，避免全量重新推理的高昂开销。

---

## 📊 基准评测结果 (Benchmark Results on VLABench)

在具身智能标准基准 **VLABench** 上的综合测试表明，LEAP 在多项极具挑战性的长程规划子任务上显著超越主流基线：

| 方法 (Method) | 骨干模型 (Backbone) | 单任务成功率 (SR-Single) | 复杂长程成功率 (SR-Long) | 规划可执行度 (Exec.) |
|---|---|:---:|:---:|:---:|
| GPT-4V (Direct Prompting) | Proprietary | 58.2% | 34.6% | 61.4% |
| PaLM-E Baseline | PaLM-E 12B | 62.4% | 41.8% | 67.2% |
| Code-as-Policies | Text-Davinci-003 | 54.1% | 36.2% | 58.9% |
| **LEAP (Ours)** | **Qwen-VL-Chat / InternVL** | **78.6%** | **61.4%** | **84.3%** |

---

## 📦 代码仓库结构 (Repository Structure)

```text
leap-code/
├── assets/
│   ├── leap_framework.png       # 架构流水线图
│   └── visual_execution.png     # 可视化轨迹推演图
├── leap/
│   ├── models/                  # 视觉语言骨干适配器与蓝图生成头
│   ├── planning/                # 因果拓扑图构建、动作校验与重规划逻辑
│   ├── alignment/               # 多模态表征对齐与对比损失函数
│   └── utils/                   # VLABench 环境交互与日志记录器
├── scripts/
│   ├── train_alignment.py       # 视觉-语言对齐微调启动脚本
│   └── evaluate_vlabench.py     # 基准评测端到端推理入口
├── configs/                     # 超参数与模型加载配置文件
└── requirements.txt             # 运行环境依赖
```

---

## 🚀 快速上手 (Quickstart)

### 1. 环境配置

```bash
git clone https://github.com/Evan-Joseph/leap-code.git
cd leap-code

# 创建 Conda 虚拟环境
conda create -n leap python=3.10 -y
conda activate leap

# 安装 PyTorch 及核心依赖
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu121
pip install -r requirements.txt
```

### 2. 下载预训练权重与评估场景

```bash
# 自动拉取示例评估场景配置
python scripts/download_assets.py --task pick_and_place_composite
```

### 3. 一键复现 VLABench 规划推理

```bash
python scripts/evaluate_vlabench.py \
    --config configs/vlabench_eval.yaml \
    --checkpoint checkpoints/leap_best.pt \
    --visualize
```

推理生成的动作执行轨迹与状态评估矩阵将自动保存于 `outputs/eval_results/`。

---

## 📑 引用格式 (BibTeX)

```bibtex
@article{leap2026embodied,
  title={LEAP: Action Blueprint Generation for Long-Horizon Embodied Planning via Vision-Language Alignment},
  author={Gao, Shenzhou and Collaborators},
  journal={arXiv preprint},
  year={2026}
}
```

---

## 📄 开源许可证 (License)

本项目遵循 [Apache 2.0 License](LICENSE) 开源协议。
