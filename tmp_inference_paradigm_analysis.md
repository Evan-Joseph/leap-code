# Inference Paradigm Analysis Report

## 1. Conclusion: One-shot Open-loop Planner
Based on the deep code audit of `scripts/evaluation/run_vlm_evaluation.py` and `VLABench/evaluation/evaluator`, we confirm that the system operates as a **One-shot Open-loop Planner**.

*   **Paradigm**: **Offline Visual Action Planning** (or Open-loop Video-Language Planning).
*   **Mechanism**: The model observes the **initial state** only and generates the **entire action sequence** in a single inference pass.
*   **Feedback**: **None**. There is no "closed-loop" interaction, no re-planning, and no environmental feedback (success/fail signals) during the generation process.

## 2. Evidence from Code

### A. Inference Loop Structure
*   **Source**: `scripts/evaluation/run_vlm_evaluation.py` (Class `CustomVLMEvaluator`) & `VLABench/.../vlm.py` (Class `VLMEvaluator`).
*   **Finding**: The evaluation loop iterates through tasks and examples. For each example, it calls `self.get_single_anwer(...)` **exactly once**.
*   **Code Reference**:
    ```python
    # run_vlm_evaluation.py
    for task_name, example_num in test_example_list:
        # One-shot call to the VLM
        answer = self.get_single_anwer(task_name, example_num, vlm, ...)
    ```
    There is no loop like `while not done: env.step(action)` inside the evaluation logic.

### B. Execution & Evaluation Logic
*   **Source**: `VLABench/evaluation/utils.py` (Function `get_final_score`).
*   **Finding**: The evaluation is **Static Benchmark**, not Dynamic Simulation.
    *   The code compares the **Textual Output** (generated skill sequence) against the **Ground Truth JSON** (standard operation sequence).
    *   It does **NOT** execute actions in a physics simulator (e.g., PyBullet/Isaac Gym) during this specific evaluation script.
    *   Scores (Skill Match, Entity Match) are calculated based on string/graph matching, not task success rates in a sim.

### C. Input/History
*   **Source**: `VLABench/.../vlm.py` (Method `load_single_input`).
*   **Finding**: The input is fixed to `input/input.png` (Initial Frame) and `input/instruction.txt`. No history of executed steps or updated frames is passed to the model.

## 3. Methodology Description Recommendations

For the "Methodology" section of the LEAP paper, use the following definitions to ensure accuracy:

1.  **Task Definition**: "Given an initial visual observation $I_0$ and a natural language instruction $L$, the model generates a complete plan $P = \{a_1, a_2, ..., a_T\}$ consisting of $T$ sequential primitives."
2.  **Planner Type**: **Open-loop**. Specifically, "The model functions as an offline planner, predicting the full action trajectory autoregressively without intermediate environmental feedback."
3.  **Correction**: Explicitly **avoid** terms like "Closed-loop", "Reactive", or "Online Replanning", as the current codebase does not support these features.
