# Methodology Section Audit Details

Based on the analysis of the training script `scripts/training/run-finetuning.py` and the dataset `data/train_151230.jsonl`:

## 1. Output Action Space Format
The model is trained to generate **Pure Natural Language Steps**.

*   **Format**: The output is a unstructured text string containing a numbered list of instructions.
*   **Structured JSON?**: **No**. There is no evidence of structured JSON output in the training targets.
*   **Example from Data**:
    > "1) Lift the horizontally placed book on the shelf with the right arm. 2) Push aside the books on the shelf with the left arm. 3) Insert the book held in the right arm into the gap cleared by the left arm on the shelf."

## 2. Prompt Template
The system uses a **Chat-based User Prompt** injected into the `processor.apply_chat_template` method.

*   **Prompt String (User Message)**:
    ```text
    You are a task planning expert.
    Please carefully observe <Frame 1>.
    Task name: {task_name}
    Based on the above description, analyze what order needs to be followed to complete the task/how this task can be broken down into several steps.
    Give the final answer directly.
    Output format: steps: 1) ... 2) ... 3) ...
    ```
*   **Action Primitives**: There is **no explicit list of "Action Primitives" or skills** defined in the prompt. The model learns the action verbs (Lift, Push, Insert, etc.) implicitly from the training examples.

## 3. Coordinate System
The methodology uses an **Object-Centric / Implicit** coordinate system.

*   **Details**: Actions are described relative to objects and body parts (e.g., "with the right arm", "on the shelf") rather than using explicit absolute (global) or relative (x, y, z) numeric coordinates.
