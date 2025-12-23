# Code Audit Report for LEAP: Logical Embodied Action Planning

## 1. Input Representation (输入表征)

*   **DataLoader Implementation**:
    *   **File**: `scripts/training/run-finetuning.py` -> `QwenVLCompletionDataset` class.
    *   **Logic**: The dataloader reads from a JSONL file where each line contains a `messages` list (user) and a `target` dictionary (assistant).
    *   **Agibot Data Handling**: The code treats robot data as standard multimodal chat data. It parses specific types (`image`, `text`, `video`) from the input JSON and converts relative image paths to absolute paths using a provided `image_root`.
    *   **Prompt Construction**: It constructs a conversation format:
        *   **User**: `{'role': 'user', 'content': [{'type': 'image', 'image': ...}, {'type': 'text', 'text': ...}]}`
        *   **Assistant**: `{'role': 'assistant', 'content': [{'type': 'text', 'text': target_steps}]}`

*   **Image Resolution (图像分辨率)**:
    *   **Mechanism**: **Dynamic Resolution**. Qwen-VL uses a mechanism based on "visual tokens" to control resolution implicitly.
    *   **Configuration**:
        *   **Code**: `scripts/training/run-finetuning.py` lines 114-117.
        *   **⚠️ Specific Implementation Detail**: The code sets `image_processor.size` based on a token budget calculation:
            ```python
            'longest_edge': int(image_tokens_max) * 32 * 32,  # e.g., 768 * 1024
            'shortest_edge': int(image_tokens_min) * 32 * 32, # e.g., 256 * 1024
            ```
        *   **Note**: This logic sets an extremely large pixel limit, effectively allowing the model's native dynamic resolution (which patches images into 14x14 tokens) to operate with a very high ceiling, constrained primarily by the `image_tokens_max` argument (default 768 tokens).

*   **Prompt Template**:
    *   **Template**: The dataset uses `processor.apply_chat_template`.
    *   **Format**: ChatML.
    *   **Structure**:
        ```
        <|im_start|>user
        <image_placeholder>
        {instruction_text}<|im_end|>
        <|im_start|>assistant
        {target_steps}<|im_end|>
        ```
    *   **Fusion**: Images are inserted at the <image_placeholder> position. The vision encoder output is projected and interleaved with text embeddings.

## 2. Model Architecture (模型架构)

*   **Base Model**:
    *   **Version**: **Qwen3-VL-2B-Instruct** (inferred from `scripts/evaluation/run_vlm_evaluation.py` defaults and usage of `AutoModelForImageTextToText`).
    *   **Loading**: Loaded with `trust_remote_code=True` and `bf16`/`fp16` precision.

*   **Fine-tuning Strategy**:
    *   **Type**: **Full Parameter Fine-tuning (Full SFT)**.
    *   **Evidence**: `scripts/training/run-finetuning.py` line 508:
        ```python
        for p in model.parameters():
            p.requires_grad = True
        ```
    *   **LoRA**: explicitly **NOT** used.

*   **Trainable Parameters**:
    *   **Count**: **~2.2 Billion**.
    *   **Scope**: Vision Encoder, Projector (Adapter), and LLM parameters are all updated jointly.

## 3. Training Details (训练细节)

*   **Loss Function**:
    *   **Type**: Standard Causal Language Modeling (Cross-Entropy) Loss.
    *   **Masking**:
        *   **Code**: `scripts/training/run-finetuning.py` -> `_build_labels`.
        *   **Logic**:
            1.  Initialize all labels to `IGNORE_INDEX (-100)`.
            2.  Locate the Assistant's response using the start token ID `77091`.
            3.  Unmask **only** the tokens corresponding to the Assistant's output (steps).
            4.  **Critical**: The User instruction and Images are masked (loss is not calculated on them).

*   **Optimizer & Scheduler**:
    *   **Optimizer**: AdamW (Hugging Face Trainer default).
    *   **Learning Rate**: **2e-5** (default argument).
    *   **Scheduler**: Linear decay with warmup (Trainer default), `warmup_steps=0`.
    *   **Batch Size**: 1 per device (accumulated as needed).

*   **Max Sequence Length**:
    *   **Setting**: **512 Tokens**.
    *   **⚠️ High Risk Warning**: This length is very short for VLM tasks, especially when including 256+ image tokens and potentially long CoT (Chain-of-Thought) reasoning steps. Truncation is highly likely if input instructions or CoT outputs are verbose.

## 4. Inference & Evaluation (推理与评估)

*   **Inference Strategy (CoT)**:
    *   **Generation**: `max_new_tokens=512`, `temperature=0.7`, `top_p=0.9`.
    *   **Stop Token**: `eos_token_id`.
    *   **CoT Extraction**: The model interacts in a chat mode. The output is expected to contain a Markdown JSON block (` ```json ... ``` `), which is parsed to extract the `skill_sequence`.

*   **Evaluation Metric (VLABench)**:
    *   **File**: `VLABench/VLABench/evaluation/utils.py`.
    *   **Metric Components**:
        1.  **Skill Match (40%)**: Accuracy of the set of skill names used.
        2.  **Entity Match (40%)**: Accuracy of target objects/containers identified.
        3.  **Skill+Entity Match (10%)**: (Corrected) Accuracy of the coupled (skill, entity) tuples.
        4.  **Exact Match (10%)**: (Corrected) Topological match of the dependency graph.
    *   **⚠️ Critical Fix Applied**: The official VLABench code contained a bug where the last two components (20% of the score) were ignored in the summation.
    *   **Status**: We successfully audited and **fixed** this issue in our local evaluation pipeline (`fixed_eva_results`). All reported scores should use the corrected values which properly account for logical precision.

## 5. Experimental Results (Corrected)

*   **Source**: `fixed_eva_results/` (Recalculated from raw logs).
*   **Best Model**: **checkpoint-7000**.
*   **Best Overall Score**: **28.84**.
*   **Baseline Score**: **25.91** (CommonSense Dimension).
*   **Detailed Performance**: Please refer to the generated charts in `fixed_eva_results` for breakdown by dimension (M&T, Semantic, Spatial, etc.).
