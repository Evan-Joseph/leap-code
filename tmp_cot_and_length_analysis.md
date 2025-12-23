# CoT & Sequence Length Analysis Report

## 1. Data Inspection (数据样本查看)

Below are 3 representative samples extracted from `data/train_151230.jsonl`:

**Sample 1:**
> **Instruction (User)**: "...analyze what order needs to be followed to complete the task... Give the final answer directly."
> **Target (Assistant)**: `"1) Lift the horizontally placed book on the shelf with the right arm. 2) Push aside the books on the shelf with the left arm. 3) Insert the book held in the right arm into the gap cleared by the left arm on the shelf. 4) Push the misplaced book on the shelf to the end with your right arm."`

**Sample 2:**
> **Instruction (User)**: "...Giving the final answer directly..."
> **Target (Assistant)**: `"1) Pick up the cloth on the table with left arm. 2) Wipe the stains on microwave oven clean with left arm. 3) Place the cloth on the table with left arm."`

**Sample 3:**
> **Instruction (User)**: "...Give the final answer directly..."
> **Target (Assistant)**: `"1) Insert the held key into the door lock with left arm. 2) Turn the key in the door with left arm. 3) Release the held key with left arm. 4) Turn the doorknob to open the door with left arm."`

**Observation**: The `target` fields contain **Direct Action Sequences** (enumerated steps). There is **NO** intermediate reasoning, thought process, or "Chain-of-Thought" text (e.g., "First, I see a book... The goal is... Therefore I should..."). The user instruction explicitly asks to "Give the final answer directly."

## 2. Token Length Stress Test (长度压力测试)

*   **Image Token Cost**:
    *   The script explicitly sets `longest_edge` corresponding to `image_tokens_max = 768` (default).
    *   For Qwen-VL, an image typically consumes **~256** tokens (min) up to **~1024** tokens depending on aspect ratio / patches. With the provided settings, we can assume a significant portion of the budget (e.g., **256+ tokens**) is reserved for the image.

*   **Text Token Cost**:
    *   **System + User Instruction**: The instruction "You are a task planning expert... Please carefully observe..." is roughly **60-80 tokens**.
    *   **Target (Action Plan)**: The steps look concise, likely **50-100 tokens**.
    *   **Special Tokens**: ChatML format `<|im_start|>` etc. adds ~10 tokens.

*   **Total Estimate**:
    *   Total $\approx$ 256 (Image) + 80 (Prompt) + 100 (Target) = **~436 Tokens**.
    *   **Risk**: This is dangerously close to the **512 limit**. If an image is encoded with higher resolution (using more tokens, e.g. 500+) or the action plan is longer, **truncation will occur**.

*   **Truncation Logic**:
    *   Code: `truncation=True` in `processor.apply_chat_template` (Line 224).
    *   Direction: Default truncation for Transformers is usually **Right** (cutting off the end).
    *   **Consequence**: If the total length exceeds 512, the **end of the action plan will be cut off**, leading to incomplete JSON/Steps and high loss.

## 3. Methodological Conclusion (方法论结论)

*   **CoT Status**: **NO Explicit CoT**.
    *   The model is performing **Direct Action Planning** (Direct Mapping from <Image, Instruction> to <Action Sequence>).
    *   The "reasoning" is implicit within the model's layers, not explicit in the output text.
    *   *Recommendation*: In the paper, do **not** claim "Chain-of-Thought" prompting unless you change the training data to include reasoning steps. Describe it as "End-to-End Visual Action Planning."

*   **Sequence Length Setting**: **512 is Risky/Insufficient**.
    *   Given the multimodal nature (Image + Text), 512 is an extremely tight constraint.
    *   *Recommendation*: Ideally, this should be increased to **1024** or **2048** to ensure training stability and strictly correct evaluations, although for the short samples observed, it might *just* barely fit.
