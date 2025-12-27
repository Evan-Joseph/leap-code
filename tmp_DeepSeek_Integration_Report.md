# DeepSeek-VL2 评估集成问题报告与环境说明

## 1. 运行环境 (Environment)
- **Transformers**: 4.57.3 (为了兼容 Qwen2.5-VL 等新模型)
- **PyTorch**: 2.4.0
- **Bitsandbytes**: 0.45.0 (用于 4-bit 量化)
- **GPU**: 32GB VRAM (需要 4-bit 才能运行 DeepSeek-VL2-Small)

## 2. DeepSeek-VL2 集成主要障碍 (Main Issues)
DeepSeek-VL2 的官方代码仓库（基于较旧的 Transformers 版本）在当前环境下存在严重的兼容性问题：

1. **类继承与方法缺失**:
   - `DeepseekV2ForCausalLM` 和 `DeepseekV2VLM` 未继承 `GenerationMixin`，导致无法直接调用 `.generate()`。
2. **Transformers API 变更**:
   - `transformers.models.llama.modeling_llama` 中已移除 `LlamaFlashAttention2`，导致导入失败。
   - 新版 Transformers 使用 `DynamicCache` 对象，而 DeepSeek 代码中调用的 `get_usable_length` 方法已更名为 `get_seq_length`。
3. **4-bit 量化与设备冲突**:
   - 在使用 `bitsandbytes` 进行 4-bit 加载时，Vision Encoder 的某些权重（如 bias）在推理时会出现 Device Mismatch (CPU vs CUDA) 或 Dtype Mismatch (BF16 vs FP16) 的错误。
4. **RoPE 与 Cache 逻辑不匹配**:
   - 尝试手动修复 Cache 调用后，在 `apply_rotary_pos_emb` 阶段会出现 `IndexError`，提示 `position_ids` 与 Cache 长度不匹配。

## 3. 建议修复方向
- 需要针对 Transformers 4.45+ 的 `Cache` API 重写 `modeling_deepseek.py` 中的 `prepare_inputs_for_generation` 和 `forward` 逻辑。
- 建议在独立的虚拟环境（降低 Transformers 版本至 4.38.2）中运行 DeepSeek 评估，或者彻底重构其 Modeling 代码以适配新版库。

## 4. Git Commit Comment (Draft)
```text
chore: update evaluation scripts and document DeepSeek-VL2 integration issues

- Optimized `run_vlm_evaluation.py` with better resume logic and task management.
- Updated `draw_evaluation_results.py` for better visualization of 6-dimension results.
- Improved `monitor_evaluation.py` for real-time GPU and progress tracking.
- Note: DeepSeek-VL2-Small integration is currently blocked by environment compatibility issues (Transformers 4.57.3 vs DeepSeek custom modeling). Reverted DeepSeek-VL2 source changes to maintain a clean state.
```

## 5. 代码变更摘要 (Git Diff Summary)
以下是 `leap-code` 仓库中已完成的工程化改进：
- `eval.sh`: 增加了并行运行逻辑。
- `scripts/evaluation/run_vlm_evaluation.py`: 完善了断点续传。
- `scripts/evaluation/draw_evaluation_results.py`: 重构了绘图逻辑，支持多模型对比。
- `scripts/utils/monitor_evaluation.py`: 增强了监控功能。
