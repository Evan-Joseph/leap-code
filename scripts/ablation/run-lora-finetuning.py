#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
run-lora-finetuning.py

LoRA 参数高效微调脚本（消融实验版本）。

设计要点：
1. 数据格式适配：与全量微调脚本一致，将原始 JSONL 转换为符合 Qwen VL Processor 的对话格式。
2. 标签构造：仅对 assistant 回复文本对应的 token 位置赋值其 token id，其他填充 IGNORE_INDEX (-100)。
3. LoRA 参数高效训练：使用 PEFT 库的 LoRA 适配器，仅训练少量参数。
4. 支持多种 LoRA 配置：可调整 r、alpha、dropout 等超参数进行消融实验。
5. 可选裁剪：支持 --max_train_samples / --max_eval_samples 用于快速验证。
6. 兼容 BF16 / FP16、Gradient Checkpointing。

消融实验设计：
- 可对比不同 LoRA rank (r=8, 16, 32, 64)
- 可对比不同 target_modules 组合
- 可对比 LoRA vs 全量微调的性能差异
"""

import os
import sys
import json
import argparse
from pathlib import Path
from dataclasses import dataclass
from typing import List, Dict, Any, Optional
import glob
from collections import deque

import torch
from torch.utils.data import Dataset
from transformers import (
    AutoProcessor,
    AutoModelForImageTextToText,
    Trainer,
    TrainingArguments,
    set_seed,
    TrainerCallback,
)
from peft import (
    LoraConfig,
    get_peft_model,
    TaskType,
    PeftModel,
)
from rich import print as rich_print
from rich.table import Table
from rich.panel import Panel

# 添加配置路径（基于脚本位置的仓库根）
REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.append(str(REPO_ROOT / "configs"))
from config import WorkspaceConfig

IGNORE_INDEX = -100

class SaveLoraCallback(TrainerCallback):
    """自定义回调：在保存checkpoint时同时保存LoRA适配器和processor配置"""
    def __init__(self, processor, peft_model):
        self.processor = processor
        self.peft_model = peft_model

    def on_save(self, args, state, control, **kwargs):
        """在每次保存checkpoint时调用"""
        checkpoint_dir = os.path.join(args.output_dir, f"checkpoint-{state.global_step}")
        if os.path.exists(checkpoint_dir):
            # 保存processor配置到checkpoint目录
            self.processor.save_pretrained(checkpoint_dir)
            rich_print(f"[dim]Processor config saved to {checkpoint_dir}[/dim]")


# =============================
# 数据集实现（与全量微调一致）
# =============================
class QwenVLCompletionDataset(Dataset):
    """针对当前 JSONL 数据集格式的自定义 Dataset。

    仅支持标准的 messages + target 格式：
    {
      "messages": [
        {"role": "user", "content": [{"type": "image", "image": "images/327-xyz.png"}, {"type": "text", "text": "..."}]}
      ],
      "target": {"steps": "1) ... 2) ..."}  
    }

    标签构造：仅 assistant 回复对应 token 计入 loss（实现语言建模式监督）。
    """
    def __init__(
        self,
        file_path: str,
        processor: Any,
        tokenizer: Any,
        image_root: str,
        max_samples: Optional[int] = None,
        model_max_length: int = 512,
        image_tokens_min: int = 256,
        image_tokens_max: int = 1024,
        video_tokens_min: int = 256,
        video_tokens_max: int = 4096,
        video_fps: Optional[float] = 2.0,
        video_num_frames: Optional[int] = None,
    ):
        self.file_path = file_path
        self.processor = processor
        self.tokenizer = tokenizer
        self.image_root = image_root
        self.model_max_length = model_max_length
        self.samples: List[Dict[str, Any]] = []
        self.video_fps = video_fps
        self.video_num_frames = video_num_frames

        if hasattr(self.processor, 'image_processor') and self.processor.image_processor is not None:
            self.processor.image_processor.size = {
                'longest_edge': int(image_tokens_max) * 32 * 32,
                'shortest_edge': int(image_tokens_min) * 32 * 32,
            }
        if hasattr(self.processor, 'video_processor') and self.processor.video_processor is not None:
            self.processor.video_processor.size = {
                'longest_edge': int(video_tokens_max) * 32 * 32,
                'shortest_edge': int(video_tokens_min) * 32 * 32,
            }

        with open(file_path, 'r', encoding='utf-8') as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                obj = json.loads(line)
                self.samples.append(obj)
                if max_samples is not None and len(self.samples) >= max_samples:
                    break

    def __len__(self):
        return len(self.samples)

    @staticmethod
    def _make_abs_image_path(image_root: str, rel_path: str) -> str:
        """将相对路径转换为绝对路径"""
        if 'demo-data/images/' in rel_path:
            rel_path = rel_path.split('demo-data/images/', 1)[1]
        
        if os.path.isabs(rel_path):
            return rel_path
        
        abs_path = os.path.abspath(os.path.join(image_root, rel_path))
        return abs_path

    def _build_messages(self, raw: Dict[str, Any]) -> tuple[List[Dict[str, Any]], str]:
        if 'messages' not in raw:
             raise ValueError("数据格式错误：缺少 'messages' 字段。")
        
        user_block = raw['messages'][0]
        assert user_block['role'] == 'user', '第一条消息必须是 user'
        new_content = []
        for c in user_block['content']:
            if c['type'] == 'image':
                abs_path = self._make_abs_image_path(self.image_root, c['image'])
                new_content.append({'type': 'image', 'image': abs_path})
            elif c['type'] == 'text':
                new_content.append({'type': 'text', 'text': c['text']})
            elif c['type'] == 'video':
                abs_path = self._make_abs_image_path(self.image_root, c['video'])
                new_content.append({'type': 'video', 'video': abs_path})
            else:
                raise ValueError(f"未支持的内容类型: {c['type']}")

        if 'target' not in raw or 'steps' not in raw['target']:
             raise ValueError("数据格式错误：缺少 'target.steps' 字段。")

        target_label = raw['target']['steps']
        assistant_text = f"{target_label}"
        messages = [
            {'role': 'user', 'content': new_content},
            {'role': 'assistant', 'content': [{'type': 'text', 'text': assistant_text}]},
        ]
        return messages, assistant_text

    def _build_labels(self, input_ids: torch.Tensor, answer_tokens: List[int]) -> torch.Tensor:
        """在 input_ids 中定位 answer_tokens 子序列位置，构建 labels 张量。"""
        seq = input_ids.tolist()
        n = len(seq)
        m = len(answer_tokens)
        start_index = -1
        for i in range(n - m + 1):
            if seq[i:i+m] == answer_tokens:
                start_index = i
                break
        labels = torch.full_like(input_ids, IGNORE_INDEX)
        if start_index != -1:
            labels[start_index:start_index + m] = torch.tensor(answer_tokens, dtype=input_ids.dtype)
        return labels

    def __getitem__(self, idx: int) -> Dict[str, Any]:
        raw = self.samples[idx]
        messages, assistant_text = self._build_messages(raw)

        processed = self.processor.apply_chat_template(
            messages,
            tokenize=True,
            return_dict=True,
            return_tensors='pt',
            truncation=True,
            max_length=self.model_max_length,
            add_generation_prompt=False,
        )

        input_ids = processed['input_ids']
        if isinstance(input_ids, list):
            input_ids = torch.tensor(input_ids).unsqueeze(0)

        if input_ids.dim() == 1:
            input_ids = input_ids.unsqueeze(0)

        labels = torch.full_like(input_ids, IGNORE_INDEX)

        input_ids_flat = input_ids[0].tolist()
        L = len(input_ids_flat)
        pos = 0
        while pos < L:
            if input_ids_flat[pos] == 77091:  # <|im_start|>assistant
                ans_start = pos + 2
                ans_end = ans_start
                while ans_end < L and input_ids_flat[ans_end] != 151645:
                    ans_end += 1
                if ans_end < L:
                    labels[0, ans_start : ans_end + 2] = input_ids[0, ans_start : ans_end + 2]
                    pos = ans_end
            pos += 1

        item = {
            'input_ids': input_ids[0],
            'labels': labels[0],
        }
        
        if 'attention_mask' in processed and processed['attention_mask'] is not None:
            item['attention_mask'] = processed['attention_mask'][0]

        for k in ['pixel_values', 'image_grid_thw', 'pixel_values_videos', 'video_grid_thw']:
            if k in processed and processed[k] is not None:
                if isinstance(processed[k], list) and len(processed[k]) > 0:
                    item[k] = processed[k][0]
                else:
                    item[k] = processed[k]
        
        return item


# =============================
# Collator（与全量微调一致）
# =============================
class DataCollatorForQwenVL:
    """按批次对齐 input_ids/labels；同时合并视觉张量。"""
    def __init__(self, tokenizer, pad_to_multiple_of: Optional[int] = None):
        self.tokenizer = tokenizer
        self.pad_to_multiple_of = pad_to_multiple_of

    def __call__(self, batch: List[Dict[str, Any]]) -> Dict[str, torch.Tensor]:
        input_ids_list = [x['input_ids'] for x in batch]
        labels_list = [x['labels'] for x in batch]

        pad_token_id = self.tokenizer.pad_token_id if self.tokenizer.pad_token_id is not None else 0

        def _pad(seqs: List[torch.Tensor], pad_value: int):
            max_len = max(s.size(0) for s in seqs)
            if self.pad_to_multiple_of:
                if max_len % self.pad_to_multiple_of != 0:
                    max_len = ((max_len // self.pad_to_multiple_of) + 1) * self.pad_to_multiple_of
            out = []
            for s in seqs:
                if s.size(0) < max_len:
                    pad_shape = (max_len - s.size(0),)
                    s = torch.cat([s, torch.full(pad_shape, pad_value, dtype=s.dtype)], dim=0)
                out.append(s)
            return torch.stack(out, dim=0)

        input_ids = _pad(input_ids_list, pad_token_id)
        labels = _pad(labels_list, IGNORE_INDEX)
        attention_mask = (input_ids != pad_token_id).long()

        batch_out = {
            'input_ids': input_ids,
            'labels': labels,
            'attention_mask': attention_mask,
        }

        vision_keys = ['pixel_values', 'image_grid_thw', 'pixel_values_videos', 'video_grid_thw']
        for k in vision_keys:
            vs = [x[k] for x in batch if k in x]
            if len(vs) > 0:
                if isinstance(vs[0], torch.Tensor):
                    try:
                        batch_out[k] = torch.cat(vs, dim=0)
                    except Exception:
                        batch_out[k] = vs
                else:
                    batch_out[k] = vs
        return batch_out


# =============================
# 评估指标（占位）
# =============================

def compute_metrics(eval_pred):
    return {}


# =============================
# 自定义日志回调
# =============================

class RichLoggingCallback(TrainerCallback):
    """使用 Rich 美化训练日志输出的回调类"""

    def __init__(self):
        self.history = deque(maxlen=3)

    def on_log(self, args, state, control, logs=None, **kwargs):
        if logs is None:
            return

        self.history.append(logs.copy())

        is_eval = 'eval_loss' in logs
        first_col_name = "Epoch" if is_eval else "Step"
        first_col_value = f"{state.epoch:.2f}" if is_eval else str(state.global_step)

        table = Table(title=f"[bold blue]{'评估' if is_eval else '训练'}日志 (LoRA) - {first_col_name} {first_col_value}[/bold blue]",
                     show_header=True, header_style="bold magenta")

        columns = [first_col_name, "Loss", "Grad Norm", "Learning Rate", "Epoch"]
        for col in columns:
            table.add_column(col, style="cyan", justify="center")

        table.add_row(*columns)

        history_list = list(self.history)
        num_history = len(history_list)

        if num_history >= 1:
            latest = history_list[-1]
            loss = f"{latest.get('loss', latest.get('eval_loss', 'N/A')):.4f}" if isinstance(latest.get('loss', latest.get('eval_loss')), (int, float)) else "N/A"
            grad_norm = f"{latest.get('grad_norm', 'N/A'):.4f}" if isinstance(latest.get('grad_norm'), (int, float)) else "N/A"
            lr = f"{latest.get('learning_rate', 'N/A'):.2e}" if isinstance(latest.get('learning_rate'), (int, float)) else "N/A"
            epoch = f"{latest.get('epoch', state.epoch or 'N/A'):.2f}" if isinstance(latest.get('epoch', state.epoch), (int, float)) else "N/A"
            step_or_epoch = f"{state.epoch:.2f}" if is_eval else str(state.global_step)
            table.add_row(step_or_epoch, loss, grad_norm, lr, epoch)

        if num_history >= 2:
            prev = history_list[-2]
            loss = f"{prev.get('loss', prev.get('eval_loss', 'N/A')):.4f}" if isinstance(prev.get('loss', prev.get('eval_loss')), (int, float)) else "N/A"
            grad_norm = f"{prev.get('grad_norm', 'N/A'):.4f}" if isinstance(prev.get('grad_norm'), (int, float)) else "N/A"
            lr = f"{prev.get('learning_rate', 'N/A'):.2e}" if isinstance(prev.get('learning_rate'), (int, float)) else "N/A"
            epoch = f"{prev.get('epoch', state.epoch or 'N/A'):.2f}" if isinstance(prev.get('epoch', state.epoch), (int, float)) else "N/A"
            step_or_epoch = f"{state.epoch:.2f}" if is_eval else str(state.global_step - args.logging_steps)
            table.add_row(step_or_epoch, loss, grad_norm, lr, epoch)

        if num_history >= 3:
            prev_prev = history_list[-3]
            loss = f"{prev_prev.get('loss', prev_prev.get('eval_loss', 'N/A')):.4f}" if isinstance(prev_prev.get('loss', prev_prev.get('eval_loss')), (int, float)) else "N/A"
            grad_norm = f"{prev_prev.get('grad_norm', 'N/A'):.4f}" if isinstance(prev_prev.get('grad_norm'), (int, float)) else "N/A"
            lr = f"{prev_prev.get('learning_rate', 'N/A'):.2e}" if isinstance(prev_prev.get('learning_rate'), (int, float)) else "N/A"
            epoch = f"{prev_prev.get('epoch', state.epoch or 'N/A'):.2f}" if isinstance(prev_prev.get('epoch', state.epoch), (int, float)) else "N/A"
            step_or_epoch = f"{state.epoch:.2f}" if is_eval else str(state.global_step - 2 * args.logging_steps)
            table.add_row(step_or_epoch, loss, grad_norm, lr, epoch)

        panel = Panel(table, border_style="green")
        rich_print(panel)


# =============================
# LoRA 配置预设
# =============================

LORA_PRESETS = {
    # 标准配置：适用于大多数场景
    "standard": {
        "r": 16,
        "lora_alpha": 32,
        "lora_dropout": 0.05,
        "target_modules": ["q_proj", "k_proj", "v_proj", "o_proj"],
    },
    # 轻量配置：最小化可训练参数
    "light": {
        "r": 8,
        "lora_alpha": 16,
        "lora_dropout": 0.1,
        "target_modules": ["q_proj", "v_proj"],
    },
    # 完整配置：覆盖更多层
    "full": {
        "r": 32,
        "lora_alpha": 64,
        "lora_dropout": 0.05,
        "target_modules": ["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"],
    },
    # 激进配置：高秩LoRA
    "aggressive": {
        "r": 64,
        "lora_alpha": 128,
        "lora_dropout": 0.05,
        "target_modules": ["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"],
    },
}


def get_lora_config(preset: str = "standard", **overrides) -> LoraConfig:
    """根据预设获取 LoRA 配置，支持参数覆盖"""
    if preset not in LORA_PRESETS:
        raise ValueError(f"未知的 LoRA 预设: {preset}，可用选项: {list(LORA_PRESETS.keys())}")
    
    config_dict = LORA_PRESETS[preset].copy()
    config_dict.update(overrides)
    
    return LoraConfig(
        task_type=TaskType.CAUSAL_LM,
        inference_mode=False,
        r=config_dict["r"],
        lora_alpha=config_dict["lora_alpha"],
        lora_dropout=config_dict["lora_dropout"],
        target_modules=config_dict["target_modules"],
        bias="none",
    )


def print_trainable_parameters(model):
    """打印可训练参数统计"""
    trainable_params = 0
    all_param = 0
    for _, param in model.named_parameters():
        all_param += param.numel()
        if param.requires_grad:
            trainable_params += param.numel()
    
    trainable_percent = 100 * trainable_params / all_param
    
    rich_print(Panel(
        f"[bold green]可训练参数: {trainable_params:,}[/bold green]\n"
        f"[bold blue]总参数: {all_param:,}[/bold blue]\n"
        f"[bold yellow]可训练比例: {trainable_percent:.4f}%[/bold yellow]",
        title="[bold magenta]LoRA 参数统计[/bold magenta]",
        border_style="green"
    ))
    
    return trainable_params, all_param, trainable_percent


# =============================
# 主函数
# =============================

def parse_args():
    parser = argparse.ArgumentParser(description='Qwen-VL LoRA 参数高效微调脚本（消融实验）')
    parser.add_argument('--model_name_or_path', type=str, required=True, help='本地已下载的预训练模型目录')
    parser.add_argument('--train_file', type=str, required=True, help='训练集 JSONL 文件路径')
    parser.add_argument('--val_file', type=str, required=False, help='验证集 JSONL 文件路径')
    parser.add_argument('--image_root', type=str, required=True, help='图片根目录')
    parser.add_argument('--output_dir', type=str, required=True, help='保存输出的目录')

    # LoRA 超参数
    parser.add_argument('--lora_preset', type=str, default='standard', 
                        choices=['standard', 'light', 'full', 'aggressive'],
                        help='LoRA 预设配置')
    parser.add_argument('--lora_r', type=int, default=None, help='LoRA rank (覆盖预设)')
    parser.add_argument('--lora_alpha', type=int, default=None, help='LoRA alpha (覆盖预设)')
    parser.add_argument('--lora_dropout', type=float, default=None, help='LoRA dropout (覆盖预设)')
    parser.add_argument('--lora_target_modules', type=str, default=None, 
                        help='LoRA target modules，以逗号分隔 (覆盖预设)')
    
    # 训练超参数
    parser.add_argument('--per_device_train_batch_size', type=int, default=2)
    parser.add_argument('--per_device_eval_batch_size', type=int, default=2)
    parser.add_argument('--learning_rate', type=float, default=2e-4, help='LoRA 通常使用更高的学习率')
    parser.add_argument('--num_train_epochs', type=int, default=1)
    parser.add_argument('--max_steps', type=int, default=None)
    parser.add_argument('--gradient_accumulation_steps', type=int, default=4)
    parser.add_argument('--warmup_steps', type=int, default=100)
    parser.add_argument('--weight_decay', type=float, default=0.01)
    parser.add_argument('--logging_steps', type=int, default=10)
    parser.add_argument('--save_steps', type=int, default=200)
    parser.add_argument('--save_total_limit', type=int, default=5)
    parser.add_argument('--bf16', action='store_true')
    parser.add_argument('--fp16', action='store_true')
    parser.add_argument('--gradient_checkpointing', action='store_true')
    parser.add_argument('--attn_implementation', type=str, default='sdpa', 
                        choices=['flash_attention_2', 'sdpa', 'eager'])
    parser.add_argument('--eval_strategy', type=str, default=None, 
                        choices=['no', 'epoch', 'steps', None])
    parser.add_argument('--eval_steps', type=int, default=50)

    parser.add_argument('--model_max_length', type=int, default=512)
    parser.add_argument('--image_tokens_min', type=int, default=256)
    parser.add_argument('--image_tokens_max', type=int, default=768)
    parser.add_argument('--video_tokens_min', type=int, default=256)
    parser.add_argument('--video_tokens_max', type=int, default=4096)
    parser.add_argument('--video_fps', type=float, default=2.0)
    parser.add_argument('--video_num_frames', type=int, default=None)
    parser.add_argument('--max_train_samples', type=int, default=None)
    parser.add_argument('--max_eval_samples', type=int, default=None)
    parser.add_argument('--seed', type=int, default=42)

    # 从已有 LoRA checkpoint 恢复
    parser.add_argument('--resume_from_lora', type=str, default=None,
                        help='从已有 LoRA checkpoint 恢复训练')

    return parser.parse_args()


def main():
    args = parse_args()

    # 将相对路径解析为绝对路径
    def _abs(p: str) -> str:
        if p is None:
            return p
        if os.path.isabs(p):
            return p
        return str((REPO_ROOT / p).resolve())

    args.model_name_or_path = _abs(args.model_name_or_path)
    args.train_file = _abs(args.train_file)
    args.val_file = _abs(args.val_file) if args.val_file else args.val_file
    args.image_root = _abs(args.image_root)
    args.output_dir = _abs(args.output_dir)
    args.resume_from_lora = _abs(args.resume_from_lora) if args.resume_from_lora else None

    os.makedirs(args.output_dir, exist_ok=True)
    set_seed(args.seed)

    rich_print("[bold blue]正在加载基础模型...[/bold blue]")
    
    dtype_sel = (
        torch.bfloat16 if args.bf16 else (torch.float16 if args.fp16 else "auto")
    )
    
    model = AutoModelForImageTextToText.from_pretrained(
        args.model_name_or_path,
        torch_dtype=dtype_sel,
        attn_implementation=args.attn_implementation,
        trust_remote_code=True,
    )
    
    if hasattr(model, 'config'):
        model.config.use_cache = False

    # 构建 LoRA 配置
    rich_print(f"[bold blue]配置 LoRA 适配器 (预设: {args.lora_preset})...[/bold blue]")
    
    overrides = {}
    if args.lora_r is not None:
        overrides['r'] = args.lora_r
    if args.lora_alpha is not None:
        overrides['lora_alpha'] = args.lora_alpha
    if args.lora_dropout is not None:
        overrides['lora_dropout'] = args.lora_dropout
    if args.lora_target_modules is not None:
        overrides['target_modules'] = [m.strip() for m in args.lora_target_modules.split(',')]
    
    lora_config = get_lora_config(args.lora_preset, **overrides)
    
    # 打印 LoRA 配置
    rich_print(Panel(
        f"[bold]r:[/bold] {lora_config.r}\n"
        f"[bold]alpha:[/bold] {lora_config.lora_alpha}\n"
        f"[bold]dropout:[/bold] {lora_config.lora_dropout}\n"
        f"[bold]target_modules:[/bold] {lora_config.target_modules}",
        title="[bold magenta]LoRA 配置[/bold magenta]",
        border_style="blue"
    ))
    
    # 应用 LoRA
    if args.resume_from_lora and os.path.exists(args.resume_from_lora):
        rich_print(f"[bold yellow]从 LoRA checkpoint 恢复: {args.resume_from_lora}[/bold yellow]")
        model = PeftModel.from_pretrained(model, args.resume_from_lora)
    else:
        model = get_peft_model(model, lora_config)
    
    # 打印可训练参数统计
    print_trainable_parameters(model)

    processor = AutoProcessor.from_pretrained(args.model_name_or_path, trust_remote_code=True)
    tokenizer = processor.tokenizer
    tokenizer.model_max_length = args.model_max_length
    tokenizer.padding_side = 'right'
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    rich_print("[bold blue]正在构建数据集...[/bold blue]")
    
    train_dataset = QwenVLCompletionDataset(
        file_path=args.train_file,
        processor=processor,
        tokenizer=tokenizer,
        image_root=args.image_root,
        max_samples=args.max_train_samples,
        model_max_length=args.model_max_length,
        image_tokens_min=args.image_tokens_min,
        image_tokens_max=args.image_tokens_max,
        video_tokens_min=args.video_tokens_min,
        video_tokens_max=args.video_tokens_max,
        video_fps=args.video_fps,
        video_num_frames=args.video_num_frames,
    )
    
    eval_dataset = None
    if args.val_file:
        eval_dataset = QwenVLCompletionDataset(
            file_path=args.val_file,
            processor=processor,
            tokenizer=tokenizer,
            image_root=args.image_root,
            max_samples=args.max_eval_samples,
            model_max_length=args.model_max_length,
            image_tokens_min=args.image_tokens_min,
            image_tokens_max=args.image_tokens_max,
            video_tokens_min=args.video_tokens_min,
            video_tokens_max=args.video_tokens_max,
            video_fps=args.video_fps,
            video_num_frames=args.video_num_frames,
        )

    collator = DataCollatorForQwenVL(tokenizer)

    if args.eval_strategy is not None:
        evaluation_strategy = args.eval_strategy
    else:
        evaluation_strategy = 'epoch' if (args.val_file is not None and len(str(args.val_file)) > 0) else 'no'

    training_args = TrainingArguments(
        output_dir=args.output_dir,
        per_device_train_batch_size=args.per_device_train_batch_size,
        per_device_eval_batch_size=args.per_device_eval_batch_size,
        learning_rate=args.learning_rate,
        num_train_epochs=args.num_train_epochs,
        max_steps=args.max_steps if args.max_steps is not None else -1,
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        warmup_steps=args.warmup_steps,
        weight_decay=args.weight_decay,
        logging_steps=args.logging_steps,
        save_steps=args.save_steps,
        save_total_limit=args.save_total_limit,
        bf16=args.bf16,
        fp16=args.fp16,
        eval_strategy=evaluation_strategy,
        eval_steps=args.eval_steps,
        report_to=[],
        gradient_checkpointing=args.gradient_checkpointing,
        dataloader_pin_memory=True,
        dataloader_num_workers=0,
        logging_strategy="steps",
        disable_tqdm=True,
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        data_collator=collator,
        compute_metrics=compute_metrics if eval_dataset is not None else None,
        callbacks=[
            RichLoggingCallback(),
            SaveLoraCallback(processor, model),
        ],
    )

    # 自动断点续训
    checkpoint_dirs = glob.glob(os.path.join(args.output_dir, "checkpoint-*"))
    resume_from_checkpoint = None

    if checkpoint_dirs:
        checkpoint_nums = []
        for ckpt_dir in checkpoint_dirs:
            try:
                num_str = os.path.basename(ckpt_dir).replace("checkpoint-", "")
                num = int(num_str)
                checkpoint_nums.append((num, ckpt_dir))
            except ValueError:
                continue

        if checkpoint_nums:
            checkpoint_nums.sort(key=lambda x: x[0], reverse=True)
            latest_checkpoint = checkpoint_nums[0][1]
            resume_from_checkpoint = latest_checkpoint
            rich_print(f"[bold yellow]检测到检查点，从 {latest_checkpoint} 恢复训练...[/bold yellow]")
        else:
            rich_print("[bold blue]未找到有效的检查点，开始全新训练...[/bold blue]")
    else:
        rich_print("[bold blue]未找到检查点，开始全新 LoRA 训练...[/bold blue]")

    rich_print("[bold green]LoRA 训练开始...[/bold green]")
    trainer.train(resume_from_checkpoint=resume_from_checkpoint)
    
    if eval_dataset is not None:
        trainer.evaluate()

    # 保存最终 LoRA 适配器
    model.save_pretrained(args.output_dir)
    processor.save_pretrained(args.output_dir)

    rich_print(f'[bold green]LoRA 训练完成: 适配器已保存到 {args.output_dir}[/bold green]')


if __name__ == '__main__':
    main()
