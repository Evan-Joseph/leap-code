#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
run-finetuning.py

全量参数微调脚本（针对最新数据集的单图+文本结构）。

设计要点：
1. 数据格式适配：将原始 JSONL 中的 `image_path` + `task_general` + `task_detail` 转换为符合 Qwen VL Processor `apply_chat_template` 的对话列表：
   [
     {'role': 'user', 'content': [ {image块}, {text块} ]},
     {'role': 'assistant', 'content': [ {'type': 'text', 'text': '理解。我已经看到了图像和任务描述。'} ]}
   ]
2. 标签构造：仅对 assistant 回复文本对应的 token 位置赋值其 token id，其他填充 IGNORE_INDEX (-100)，实现语言建模式监督。
   为稳健性：通过查找特定 token id 来定位 assistant 回复的起止位置。
3. 全量参数训练：不启用 LoRA；直接加载模型并确保所有参数 requires_grad=True。
4. 评估：提供验证集 (val_file)，以相同标签规则计算 loss，Trainer 的 evaluation_loss 可用于监控。
5. 可选裁剪：支持 --max_train_samples / --max_eval_samples 用于快速烟囱测试。
6. 多模态路径：图像相对路径拼接 --image_root 形成绝对路径，传递给 processor。
7. 兼容 BF16 / FP16、Gradient Checkpointing、DeepSpeed。
8. 单图全量微调：专为单个图像+任务描述的微调场景优化，支持完整语言建模训练。

注意：本脚本针对新的数据格式独立运行，避免与旧的多图标注逻辑冲突。
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
from rich import print as rich_print
from rich.table import Table
from rich.panel import Panel

# 添加配置路径（基于脚本位置的仓库根）
REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.append(str(REPO_ROOT / "configs"))
from config import WorkspaceConfig

IGNORE_INDEX = -100

class SaveProcessorCallback(TrainerCallback):
    """自定义回调：在保存checkpoint时同时保存processor配置"""
    def __init__(self, processor):
        self.processor = processor

    def on_save(self, args, state, control, **kwargs):
        """在每次保存checkpoint时调用"""
        checkpoint_dir = os.path.join(args.output_dir, f"checkpoint-{state.global_step}")
        if os.path.exists(checkpoint_dir):
            # 保存processor配置到checkpoint目录
            self.processor.save_pretrained(checkpoint_dir)
            rich_print(f"[dim]Processor config saved to {checkpoint_dir}[/dim]")

# =============================
# 数据集实现
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
        # 记录视频控制参数（在 apply_chat_template 时传入）
        self.video_fps = video_fps
        self.video_num_frames = video_num_frames

        # 控制视觉 token 预算：将图像/视频的“总像素预算”设为基于 token 数的值（与官方 README 一致的思路）
        # 说明：Qwen3-VL 将视觉 token 与像素预算约按 32*32 的压缩比对应，此处用 token 数 * 32 * 32 进行设置。
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
        """将相对路径转换为绝对路径
        
        特殊处理：如果 rel_path 中包含 'demo-data/images/'，
        会自动从image_root中提取子路径
        """
        # 被犀改映射：从 'demo-data/images/xxx' 提取 'xxx' 部分
        if 'demo-data/images/' in rel_path:
            rel_path = rel_path.split('demo-data/images/', 1)[1]
        
        # 如果是绝对路径，直接返回
        if os.path.isabs(rel_path):
            return rel_path
        
        # 其他情况与 image_root 拼接
        abs_path = os.path.abspath(os.path.join(image_root, rel_path))
        return abs_path

    def _build_messages(self, raw: Dict[str, Any]) -> tuple[List[Dict[str, Any]], str]:
        # 输入 raw: 原始单行 JSON 对象
        # 输出 messages: 符合 processor.apply_chat_template 格式的对话列表
        
        # 强制检查数据格式
        if 'messages' not in raw:
             raise ValueError("数据格式错误：缺少 'messages' 字段。请确保使用包含 'messages' 和 'target' 的完整数据格式。")
        
        # 格式一：messages + target 示例
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

        # 提取回复
        if 'target' not in raw or 'steps' not in raw['target']:
             raise ValueError("数据格式错误：缺少 'target.steps' 字段，无法提取训练标签。")

        target_label = raw['target']['steps']
        assistant_text = f"{target_label}"
        messages = [
            {'role': 'user', 'content': new_content},
            {'role': 'assistant', 'content': [{'type': 'text', 'text': assistant_text}]},
        ]
        return messages, assistant_text

    def _build_labels(self, input_ids: torch.Tensor, answer_tokens: List[int]) -> torch.Tensor:
        """在 input_ids 中定位 answer_tokens 子序列位置，构建 labels 张量。

        若无法匹配，则全部设为 IGNORE_INDEX（保守退化）。
        """
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

    # 在 QwenVLCompletionDataset 类里面
    def __getitem__(self, idx: int) -> Dict[str, Any]:
        raw = self.samples[idx]
        messages, assistant_text = self._build_messages(raw)

        # ------------------- 唯一的、关键的修改在这里！-------------------
        # 我们不再使用模糊的**kwargs，而是清晰、明确地调用函数
        # 并把我们构造好的 `messages` 列表，传递给正确的 `conversation` 参数
        
        processed = self.processor.apply_chat_template(
            messages,  # 直接传递 messages 列表
            tokenize=True,
            return_dict=True,
            return_tensors='pt',
            truncation=True,
            max_length=self.model_max_length,
            add_generation_prompt=False, # 训练时不需要添加生成提示
        )
        # ----------------------------------------------------------------

        # 提取 input_ids
        input_ids = processed['input_ids']
        if isinstance(input_ids, list):
            input_ids = torch.tensor(input_ids).unsqueeze(0)

        # 确保 input_ids 是二维的 [batch_size, seq_len]
        if input_ids.dim() == 1:
            input_ids = input_ids.unsqueeze(0)

        # 手动构建 labels：除了 assistant 回答部分，其他都设为 IGNORE_INDEX
        labels = torch.full_like(input_ids, IGNORE_INDEX)

        # 找到 assistant 回答的开始和结束位置
        input_ids_flat = input_ids[0].tolist()
        L = len(input_ids_flat)
        pos = 0
        while pos < L:
            # 查找 assistant 回答的开始标记（通常是特定的 token id）
            # Qwen 的 assistant 标记通常是 77091
            if input_ids_flat[pos] == 77091:  # <|im_start|>assistant
                ans_start = pos + 2  # 跳过标记和换行
                ans_end = ans_start
                # 查找结束标记 151645（通常是 <|im_end|>）
                while ans_end < L and input_ids_flat[ans_end] != 151645:
                    ans_end += 1
                if ans_end < L:
                    labels[0, ans_start : ans_end + 2] = input_ids[0, ans_start : ans_end + 2]
                    pos = ans_end
            pos += 1

        # 输出组装
        item = {
            'input_ids': input_ids[0],
            'labels': labels[0],
        }
        
        # 确保attention_mask存在且正确
        if 'attention_mask' in processed and processed['attention_mask'] is not None:
            item['attention_mask'] = processed['attention_mask'][0]

        # 动态添加视觉相关的字段
        for k in ['pixel_values', 'image_grid_thw', 'pixel_values_videos', 'video_grid_thw']:
            if k in processed and processed[k] is not None:
                # 有些processor会返回一个列表，我们需要取出里面的张量
                if isinstance(processed[k], list) and len(processed[k]) > 0:
                    item[k] = processed[k][0]
                else:
                    item[k] = processed[k]
        
        return item

# =============================
# Collator
# =============================
class DataCollatorForQwenVL:
    """按批次对齐 input_ids/labels；同时合并视觉张量。

    中文说明：
    - 将不同长度序列进行 pad（使用 tokenizer.pad_token_id）。
    - labels 对齐时 pad 位置填充 IGNORE_INDEX。
    - 视觉张量（pixel_values 等）直接 cat。真实场景中应考虑不同形状的对齐，这里假设 processor 已统一尺寸。
    """
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
                # 向上对齐到给定倍数
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

        # 处理视觉字段（简单 cat，不做复杂对齐）
        vision_keys = ['pixel_values', 'image_grid_thw', 'pixel_values_videos', 'video_grid_thw']
        for k in vision_keys:
            vs = [x[k] for x in batch if k in x]
            if len(vs) > 0:
                if isinstance(vs[0], torch.Tensor):
                    try:
                        batch_out[k] = torch.cat(vs, dim=0)
                    except Exception:
                        # 若形状不匹配，可在此处加入更复杂的 pad 逻辑
                        batch_out[k] = vs
                else:
                    batch_out[k] = vs
        return batch_out

# =============================
# 评估指标（占位）
# =============================

def compute_metrics(eval_pred):
    # 这里默认返回空字典，仅使用 evaluation loss。
    # 后续可扩展：解析 assistant 预测文本并与 ground truth 比较，计算分类准确率。
    return {}

# =============================
# 自定义日志回调
# =============================

class RichLoggingCallback(TrainerCallback):
    """使用 Rich 美化训练日志输出的回调类，支持历史记录和横向表格"""

    def __init__(self):
        self.history = deque(maxlen=3)  # 存储最近3条日志记录

    def on_log(self, args, state, control, logs=None, **kwargs):
        if logs is None:
            return

        # 将当前日志添加到历史记录
        self.history.append(logs.copy())

        # 确定表格类型：训练日志还是评估日志
        is_eval = 'eval_loss' in logs
        first_col_name = "Epoch" if is_eval else "Step"
        first_col_value = f"{state.epoch:.2f}" if is_eval else str(state.global_step)

        # 创建横向表格
        table = Table(title=f"[bold blue]{'评估' if is_eval else '训练'}日志 - {first_col_name} {first_col_value}[/bold blue]",
                     show_header=True, header_style="bold magenta")

        # 定义列
        columns = [first_col_name, "Loss", "Grad Norm", "Learning Rate", "Epoch"]
        for col in columns:
            table.add_column(col, style="cyan", justify="center")

        # 添加表头行（实际上是列名，但我们用作第一行）
        table.add_row(*columns)

        # 获取历史记录（最多3条）
        history_list = list(self.history)
        num_history = len(history_list)

        # 添加最新值行
        if num_history >= 1:
            latest = history_list[-1]
            loss = f"{latest.get('loss', latest.get('eval_loss', 'N/A')):.4f}" if isinstance(latest.get('loss', latest.get('eval_loss')), (int, float)) else "N/A"
            grad_norm = f"{latest.get('grad_norm', 'N/A'):.4f}" if isinstance(latest.get('grad_norm'), (int, float)) else "N/A"
            lr = f"{latest.get('learning_rate', 'N/A'):.2e}" if isinstance(latest.get('learning_rate'), (int, float)) else "N/A"
            epoch = f"{latest.get('epoch', state.epoch or 'N/A'):.2f}" if isinstance(latest.get('epoch', state.epoch), (int, float)) else "N/A"
            step_or_epoch = f"{state.epoch:.2f}" if is_eval else str(state.global_step)
            table.add_row(step_or_epoch, loss, grad_norm, lr, epoch)

        # 添加上一次行
        if num_history >= 2:
            prev = history_list[-2]
            loss = f"{prev.get('loss', prev.get('eval_loss', 'N/A')):.4f}" if isinstance(prev.get('loss', prev.get('eval_loss')), (int, float)) else "N/A"
            grad_norm = f"{prev.get('grad_norm', 'N/A'):.4f}" if isinstance(prev.get('grad_norm'), (int, float)) else "N/A"
            lr = f"{prev.get('learning_rate', 'N/A'):.2e}" if isinstance(prev.get('learning_rate'), (int, float)) else "N/A"
            epoch = f"{prev.get('epoch', state.epoch or 'N/A'):.2f}" if isinstance(prev.get('epoch', state.epoch), (int, float)) else "N/A"
            step_or_epoch = f"{state.epoch:.2f}" if is_eval else str(state.global_step - args.logging_steps)
            table.add_row(step_or_epoch, loss, grad_norm, lr, epoch)

        # 添加上上次行
        if num_history >= 3:
            prev_prev = history_list[-3]
            loss = f"{prev_prev.get('loss', prev_prev.get('eval_loss', 'N/A')):.4f}" if isinstance(prev_prev.get('loss', prev_prev.get('eval_loss')), (int, float)) else "N/A"
            grad_norm = f"{prev_prev.get('grad_norm', 'N/A'):.4f}" if isinstance(prev_prev.get('grad_norm'), (int, float)) else "N/A"
            lr = f"{prev_prev.get('learning_rate', 'N/A'):.2e}" if isinstance(prev_prev.get('learning_rate'), (int, float)) else "N/A"
            epoch = f"{prev_prev.get('epoch', state.epoch or 'N/A'):.2f}" if isinstance(prev_prev.get('epoch', state.epoch), (int, float)) else "N/A"
            step_or_epoch = f"{state.epoch:.2f}" if is_eval else str(state.global_step - 2 * args.logging_steps)
            table.add_row(step_or_epoch, loss, grad_norm, lr, epoch)

        # 使用 Panel 包装表格
        panel = Panel(table, border_style="blue")
        rich_print(panel)

# =============================
# 主函数
# =============================

def parse_args():
    parser = argparse.ArgumentParser(description='Qwen-VL 全量参数微调脚本')
    parser.add_argument('--model_name_or_path', type=str, required=True, help='本地已下载的预训练模型目录')
    parser.add_argument('--train_file', type=str, required=True, help='训练集 JSONL 文件路径')
    parser.add_argument('--val_file', type=str, required=False, help='验证集 JSONL 文件路径')
    parser.add_argument('--image_root', type=str, required=True, help='图片根目录（与JSON中相对路径拼接）')
    parser.add_argument('--output_dir', type=str, required=True, help='保存输出的目录')

    # 训练超参数
    parser.add_argument('--per_device_train_batch_size', type=int, default=1)
    parser.add_argument('--per_device_eval_batch_size', type=int, default=1)
    parser.add_argument('--learning_rate', type=float, default=2e-5)
    parser.add_argument('--num_train_epochs', type=int, default=1)
    parser.add_argument('--max_steps', type=int, default=None, help='若设置则覆盖 epoch 训练步数，进行固定步数训练')
    parser.add_argument('--gradient_accumulation_steps', type=int, default=1)
    parser.add_argument('--warmup_steps', type=int, default=0)
    parser.add_argument('--weight_decay', type=float, default=0.0)
    parser.add_argument('--logging_steps', type=int, default=10)
    parser.add_argument('--save_steps', type=int, default=200)
    parser.add_argument('--save_total_limit', type=int, default=3)
    parser.add_argument('--bf16', action='store_true')
    parser.add_argument('--fp16', action='store_true')
    parser.add_argument('--gradient_checkpointing', action='store_true')
    parser.add_argument('--deepspeed', type=str, default=None, help='DeepSpeed配置JSON文件路径')
    parser.add_argument('--attn_implementation', type=str, default='sdpa', choices=['flash_attention_2', 'sdpa', 'eager'])
    parser.add_argument('--eval_strategy', type=str, default=None, choices=['no', 'epoch', 'steps', None], help='评估策略；默认根据是否提供验证集自动选择')
    parser.add_argument('--eval_steps', type=int, default=50, help='当 eval_strategy=steps 时的评估步数')

    parser.add_argument('--model_max_length', type=int, default=512)
    # 视觉 token 预算控制（单位：视觉token数，内部会乘以32*32转为像素预算）
    parser.add_argument('--image_tokens_min', type=int, default=256)
    parser.add_argument('--image_tokens_max', type=int, default=768)
    parser.add_argument('--video_tokens_min', type=int, default=256)
    parser.add_argument('--video_tokens_max', type=int, default=4096)
    parser.add_argument('--video_fps', type=float, default=2.0)
    parser.add_argument('--video_num_frames', type=int, default=None)
    parser.add_argument('--max_train_samples', type=int, default=None, help='调试/烟囱测试：限制训练样本数')
    parser.add_argument('--max_eval_samples', type=int, default=None, help='调试/烟囱测试：限制验证样本数')
    parser.add_argument('--seed', type=int, default=42)

    return parser.parse_args()


def main():
    args = parse_args()

    # 将相对路径解析为基于仓库根的绝对路径，避免依赖当前工作目录
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

    os.makedirs(args.output_dir, exist_ok=True)
    set_seed(args.seed)

    rich_print("[bold blue]正在加载模型和处理器...[/bold blue]")
    # 加载模型/处理器（与官方README一致的精简方式）
    # 说明：使用 AutoModelForImageTextToText + trust_remote_code=True，避免特定 transformers 版本缺少专用类导致的 ImportError。
    # dtype 策略：若未显式指定 fp16/bf16，则采用 "auto"，让HF根据权重与设备自动选择；训练场景不设置 device_map（由Trainer/Accelerate管理）。
    dtype_sel = (
        torch.bfloat16 if args.bf16 else (torch.float16 if args.fp16 else "auto")
    )
    model = AutoModelForImageTextToText.from_pretrained(
        args.model_name_or_path,
        dtype=dtype_sel,
        attn_implementation=args.attn_implementation,
        trust_remote_code=True,
    )
    # 训练时关闭缓存，便于梯度检查点等功能
    if hasattr(model, 'config'):
        model.config.use_cache = False

    # 全量参数训练：确保全部参数可训练
    for p in model.parameters():
        p.requires_grad = True

    processor = AutoProcessor.from_pretrained(args.model_name_or_path, trust_remote_code=True)
    tokenizer = processor.tokenizer
    tokenizer.model_max_length = args.model_max_length
    tokenizer.padding_side = 'right'
    if tokenizer.pad_token is None:
        # 有些模型 pad_token 未设置，使用 eos_token 作为 pad
        tokenizer.pad_token = tokenizer.eos_token

    rich_print("[bold blue]正在构建数据集...[/bold blue]")
    # 构建数据集
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

    # 根据是否提供 val_file 与用户设置，确定评估策略
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
        deepspeed=args.deepspeed,
        eval_strategy=evaluation_strategy,
        eval_steps=args.eval_steps,
        report_to=[],  # 关闭默认W&B等
        gradient_checkpointing=args.gradient_checkpointing,
        dataloader_pin_memory=True,
        dataloader_num_workers=0,
        logging_strategy="steps",  # 确保只在指定步骤记录日志
        disable_tqdm=True,  # 禁用默认进度条，避免与自定义日志冲突
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        data_collator=collator,
        # tokenizer=tokenizer,
        compute_metrics=compute_metrics if eval_dataset is not None else None,
        callbacks=[
            RichLoggingCallback(),  # 自定义日志回调
            SaveProcessorCallback(processor),  # 保存processor配置的回调
        ],
    )

    # =============================
    # 自动断点续训逻辑
    # =============================
    # 1. 检查输出目录下是否存在 checkpoint-* 文件夹
    checkpoint_dirs = glob.glob(os.path.join(args.output_dir, "checkpoint-*"))
    resume_from_checkpoint = None

    if checkpoint_dirs:
        # 2. 找出最新的检查点（按数字编号最大的）
        checkpoint_nums = []
        for ckpt_dir in checkpoint_dirs:
            try:
                # 提取 checkpoint- 后的数字
                num_str = os.path.basename(ckpt_dir).replace("checkpoint-", "")
                num = int(num_str)
                checkpoint_nums.append((num, ckpt_dir))
            except ValueError:
                continue

        if checkpoint_nums:
            # 按数字排序，选最大的
            checkpoint_nums.sort(key=lambda x: x[0], reverse=True)
            latest_checkpoint = checkpoint_nums[0][1]
            resume_from_checkpoint = latest_checkpoint
            rich_print(f"[bold yellow]检测到检查点，从 {latest_checkpoint} 恢复训练...[/bold yellow]")
        else:
            rich_print("[bold blue]未找到有效的检查点，开始全新训练...[/bold blue]")
    else:
        rich_print("[bold blue]未找到检查点，开始全新训练...[/bold blue]")

    # 3. 开始训练，如果有检查点则自动续训
    rich_print("[bold green]训练开始...[/bold green]")
    trainer.train(resume_from_checkpoint=resume_from_checkpoint)
    if eval_dataset is not None:
        trainer.evaluate()

    # 保存最终模型与处理器
    trainer.save_model(args.output_dir)
    processor.save_pretrained(args.output_dir)

    rich_print(f'[bold green]训练完成: 模型与处理器已保存到 {args.output_dir}[/bold green]')


if __name__ == '__main__':
    main()
