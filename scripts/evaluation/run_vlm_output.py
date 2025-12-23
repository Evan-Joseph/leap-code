#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
对 blind_10.jsonl 的快速输出脚本：
- 读取 data/blind_10.jsonl 的10条图文+提示词
- 对 Baseline (./models/Qwen3-VL-2B-Instruct) 以及 ./output 下所有 checkpoint-* 逐一推理
- 将 输入图文信息、模型输出、参考答案 汇总到一个 CSV

依赖：
- transformers >= 4.42
- torch
- pillow

使用示例：
python scripts/evaluation/run_vlm_output.py \
  --data_file ./data/blind_10.jsonl \
  --baseline_model ./models/Qwen3-VL-2B-Instruct \
  --checkpoints_dir ./output \
  --save_dir ./eva_results/blind_10 \
  --device cuda:0

可选：
  --max_checkpoints None    # 评估所有ckpt（默认）
  --max_checkpoints 5       # 只评估前5个步数最小的ckpt
  --max_models 6            # 同时并驻留显存中的模型数量上限（默认1，按需提高）
"""

import os
import re
import csv
import gc
import json
import glob
import argparse
from datetime import datetime
from typing import List, Dict, Any

import torch
from PIL import Image
from transformers import AutoModelForImageTextToText, AutoProcessor
from pathlib import Path


def discover_models(baseline_path: str, checkpoints_dir: str, max_checkpoints: int | None = None):
    """发现 Baseline + checkpoints。
    返回两个列表：model_paths, model_names（顺序一致）。
    """
    model_paths: List[str] = []
    model_names: List[str] = []

    # Baseline
    if baseline_path and os.path.exists(baseline_path):
        model_paths.append(baseline_path)
        model_names.append("Baseline")
        print(f"✓ 发现基线模型: {baseline_path}")
    else:
        print("! 警告：未发现基线模型，跳过 Baseline")

    # checkpoints
    if checkpoints_dir and os.path.isdir(checkpoints_dir):
        ckpt_dirs = [d for d in glob.glob(os.path.join(checkpoints_dir, "checkpoint-*")) if os.path.isdir(d)]
        pairs = []
        for d in ckpt_dirs:
            m = re.search(r"checkpoint-(\d+)", os.path.basename(d))
            if m:
                step = int(m.group(1))
                pairs.append((step, d))
        pairs.sort(key=lambda x: x[0])
        if max_checkpoints is not None:
            pairs = pairs[:max_checkpoints]
        for step, d in pairs:
            model_paths.append(d)
            model_names.append(f"checkpoint-{step}")
        print(f"✓ 发现 {len(pairs)} 个 checkpoints 于 {checkpoints_dir}")
    else:
        print("! 警告：未发现 checkpoints 目录，跳过")

    return model_paths, model_names


class Qwen3VLInfer:
    """最小化推理封装，适配 Qwen3-VL 系列 AutoModelForImageTextToText。
    """

    def __init__(self, model_path: str, device: str = "cuda:0") -> None:
        self.model_path = model_path
        self.device = device
        print(f"\n加载模型: {model_path}")
        self._load()
        self.model.eval()

    def _load(self):
        # bfloat16 + 自动映射到可用设备
        self.model = AutoModelForImageTextToText.from_pretrained(
            self.model_path,
            torch_dtype=torch.bfloat16,
            trust_remote_code=True,
            device_map="auto",
        )
        self.processor = AutoProcessor.from_pretrained(self.model_path, trust_remote_code=True)

    @torch.inference_mode()
    def generate_from_messages(self, messages: List[Dict[str, Any]],
                               max_new_tokens: int = 256,
                               temperature: float = 0.7,
                               top_p: float = 0.9) -> str:
        """
        参数 messages: 形如 blind_10.jsonl 中的对话结构，只需包含单轮 user，内容含若干 image/text。
        返回生成的文本。
        """
        # 组装 Qwen3-VL 所需的 content 与 images 列表
        assert len(messages) >= 1, "messages 至少包含一轮"
        content = []
        image_list = []

        for msg in messages:
            if msg.get("role") != "user":
                # 本脚本只处理 user 提示
                continue
            for item in msg.get("content", []):
                if item.get("type") == "text":
                    content.append({"type": "text", "text": item.get("text", "")})
                elif item.get("type") == "image":
                    img_path = item.get("image")
                    if img_path is None:
                        continue
                    # 转绝对路径：基于仓库根解析相对路径，避免依赖当前工作目录
                    abs_path = img_path
                    if not os.path.isabs(abs_path):
                        abs_path = str((Path(__file__).resolve().parents[2] / img_path).resolve())
                    if not os.path.exists(abs_path):
                        raise FileNotFoundError(f"找不到图像文件: {abs_path}")
                    image_list.append(Image.open(abs_path).convert("RGB"))
                    content.append({"type": "image"})

        # 包装成对话
        conv = [{"role": "user", "content": content}]

        # 模板与处理
        text = self.processor.apply_chat_template(conv, tokenize=False, add_generation_prompt=True)
        inputs = self.processor(text=[text], images=image_list if image_list else None,
                                padding=True, return_tensors="pt")
        # 放到设备（当 device_map=auto 时，尽量不手动 .to，但输入张量仍需到首要设备）
        inputs = {k: v.to(self.model.device) if hasattr(self.model, "device") else v for k, v in inputs.items()}

        # 生成
        generated_ids = self.model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            do_sample=True,
            temperature=temperature,
            top_p=top_p,
            pad_token_id=self.processor.tokenizer.eos_token_id,
        )
        # 去除提示部分
        input_len = inputs["input_ids"].shape[1]
        gen_ids = generated_ids[:, input_len:]
        out_text = self.processor.batch_decode(gen_ids, skip_special_tokens=True, clean_up_tokenization_spaces=False)[0]
        return out_text

    def unload(self):
        try:
            del self.model
            del self.processor
        except Exception:
            pass
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        gc.collect()


def read_jsonl(path: str) -> List[Dict[str, Any]]:
    data = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            data.append(json.loads(line))
    return data


def ensure_dir(path: str):
    os.makedirs(path, exist_ok=True)


def sanitize_csv_field(text: Any) -> str:
    """将任意字段转为不含换行的字符串，以避免在CSV中出现换行。
    - 替换 \r\n、\n、\r 为空格
    - 去除首尾空白
    - 将非字符串安全转换为字符串
    """
    if text is None:
        return ""
    s = str(text)
    s = s.replace("\r\n", " ").replace("\n", " ").replace("\r", " ")
    return s.strip()


def chunk_list(lst, n):
    """将列表按 n 个一组分块。"""
    for i in range(0, len(lst), n):
        yield lst[i:i + n]


def main():
    parser = argparse.ArgumentParser(description="blind_10.jsonl 多模型输出到CSV")
    parser.add_argument("--data_file", type=str, default="./data/blind_10.jsonl", help="jsonl 数据文件路径")
    parser.add_argument("--baseline_model", type=str, default="./models/Qwen3-VL-2B-Instruct", help="Baseline 模型路径")
    parser.add_argument("--checkpoints_dir", type=str, default="./output", help="checkpoints 目录")
    parser.add_argument("--save_dir", type=str, default="./eva_results/blind_10", help="CSV 保存目录")
    parser.add_argument("--device", type=str, default="cuda:0", help="推理设备标识")
    parser.add_argument("--max_checkpoints", type=lambda x: None if x == 'None' else int(x), default=None, help="最多评估多少个ckpt，None表示全部")
    parser.add_argument("--max_new_tokens", type=int, default=256)
    parser.add_argument("--temperature", type=float, default=0.7)
    parser.add_argument("--top_p", type=float, default=0.9)
    parser.add_argument("--max_models", type=int, default=1, help="同时常驻显存的模型数量上限（根据显存大小设置，例如 6）")
    args = parser.parse_args()

    # 将相对路径解析为基于仓库根的绝对路径，保证从任意 cwd 调用脚本都能正确找到资源
    REPO_ROOT = Path(__file__).resolve().parents[2]
    def _abs(p: str) -> str:
        if p is None:
            return p
        if os.path.isabs(p):
            return p
        return str((REPO_ROOT / p).resolve())

    args.data_file = _abs(args.data_file)
    args.baseline_model = _abs(args.baseline_model)
    args.checkpoints_dir = _abs(args.checkpoints_dir)
    args.save_dir = _abs(args.save_dir)

    ensure_dir(args.save_dir)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    out_csv = os.path.join(args.save_dir, f"blind10_outputs_{timestamp}.csv")

    # 发现模型
    model_paths, model_names = discover_models(args.baseline_model, args.checkpoints_dir, args.max_checkpoints)
    if not model_paths:
        print("未发现任何模型，退出。")
        return

    # 读取数据
    items = read_jsonl(args.data_file)
    print(f"读取到 {len(items)} 条数据。")

    # 写 CSV 头
    fieldnames = [
        "model_name",
        "sample_idx",
        "image_paths",
        "prompt_text",
        "model_output",
        "reference_steps",
    ]
    with open(out_csv, "w", encoding="utf-8", newline="") as fw:
        writer = csv.DictWriter(fw, fieldnames=fieldnames)
        writer.writeheader()
        # 分批同时加载多个模型以充分利用显存
        max_models = max(1, int(args.max_models))
        total_models = len(model_paths)
        batch_id = 0

        for batch_paths in chunk_list(model_paths, max_models):
            batch_names = model_names[batch_id * max_models: batch_id * max_models + len(batch_paths)]
            batch_id += 1
            print(f"\n加载模型批次 {batch_id}: {len(batch_paths)}/{total_models} 模型常驻显存")

            # 加载本批模型
            infers: List[Qwen3VLInfer] = []
            for m_path, m_name in zip(batch_paths, batch_names):
                print(f"尝试加载: {m_name} -> {m_path}")
                try:
                    infers.append(Qwen3VLInfer(m_path, device=args.device))
                except Exception as e:
                    print(f"模型加载失败，跳过 {m_name}: {e}")

            if not infers:
                print("本批无可用模型，继续下一批。")
                continue

            # 对每条样本，逐个模型推理并写出
            for idx, example in enumerate(items):
                messages = example.get("messages", [])
                target = example.get("target", {})
                # 收集输入文本与图片路径（便于落盘）
                img_paths: List[str] = []
                prompt_text_parts: List[str] = []
                if messages:
                    for msg in messages:
                        if msg.get("role") != "user":
                            continue
                        for c in msg.get("content", []):
                            if c.get("type") == "image" and c.get("image"):
                                img_paths.append(c["image"])
                            elif c.get("type") == "text" and c.get("text"):
                                prompt_text_parts.append(c["text"])
                prompt_text = "\n\n".join(prompt_text_parts)
                prompt_text = sanitize_csv_field(prompt_text)

                for infer, m_name in zip(infers, batch_names):
                    try:
                        output_text = infer.generate_from_messages(
                            messages,
                            max_new_tokens=args.max_new_tokens,
                            temperature=args.temperature,
                            top_p=args.top_p,
                        )
                    except Exception as e:
                        output_text = f"<inference_error>: {e}"

                    row = {
                        "model_name": sanitize_csv_field(m_name),
                        "sample_idx": idx,
                        "image_paths": sanitize_csv_field(";".join(img_paths)),
                        "prompt_text": prompt_text,
                        "model_output": sanitize_csv_field(output_text),
                        "reference_steps": sanitize_csv_field(target.get("steps", "")),
                    }
                    writer.writerow(row)

            # 卸载本批次全部模型以释放显存
            for infer in infers:
                try:
                    infer.unload()
                except Exception:
                    pass

    print(f"\n已保存 CSV: {out_csv}")


if __name__ == "__main__":
    main()
