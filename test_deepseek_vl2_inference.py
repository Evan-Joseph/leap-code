#!/usr/bin/env python3
"""
DeepSeek-VL2 推理验证脚本
========================

验证 DeepSeek-VL2 模型是否能正常工作。

使用方法:
    python test_deepseek_vl2_inference.py --model_path ./models/deepseek-vl2-small

硬件要求:
    - 2x 32GB GPU 或 1x 80GB GPU
    - 不支持 4-bit 量化 (MoE 架构限制)

环境要求:
    - transformers==4.38.2
    - 激活 deepseek-vl2-env 环境
"""

import sys
import os
import argparse
from pathlib import Path

# 添加 DeepSeek-VL2 到路径
SCRIPT_DIR = Path(__file__).resolve().parent
DEEPSEEK_VL2_PATH = SCRIPT_DIR / "DeepSeek-VL2"
if DEEPSEEK_VL2_PATH.exists():
    sys.path.insert(0, str(DEEPSEEK_VL2_PATH))

import torch
from PIL import Image


def main(args):
    print("=" * 60)
    print("  DeepSeek-VL2 推理验证")
    print("=" * 60)
    
    # 环境信息
    print(f"\nPython: {sys.version.split()[0]}")
    print(f"PyTorch: {torch.__version__}")
    print(f"CUDA: {torch.cuda.is_available()}")
    if torch.cuda.is_available():
        print(f"GPU 数量: {torch.cuda.device_count()}")
        for i in range(torch.cuda.device_count()):
            props = torch.cuda.get_device_properties(i)
            print(f"  GPU {i}: {props.name} ({props.total_memory / 1024**3:.1f} GB)")
    
    import transformers
    print(f"Transformers: {transformers.__version__}")
    
    if transformers.__version__ != "4.38.2":
        print(f"\n⚠️  警告: 推荐使用 transformers==4.38.2, 当前版本 {transformers.__version__}")
    
    # 导入 DeepSeek-VL2
    print("\n[1/4] 导入模块...")
    try:
        from deepseek_vl2.models import DeepseekVLV2Processor
        from transformers import AutoModelForCausalLM
        print("✅ 模块导入成功")
    except ImportError as e:
        print(f"❌ 导入失败: {e}")
        print("请确保已克隆 DeepSeek-VL2 仓库并安装依赖")
        return 1
    
    # 加载 Processor
    print("\n[2/4] 加载 Processor...")
    processor = DeepseekVLV2Processor.from_pretrained(args.model_path)
    tokenizer = processor.tokenizer
    print("✅ Processor 加载成功")
    
    # 加载模型
    print("\n[3/4] 加载模型 (bfloat16, 自动分配多卡)...")
    model = AutoModelForCausalLM.from_pretrained(
        args.model_path,
        trust_remote_code=True,
        torch_dtype=torch.bfloat16,
        device_map="auto",
    )
    model.eval()
    print("✅ 模型加载成功")
    
    # 显存使用
    if torch.cuda.is_available():
        for i in range(torch.cuda.device_count()):
            allocated = torch.cuda.memory_allocated(i) / 1024**3
            total = torch.cuda.get_device_properties(i).total_memory / 1024**3
            print(f"   GPU {i}: {allocated:.1f}/{total:.1f} GB")
    
    # 测试推理
    print("\n[4/4] 测试推理...")
    
    # 创建测试图片
    test_image = Image.new("RGB", (384, 384), color=(100, 150, 200))
    test_image_path = "/tmp/deepseek_test.jpg"
    test_image.save(test_image_path)
    
    conversation = [
        {
            "role": "<|User|>",
            "content": "<image>\nDescribe this image in one sentence.",
            "images": [test_image_path],
        },
        {"role": "<|Assistant|>", "content": ""},
    ]
    
    inputs = processor(
        conversations=conversation,
        images=[test_image],
        force_batchify=True,
        system_prompt=""
    ).to(model.device)
    
    with torch.no_grad():
        inputs_embeds = model.prepare_inputs_embeds(**inputs)
        
        outputs = model.language.generate(
            inputs_embeds=inputs_embeds,
            attention_mask=inputs.attention_mask,
            pad_token_id=tokenizer.eos_token_id,
            max_new_tokens=50,
            do_sample=False,
        )
    
    response = tokenizer.decode(outputs[0].cpu().tolist(), skip_special_tokens=True)
    print("✅ 推理成功!")
    print(f"\n模型输出:\n{response}\n")
    
    print("=" * 60)
    print("✅ 所有测试通过! DeepSeek-VL2 可以正常工作。")
    print("=" * 60)
    
    return 0


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="DeepSeek-VL2 推理验证")
    parser.add_argument(
        "--model_path",
        type=str,
        default="./models/deepseek-vl2-small",
        help="模型路径"
    )
    args = parser.parse_args()
    
    exit(main(args))
