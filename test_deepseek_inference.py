import torch
from transformers import AutoModelForCausalLM, BitsAndBytesConfig
from PIL import Image
import os
import sys

# Add DeepSeek-VL2 to path
sys.path.insert(0, "/root/autodl-tmp/leap-code/DeepSeek-VL2")

from deepseek_vl2.models import DeepseekVLV2Processor, DeepseekVLV2ForCausalLM

def test_inference():
    model_path = "/root/autodl-tmp/leap-code/models/deepseek-vl2-small"
    image_path = "dataset/vlm_evaluation_v1.0/CommenSence/add_condiment_common_sense/example0/input/input.png"
    
    print(f"Loading processor from {model_path}...")
    vl_chat_processor = DeepseekVLV2Processor.from_pretrained(model_path)
    tokenizer = vl_chat_processor.tokenizer

    print(f"Loading model with 4-bit quantization...")
    quantization_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_compute_dtype=torch.bfloat16,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_use_double_quant=True,
    )

    vl_gpt = AutoModelForCausalLM.from_pretrained(
        model_path,
        trust_remote_code=True,
        quantization_config=quantization_config,
        device_map="auto",
        torch_dtype=torch.bfloat16
    )
    vl_gpt.eval()

    # Prepare conversation
    conversation = [
        {
            "role": "<|User|>",
            "content": "<image>\nWhat is in this image? Please describe it briefly.",
            "images": [image_path],
        },
        {"role": "<|Assistant|>", "content": ""},
    ]

    # Load image
    pil_image = Image.open(image_path).convert("RGB")
    
    # Prepare inputs
    prepare_inputs = vl_chat_processor(
        conversations=conversation,
        images=[pil_image],
        force_batchify=True,
        system_prompt=""
    ).to(vl_gpt.device)

    # Run image encoder to get the image embeddings
    print("Running inference...")
    with torch.no_grad():
        inputs_embeds = vl_gpt.prepare_inputs_embeds(**prepare_inputs)
        
        outputs = vl_gpt.language.generate(
            inputs_embeds=inputs_embeds,
            attention_mask=prepare_inputs.attention_mask,
            pad_token_id=tokenizer.eos_token_id,
            bos_token_id=tokenizer.bos_token_id,
            eos_token_id=tokenizer.eos_token_id,
            max_new_tokens=128,
            do_sample=False,
            use_cache=True
        )

    answer = tokenizer.decode(outputs[0].cpu().tolist(), skip_special_tokens=False)
    print("\n" + "="*50)
    print("Model Output:")
    print(answer)
    print("="*50)

if __name__ == "__main__":
    test_inference()
