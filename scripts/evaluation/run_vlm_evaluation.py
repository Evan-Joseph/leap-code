#!/usr/bin/env python3
"""
简化的VLM任务规划能力评估脚本
基于VLABench官方架构，充分利用32G显存
"""
import os
import sys
import json
import argparse
import time
import gc
import warnings
import shutil
import traceback
from datetime import datetime
import torch
from PIL import Image
from pathlib import Path

# 设置仓库根路径（脚本位置的上两级为 repo 根），并将 VLABench 根加入环境
REPO_ROOT = Path(__file__).resolve().parents[2]
# 默认结果根目录改为 results
EVA_RESULTS_ROOT = REPO_ROOT / 'results'
BACKUP_EVA_RESULTS_ROOT = REPO_ROOT / 'results' / 'backup'
vlabench_root = REPO_ROOT / 'VLABench' / 'VLABench'
os.environ['VLABENCH_ROOT'] = str(vlabench_root)
sys.path.insert(0, str(REPO_ROOT / 'VLABench'))

# 导入VLABench评估组件
from VLABench.evaluation.evaluator import VLMEvaluator
from VLABench.evaluation.model.vlm.base import BaseVLM, get_ti_list
from VLABench.evaluation.utils import get_final_score

# 创建自定义VLMEvaluator子类，修复get_result_save_path方法
class CustomVLMEvaluator(VLMEvaluator):
    """自定义评估器，修复get_result_save_path方法"""
    
    def get_result_save_path(self, vlm_name, few_shot_num=0, with_CoT=False, eval_dim=None):
        """重写get_result_save_path方法，使用初始化时传入的save_path"""
        # 构造模型特定的保存目录名，包含 few-shot 信息
        suffix = f"_{few_shot_num}shot" if few_shot_num > 0 else ""
        if with_CoT:
            suffix += "_CoT"
        
        # 返回模型特定的保存路径，基于 self.save_path (即维度目录)
        return os.path.join(self.save_path, f"{vlm_name}{suffix}")
    
    def evaluate(self, vlm, task_list=None, save_interval=5, few_shot_num=0, with_CoT=False, eval_dim=None):
        """重写evaluate方法，确保正确传递eval_dim参数"""
        # 如果没有提供eval_dim，使用默认值
        if eval_dim is None:
            eval_dim = "default"
            
        print(Fore.YELLOW + Style.BRIGHT + "\n\nworking on ",end = "")
        print(Fore.BLUE + Style.BRIGHT + vlm.name)

        if task_list is None or len(task_list) == 0:
            task_list = self.eval_tasks
            
        # 使用重写的get_result_save_path方法，传递eval_dim参数
        model_result_save_path = self.get_result_save_path(vlm.name, few_shot_num, with_CoT, eval_dim)
        if not os.path.exists(model_result_save_path):
            os.makedirs(model_result_save_path)

        model_result_output_save_file = os.path.join(model_result_save_path, "output.json")
        if os.path.exists(model_result_output_save_file):
            with open(model_result_output_save_file) as f:
                model_output = json.load(f)
        else:
            model_output = {}
            model_output["benchmeatinfo"] = {}
            model_output["benchmeatinfo"]["existing_num"] = 0
            model_output["benchmeatinfo"]["already_running_time"] = 0

        test_example_list = []
        is_resuming = False
        existing_num = 0
        for task_name in task_list:
            for example_num in range(len(os.listdir(os.path.join(self.data_path, task_name)))):
                if task_name in model_output and str(example_num) in model_output[task_name]:
                    if self.check_filled_output(model_output[task_name][str(example_num)]):
                        is_resuming = True
                        existing_num += 1
                        continue
                test_example_list.append((task_name, str(example_num)))
                
        if len(test_example_list) == 0:
            print(Fore.MAGENTA + Style.BRIGHT + "All examples are already exist")
            return model_output

        if model_output["benchmeatinfo"]["existing_num"] == 0:
            model_output["benchmeatinfo"]["existing_num"] = existing_num
            model_output["benchmeatinfo"]["already_running_time"] = existing_num * 10

        elif model_output["benchmeatinfo"]["existing_num"] != existing_num:
            model_output["benchmeatinfo"]["already_running_time"] = model_output["benchmeatinfo"]["already_running_time"] * existing_num / model_output["benchmeatinfo"]["existing_num"]
            model_output["benchmeatinfo"]["existing_num"] = existing_num
        already_running_time = model_output["benchmeatinfo"]["already_running_time"]

        test_example_num = len(test_example_list)
        working_start_time = time.time()
        working_number = 0
        print(Fore.YELLOW + Style.BRIGHT + "working start at " + time.strftime("%Y-%m-%d %H:%M:%S", time.localtime()))

        if is_resuming: 
            print(Fore.MAGENTA + Style.BRIGHT + "{} example existed. Resume start at task: {}, example: {}".format(existing_num, test_example_list[0][0], test_example_list[0][1]))
        
        for task_name, example_num in test_example_list:
            try:
                if task_name not in model_output:
                    model_output[task_name] = {}
                if example_num not in model_output[task_name]:
                    model_output[task_name][example_num] = {}
                #   answer should be a dict with keys: operation_sequence / format_error
                #   if format_error is not in dict, it means the answer is not valid
                answer = self.get_single_anwer(task_name, example_num, vlm, few_shot_num, with_CoT)
                model_output[task_name][example_num] = answer
            except Exception as e:
                print("\n\nError in task: ", task_name, " example: ", example_num)
                print(e)
                traceback.print_exc()
                new_existing_num = existing_num + working_number
                model_output["benchmeatinfo"]["existing_num"] = new_existing_num
                model_output["benchmeatinfo"]["already_running_time"] = time.time() - working_start_time + already_running_time
                with open(model_result_output_save_file, 'w', encoding="utf-8") as f:
                    json.dump(model_output, f, ensure_ascii=False, indent=4)
                raise e
            
            if len(model_output) % save_interval == 0:
                new_existing_num = existing_num + working_number
                model_output["benchmeatinfo"]["existing_num"] = new_existing_num
                model_output["benchmeatinfo"]["already_running_time"] = time.time() - working_start_time + already_running_time
                with open(model_result_output_save_file, 'w', encoding="utf-8") as f:
                    json.dump(model_output, f, ensure_ascii=False, indent=4)

            working_number += 1

            now_time = time.time()
            total_using_time = now_time - working_start_time + already_running_time
            average_using_time = total_using_time / (working_number + existing_num)
            predict_time = average_using_time * (test_example_num - working_number)
            question_percentage = int((working_number + existing_num) / (test_example_num + existing_num) * 100)
            print(Fore.GREEN + question_percentage*'-',end='', flush=True)
            print(Fore.RED + (100-question_percentage)*'-',end='', flush=True)
            print(Fore.GREEN +  "{:>3}%({:>3}/{:>3})".format(question_percentage, (working_number + existing_num), (test_example_num + existing_num)),end='', flush=True)
            print(Fore.GREEN + " using:{:>3}h{:>2}m{:>2}s, ".format(int(total_using_time/3600), int((total_using_time%3600)/60), int(total_using_time%60)),end='', flush=True)
            print(Fore.GREEN + "remain:{:>3}h{:>2}m{:>2}s".format(int(predict_time/3600), int((predict_time%3600)/60), int(predict_time%60)),end='         \r', flush=True)

        new_existing_num = existing_num + working_number
        model_output["benchmeatinfo"]["existing_num"] = new_existing_num
        model_output["benchmeatinfo"]["already_running_time"] = time.time() - working_start_time + already_running_time
        with open(model_result_output_save_file, 'w', encoding="utf-8") as f:
            json.dump(model_output, f, ensure_ascii=False, indent=4)
        print(Fore.YELLOW + Style.BRIGHT + "working end at " + time.strftime("%Y-%m-%d %H:%M:%S", time.localtime()))
        
        return model_output
    
    def get_final_score_dict(self, vlm_name, few_shot_num=0, with_CoT=False, eval_dim=None):
        """重写get_final_score_dict方法，添加eval_dim参数"""
        # 如果没有提供eval_dim，使用默认值
        if eval_dim is None:
            eval_dim = "default"
            
        output_file = os.path.join(self.get_result_save_path(vlm_name, few_shot_num, with_CoT, eval_dim), "output.json")
        if not os.path.exists(output_file):
            print(Fore.RED + Style.BRIGHT + f"output file not exist for model: {vlm_name}, few_shot_num: {few_shot_num}, with_CoT: {with_CoT}, eval_dim: {eval_dim}")
            return None
        with open(output_file) as f:
            model_output = json.load(f)

        final_score_dict = {}

        for task_name in model_output:
            if task_name == "benchmeatinfo":
                continue
            if task_name not in final_score_dict:
                final_score_dict[task_name] = {}
            for example_num in model_output[task_name]:
                if example_num not in final_score_dict[task_name]:
                    final_score_dict[task_name][example_num] = {}

                if "format_error" in model_output[task_name][example_num]:
                    final_score_dict[task_name][example_num] = {
                        "skill_match_score": 0,
                        "entity_match_score": 0,
                        "skill_with_entity_match_score": 0,
                        "exact_match_score": 0,
                        "total_score": 0
                    }
                    continue
                standard_output = self.load_single_output(task_name, example_num)["skill_sequence"]
                try:
                    model_skill_sequence = model_output[task_name][example_num]["skill_sequence"]
                    dependency = "Sequential" if task_name not in self.seq_independent_task else "Seq-independent"
                    example_score = get_final_score(standard_output, model_skill_sequence, dependency=dependency)
                    final_score_dict[task_name][example_num] = example_score
                except:
                    final_score_dict[task_name][example_num] = {
                        "skill_match_score": 0,
                        "entity_match_score": 0,
                        "skill_with_entity_match_score": 0,
                        "exact_match_score": 0,
                        "total_score": 0
                    }
                
        final_score_dict_save_path = os.path.join(self.get_result_save_path(vlm_name, few_shot_num, with_CoT, eval_dim), "final_score.json")
        with open(final_score_dict_save_path, 'w', encoding="utf-8") as f:
            json.dump(final_score_dict, f, ensure_ascii=False, indent=4)
        return final_score_dict

# 禁用warnings
warnings.filterwarnings("ignore")
try:
    from colorama import Fore, Back, Style, init
    init(autoreset=True)
except ImportError:
    class Fore:
        YELLOW = ""
        BLUE = ""
        GREEN = ""
        RED = ""
        MAGENTA = ""
    class Style:
        BRIGHT = ""


class QwenVLAdapter(BaseVLM):
    """
    Qwen-VL 系列适配器 (支持 Qwen2-VL, Qwen2.5-VL, Qwen3-VL)
    """
    def __init__(self, model_path: str, device: str = "cuda:0", batch_size: int = 4, baseline_path: str = None) -> None:
        self.model_path = model_path
        self.baseline_path = baseline_path
        self.device = device
        self.batch_size = batch_size
        super().__init__()
        
        self.total_gpu_memory = torch.cuda.get_device_properties(device).total_memory / 1024**3 if torch.cuda.is_available() else 0
        
        print(f"{Fore.YELLOW}正在加载 Qwen-VL 模型: {model_path}")
        self._load_model()
        self.model.eval()
        self.print_memory_usage("模型加载完成")
    
    def _load_model(self):
        from transformers import AutoModelForImageTextToText, AutoProcessor
        print(f"{Fore.GREEN}✓ 使用BF16精度{Style.RESET_ALL}")
        
        self.model = AutoModelForImageTextToText.from_pretrained(
            self.model_path,
            torch_dtype=torch.bfloat16,
            trust_remote_code=True,
            device_map="auto"
        )
        self.processor = AutoProcessor.from_pretrained(self.model_path)
        
        # 尝试加载模板
        baseline_template = None
        if self.baseline_path and os.path.exists(self.baseline_path):
            bt_path = os.path.join(self.baseline_path, "chat_template.json")
            if os.path.exists(bt_path):
                try:
                    with open(bt_path, 'r') as f:
                        data = json.load(f)
                        baseline_template = data.get("chat_template")
                        if baseline_template:
                            print(f"{Fore.GREEN}✓ 已成功沿用 Baseline 模板{Style.RESET_ALL}")
                except: pass

        if baseline_template:
            self.processor.chat_template = baseline_template
        elif not hasattr(self.processor, 'chat_template') or self.processor.chat_template is None:
            chat_template_path = os.path.join(self.model_path, "chat_template.json")
            if os.path.exists(chat_template_path):
                with open(chat_template_path, 'r') as f:
                    template_data = json.load(f)
                    if isinstance(template_data, dict) and "chat_template" in template_data:
                        self.processor.chat_template = template_data["chat_template"]
    
    def print_memory_usage(self, prefix: str = ""):
        if not torch.cuda.is_available(): return
        allocated = torch.cuda.memory_allocated(self.device) / 1024**3
        print(f"{prefix} - 显存使用: {allocated:.1f}GB/{self.total_gpu_memory:.1f}GB")
    
    def evaluate(self, input_dict, language="en", with_CoT=False):
        self.print_memory_usage("开始推理")
        ti_list = get_ti_list(input_dict, language, with_CoT=with_CoT)
        
        content = []
        image_list = []
        for ti in ti_list:
            if ti[0] == "text":
                content.append({"type": "text", "text": ti[1]})
            elif ti[0] == "image":
                content.append({"type": "image"})
                image_list.append(Image.open(ti[1]).convert('RGB'))
        
        messages = [{"role": "user", "content": content}]
        text = self.processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        
        inputs = self.processor(text=[text], images=image_list if image_list else None, padding=True, return_tensors="pt")
        inputs = {k: v.to(self.device) for k, v in inputs.items()}
        
        try:
            with torch.no_grad():
                generated_ids = self.model.generate(
                    **inputs, max_new_tokens=512, do_sample=False,
                    pad_token_id=self.processor.tokenizer.eos_token_id
                )
                input_len = inputs["input_ids"].shape[1]
                output_text = self.processor.batch_decode(generated_ids[:, input_len:], skip_special_tokens=True)[0]
            
            output = {"origin_output": output_text}
            try:
                json_data = output_text.split("```json")[1].split("```")[0]
                output["skill_sequence"] = json.loads(json_data)
            except:
                output["format_error"] = "format_error"
        except Exception as e:
            output = {"origin_output": "", "format_error": "inference_error"}
        
        if torch.cuda.is_available(): torch.cuda.empty_cache()
        return output

    def get_name(self):
        return os.path.basename(self.model_path)


class InternVLAdapter(BaseVLM):
    """
    InternVL2 / InternVL2.5 适配器
    """
    def __init__(self, model_path: str, device: str = "cuda:0", batch_size: int = 4, baseline_path: str = None) -> None:
        self.model_path = model_path
        self.device = device
        super().__init__()
        
        print(f"{Fore.YELLOW}正在加载 InternVL 模型: {model_path}")
        from transformers import AutoModel, AutoTokenizer
        self.model = AutoModel.from_pretrained(
            model_path, torch_dtype=torch.bfloat16, low_cpu_mem_usage=True,
            trust_remote_code=True, device_map="auto"
        ).eval()
        self.tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True, use_fast=False)
        
        # 导入必要的转换函数
        import torchvision.transforms as T
        from torchvision.transforms.functional import InterpolationMode
        self.IMAGENET_MEAN = (0.485, 0.456, 0.406)
        self.IMAGENET_STD = (0.229, 0.224, 0.225)
        self.transform = T.Compose([
            T.Lambda(lambda img: img.convert('RGB') if img.mode != 'RGB' else img),
            T.Resize((448, 448), interpolation=InterpolationMode.BICUBIC),
            T.ToTensor(),
            T.Normalize(mean=self.IMAGENET_MEAN, std=self.IMAGENET_STD)
        ])

    def dynamic_preprocess(self, image, min_num=1, max_num=12, image_size=448, use_thumbnail=False):
        orig_width, orig_height = image.size
        aspect_ratio = orig_width / orig_height
        target_ratios = set((i, j) for n in range(min_num, max_num + 1) for i in range(1, n + 1) for j in range(1, n + 1) if i * j <= max_num and i * j >= min_num)
        target_ratios = sorted(target_ratios, key=lambda x: x[0] * x[1])
        
        best_ratio_diff = float('inf')
        best_ratio = (1, 1)
        area = orig_width * orig_height
        for ratio in target_ratios:
            target_aspect_ratio = ratio[0] / ratio[1]
            ratio_diff = abs(aspect_ratio - target_aspect_ratio)
            if ratio_diff < best_ratio_diff:
                best_ratio_diff = ratio_diff
                best_ratio = ratio
            elif ratio_diff == best_ratio_diff:
                if area > 0.5 * image_size * image_size * ratio[0] * ratio[1]:
                    best_ratio = ratio
        
        target_width = image_size * best_ratio[0]
        target_height = image_size * best_ratio[1]
        blocks = best_ratio[0] * best_ratio[1]
        resized_img = image.resize((target_width, target_height))
        processed_images = []
        for i in range(blocks):
            box = ((i % best_ratio[0]) * image_size, (i // best_ratio[0]) * image_size, ((i % best_ratio[0]) + 1) * image_size, ((i // best_ratio[0]) + 1) * image_size)
            processed_images.append(resized_img.crop(box))
        if use_thumbnail and len(processed_images) != 1:
            processed_images.append(image.resize((image_size, image_size)))
        return processed_images

    def evaluate(self, input_dict, language="en", with_CoT=False):
        ti_list = get_ti_list(input_dict, language, with_CoT=with_CoT)
        prompt = ""
        pixel_values_list = []
        
        for ti in ti_list:
            if ti[0] == "text":
                prompt += ti[1]
            elif ti[0] == "image":
                prompt += "<image>\n"
                img = Image.open(ti[1]).convert('RGB')
                patches = self.dynamic_preprocess(img, image_size=448, use_thumbnail=True, max_num=12)
                pixel_values = torch.stack([self.transform(p) for p in patches])
                pixel_values_list.append(pixel_values)
        
        if pixel_values_list:
            pixel_values = torch.cat(pixel_values_list, dim=0).to(torch.bfloat16).to(self.device)
        else:
            pixel_values = None
            
        generation_config = dict(max_new_tokens=512, do_sample=False)
        try:
            response = self.model.chat(self.tokenizer, pixel_values, prompt, generation_config)
            output = {"origin_output": response}
            try:
                json_data = response.split("```json")[1].split("```")[0]
                output["skill_sequence"] = json.loads(json_data)
            except:
                output["format_error"] = "format_error"
        except Exception as e:
            output = {"origin_output": "", "format_error": "inference_error"}
        
        if torch.cuda.is_available(): torch.cuda.empty_cache()
        return output

    def get_name(self):
        return os.path.basename(self.model_path)


class MiniCPMVAdapter(BaseVLM):
    """
    MiniCPM-V 适配器
    """
    def __init__(self, model_path: str, device: str = "cuda:0", batch_size: int = 4, baseline_path: str = None, quantization: str = None) -> None:
        self.model_path = model_path
        self.device = device
        super().__init__()
        
        print(f"{Fore.YELLOW}正在加载 MiniCPM-V 模型: {model_path} (Quantization: {quantization})")
        from transformers import AutoModel, AutoTokenizer, BitsAndBytesConfig
        
        load_params = {
            "trust_remote_code": True,
            "attn_implementation": 'sdpa',
            "torch_dtype": torch.bfloat16
        }
        
        if quantization == "4bit":
            load_params["quantization_config"] = BitsAndBytesConfig(
                load_in_4bit=True,
                bnb_4bit_compute_dtype=torch.bfloat16,
                bnb_4bit_quant_type="nf4",
                bnb_4bit_use_double_quant=True,
            )
            load_params["device_map"] = "auto"
        elif quantization == "8bit":
            load_params["load_in_8bit"] = True
            load_params["device_map"] = "auto"
        else:
            load_params["device_map"] = "auto"

        self.model = AutoModel.from_pretrained(model_path, **load_params).eval()
        if quantization is None:
            self.model = self.model.cuda()
            
        self.tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)

    def evaluate(self, input_dict, language="en", with_CoT=False):
        ti_list = get_ti_list(input_dict, language, with_CoT=with_CoT)
        content = []
        for ti in ti_list:
            if ti[0] == "text":
                content.append(ti[1])
            elif ti[0] == "image":
                content.append(Image.open(ti[1]).convert('RGB'))
        
        msgs = [{'role': 'user', 'content': content}]
        try:
            response = self.model.chat(image=None, msgs=msgs, tokenizer=self.tokenizer)
            output = {"origin_output": response}
            try:
                json_data = response.split("```json")[1].split("```")[0]
                output["skill_sequence"] = json.loads(json_data)
            except:
                output["format_error"] = "format_error"
        except Exception as e:
            output = {"origin_output": "", "format_error": "inference_error"}
            
        if torch.cuda.is_available(): torch.cuda.empty_cache()
        return output

    def get_name(self):
        return os.path.basename(self.model_path)


def get_adapter(model_path, device="cuda:0", batch_size=4, baseline_path=None, quantization=None):
    """模型适配器工厂"""
    model_path_lower = model_path.lower()
    if "internvl" in model_path_lower:
        return InternVLAdapter(model_path, device, batch_size, baseline_path)
    elif "minicpm" in model_path_lower:
        return MiniCPMVAdapter(model_path, device, batch_size, baseline_path, quantization)
    else:
        return QwenVLAdapter(model_path, device, batch_size, baseline_path)


def discover_models(baseline_path: str, checkpoints_dir: str, max_checkpoints: int = None, skip_baseline: bool = False):
    """自动发现所有待评估的模型"""
    import glob
    import re
    
    model_paths = []
    model_names = []
    
    # 添加基线模型
    if not skip_baseline and os.path.exists(baseline_path):
        model_paths.append(baseline_path)
        model_names.append("Baseline")
        print(f"{Fore.GREEN}✓ 发现基线模型: {baseline_path}{Style.RESET_ALL}")
    
    # 扫描checkpoint目录
    if os.path.exists(checkpoints_dir):
        checkpoint_dirs = glob.glob(os.path.join(checkpoints_dir, "checkpoint-*"))
        checkpoint_info = []
        for ckpt_dir in checkpoint_dirs:
            # 修复正则表达式，使用单反斜杠
            match = re.search(r'checkpoint-(\d+)', ckpt_dir)
            if match and os.path.isdir(ckpt_dir):
                step = int(match.group(1))
                checkpoint_info.append((step, ckpt_dir))
        
        # 按步数排序
        checkpoint_info.sort(key=lambda x: x[0])
        
        # 如果max_checkpoints为None，则包含所有checkpoint
        checkpoints_to_include = checkpoint_info if max_checkpoints is None else checkpoint_info[:max_checkpoints]
        
        for step, ckpt_dir in checkpoints_to_include:
            model_paths.append(ckpt_dir)
            model_names.append(f"checkpoint-{step}")
            print(f"{Fore.GREEN}✓ 发现checkpoint: {ckpt_dir}{Style.RESET_ALL}")
        
        if max_checkpoints is None:
            print(f"{Fore.GREEN}✓ 包含所有 {len(checkpoint_info)} 个checkpoints{Style.RESET_ALL}")
        else:
            print(f"{Fore.GREEN}✓ 选择前 {min(max_checkpoints, len(checkpoint_info))} 个checkpoints{Style.RESET_ALL}")
    
    return model_paths, model_names


def prepare_dimension_save_path(dimension: str, model_name: str = None, base_results_dir: str = None):
    """为每个维度准备保存路径。
    
    不再自动备份，以支持断点续传。
    """
    if base_results_dir:
        base_dir = base_results_dir
    else:
        base_dir = str(EVA_RESULTS_ROOT)
        
    dimension_dir = os.path.join(base_dir, dimension)

    # 确保目录存在
    os.makedirs(dimension_dir, exist_ok=True)
    
    if model_name:
        model_dir = os.path.join(dimension_dir, model_name)
        os.makedirs(model_dir, exist_ok=True)
        if os.path.exists(os.path.join(model_dir, "output.json")):
            print(f"{Fore.GREEN}发现已存在的模型结果，将尝试断点续传: {model_dir}{Style.RESET_ALL}")
            
    return dimension_dir


def single_model_evaluate(args, model_name=None):
    """单模型评估"""
    print(f"{Fore.BLUE}{'='*60}")
    print("VLM 任务规划能力评估系统 (基于VLABench)")
    print(f"{'='*60}{Style.RESET_ALL}")
    print(f"评估维度: {args.dimension}")
    print(f"模型路径: {args.model_path}")
    print(f"每任务样本数: {args.num_episodes}")
    print(f"数据路径: {args.data_path}")
    print(f"GPU设备: {args.device}")
    print(f"批次大小: {args.batch_size}")
    print(f"{'='*60}")
    
    # 如果是baseline维度，使用所有可用维度
    if args.dimension == "baseline":
        all_dimensions = ["M&T", "CommenSence", "Semantic", "Spatial", "PhysicsLaw", "Complex"]
        print(f"{Fore.YELLOW}Baseline模式: 将评估所有维度{Style.RESET_ALL}")
        all_results = {}
        
        # 创建模型适配器 (移出循环，只加载一次)
        vlm = get_adapter(
            args.model_path, 
            device=args.device,
            batch_size=args.batch_size,
            baseline_path=args.baseline_model,
            quantization=args.quantization
        )
        
        # 如果提供了model_name参数，则使用它作为模型名称
        if model_name:
            vlm.name = model_name

        for dim in all_dimensions:
            print(f"\n{Fore.CYAN}开始评估维度: {dim}{Style.RESET_ALL}")
            print(f"{'-'*40}")
            
            # 检查数据集路径
            dim_path = os.path.join(args.data_path, dim)
            if not os.path.exists(dim_path):
                print(f"{Fore.RED}警告: 维度路径不存在: {dim_path}，跳过{Style.RESET_ALL}")
                continue
            
            # 获取任务列表，限制最多20个
            all_tasks = [d for d in os.listdir(dim_path) if os.path.isdir(os.path.join(dim_path, d))]
            task_list = all_tasks[:min(len(all_tasks), args.max_tasks)]
            
            print(f"发现 {len(all_tasks)} 个任务，评估前 {len(task_list)} 个: {task_list[:3]}...")
            
            # 为每个维度准备保存路径
            save_path = prepare_dimension_save_path(dim, vlm.name)
            
            # 创建评估器
            evaluator = CustomVLMEvaluator(
                tasks=task_list,
                n_episodes=args.num_episodes,
                data_path=dim_path,
                save_path=save_path,
                language="en"
            )
            
            # 执行评估
            result = evaluator.evaluate(
                vlm,
                save_interval=5,
                few_shot_num=args.few_shot_num,
                with_CoT=args.with_cot,
                eval_dim=dim
            )
            
            # 获取最终评分
            final_scores = evaluator.get_final_score_dict(vlm.name, few_shot_num=args.few_shot_num, with_CoT=args.with_cot, eval_dim=dim)
            all_results[dim] = final_scores
            
            if final_scores:
                print(f"{Fore.GREEN}✓ {dim} 维度评估完成{Style.RESET_ALL}")
            else:
                print(f"{Fore.RED}✗ {dim} 维度评估失败{Style.RESET_ALL}")
            
            # 清理GPU缓存
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
                gc.collect()
        
        # 生成汇总报告
        print(f"\n{Fore.BLUE}{'='*60}")
        print("Baseline评估完成 - 结果汇总")
        print(f"{'='*60}{Style.RESET_ALL}")
        
        success_dims = [dim for dim, result in all_results.items() if result is not None]
        if success_dims:
            print(f"\n{Fore.GREEN}成功评估 {len(success_dims)}/{len(all_dimensions)} 个维度: {', '.join(success_dims)}{Style.RESET_ALL}")
        else:
            print(f"\n{Fore.RED}所有维度评估均失败{Style.RESET_ALL}")
            
        return all_results
    
    # 非baseline模式，正常处理单个维度
    # 检查数据集路径
    dim_path = os.path.join(args.data_path, args.dimension)
    if not os.path.exists(dim_path):
        print(f"{Fore.RED}错误: 评估维度路径不存在: {dim_path}{Style.RESET_ALL}")
        return None
    
    # 获取任务列表，限制最多20个
    all_tasks = [d for d in os.listdir(dim_path) if os.path.isdir(os.path.join(dim_path, d))]
    task_list = all_tasks[:min(len(all_tasks), args.max_tasks)]
    
    print(f"\\n发现 {len(all_tasks)} 个任务，评估前 {len(task_list)} 个: {task_list[:3]}...")
    if len(all_tasks) > len(task_list):
        print(f"{Fore.YELLOW}注意: 由于max_tasks={args.max_tasks}限制，跳过 {len(all_tasks)-len(task_list)} 个任务{Style.RESET_ALL}")
    
    # 创建模型适配器
    vlm = get_adapter(
        args.model_path, 
        device=args.device,
        batch_size=args.batch_size,
        baseline_path=args.baseline_model,
        quantization=args.quantization
    )
    
    # 如果提供了model_name参数，则使用它作为模型名称
    if model_name:
        vlm.name = model_name
    
    # 为每个维度准备保存路径，不使用时间戳
    save_path = prepare_dimension_save_path(args.dimension, vlm.name, args.output_dir)
    
    # 创建评估器
    evaluator = CustomVLMEvaluator(
        tasks=task_list,
        n_episodes=args.num_episodes,
        data_path=dim_path,
        save_path=save_path,
        language="en"
    )
    
    # 执行评估
    print(f"\\n{Fore.YELLOW}{'='*60}")
    print("开始评估...")
    print(f"{'='*60}{Style.RESET_ALL}")
    
    result = evaluator.evaluate(
        vlm,
        save_interval=5,
        few_shot_num=args.few_shot_num,
        with_CoT=args.with_cot,
        eval_dim=args.dimension
    )
    
    # 获取最终评分
    print(f"\n{Fore.YELLOW}{'='*60}")
    print("计算最终评分...")
    print(f"{'='*60}{Style.RESET_ALL}")
    
    final_scores = evaluator.get_final_score_dict(vlm.name, few_shot_num=args.few_shot_num, with_CoT=args.with_cot, eval_dim=args.dimension)
    
    if final_scores:
        print(f"\n{Fore.GREEN}评估完成!{Style.RESET_ALL}")
        print(f"详细结果保存在: {evaluator.save_path}")
    
    print(f"\n{Fore.GREEN}{'='*60}")
    print("评估完成!")
    print(f"{'='*60}{Style.RESET_ALL}")
    
    return final_scores


def batch_evaluate(args):
    """批量评估模式"""
    print(f"{Fore.BLUE}{'='*60}")
    print("批量VLM评估模式")
    print(f"{'='*60}{Style.RESET_ALL}")
    
    # 发现所有模型
    model_paths, model_names = discover_models(
        args.baseline_model,
        args.checkpoints_dir,
        args.max_checkpoints,
        skip_baseline=args.skip_baseline
    )
    
    if not model_paths:
        print(f"{Fore.RED}错误: 未找到任何模型{Style.RESET_ALL}")
        return None
    
    checkpoint_info = "所有" if args.max_checkpoints is None else f"前{args.max_checkpoints}个"
    task_info = f"最多{args.max_tasks}个" if args.max_tasks < 100 else "所有"
    
    print(f"\\n总计发现 {len(model_paths)} 个模型 ({checkpoint_info})")
    print(f"评估维度: {args.dimension}")
    print(f"每任务样本数: {args.num_episodes}")
    print(f"GPU设备: {args.device}")
    print(f"批次大小: {args.batch_size}")
    print(f"任务限制: {task_info}")
    print(f"{'='*60}")
    
    # 为整个批量评估准备一个共享的维度目录
    if args.output_dir:
        dimension_save_path = os.path.join(args.output_dir, args.dimension)
    else:
        dimension_save_path = os.path.join(str(EVA_RESULTS_ROOT), args.dimension)
        
    os.makedirs(dimension_save_path, exist_ok=True)
    
    all_results = []
    for idx, (model_path, model_name) in enumerate(zip(model_paths, model_names), 1):
        print(f"\\n{Fore.YELLOW}[{idx}/{len(model_paths)}] 开始评估: {model_name}{Style.RESET_ALL}")
        print(f"路径: {model_path}")
        print(f"{'-'*60}")
        
        try:
            # 临时修改args
            args.model_path = model_path
            
            # 显存检查
            if torch.cuda.is_available():
                initial_memory = torch.cuda.memory_allocated(args.device) / 1024**3
                print(f"模型加载前显存使用: {initial_memory:.1f}GB")
            
            result = single_model_evaluate(args, model_name)
            
            if result:
                all_results.append({
                    "model_name": model_name,
                    "model_path": model_path,
                    "result": result
                })
                print(f"{Fore.GREEN}✓ {model_name} 评估完成{Style.RESET_ALL}")
            else:
                print(f"{Fore.RED}✗ {model_name} 评估失败{Style.RESET_ALL}")
            
            # 强制清理GPU缓存
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
                gc.collect()
                final_memory = torch.cuda.memory_allocated(args.device) / 1024**3
                print(f"模型清理后显存使用: {final_memory:.1f}GB")
            
        except Exception as e:
            print(f"{Fore.RED}✗ {model_name} 评估出错: {str(e)}{Style.RESET_ALL}")
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
                gc.collect()
            continue
    
    # 生成汇总报告
    print(f"\\n{Fore.BLUE}{'='*60}")
    print("批量评估完成 - 结果汇总")
    print(f"{'='*60}{Style.RESET_ALL}")
    
    if all_results:
        print(f"\\n{Fore.GREEN}所有模型评估完成！共 {len(all_results)} 个模型{Style.RESET_ALL}")
        print(f"结果保存在维度目录: {dimension_save_path}")
    
    return all_results


def main():
    parser = argparse.ArgumentParser(description="简化的VLM任务规划能力评估系统")
    parser.add_argument("--dimension", type=str, default="M&T", 
                        choices=["M&T", "CommenSence", "Semantic", "Spatial", "PhysicsLaw", "Complex", "baseline"],
                        help="评估维度")
    parser.add_argument("--model_path", type=str, default="./models/Qwen3-VL-2B-Instruct",
                        help="模型路径")
    parser.add_argument("--num_episodes", type=int, default=5,
                        help="每个任务评估的样本数")
    parser.add_argument("--data_path", type=str, default="./dataset/vlm_evaluation_v1.0",
                        help="评估数据集路径")
    parser.add_argument("--device", type=str, default="cuda:0",
                        help="GPU设备")
    parser.add_argument("--batch_size", type=int, default=4,
                        help="批次大小 (32G显存建议8-16)")
    parser.add_argument("--batch_mode", action="store_true",
                        help="批量模式：评估基线模型+所有checkpoints")
    parser.add_argument("--baseline_model", type=str, default="./models/Qwen3-VL-2B-Instruct",
                        help="批量模式下的基线模型路径")
    parser.add_argument("--checkpoints_dir", type=str, default="./output",
                        help="批量模式下的checkpoints目录")
    parser.add_argument("--max_checkpoints", type=lambda x: None if x == 'None' else int(x), default=None,
                        help="批量模式下最多评估的checkpoint数量，'None'表示全部")
    parser.add_argument("--max_tasks", type=int, default=20,
                        help="每个维度最多评估的任务数量，默认20个")
    parser.add_argument("--few-shot-num", type=int, default=0, dest="few_shot_num",
                        help="Few-shot样本数量，0表示zero-shot，1表示one-shot，默认0")
    parser.add_argument("--with-cot", action="store_true", dest="with_cot",
                        help="是否使用Chain-of-Thought推理，默认不使用")
    parser.add_argument("--skip-baseline", action="store_true",
                        help="批量模式下跳过基线模型评估")
    parser.add_argument("--quantization", type=str, default=None, choices=["4bit", "8bit"],
                        help="模型量化方式 (仅支持 MiniCPM-V)")
    parser.add_argument("--output_dir", type=str, default=None,
                        help="结果保存的根目录 (例如 results/internvl2.5_2b_0shot)")
    
    args = parser.parse_args()

    # 将相对路径解析为基于仓库根的绝对路径，保证从任意 cwd 调用脚本都能正确找到资源
    def _abs(p: str) -> str:
        if p is None:
            return p
        if os.path.isabs(p):
            return p
        return str((REPO_ROOT / p).resolve())

    args.model_path = _abs(args.model_path)
    args.data_path = _abs(args.data_path)
    args.baseline_model = _abs(args.baseline_model)
    args.checkpoints_dir = _abs(args.checkpoints_dir)
    
    # 导入torch (放在参数解析后面避免import问题)
    import torch
    
    # 根据模式选择评估方式
    if args.batch_mode:
        return batch_evaluate(args)
    else:
        return single_model_evaluate(args)


if __name__ == "__main__":
    main()