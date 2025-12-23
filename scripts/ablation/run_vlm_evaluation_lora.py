#!/usr/bin/env python3
"""
LoRA 消融实验评估脚本
支持加载 PEFT LoRA 适配器进行 VLABench 评估
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

# 设置仓库根路径
REPO_ROOT = Path(__file__).resolve().parents[2]
EVA_RESULTS_ROOT = REPO_ROOT / 'eva_results_lora'
vlabench_root = REPO_ROOT / 'VLABench' / 'VLABench'
os.environ['VLABENCH_ROOT'] = str(vlabench_root)
sys.path.insert(0, str(REPO_ROOT / 'VLABench'))

# 导入VLABench评估组件
from VLABench.evaluation.evaluator import VLMEvaluator
from VLABench.evaluation.model.vlm.base import BaseVLM, get_ti_list
from VLABench.evaluation.utils import get_final_score

# 禁用warnings
warnings.filterwarnings("ignore")
try:
    from colorama import Fore, Back, Style, init
    init(autoreset=True)
except ImportError:
    class DummyColor:
        RESET_ALL = ""
        GREEN = ""
        RED = ""
        YELLOW = ""
        BLUE = ""
        CYAN = ""
        MAGENTA = ""
        BRIGHT = ""
    Fore = DummyColor()
    Style = DummyColor()


class CustomVLMEvaluator(VLMEvaluator):
    """自定义评估器，修复get_result_save_path方法"""
    
    def get_result_save_path(self, vlm_name, few_shot_num=0, with_CoT=False, eval_dim=None):
        """重写get_result_save_path方法"""
        if eval_dim is None:
            eval_dim = "default"
        save_path = os.path.join(self.save_path, vlm_name)
        os.makedirs(save_path, exist_ok=True)
        return save_path
    
    def evaluate(self, vlm, task_list=None, save_interval=5, few_shot_num=0, with_CoT=False, eval_dim=None):
        """重写evaluate方法"""
        if eval_dim is None:
            eval_dim = "default"
            
        print(Fore.YELLOW + Style.BRIGHT + "\n\nworking on ",end = "")
        print(Fore.BLUE + Style.BRIGHT + vlm.name)

        if task_list is None or len(task_list) == 0:
            task_list = self.eval_tasks
            
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
        """重写get_final_score_dict方法"""
        if eval_dim is None:
            eval_dim = "default"
            
        output_file = os.path.join(self.get_result_save_path(vlm_name, few_shot_num, with_CoT, eval_dim), "output.json")
        if not os.path.exists(output_file):
            print(Fore.RED + Style.BRIGHT + f"output file not exist for model: {vlm_name}")
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


class Qwen3VLLoRAAdapter(BaseVLM):
    """
    Qwen3-VL + LoRA 适配器
    支持加载 PEFT LoRA 权重进行评估
    """
    def __init__(self, base_model_path: str, lora_adapter_path: str, model_name: str, device: str = "cuda:0") -> None:
        self.base_model_path = base_model_path
        self.lora_adapter_path = lora_adapter_path
        self.device = device
        self._name = model_name
        super().__init__()
        
        self.total_gpu_memory = torch.cuda.get_device_properties(device).total_memory / 1024**3 if torch.cuda.is_available() else 0
        
        print(f"{Fore.YELLOW}正在加载 LoRA 模型...")
        print(f"基座模型: {base_model_path}")
        print(f"LoRA 适配器: {lora_adapter_path}")
        print(f"目标设备: {device}")
        if torch.cuda.is_available():
            print(f"GPU总显存: {self.total_gpu_memory:.1f} GB")
        print(f"{Style.RESET_ALL}")
        
        self._load_model()
        self.model.eval()
        
        self.print_memory_usage("模型加载完成")
    
    @property
    def name(self):
        return self._name
    
    @name.setter
    def name(self, value):
        self._name = value
    
    def _load_model(self):
        """加载基座模型 + LoRA 适配器"""
        from transformers import AutoModelForImageTextToText, AutoProcessor
        from peft import PeftModel
        
        print(f"{Fore.GREEN}✓ 加载基座模型...{Style.RESET_ALL}")
        
        # 加载基座模型
        base_model = AutoModelForImageTextToText.from_pretrained(
            self.base_model_path,
            torch_dtype=torch.bfloat16,
            trust_remote_code=True,
            device_map="auto"
        )
        
        print(f"{Fore.GREEN}✓ 加载 LoRA 适配器...{Style.RESET_ALL}")
        
        # 加载 LoRA 适配器
        self.model = PeftModel.from_pretrained(base_model, self.lora_adapter_path)
        
        # 可选：合并权重以加速推理
        # self.model = self.model.merge_and_unload()
        
        # 加载处理器
        self.processor = AutoProcessor.from_pretrained(self.base_model_path)
        
        print(f"{Fore.GREEN}✓ LoRA 模型加载完成{Style.RESET_ALL}")
    
    def print_memory_usage(self, prefix: str = ""):
        """打印当前显存使用情况"""
        if not torch.cuda.is_available():
            return
        
        allocated = torch.cuda.memory_allocated(self.device) / 1024**3
        reserved = torch.cuda.memory_reserved(self.device) / 1024**3
        utilization = (allocated / self.total_gpu_memory) * 100
        
        print(f"{prefix} - 显存使用: {allocated:.1f}GB/{self.total_gpu_memory:.1f}GB ({utilization:.1f}%)")
    
    def evaluate(self, input_dict, language="en", with_CoT=False):
        """VLABench标准评估接口"""
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
        
        text = self.processor.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True
        )
        
        inputs = self.processor(
            text=[text],
            images=image_list if image_list else None,
            padding=True,
            return_tensors="pt",
        )
        
        inputs = {k: v.to(self.device) for k, v in inputs.items()}
        
        try:
            with torch.no_grad():
                generated_ids = self.model.generate(
                    **inputs, 
                    max_new_tokens=512,
                    do_sample=True,
                    temperature=0.7,
                    top_p=0.9,
                    pad_token_id=self.processor.tokenizer.eos_token_id
                )
                input_len = inputs["input_ids"].shape[1]
                generated_ids_trimmed = generated_ids[:, input_len:]
                output_text = self.processor.batch_decode(
                    generated_ids_trimmed, 
                    skip_special_tokens=True, 
                    clean_up_tokenization_spaces=False
                )[0]
            
            output = {"origin_output": output_text}
            try:
                json_data = output_text.split("```json")[1].split("```")[0]
                output["skill_sequence"] = json.loads(json_data)
            except:
                output["format_error"] = "format_error"
            
        except Exception as e:
            print(f"{Fore.RED}推理错误: {str(e)[:100]}{Style.RESET_ALL}")
            output = {"origin_output": "", "format_error": "inference_error"}
        
        del inputs, generated_ids, generated_ids_trimmed
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        
        self.print_memory_usage("推理完成")
        
        return output
    
    def get_name(self):
        return self._name


def main():
    parser = argparse.ArgumentParser(description="LoRA 消融实验评估脚本")
    parser.add_argument("--dimension", type=str, required=True,
                        choices=["M&T", "CommonSense", "Semantic", "Spatial", "PhysicalLaw", "Complex"],
                        help="评估维度")
    parser.add_argument("--base_model_path", type=str, required=True,
                        help="基座模型路径")
    parser.add_argument("--lora_adapter_path", type=str, required=True,
                        help="LoRA 适配器路径")
    parser.add_argument("--model_name", type=str, required=True,
                        help="模型名称（用于保存结果）")
    parser.add_argument("--data_path", type=str, default="./dataset/vlm_evaluation_v1.0",
                        help="评估数据集路径")
    parser.add_argument("--output_dir", type=str, required=True,
                        help="评估结果保存目录")
    parser.add_argument("--device", type=str, default="cuda:0",
                        help="GPU设备")
    parser.add_argument("--max_tasks", type=int, default=20,
                        help="每个维度最多评估的任务数量")
    parser.add_argument("--num_episodes", type=int, default=5,
                        help="每个任务评估的样本数")
    
    args = parser.parse_args()
    
    # 转换为绝对路径
    def _abs(p: str) -> str:
        if os.path.isabs(p):
            return p
        return str((REPO_ROOT / p).resolve())
    
    args.base_model_path = _abs(args.base_model_path)
    args.lora_adapter_path = _abs(args.lora_adapter_path)
    args.data_path = _abs(args.data_path)
    args.output_dir = _abs(args.output_dir)
    
    print(f"{Fore.BLUE}{'='*60}")
    print("LoRA 消融实验评估")
    print(f"{'='*60}{Style.RESET_ALL}")
    print(f"评估维度: {args.dimension}")
    print(f"模型名称: {args.model_name}")
    print(f"基座模型: {args.base_model_path}")
    print(f"LoRA 适配器: {args.lora_adapter_path}")
    print(f"输出目录: {args.output_dir}")
    print(f"{'='*60}")
    
    # 检查路径
    dim_path = os.path.join(args.data_path, args.dimension)
    if not os.path.exists(dim_path):
        print(f"{Fore.RED}错误: 维度路径不存在: {dim_path}{Style.RESET_ALL}")
        return None
    
    # 获取任务列表
    all_tasks = [d for d in os.listdir(dim_path) if os.path.isdir(os.path.join(dim_path, d))]
    task_list = all_tasks[:min(len(all_tasks), args.max_tasks)]
    
    print(f"\n发现 {len(all_tasks)} 个任务，评估前 {len(task_list)} 个")
    
    # 创建 LoRA 模型适配器
    vlm = Qwen3VLLoRAAdapter(
        args.base_model_path,
        args.lora_adapter_path,
        args.model_name,
        device=args.device
    )
    
    # 创建评估器
    evaluator = CustomVLMEvaluator(
        tasks=task_list,
        n_episodes=args.num_episodes,
        data_path=dim_path,
        save_path=args.output_dir,
        language="en"
    )
    
    # 执行评估
    print(f"\n{Fore.YELLOW}开始评估...{Style.RESET_ALL}")
    
    result = evaluator.evaluate(
        vlm,
        save_interval=5,
        few_shot_num=0,
        with_CoT=False,
        eval_dim=args.dimension
    )
    
    # 获取最终评分
    final_scores = evaluator.get_final_score_dict(vlm.name, eval_dim=args.dimension)
    
    if final_scores:
        print(f"\n{Fore.GREEN}评估完成!{Style.RESET_ALL}")
        print(f"结果保存在: {args.output_dir}")
    
    # 清理
    del vlm
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        gc.collect()
    
    return final_scores


if __name__ == "__main__":
    main()
