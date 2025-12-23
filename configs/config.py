#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
统一配置管理模块
避免硬编码路径和参数，便于维护和修改
"""
import os
from pathlib import Path
from typing import Dict, Any, Optional

class WorkspaceConfig:
    """工作区配置类"""
    
    # 自动检测工作区根目录
    ROOT_DIR = Path(__file__).parent.parent
    
    # 核心目录配置
    DATA_ROOT = ROOT_DIR / "data"
    MODELS_ROOT = ROOT_DIR / "models"
    OUTPUT_ROOT = ROOT_DIR / "output"
    RESULTS_ROOT = ROOT_DIR / "results"
    SCRIPTS_ROOT = ROOT_DIR / "scripts"
    VLABENCH_ROOT = ROOT_DIR / "VLABench"
    
    # 脚本子目录
    TRAINING_SCRIPTS = SCRIPTS_ROOT / "training"
    EVALUATION_SCRIPTS = SCRIPTS_ROOT / "evaluation"
    UTILS_SCRIPTS = SCRIPTS_ROOT / "utils"
    DOWNLOAD_SCRIPTS = SCRIPTS_ROOT / "download"
    
    # 数据集配置
    DATASET_VLM_EVAL = DATA_ROOT / "vlm_evaluation_v1.0"
    TRAIN_FILE = DATA_ROOT / "train_151230.jsonl"
    TEST_FILE = DATA_ROOT / "test_16812.jsonl"
    
    # 模型配置
    BASELINE_MODEL = MODELS_ROOT / "Qwen3-VL-2B-Instruct"
    
    # VLABench环境变量设置
    @staticmethod
    def setup_vlabench_env():
        """设置VLABench环境变量"""
        os.environ['VLABENCH_ROOT'] = str(WorkspaceConfig.VLABENCH_ROOT)
    
    # 评估维度配置
    EVALUATION_DIMENSIONS = ["M&T", "CommonSense", "Semantic", "Spatial", "PhysicsLaw", "Complex"]
    
    # 默认评估参数
    DEFAULT_EVAL_CONFIG = {
        "num_episodes": 15,
        "max_tasks": 20,
        "batch_size": 4,
        "device": "cuda:0"
    }
    
    # 默认训练参数
    DEFAULT_TRAIN_CONFIG = {
        "epochs": 2,
        "batch_size": 6,
        "gradient_accumulation": 6,
        "learning_rate": 1.5e-5,
        "warmup_steps": 800,
        "weight_decay": 0.01,
        "logging_steps": 10,
        "save_steps": 200,
        "save_total_limit": 100,
        "seed": 42,
        "attn_implementation": "sdpa",
        "bf16": True,
        "gradient_checkpointing": True,
        "model_max_length": 4096,
        "image_tokens_max": 768
    }
    
    @classmethod
    def ensure_directories(cls):
        """确保所有目录存在"""
        directories = [
            cls.DATA_ROOT,
            cls.MODELS_ROOT, 
            cls.OUTPUT_ROOT,
            cls.RESULTS_ROOT,
            cls.TRAINING_SCRIPTS,
            cls.EVALUATION_SCRIPTS,
            cls.UTILS_SCRIPTS,
            cls.DOWNLOAD_SCRIPTS
        ]
        
        for directory in directories:
            directory.mkdir(parents=True, exist_ok=True)
    
    @classmethod
    def get_absolute_path(cls, relative_path: str) -> str:
        """获取绝对路径"""
        return str(cls.ROOT_DIR / relative_path)
    
    @classmethod
    def update_config(cls, config_dict: Dict[str, Any]) -> Dict[str, Any]:
        """更新配置字典"""
        updated_config = cls.DEFAULT_EVAL_CONFIG.copy()
        updated_config.update(config_dict)
        return updated_config

class ModelConfig:
    """模型配置类"""
    
    # Qwen3VL模型信息
    QWEN3VL_INFO = {
        "name": "Qwen3-VL-2B-Instruct",
        "path": "Qwen/Qwen3-VL-2B-Instruct",
        "size": "2B",
        "type": "vision-language"
    }
    
    # 支持的模型列表
    SUPPORTED_MODELS = {
        "qwen3vl": QWEN3VL_INFO,
        # 可以添加更多模型
    }
    
    @classmethod
    def get_model_info(cls, model_name: str) -> Optional[Dict[str, Any]]:
        """获取模型信息"""
        return cls.SUPPORTED_MODELS.get(model_name.lower())

class EvaluationConfig:
    """评估配置类"""
    
    # 评分指标权重
    METRIC_WEIGHTS = {
        "skill_match_score": 0.4,
        "entity_match_score": 0.2, 
        "skill_with_entity_match_score": 0.3,
        "exact_match_score": 0.1
    }
    
    # 评估任务限制
    MAX_EPISODES_PER_TASK = 50
    MAX_TASKS_PER_DIMENSION = 50
    
    # 输出格式配置
    OUTPUT_FORMATS = {
        "json": True,
        "csv": True,
        "plots": True
    }

# 全局配置实例
config = WorkspaceConfig()
model_config = ModelConfig()
eval_config = EvaluationConfig()

# 初始化目录结构
if __name__ == "__main__":
    WorkspaceConfig.ensure_directories()
    print("✓ 工作区目录结构初始化完成")
    print(f"根目录: {WorkspaceConfig.ROOT_DIR}")
    print(f"配置文件: {__file__}")
