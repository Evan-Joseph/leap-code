#!/usr/bin/env python3
"""
VLM评估进度监控脚本
使用rich库实时显示6个维度的评估进度
"""

import os
import re
import time
import threading
from datetime import datetime
from pathlib import Path
from typing import Optional

from rich.console import Console
from rich.layout import Layout
from rich.panel import Panel
from rich.progress import Progress, BarColumn, TextColumn, TimeRemainingColumn
from rich.table import Table
from rich.live import Live
from rich.text import Text
from rich.align import Align


class EvaluationMonitor:
    def __init__(self, log_dir: Optional[Path] = None):
        self.console = Console()
        repo_root = Path(os.environ.get("PROJECT_ROOT", Path(__file__).resolve().parent))
        default_logs = repo_root / "logs"
        self.log_dir = Path(os.environ.get("PROJECT_LOG_DIR", log_dir or default_logs))
        self.dimensions = {
            "M&T": "log_mt.log",
            "CommonSense": "log_commonsense.log",  # 修正拼写
            "Semantic": "log_semantic.log",
            "Spatial": "log_spatial.log",
            "PhysicalLaw": "log_physics.log",     # 修正拼写
            "Complex": "log_complex.log"
        }
        self.status = {}
        self.init_status()
        
    def init_status(self):
        """初始化各维度状态"""
        for dim in self.dimensions:
            self.status[dim] = {
                "model": "Unknown",
                "current": 0,
                "total": 100,
                "percent": 0.0,
                "eta": "Unknown",
                "memory": "Unknown",
                "start_time": None,
                "last_update": None,
                "is_running": False,
                "tasks": []
            }
    
    def parse_log_file(self, log_file):
        """解析日志文件，提取评估状态信息"""
        if not os.path.exists(log_file):
            return None
            
        try:
            with open(log_file, 'r', encoding='utf-8') as f:
                lines = f.readlines()
                
            # 获取最后几行用于状态检测
            last_lines = lines[-20:] if len(lines) > 20 else lines
            content = ''.join(lines)
            recent_content = ''.join(last_lines)
                
            # 提取模型名称 - 优先使用"working on "后面的内容
            model = "Unknown"
            
            # 首先尝试查找所有"working on "行，并使用最后一个匹配
            working_on_matches = re.findall(r'working on (.+)', content)
            if working_on_matches:
                model = working_on_matches[-1]  # 使用最后一个匹配项
            else:
                # 如果没有找到"working on"，尝试其他模式
                model_patterns = [
                    r'\[\d+/\d+\] 开始评估: (.+)',  # 批量模式的模型标识
                    r'模型路径: (.+)',                # 模型路径
                    r'checkpoint-(\d+)',             # 直接匹配checkpoint
                    r'开始评估: (.+)'                 # 其他可能的评估开始行
                ]
                
                for pattern in model_patterns:
                    match = re.search(pattern, content)
                    if match:
                        model = match.group(1)
                        # 如果是路径，只取文件名部分
                        if '/' in model or '\\' in model:
                            model = os.path.basename(model)
                        break
            
            # 提取开始时间
            start_time_match = re.search(r'working start at (.+)', content)
            start_time = start_time_match.group(1) if start_time_match else None
            
            # 提取任务列表
            tasks_match = re.search(r'发现 (\d+) 个任务，评估前 \d+ 个: \[(.+?)\]', content)
            tasks = []
            if tasks_match:
                task_str = tasks_match.group(2)
                tasks = [t.strip().strip("'") for t in task_str.split(",")]
            
            # 提取进度信息 - 查找最后一行进度
            progress_pattern = r'(\d+)%\(\s*(\d+)/(\d+)\).*?remain:\s*(.+?)\s*$'
            progress_matches = re.findall(progress_pattern, content, re.MULTILINE)
            
            current = 0
            total = 100
            percent = 0.0
            eta = "Unknown"
            
            if progress_matches:
                # 获取最后一个匹配项（最新进度）
                last_match = progress_matches[-1]
                percent = float(last_match[0])
                current = int(last_match[1])
                total = int(last_match[2])
                eta = last_match[3].strip()
            
            # 检查是否正在运行 - 改进状态检测逻辑
            is_running = False
            
            # 检查最近的日志中是否有推理活动
            recent_inference = "推理完成" in recent_content or "开始推理" in recent_content
            
            # 检查是否有进度信息
            if progress_matches:
                # 如果进度小于100%，认为正在运行
                is_running = percent < 100 and recent_inference
            else:
                # 如果没有进度信息但有推理完成记录，认为正在运行
                is_running = recent_inference
            
            # 检查是否已完成
            if percent >= 100:
                is_running = False
            
            # 检查是否有错误或异常终止
            if "Traceback" in recent_content or "Error" in recent_content:
                is_running = False
            
            return {
                "model": model,
                "current": current,
                "total": total,
                "percent": percent,
                "eta": eta,
                "start_time": start_time,
                "is_running": is_running,
                "tasks": tasks[:5]  # 只显示前5个任务
            }
        except Exception as e:
            self.console.print(f"解析日志文件 {log_file} 时出错: {e}", style="red")
            return None
    
    def update_status(self):
        """更新所有维度的状态"""
        for dim, log_file in self.dimensions.items():
            log_path = self.log_dir / log_file
            data = self.parse_log_file(log_path)
            if data:
                self.status[dim].update(data)
                self.status[dim]["last_update"] = datetime.now().strftime("%H:%M:%S")
    
    def create_dimension_table(self, dimension):
        """为单个维度创建状态表格"""
        status = self.status[dimension]
        
        # 状态颜色
        if status["is_running"]:
            status_color = "green"
            status_text = "运行中"
        elif status["percent"] >= 100:
            status_color = "blue"
            status_text = "已完成"
        else:
            status_color = "yellow"
            status_text = "未知"
        
        # 创建表格
        table = Table(show_header=False, box=None, padding=0)
        table.add_column("Property", style="bold cyan")
        table.add_column("Value")
        
        table.add_row("状态:", f"[{status_color}]{status_text}[/{status_color}]")
        table.add_row("模型:", status["model"])
        table.add_row("进度:", f"{status['percent']:.1f}% ({status['current']}/{status['total']})")
        table.add_row("预计剩余:", status["eta"])
        
        if status["start_time"]:
            table.add_row("开始时间:", status["start_time"])
        
        if status["tasks"]:
            task_list = ", ".join(status["tasks"][:3])
            if len(status["tasks"]) > 3:
                task_list += "..."
            table.add_row("任务:", task_list)
        
        return Panel(
            table,
            title=f"[bold]{dimension}[/bold]",
            border_style=status_color,
            padding=(1, 1)
        )
    
    def create_layout(self):
        """创建整体布局"""
        layout = Layout()
        
        # 创建标题 - 增加高度并添加系统信息
        current_time = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        
        # 获取GPU信息
        try:
            import subprocess
            result = subprocess.run(['nvidia-smi', '--query-gpu=memory.used,memory.total,utilization.gpu', 
                                   '--format=csv,noheader,nounits'], 
                                  capture_output=True, text=True)
            if result.returncode == 0:
                gpu_info = result.stdout.strip().split(', ')
                memory_used = gpu_info[0]
                memory_total = gpu_info[1]
                gpu_util = gpu_info[2]
                gpu_text = f"GPU: {gpu_util}% | 显存: {memory_used}/{memory_total}MB"
            else:
                gpu_text = "GPU信息获取失败"
        except Exception:
            gpu_text = "GPU信息不可用"
        
        title_content = Table.grid(padding=0)
        title_content.add_column(justify="center")
        title_content.add_row(
            Text(f"{gpu_text} | {current_time}", style="white")
        )
        
        title = Panel(
            Align.center(title_content),
            padding=(1, 1)
        )
        
        # 创建维度表格 - 每行两个维度，共三行
        top_row = Layout()
        top_row.split_column(
            Layout(self.create_dimension_table("M&T")),
            Layout(self.create_dimension_table("CommonSense"))  # 修正拼写
        )
        
        middle_row = Layout()
        middle_row.split_column(
            Layout(self.create_dimension_table("Semantic")),
            Layout(self.create_dimension_table("Spatial"))
        )
        
        bottom_row = Layout()
        bottom_row.split_column(
            Layout(self.create_dimension_table("PhysicalLaw")),  # 修正拼写
            Layout(self.create_dimension_table("Complex"))
        )
        
        dimensions_layout = Layout()
        dimensions_layout.split_row(
            Layout(top_row),
            Layout(middle_row),
            Layout(bottom_row)
        )
        
        # 组合布局 - 增加标题栏高度
        layout.split_column(
            Layout(title, size=5),  # 增加标题栏高度
            Layout(dimensions_layout)
        )
        
        return layout
    
    def run(self):
        """运行监控"""
        with Live(self.create_layout(), refresh_per_second=1, screen=True) as live:
            try:
                while True:
                    self.update_status()
                    live.update(self.create_layout())
                    time.sleep(1)
            except KeyboardInterrupt:
                self.console.print("\n监控已停止", style="bold green")


if __name__ == "__main__":
    monitor = EvaluationMonitor()
    monitor.run()