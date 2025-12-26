import os
import time
import re
from rich.live import Live
from rich.table import Table
from rich.panel import Panel
from rich.console import Console
from rich.progress import Progress, BarColumn, TextColumn
from rich.layout import Layout
from pathlib import Path

console = Console()

def get_gpu_info():
    try:
        import subprocess
        res = subprocess.check_output(["nvidia-smi", "--query-gpu=memory.used,memory.total,utilization.gpu", "--format=csv,noheader,nounits"])
        info = res.decode().strip().split(",")
        used = info[0].strip()
        total = info[1].strip()
        util = info[2].strip()
        return f"{used}MB / {total}MB", f"{util}%"
    except:
        return "N/A", "N/A"

def get_progress_from_log(log_file):
    if not os.path.exists(log_file):
        return 0, 0, "等待中", "N/A", "N/A"
    
    try:
        with open(log_file, 'r') as f:
            lines = f.readlines()
            
        # 查找当前维度 (从后往前找全文)
        current_dim = "初始化"
        for line in reversed(lines):
            if "评估维度:" in line or "开始评估维度:" in line:
                # 兼容两种打印格式
                parts = line.split("维度:")
                if len(parts) > 1:
                    current_dim = parts[1].strip().replace("-", "")
                    # 去除颜色代码 (如果有)
                    current_dim = re.sub(r'\x1b\[[0-9;]*m', '', current_dim)
                    break
            elif "working on" in line and current_dim == "初始化":
                # 兜底逻辑：如果还没看到维度开始，可能正在加载模型
                current_dim = "加载中"
        
        content = "".join(lines[-100:]) # 增加搜索范围到最后100行
        
        # 查找进度百分比和剩余时间
        # 匹配格式: 0%(  4/900) using:  0h 0m 7s, remain:  0h26m40s
        matches = re.findall(r'(\d+)%\(\s*(\d+)/(\d+)\).*?remain:\s*([\dhms\s:]+)', content)
        if matches:
            last_match = matches[-1]
            percent = int(last_match[0])
            current = int(last_match[1])
            total = int(last_match[2])
            eta = last_match[3].strip()
            return current, total, f"{percent}%", current_dim, eta
        
        if "评估完成!" in content or "Baseline评估完成" in content:
            return 100, 100, "已完成", "全部", "0s"
            
        return 0, 0, "运行中", current_dim, "计算中..."
    except Exception as e:
        return 0, 0, "读取错误", "N/A", str(e)

def get_latest_log_files():
    """动态获取最近活跃的日志文件列表"""
    logs_root = Path("/root/autodl-tmp/leap-code/logs")
    if not logs_root.exists():
        return []
    
    # 获取所有子目录
    subdirs = [d for d in logs_root.iterdir() if d.is_dir()]
    if not subdirs:
        return []
    
    # 按修改时间排序，取最新的一个
    latest_dir = max(subdirs, key=lambda d: d.stat().st_mtime)
    
    # 获取该目录下所有的 .log 文件
    log_files = sorted(list(latest_dir.glob("*.log")))
    return [str(f) for f in log_files]

def main():
    with Live(console=console, refresh_per_second=1, vertical_overflow="visible") as live:
        while True:
            mem, util = get_gpu_info()
            
            # 动态获取日志文件
            log_files = get_latest_log_files()
            
            table = Table(show_header=True, header_style="bold magenta", box=None)
            table.add_column("任务", style="white", width=15)
            table.add_column("当前维度", style="cyan", width=15)
            table.add_column("进度", style="green", width=15)
            table.add_column("样本数", style="yellow", width=12)
            table.add_column("剩余时间 (ETA)", style="bold red", width=20)
            
            if not log_files:
                table.add_row("无活跃任务", "-", "-", "-", "-")
            else:
                for log_file in log_files:
                    current, total, status, dim, eta = get_progress_from_log(log_file)
                    # 优化显示：如果已完成，进度显示 100%
                    display_status = status if "%" in status else ("100%" if status == "已完成" else status)
                    table.add_row(
                        Path(log_file).stem.upper(),
                        dim, 
                        display_status,
                        f"{current}/{total}",
                        eta
                    )
            
            # 组合面板，减少空白
            summary_text = f"[bold green]GPU利用率:[/bold green] {util}  |  [bold blue]显存占用:[/bold blue] {mem}"
            
            main_panel = Panel(
                table, 
                title="[bold]VLM 并行评估实时监控[/bold]", 
                subtitle=summary_text,
                border_style="bright_blue",
                expand=False # 关键：不扩展以减少空白
            )
            
            live.update(main_panel)
            time.sleep(1)

if __name__ == "__main__":
    main()
