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
        # 获取所有 GPU 的信息
        res = subprocess.check_output(["nvidia-smi", "--query-gpu=memory.used,memory.total,utilization.gpu", "--format=csv,noheader,nounits"])
        lines = res.decode().strip().split("\n")
        
        total_used = 0
        total_capacity = 0
        total_util = 0
        count = 0
        
        for line in lines:
            parts = line.split(",")
            total_used += int(parts[0].strip())
            total_capacity += int(parts[1].strip())
            total_util += int(parts[2].strip())
            count += 1
            
        if count == 0:
            return "N/A", "N/A"
            
        # 格式化输出: 总已用 / 总容量 (平均利用率%)
        return f"{total_used}MB / {total_capacity}MB", f"{int(total_util / count)}%"
    except Exception as e:
        return "N/A", "N/A"

def get_progress_from_log(log_file):
    if not os.path.exists(log_file):
        return 0, 0, "等待中", "N/A", "N/A", "N/A"
    
    try:
        log_path = Path(log_file)
        config_name = log_path.parent.name
        
        with open(log_file, 'r') as f:
            lines = f.readlines()
            
        if not lines:
            return 0, 0, "队列中", "等待调度", "-", config_name

        # 查找当前维度和模型名
        current_dim = "初始化"
        model_name = ""
        
        for line in reversed(lines):
            if "评估维度:" in line or "开始评估维度:" in line:
                parts = line.split("维度:")
                if len(parts) > 1:
                    current_dim = parts[1].strip().replace("-", "")
                    current_dim = re.sub(r'\x1b\[[0-9;]*m', '', current_dim)
            
            if "评估模型:" in line:
                model_name = line.split("模型:")[1].strip()
            elif "working on" in line and not model_name:
                model_name = line.split("working on")[1].strip()
            
            if current_dim != "初始化" and model_name:
                break
        
        # 清理颜色代码
        model_name = re.sub(r'\x1b\[[0-9;]*m', '', model_name) if model_name else "加载中..."
        display_name = f"{config_name} > {model_name}"

        content = "".join(lines[-100:])
        
        # 查找进度
        matches = re.findall(r'(\d+)%\(\s*(\d+)/(\d+)\).*?remain:\s*([\dhms\s:]+)', content)
        if matches:
            last_match = matches[-1]
            percent, current, total, eta = last_match
            return int(current), int(total), f"{percent}%", current_dim, eta.strip(), display_name
        
        if "评估完成!" in content or "Baseline评估完成" in content:
            # 尝试从全文找一次维度名，防止完成时丢失
            if current_dim == "初始化":
                for line in lines:
                    if "评估维度:" in line:
                        current_dim = line.split("维度:")[1].strip()
                        break
            return 100, 100, "已完成", current_dim, "0s", display_name
            
        return 0, 0, "运行中", current_dim, "计算中...", display_name
    except Exception as e:
        return 0, 0, "读取错误", "N/A", str(e), "N/A"

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
            
            # 动态调整表格宽度 - 使用 Rich 的 expand 功能
            table = Table(
                show_header=True, 
                header_style="bold magenta", 
                box=None,
                padding=(0, 1), # 减少 padding，让 expand 自己分配
                expand=True,
                collapse_padding=True
            )
            
            # 重新分配 ratio 和对齐方式
            table.add_column("模型 Checkpoint", style="bright_white", justify="left", ratio=3)
            table.add_column("维度", style="cyan", justify="center", ratio=1)
            table.add_column("进度", style="green", justify="center", ratio=2)
            table.add_column("样本", style="yellow", justify="center", ratio=1)
            table.add_column("剩余时间", style="bold red", justify="center", ratio=1)
            
            if not log_files:
                table.add_row("无活跃任务", "-", "-", "-", "-")
            else:
                for log_file in log_files:
                    current, total, status, dim, eta, model = get_progress_from_log(log_file)
                    
                    # 精简显示名称
                    model_display = model.replace("lora_ablation > ", "").replace("lora_", "")
                    
                    # 进度条可视化 (缩短长度到 10，防止换行)
                    bar_len = 10
                    if total > 0 and "%" in status:
                        pct = current / total
                        filled = int(pct * bar_len)
                        bar = "━" * filled + "┄" * (bar_len - filled)
                        display_status = f"{status} {bar}" # 去掉中括号，更紧凑
                    elif status == "已完成":
                        display_status = "[bold green]✅[/bold green]" # 简短图标
                    else:
                        display_status = status

                    table.add_row(
                        model_display,
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
