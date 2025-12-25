import os
import re
from pathlib import Path
from rich.console import Console
from rich.table import Table

def get_progress(log_path):
    if not os.path.exists(log_path):
        return "Missing", 0, 0
    
    with open(log_path, 'r', encoding='utf-8', errors='ignore') as f:
        content = f.read()
        
    if "评估完成!" in content:
        return "Completed", 100, 100
    
    # Find progress like "857/900" or "95%"
    # The log format is: "开始推理 - ... 1h37m38s, remain:  0h 4m53s"
    # But the progress bar is usually above it.
    # Let's look for the last occurrence of "task: ..., example: ..."
    
    # Try to find total tasks
    total_match = re.search(r"Total tasks: (\d+)", content)
    total = int(total_match.group(1)) if total_match else 0
    
    # Try to find current progress from the tqdm-like bar if present
    # Or just count "推理完成"
    done_count = content.count("推理完成")
    
    # If it's resuming, we need to add the existing count
    resume_match = re.search(r"(\d+) example existed", content)
    if resume_match:
        done_count += int(resume_match.group(1))
        
    if total == 0:
        # Try to infer total from dimension
        if "M&T" in log_path: total = 1000
        elif "CommenSence" in log_path: total = 800
        elif "Semantic" in log_path: total = 900
        elif "Spatial" in log_path: total = 1000
        elif "PhysicsLaw" in log_path: total = 600
        elif "Complex" in log_path: total = 500
        else: total = 1000
        
    percent = (done_count / total * 100) if total > 0 else 0
    status = "Running" if (time.time() - os.path.getmtime(log_path)) < 300 else "Stalled"
    
    return status, done_count, total

import time
console = Console()
table = Table(title="LoRA Evaluation Progress")
table.add_column("Task", style="cyan")
table.add_column("Status", style="magenta")
table.add_column("Progress", style="green")
table.add_column("Percent", style="yellow")

log_dir = Path("logs_lora")
logs = sorted(list(log_dir.glob("lora_standard_step*.log")))

for log in logs:
    status, done, total = get_progress(str(log))
    percent = f"{done/total*100:.1f}%" if total > 0 else "0%"
    table.add_row(log.name, status, f"{done}/{total}", percent)

console.print(table)
