import subprocess
import sys
import time
import argparse
import os
import threading
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from queue import Queue

# GPU 资源管理队列
gpu_queue = Queue()

def run_command(cmd, log_file, use_gpu_scheduling=False):
    env = os.environ.copy()
    gpu_id = None
    
    if use_gpu_scheduling:
        # 获取一个可用的 GPU 槽位
        gpu_id = gpu_queue.get()
        
        # 关键优化：强制物理隔离 GPU，防止进程同时占用多张卡的显存
        env["CUDA_VISIBLE_DEVICES"] = str(gpu_id)
        
        # 由于设置了 CUDA_VISIBLE_DEVICES，物理 GPU x 变成了逻辑上的 cuda:0
        # 所以必须将命令中的 __DEVICE__ 替换为 cuda:0
        cmd = cmd.replace("__DEVICE__", "cuda:0")
        
    print(f"[START] (GPU: {gpu_id if gpu_id is not None else 'N/A'}) {cmd}")
    
    log_path = Path(log_file)
    log_path.parent.mkdir(parents=True, exist_ok=True)
    
    try:
        with open(log_file, "w") as f:
            process = subprocess.Popen(
                cmd,
                shell=True,
                stdout=f,
                stderr=subprocess.STDOUT,
                text=True,
                env=env
            )
            process.wait()
            
        status = "SUCCESS" if process.returncode == 0 else f"FAILED({process.returncode})"
        print(f"[{status}] (GPU: {gpu_id if gpu_id is not None else 'N/A'}) {cmd}")
        return process.returncode
        
    finally:
        if use_gpu_scheduling and gpu_id is not None:
            # 释放 GPU 槽位
            gpu_queue.put(gpu_id)

def main():
    parser = argparse.ArgumentParser(description="Task Runner with GPU Scheduling")
    parser.add_argument("--commands", nargs="+", required=True, help="List of commands to run")
    parser.add_argument("--logs", nargs="+", required=True, help="List of log files")
    parser.add_argument("--concurrency", type=int, default=2, help="Max parallel tasks (ignored if --gpus is set)")
    parser.add_argument("--gpus", type=str, help="Comma-separated GPU IDs (e.g., '0,1')")
    parser.add_argument("--tasks-per-gpu", type=int, default=1, help="Number of concurrent tasks per GPU")
    
    args = parser.parse_args()
    
    if len(args.commands) != len(args.logs):
        print("Error: Number of commands and logs must match.")
        sys.exit(1)
        
    tasks = list(zip(args.commands, args.logs))
    
    # 配置并发控制
    max_workers = args.concurrency
    use_gpu_scheduling = False
    
    if args.gpus:
        use_gpu_scheduling = True
        gpu_ids = [int(x.strip()) for x in args.gpus.split(",")]
        # 初始化 GPU 队列
        for gpu in gpu_ids:
            for _ in range(args.tasks_per_gpu):
                gpu_queue.put(gpu)
        
        max_workers = len(gpu_ids) * args.tasks_per_gpu
        print(f"Enabled GPU scheduling: GPUs={gpu_ids}, Tasks/GPU={args.tasks_per_gpu}, Total Concurrency={max_workers}")
    else:
        print(f"Starting tasks with static concurrency={max_workers}")
    
    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        futures = [executor.submit(run_command, cmd, log, use_gpu_scheduling) for cmd, log in tasks]
        for future in as_completed(futures):
            try:
                future.result()
            except Exception as e:
                print(f"Task generated an exception: {e}")

if __name__ == "__main__":
    main()
