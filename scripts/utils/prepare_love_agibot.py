#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
LOVE-Agibot-Beta 数据集下载与处理脚本

一、数据纳入策略指北：

2. 该脚本的职责：
   - 仅下载并提取首帧图像到 data/images/ 目录
   - 不处理 JSONL 数据文件
   - 旧的 JSONL 文件需要手动转移到 data/ 目录

3. 使用起来简单–只需运行：
   python prepare_love_agibot.py --num-workers 4

核心依赖:
- huggingface_hub: 用于从Hugging Face下载文件
- tqdm: 用于显示进度条
- rich: 用于美化日志输出
"""

import os
import json
import tarfile
from pathlib import Path
from huggingface_hub import HfApi, hf_hub_download
from tqdm import tqdm
from rich.console import Console
import argparse
import threading
from concurrent.futures import ThreadPoolExecutor, as_completed
import signal
import sys

# 设置Hugging Face镜像源
os.environ['HF_ENDPOINT'] = 'https://hf-mirror.com'

# 初始化API客户端和控制台输出
console = Console()

# 全局变量用于优雅退出
shutdown_event = threading.Event()

# 定义数据集名称和本地存储路径
dataset_repo = "EvanSirius/LOVE-Agibot-Beta"
REPO_ROOT = Path(__file__).resolve().parents[2]
data_root = str(REPO_ROOT / "data")
images_dir = os.path.join(data_root, "images")
temp_dir = os.path.join(data_root, "temp")

def setup_directories():
    """创建必要的目录结构"""
    console.print("[blue]正在创建目录结构...[/blue]")
    Path(images_dir).mkdir(parents=True, exist_ok=True)
    Path(temp_dir).mkdir(parents=True, exist_ok=True)
    console.print("[green]目录结构创建完成[/green]")

def get_file_lists():
    """获取图像压缩包列表"""
    console.print("[blue]正在获取数据集文件列表...[/blue]")
    try:
        # 使用镜像源创建HfApi实例
        api = HfApi()
        
        files = api.list_repo_files(repo_id=dataset_repo, repo_type="dataset")
    except Exception as e:
        console.print(f"[red]获取文件列表时出错: {str(e)}[/red]")
        return []
    
    image_files = [f for f in files if f.startswith('image/') and f.endswith('.tar.gz')]
    
    console.print(f"[green]找到 {len(image_files)} 个图像压缩包[/green]")
    return image_files

def extract_task_id_from_filename(filename):
    """从文件名中提取任务ID"""
    # 处理 image/task_327.tar.gz 或 json/output_327.json 格式
    basename = os.path.basename(filename)
    if basename.startswith('task_'):
        return basename.split('_')[1].split('.')[0]
    elif basename.startswith('output_'):
        return basename.split('_')[1].split('.')[0]
    return None

def process_single_archive(image_file):
    """处理单个图像压缩包，仅提取图片"""
    # 检查是否需要退出
    if shutdown_event.is_set():
        return
    
    # 1. 确定任务 ID
    task_id = extract_task_id_from_filename(image_file)
    if not task_id:
        console.print(f"[red]无法从文件名 {image_file} 提取任务ID[/red]")
        return
    
    # 2. 断点续传检查
    # 检查是否存在任意以task_id开头的图片，如果存在则跳过
    existing_images = list(Path(images_dir).glob(f"{task_id}-*.png"))
    if existing_images:
        console.print(f"[yellow]任务 {task_id} 的图片已存在，跳过处理[/yellow]")
        return
    
    # 3. 下载图像压缩包
    console.print(f"[blue]正在下载任务 {task_id} 的图像压缩包...[/blue]")
    try:
        local_tar_path = hf_hub_download(
            repo_id=dataset_repo,
            filename=image_file,
            repo_type="dataset",
            cache_dir=temp_dir,
            endpoint=os.environ.get('HF_ENDPOINT', 'https://huggingface.co')
        )
    except Exception as e:
        console.print(f"[red]下载图像压缩包 {image_file} 时出错: {str(e)}[/red]")
        return
    
    # 检查是否需要退出
    if shutdown_event.is_set():
        if os.path.exists(local_tar_path):
            os.remove(local_tar_path)
        return
    
    # 4. 解压压缩包
    console.print(f"[blue]正在解压任务 {task_id} 的图像压缩包...[/blue]")
    try:
        with tarfile.open(local_tar_path, 'r:gz') as tar:
            tar.extractall(path=temp_dir)
    except Exception as e:
        console.print(f"[red]解压图像压缩包 {local_tar_path} 时出错: {str(e)}[/red]")
        return
    
    # 检查是否需要退出
    if shutdown_event.is_set():
        if os.path.exists(local_tar_path):
            os.remove(local_tar_path)
        extracted_folder = os.path.join(temp_dir, task_id)
        if os.path.exists(extracted_folder):
            import shutil
            shutil.rmtree(extracted_folder)
        return
    
    # 5. 提取每个episode的首帧到输出目录
    extracted_folder = os.path.join(temp_dir, task_id)
    if os.path.exists(extracted_folder):
        # 遍历task_id目录下的所有episode文件夹
        image_count = 0
        for episode_folder in sorted(os.listdir(extracted_folder)):
            episode_path = os.path.join(extracted_folder, episode_folder)
            if not os.path.isdir(episode_path):
                continue
            
            # 获取episode中的所有PNG文件，按名称排序（首帧通常为第一个）
            png_files = sorted([f for f in os.listdir(episode_path) if f.endswith('.png')])
            
            if png_files:
                # 只取首帧
                first_frame = png_files[0]
                src_path = os.path.join(episode_path, first_frame)
                # 使用 taskid-episodeid.png 的格式命名
                new_filename = f"{task_id}-{episode_folder}.png"
                dst_path = os.path.join(images_dir, new_filename)
                
                try:
                    with open(src_path, 'rb') as src, open(dst_path, 'wb') as dst:
                        dst.write(src.read())
                    image_count += 1
                except Exception as e:
                    console.print(f"[red]复制图片 {src_path} 时出错: {str(e)}[/red]")
                    continue
        
        console.print(f"[green]任务 {task_id} 处理完成，共提取 {image_count} 个episode的首帧[/green]")
    
    # 6. 清理临时文件
    console.print(f"[blue]正在清理任务 {task_id} 的临时文件...[/blue]")
    if os.path.exists(local_tar_path):
        os.remove(local_tar_path)
    if os.path.exists(extracted_folder):
        import shutil
        shutil.rmtree(extracted_folder)

def parse_args():
    """解析命令行参数"""
    parser = argparse.ArgumentParser(description="LOVE-Agibot-Beta 数据集下载与处理脚本")
    parser.add_argument('--num-workers', type=int, default=4, help='并发处理的线程数量')
    return parser.parse_args()

def signal_handler(signum, frame):
    """信号处理函数，用于优雅退出"""
    console.print("\n[yellow]收到中断信号，正在优雅退出...[/yellow]")
    shutdown_event.set()
    sys.exit(130)  # 130 是标准的被信号中断的退出码（SIGINT）

def main():
    """主函数"""
    # 注册信号处理器
    signal.signal(signal.SIGINT, signal_handler)
    signal.signal(signal.SIGTERM, signal_handler)
    
    # 解析命令行参数
    args = parse_args()
    
    console.print(f"[bold blue]开始处理 LOVE-Agibot-Beta 数据集，使用 {args.num_workers} 个线程[/bold blue]")
    
    # 1. 初始化与设置
    setup_directories()
    
    # 2. 获取文件列表
    image_files = get_file_lists()
    
    # 3. 主处理逻辑（多线程）
    console.print("[bold blue]开始处理图像压缩包...[/bold blue]")
    
    # 使用ThreadPoolExecutor进行多线程处理
    try:
        with ThreadPoolExecutor(max_workers=args.num_workers) as executor:
            # 提交所有任务
            future_to_image = {
                executor.submit(process_single_archive, image_file): image_file 
                for image_file in image_files
            }
            
            # 使用tqdm显示进度
            completed_count = 0
            total_count = len(image_files)
            
            with tqdm(total=total_count, desc="处理进度") as pbar:
                # 处理完成的任务
                for future in as_completed(future_to_image):
                    image_file = future_to_image[future]
                    try:
                        future.result()  # 获取结果以触发异常
                    except Exception as e:
                        if not shutdown_event.is_set():
                            console.print(f"[red]处理 {image_file} 时出错: {str(e)}[/red]")
                    
                    completed_count += 1
                    pbar.update(1)
                    
                    # 如果收到退出信号，则取消剩余的任务
                    if shutdown_event.is_set():
                        console.print("[yellow]正在取消剩余任务...[/yellow]")
                        for f in future_to_image.keys():
                            if not f.done():
                                f.cancel()
                        break
            
            # 等待所有任务完成或取消
            for future in future_to_image.keys():
                if not future.done():
                    future.cancel()
        
        if shutdown_event.is_set():
            console.print("[yellow]程序已被中断[/yellow]")
            sys.exit(0)
        else:
            console.print("[bold green]所有任务处理完成![/bold green]")
    except KeyboardInterrupt:
        console.print("\n[yellow]程序被键盘中断[/yellow]")
        sys.exit(130)
    except Exception as e:
        console.print(f"[red]发生未预期的错误: {str(e)}[/red]")
        sys.exit(1)

if __name__ == "__main__":
    main()