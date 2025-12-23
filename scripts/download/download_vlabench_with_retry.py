#!/usr/bin/env python3
"""
智能下载VLABench数据集脚本
- 自动检测API速率限制（429错误）
- 遇到限流后休眠并重试
- 支持断点续传
- 显示下载进度
"""

import os
import subprocess
import time
import re
import sys
from datetime import datetime
from pathlib import Path

class VLABenchDownloader:
    def __init__(
        self,
        repo_id="VLABench/vlm_evaluation_v1.0",
        local_dir=None,
        include_patterns=["M&T/**"],
        max_workers=2,
        initial_sleep=60,
        max_sleep=600,
        backoff_factor=2.0
    ):
        """
        Args:
            repo_id: HuggingFace仓库ID
            local_dir: 本地保存目录
            include_patterns: 要下载的文件模式列表
            max_workers: 并发worker数
            initial_sleep: 初始休眠时间（秒）
            max_sleep: 最大休眠时间（秒）
            backoff_factor: 退避因子（每次失败后休眠时间翻倍）
        """
        self.repo_id = repo_id
        # 使用脚本文件位置的仓库根作为默认本地目录
        if local_dir is None:
            script_root = Path(__file__).resolve().parents[2]
            self.local_dir = str(script_root / "dataset" / "vlm_evaluation_v1.0")
        else:
            self.local_dir = local_dir
        self.include_patterns = include_patterns
        self.max_workers = max_workers
        self.initial_sleep = initial_sleep
        self.max_sleep = max_sleep
        self.backoff_factor = backoff_factor
        
        self.retry_count = 0
        self.current_sleep = initial_sleep
    
    def build_command(self):
        """构建huggingface-cli下载命令"""
        cmd = [
            "huggingface-cli", "download",
            self.repo_id,  # repo_id 应该紧跟在 download 命令之后
            "--repo-type", "dataset",
            "--local-dir", self.local_dir,
            "--max-workers", str(self.max_workers),
            "--resume-download"
        ]
        
        # 添加include模式
        for pattern in self.include_patterns:
            cmd.extend(["--include", pattern])
        
        return cmd
    
    def check_rate_limit(self, output):
        """检查输出中是否包含速率限制错误"""
        rate_limit_patterns = [
            r"429",  # HTTP 429 Too Many Requests
            r"rate limit",
            r"too many requests",
            r"HTTPError.*429",
            r"ReadTimeoutError",
            r"Connection.*reset",
        ]
        
        for pattern in rate_limit_patterns:
            if re.search(pattern, output, re.IGNORECASE):
                return True
        return False
    
    def log(self, message, level="INFO"):
        """带时间戳的日志输出"""
        timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        prefix = {
            "INFO": "ℹ️",
            "SUCCESS": "✅",
            "ERROR": "❌",
            "WARN": "⚠️",
            "SLEEP": "😴"
        }.get(level, "ℹ️")
        
        print(f"[{timestamp}] {prefix} {message}", flush=True)
    
    def download_with_retry(self):
        """执行下载，遇到限流自动重试"""
        self.log(f"开始下载 {self.repo_id}")
        self.log(f"保存目录: {self.local_dir}")
        self.log(f"下载模式: {', '.join(self.include_patterns)}")
        self.log(f"并发数: {self.max_workers}")
        print("=" * 60)
        
        while True:
            try:
                cmd = self.build_command()
                self.log(f"执行命令: {' '.join(cmd)}")
                
                # 执行下载命令
                env = os.environ.copy()
                env["HF_ENDPOINT"] = "https://hf-mirror.com"

                process = subprocess.Popen(
                    cmd,
                    stdout=subprocess.PIPE,
                    stderr=subprocess.STDOUT,
                    text=True,
                    bufsize=1,
                    env=env
                )
                
                output_buffer = []
                rate_limited = False
                
                # 实时读取输出
                for line in process.stdout:
                    print(line, end='', flush=True)
                    output_buffer.append(line)
                    
                    # 检查是否遇到速率限制
                    if self.check_rate_limit(line):
                        rate_limited = True
                
                process.wait()
                
                # 检查退出码
                if process.returncode == 0:
                    self.log("下载完成！", "SUCCESS")
                    return True
                
                # 检查是否因为速率限制失败
                full_output = ''.join(output_buffer)
                if rate_limited or self.check_rate_limit(full_output):
                    self.retry_count += 1
                    self.log(f"检测到API速率限制（第{self.retry_count}次）", "WARN")
                    self.log(f"休眠 {self.current_sleep} 秒后重试...", "SLEEP")
                    
                    time.sleep(self.current_sleep)
                    
                    # 指数退避：每次失败后增加休眠时间
                    self.current_sleep = min(
                        self.current_sleep * self.backoff_factor,
                        self.max_sleep
                    )
                    
                    self.log("继续下载...", "INFO")
                    continue
                else:
                    # 其他错误
                    self.log(f"下载失败，退出码: {process.returncode}", "ERROR")
                    self.log("非速率限制错误，建议手动检查", "WARN")
                    return False
                    
            except KeyboardInterrupt:
                self.log("用户中断下载", "WARN")
                return False
            except Exception as e:
                self.log(f"发生异常: {str(e)}", "ERROR")
                self.log(f"休眠 {self.current_sleep} 秒后重试...", "SLEEP")
                time.sleep(self.current_sleep)
                self.current_sleep = min(
                    self.current_sleep * self.backoff_factor,
                    self.max_sleep
                )


def main():
    import argparse
    
    parser = argparse.ArgumentParser(description="智能下载VLABench数据集（自动处理速率限制）")
    parser.add_argument("--dimensions", nargs='+', default=["M&T"], 
                        help="要下载的评估维度 (默认: M&T)")
    parser.add_argument("--local-dir", default=None,
                        help="本地保存目录 (默认: 仓库根下 dataset/vlm_evaluation_v1.0)")
    parser.add_argument("--max-workers", type=int, default=2,
                        help="并发worker数（默认2，避免过快触发限流）")
    parser.add_argument("--initial-sleep", type=int, default=60,
                        help="初始休眠时间（秒，默认60）")
    parser.add_argument("--max-sleep", type=int, default=600,
                        help="最大休眠时间（秒，默认600）")
    
    args = parser.parse_args()

    repo_root = Path(__file__).resolve().parents[2]

    def _abs(path):
        if path is None:
            return None
        p = Path(path)
        if p.is_absolute():
            return str(p)
        return str((repo_root / p).resolve())

    if args.local_dir is None:
        args.local_dir = str((repo_root / "dataset" / "vlm_evaluation_v1.0").resolve())
    else:
        args.local_dir = _abs(args.local_dir)
    
    # 构建include模式
    include_patterns = [f"{dim}/**" for dim in args.dimensions]
    
    # 创建下载器
    downloader = VLABenchDownloader(
    local_dir=args.local_dir,
        include_patterns=include_patterns,
        max_workers=args.max_workers,
        initial_sleep=args.initial_sleep,
        max_sleep=args.max_sleep,
        backoff_factor=2.0
    )
    
    # 执行下载
    success = downloader.download_with_retry()
    
    if success:
        print("\n" + "=" * 60)
        print("🎉 数据集下载成功！")
        print("=" * 60)
        sys.exit(0)
    else:
        print("\n" + "=" * 60)
        print("😞 数据集下载失败，请检查错误信息")
        print("=" * 60)
        sys.exit(1)


if __name__ == "__main__":
    main()
