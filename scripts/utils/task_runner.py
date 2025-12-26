import subprocess
import sys
import time
import argparse
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

def run_command(cmd, log_file):
    print(f"[START] {cmd}")
    log_path = Path(log_file)
    log_path.parent.mkdir(parents=True, exist_ok=True)
    
    with open(log_file, "w") as f:
        process = subprocess.Popen(
            cmd,
            shell=True,
            stdout=f,
            stderr=subprocess.STDOUT,
            text=True
        )
        process.wait()
        
    status = "SUCCESS" if process.returncode == 0 else f"FAILED({process.returncode})"
    print(f"[{status}] {cmd}")
    return process.returncode

def main():
    parser = argparse.ArgumentParser(description="Simple Task Runner with Concurrency Control")
    parser.add_argument("--commands", nargs="+", required=True, help="List of commands to run")
    parser.add_argument("--logs", nargs="+", required=True, help="List of log files for each command")
    parser.add_argument("--concurrency", type=int, default=2, help="Max parallel tasks")
    
    args = parser.parse_args()
    
    if len(args.commands) != len(args.logs):
        print("Error: Number of commands and logs must match.")
        sys.exit(1)
        
    tasks = list(zip(args.commands, args.logs))
    
    print(f"Starting {len(tasks)} tasks with concurrency={args.concurrency}")
    
    with ThreadPoolExecutor(max_workers=args.concurrency) as executor:
        futures = [executor.submit(run_command, cmd, log) for cmd, log in tasks]
        for future in as_completed(futures):
            try:
                future.result()
            except Exception as e:
                print(f"Task generated an exception: {e}")

if __name__ == "__main__":
    main()
