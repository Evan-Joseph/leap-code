#!/usr/bin/env python3
"""Utility to upload processed Agibot data to Hugging Face datasets.

This script uses the official 'hf' CLI command for uploads.
Automatically excludes the images/ directory since images.tar.gz is already included.

Example:
    python tmp_upload_dataset.py
    python tmp_upload_dataset.py --repo-id YourName/dataset --data-dir ./data

Prerequisites:
    pip install huggingface_hub rich
    huggingface-cli login  # or export HF_TOKEN=<token>
"""

from __future__ import annotations

import argparse
import os
import subprocess
import sys
from pathlib import Path
from typing import List, Optional

from rich.console import Console
from rich.panel import Panel
from rich.table import Table
from rich.prompt import Confirm
from rich.progress import Progress, SpinnerColumn, TextColumn, BarColumn, TaskProgressColumn
from rich import box


console = Console()


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Upload dataset to Hugging Face, excluding images/ directory.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument(
        "--repo-id",
        default="EvanSirius/leap-agibot-processed",
        help="Target HuggingFace dataset repo. Default: EvanSirius/leap-agibot-processed",
    )
    parser.add_argument(
        "--data-dir",
        default="./data",
        help="Local directory containing dataset files. Default: ./data",
    )
    parser.add_argument(
        "--private",
        action="store_true",
        help="Create the dataset repo as private (default: public).",
    )
    parser.add_argument(
        "--token",
        type=str,
        default=None,
        help="HuggingFace access token (or use HF_TOKEN env var).",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Show what would be uploaded without actually uploading.",
    )
    parser.add_argument(
        "--yes",
        action="store_true",
        help="Skip interactive confirmation.",
    )
    return parser.parse_args()


def check_hf_cli() -> bool:
    """Check if 'hf' CLI is available."""
    try:
        result = subprocess.run(
            ["hf", "version"],
            capture_output=True,
            text=True,
            timeout=5
        )
        return result.returncode == 0
    except (subprocess.TimeoutExpired, FileNotFoundError):
        return False


def format_bytes(num: int) -> str:
    """Format bytes as human-readable string."""
    for unit in ("B", "KB", "MB", "GB", "TB"):
        if num < 1024 or unit == "TB":
            return f"{num:.2f} {unit}"
        num /= 1024
    return f"{num:.2f} PB"


def get_files_to_upload(data_dir: Path) -> List[Path]:
    """Get list of files to upload, excluding images directory."""
    files = []
    
    for item in data_dir.iterdir():
        if item.is_file():
            files.append(item)
        elif item.is_dir() and item.name != "images":
            # Include non-image directories if needed
            for subfile in item.rglob("*"):
                if subfile.is_file():
                    files.append(subfile)
    
    return sorted(files)


def show_summary(
    files: List[Path],
    data_root: Path,
    repo_id: str,
) -> None:
    """Display upload summary."""
    
    total_size = sum(f.stat().st_size for f in files)
    
    table = Table(box=box.ROUNDED, show_header=False, padding=(0, 2))
    table.add_column("Metric", style="cyan bold", justify="right")
    table.add_column("Value", style="white")

    table.add_row("📁 Source Directory", str(data_root))
    table.add_row("🎯 Target Repo", repo_id)
    table.add_row("📄 Files to Upload", f"{len(files):,}")
    table.add_row("💾 Total Size", format_bytes(total_size))
    table.add_row("📦 Upload Method", "hf upload (per-file)")
    table.add_row("🚫 Excluded", "images/ directory")

    console.print(Panel(table, title="📊 Upload Summary", border_style="cyan"))

    # Show all files since there should only be 6
    if files:
        preview_table = Table(
            title=f"Files to upload",
            box=box.SIMPLE,
        )
        preview_table.add_column("File", style="dim")
        preview_table.add_column("Size", justify="right", style="cyan")

        for path in files:
            relative = path.relative_to(data_root)
            size = format_bytes(path.stat().st_size)
            preview_table.add_row(str(relative), size)

        console.print(preview_table)


def create_repo_if_needed(repo_id: str, private: bool, token: Optional[str]) -> bool:
    """Create repository if it doesn't exist."""
    cmd = ["hf", "repo", "create", repo_id, "--repo-type", "dataset"]
    
    if private:
        cmd.append("--private")
    
    if token:
        cmd.extend(["--token", token])
    
    try:
        result = subprocess.run(
            cmd,
            capture_output=True,
            text=True,
            timeout=30
        )
        return result.returncode in [0, 1]
    except Exception:
        return True  # Continue anyway


def upload_files_individually(
    repo_id: str,
    files: List[Path],
    data_root: Path,
    token: Optional[str],
) -> bool:
    """Upload files one by one using hf upload command."""
    
    console.print("\n[bold cyan]📤 Starting upload...[/bold cyan]\n")
    
    success_count = 0
    failed_files = []
    
    for idx, file_path in enumerate(files, 1):
        relative_path = file_path.relative_to(data_root)
        file_size = file_path.stat().st_size
        
        console.print(
            f"[cyan]Uploading {idx}/{len(files)}:[/cyan] "
            f"{relative_path} [dim]({format_bytes(file_size)})[/dim]"
        )
        
        cmd = [
            "hf",
            "upload",
            repo_id,
            str(file_path),
            str(relative_path),
            "--repo-type=dataset",
        ]
        
        if token:
            cmd.append(f"--token={token}")
        
        try:
            # Don't capture output so user can see progress
            # Set timeout based on file size: ~1 hour per 10GB
            timeout_seconds = max(3600, int(file_size / (10 * 1024**3) * 3600))
            
            result = subprocess.run(
                cmd,
                text=True,
                timeout=timeout_seconds
            )
            
            if result.returncode == 0:
                success_count += 1
                console.print(f"[green]✓ Successfully uploaded {relative_path}[/green]\n")
            else:
                failed_files.append(str(relative_path))
                console.print(f"[red]✗ Failed to upload {relative_path}[/red]\n")
                
        except subprocess.TimeoutExpired:
            failed_files.append(str(relative_path))
            console.print(
                f"[red]✗ Timeout uploading {relative_path} "
                f"(limit: {timeout_seconds}s)[/red]\n"
            )
        except KeyboardInterrupt:
            console.print(f"\n[yellow]⚠  Upload interrupted by user.[/yellow]")
            raise
        except Exception as e:
            failed_files.append(str(relative_path))
            console.print(f"[red]✗ Error uploading {relative_path}: {e}[/red]\n")
    
    # Summary
    if success_count == len(files):
        console.print(f"[bold green]✓ Successfully uploaded all {len(files)} files![/bold green]")
    else:
        console.print(
            f"[yellow]⚠  Uploaded {success_count}/{len(files)} files. "
            f"{len(failed_files)} failed.[/yellow]"
        )
        if failed_files:
            console.print("[yellow]Failed files:[/yellow]")
            for f in failed_files:
                console.print(f"  [dim]- {f}[/dim]")
    
    return success_count == len(files)


def main() -> None:
    args = parse_args()

    # Print header
    console.print(
        Panel(
            "[bold cyan]🤗 HuggingFace Dataset Uploader[/bold cyan]\n"
            f"Target: [yellow]{args.repo_id}[/yellow]",
            box=box.DOUBLE,
            border_style="cyan",
        )
    )

    # Check if hf CLI is available
    if not check_hf_cli():
        console.print(
            Panel(
                "[red]'hf' CLI not found![/red]\n\n"
                "Please install huggingface_hub:\n"
                "  [cyan]pip install huggingface_hub[/cyan]\n\n"
                "And authenticate:\n"
                "  [cyan]huggingface-cli login[/cyan]",
                title="❌ Missing Dependency",
                border_style="red",
            )
        )
        raise SystemExit(1)

    # Validate data directory
    data_root = Path(args.data_dir).expanduser().resolve()
    if not data_root.exists():
        console.print(
            Panel(
                f"[red]Directory not found:[/red] {data_root}",
                title="❌ Error",
                border_style="red",
            )
        )
        raise SystemExit(2)

    # Create or verify repository
    with console.status("[bold cyan]Checking repository...", spinner="dots"):
        create_repo_if_needed(args.repo_id, args.private, args.token)
        repo_url = f"https://huggingface.co/datasets/{args.repo_id}"
        console.print(f"[green]✓ Repository ready:[/green] [link]{repo_url}[/link]\n")

    # Get files to upload (excluding images directory)
    with console.status("[bold cyan]Scanning files...", spinner="dots"):
        files = get_files_to_upload(data_root)

    if not files:
        console.print(
            "[yellow]⚠  No files found to upload.[/yellow]"
        )
        return

    # Show summary
    show_summary(files, data_root, args.repo_id)

    # Dry run check
    if args.dry_run:
        console.print(
            Panel(
                "[green]Dry-run complete. No files were uploaded.[/green]",
                title="✓ Dry Run",
                border_style="green",
            )
        )
        return

    # Confirm upload
    if not args.yes:
        proceed = Confirm.ask(
            "[bold yellow]Proceed with upload?[/bold yellow]", default=True
        )
        if not proceed:
            console.print("[yellow]Upload cancelled.[/yellow]")
            return

    # Upload files
    import time
    start_time = time.time()

    success = upload_files_individually(
        repo_id=args.repo_id,
        files=files,
        data_root=data_root,
        token=args.token,
    )

    if not success:
        console.print("[yellow]⚠  Some files failed to upload.[/yellow]")

    elapsed = time.time() - start_time
    total_size = sum(f.stat().st_size for f in files)

    # Final summary
    console.print(
        Panel(
            f"[bold green]✓ Upload Complete![/bold green]\n\n"
            f"Files uploaded: [cyan]{len(files):,}[/cyan]\n"
            f"Total size: [cyan]{format_bytes(total_size)}[/cyan]\n"
            f"Time elapsed: [cyan]{elapsed:.1f}s[/cyan]\n"
            f"Dataset URL: [link]https://huggingface.co/datasets/{args.repo_id}[/link]",
            title="🎉 Success",
            border_style="green",
            box=box.DOUBLE,
        )
    )


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        console.print("\n[yellow]⚠  Upload interrupted by user.[/yellow]")
        sys.exit(130)
    except Exception as e:
        console.print(
            Panel(
                f"[red]Unexpected error:[/red]\n{e}",
                title="❌ Error",
                border_style="red",
            )
        )
        import traceback
        console.print("[dim]" + traceback.format_exc() + "[/dim]")
        sys.exit(1)