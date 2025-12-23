#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
检查JSONL文件中的图片路径完整性，并清理缺失的task_id对应的所有图片。

功能：
1. 从all_168042.jsonl文件中提取图片路径
2. 检查对应的图片文件是否真实存在
3. 识别缺失的task_id（三位数前缀）
4. 删除不完整task_id对应的所有其他图片（支持DryRun模式）
"""

import json
import os
import re
from pathlib import Path
from collections import defaultdict
from typing import Set, Dict, List, Optional

# 仓库根，用于将相对路径解析为仓库内路径
REPO_ROOT = Path(__file__).resolve().parents[2]

class ImageValidator:
    def __init__(self, data_dir: str = "data"):
        # 支持相对路径：基于仓库根解析
        if not os.path.isabs(str(data_dir)):
            self.data_dir = REPO_ROOT / data_dir
        else:
            self.data_dir = Path(data_dir)
        self.images_dir = self.data_dir / "images"
        self.missing_images = []
        self.missing_task_ids = set()
        self.image_files_by_task_id = defaultdict(list)
        
    def extract_task_id(self, image_path: str) -> str:
        """
        从图片路径中提取task_id（前三位数字）。
        格式：data/images/{task_id}-{episode_id}.png
        """
        # 提取文件名
        filename = Path(image_path).name
        # 提取前三位数字
        match = re.match(r'(\d{3})-', filename)
        if match:
            return match.group(1)
        return None
    
    def check_images_in_jsonl(self, jsonl_file: str) -> Dict[str, Set[str]]:
        """
        检查JSONL文件中的图片是否存在。
        返回：(存在的task_ids, 缺失的task_ids)
        """
        existing_images = set()
        missing_images = set()
        
        jsonl_path = self.data_dir / jsonl_file
        if not jsonl_path.exists():
            print(f"❌ 文件不存在: {jsonl_path}")
            return existing_images, missing_images
        
        print(f"📋 正在检查 {jsonl_file}...", end='', flush=True)
        
        # 预先加载所有图片文件以加快查找
        existing_files = set()
        if self.images_dir.exists():
            existing_files = set(f.name for f in self.images_dir.glob('*.png'))
        
        processed = 0
        with open(jsonl_path, 'r', encoding='utf-8') as f:
            for line_num, line in enumerate(f, 1):
                try:
                    data = json.loads(line)
                    messages = data.get('messages', [])
                    
                    for msg in messages:
                        content = msg.get('content', [])
                        for item in content:
                            if item.get('type') == 'image':
                                image_path = item.get('image')
                                if image_path:
                                    filename = Path(image_path).name
                                    
                                    if filename in existing_files:
                                        existing_images.add(image_path)
                                        task_id = self.extract_task_id(image_path)
                                        if task_id:
                                            self.image_files_by_task_id[task_id].append(image_path)
                                    else:
                                        missing_images.add(image_path)
                                        self.missing_images.append({
                                            'file': jsonl_file,
                                            'line': line_num,
                                            'path': image_path
                                        })
                                        task_id = self.extract_task_id(image_path)
                                        if task_id:
                                            self.missing_task_ids.add(task_id)
                    
                    processed += 1
                    if processed % 10000 == 0:
                        print(f"\r📋 正在检查 {jsonl_file}... ({processed:,} 行处理中)", end='', flush=True)
                        
                except json.JSONDecodeError as e:
                    print(f"\n⚠️  行 {line_num} 的JSON解析错误: {e}")
        
        print(f"\r✅ {jsonl_file} 检查完成", flush=True)
        return existing_images, missing_images
    
    def validate_task_completeness(self) -> Dict[str, List[str]]:
        """
        检查每个task_id是否有缺失的图片。
        返回不完整的task_id及其缺失的图片。
        """
        incomplete_tasks = {}
        
        # 为每个缺失的task_id收集对应的缺失图片
        missing_images_by_task = defaultdict(list)
        for missing_item in self.missing_images:
            task_id = self.extract_task_id(missing_item['path'])
            if task_id:
                missing_images_by_task[task_id].append(missing_item['path'])
        
        # 返回不完整的task_id及其对应的缺失图片
        return dict(missing_images_by_task)
    
    def find_orphan_images(self, incomplete_task_ids: Set[str]) -> List[str]:
        """
        找到属于不完整task_id的所有图片文件。
        """
        orphan_images = []
        
        if not self.images_dir.exists():
            print(f"❌ 图片目录不存在: {self.images_dir}")
            return orphan_images
        
        for image_file in self.images_dir.glob('*.png'):
            filename = image_file.name
            task_id = self.extract_task_id(filename)
            
            if task_id and task_id in incomplete_task_ids:
                orphan_images.append(str(image_file))
        
        return orphan_images
    
    def delete_orphan_images(self, orphan_images: List[str], dry_run: bool = True):
        """
        删除孤立的图片文件。
        dry_run=True 时只显示要删除的文件，不实际删除。
        """
        if not orphan_images:
            print("✅ 没有要删除的图片文件")
            return
        
        if dry_run:
            print(f"\n🔍 DRY RUN 模式：以下 {len(orphan_images)} 个文件将被删除：")
            for img_path in sorted(orphan_images):
                print(f"  - {img_path}")
        else:
            print(f"\n🗑️  删除 {len(orphan_images)} 个图片文件...")
            deleted_count = 0
            for img_path in orphan_images:
                try:
                    os.remove(img_path)
                    print(f"  ✓ 已删除: {img_path}")
                    deleted_count += 1
                except Exception as e:
                    print(f"  ✗ 删除失败: {img_path} - {e}")
            print(f"✅ 完成：成功删除 {deleted_count}/{len(orphan_images)} 个文件")
    
    def generate_report(self) -> str:
        """
        生成检查报告。
        """
        report = []
        report.append("\n" + "="*60)
        report.append("📊 图片完整性检查报告")
        report.append("="*60)
        
        # 缺失图片统计
        report.append(f"\n📉 缺失图片总数: {len(self.missing_images)}")
        if self.missing_images:
            report.append("\n缺失的图片详情:")
            for item in self.missing_images[:10]:  # 只显示前10个
                report.append(f"  - {item['file']} 第 {item['line']} 行: {item['path']}")
            if len(self.missing_images) > 10:
                report.append(f"  ... 还有 {len(self.missing_images) - 10} 个缺失图片")
        
        # 缺失task_id统计
        report.append(f"\n⚠️  缺失的 task_id 数量: {len(self.missing_task_ids)}")
        if self.missing_task_ids:
            report.append(f"\n不完整的 task_id 清单:")
            for task_id in sorted(self.missing_task_ids):
                existing_count = len(self.image_files_by_task_id.get(task_id, []))
                report.append(f"  - task_id: {task_id} (已有 {existing_count} 张图片)")
        
        report.append("\n" + "="*60)
        
        return "\n".join(report)
    
    def save_missing_task_ids(self, output_file: Optional[str] = None):
        """
        将缺失的task_id保存到文件，默认写入 data 目录下的 missing_task_ids.txt。
        当 output_file 为相对路径时，会相对于数据目录解析，避免在任意 cwd 下生成新的 data 目录。
        """
        if output_file is None:
            output_path = self.data_dir / "missing_task_ids.txt"
        else:
            candidate = Path(output_file)
            if candidate.is_absolute():
                output_path = candidate
            else:
                output_path = (self.data_dir / candidate).resolve()

        output_path.parent.mkdir(parents=True, exist_ok=True)

        with open(output_path, 'w', encoding='utf-8') as f:
            f.write("不完整的 task_id 清单\n")
            f.write("="*40 + "\n\n")
            
            for task_id in sorted(self.missing_task_ids):
                existing_count = len(self.image_files_by_task_id.get(task_id, []))
                f.write(f"{task_id}\n")
            
            f.write(f"\n总计: {len(self.missing_task_ids)} 个不完整的 task_id\n")
        
        print(f"\n💾 缺失 task_id 列表已保存到: {output_path}")


def main():
    import argparse
    
    parser = argparse.ArgumentParser(description='检查和清理不完整的task_id对应的图片')
    parser.add_argument('--data-dir', default='data', help='数据目录路径')
    parser.add_argument('--delete', action='store_true', help='实际删除文件（默认为DryRun模式）')
    parser.add_argument('--no-dry-run', action='store_true', help='禁用DryRun模式并实际删除')
    
    args = parser.parse_args()
    
    validator = ImageValidator(data_dir=args.data_dir)
    
    # 检查all_168042.jsonl文件
    all_existing, all_missing = validator.check_images_in_jsonl('all_168042.jsonl')
    
    print(f"\n✅ all_168042.jsonl: {len(all_existing)} 张现存图片, {len(all_missing)} 张缺失图片")
    
    # 生成报告
    report = validator.generate_report()
    print(report)
    
    # 保存缺失的task_id列表
    validator.save_missing_task_ids()
    
    # 处理删除
    if validator.missing_task_ids:
        orphan_images = validator.find_orphan_images(validator.missing_task_ids)
        
        if orphan_images:
            dry_run = not args.no_dry_run
            validator.delete_orphan_images(orphan_images, dry_run=dry_run)
            
            if dry_run:
                print("\n💡 要实际删除这些文件，请运行: python check_and_clean_images.py --no-dry-run")
        else:
            print("✅ 没有找到属于缺失task_id的图片文件")
    else:
        print("\n✅ 所有 task_id 都是完整的，无需清理")


if __name__ == '__main__':
    main()
