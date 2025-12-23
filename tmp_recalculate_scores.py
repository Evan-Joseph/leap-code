import os
import json
import glob
from pathlib import Path
import shutil

def recalculate_total_score(scores):
    """
    根据正确的权重重新计算 total_score
    weights:
        skill_match_score: 0.4
        entity_match_score: 0.4
        skill_with_entity_match_score: 0.1
        exact_match_score: 0.1
    """
    skill = scores.get("skill_match_score", 0.0)
    entity = scores.get("entity_match_score", 0.0)
    skill_with_entity = scores.get("skill_with_entity_match_score", 0.0)
    exact = scores.get("exact_match_score", 0.0)
    
    total = (skill * 0.4) + (entity * 0.4) + (skill_with_entity * 0.1) + (exact * 0.1)
    return total

def process_file(file_path, output_path):
    with open(file_path, 'r', encoding='utf-8') as f:
        data = json.load(f)
    
    modified = False
    
    # 遍历所有任务
    for task_name, task_data in data.items():
        if not isinstance(task_data, dict):
            continue
            
        # 遍历所有样本
        for sample_id, scores in task_data.items():
            if not isinstance(scores, dict):
                continue
            
            # 检查是否有分数键
            if "total_score" in scores:
                old_total = scores["total_score"]
                new_total = recalculate_total_score(scores)
                
                # 更新分数
                scores["total_score"] = new_total
                
                # 如果有浮点数差异，标记为已修改
                if abs(old_total - new_total) > 1e-5:
                    modified = True
    
    # 确保输出目录存在
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    
    # 保存文件
    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump(data, f, indent=4, ensure_ascii=False)
        
    return modified

def main():
    src_root = "eva_results"
    dst_root = "fixed_eva_results"
    
    print(f"开始修复分数: {src_root} -> {dst_root}")
    
    # 复制整个目录结构作为基础（包含图片等非JSON文件）
    if os.path.exists(dst_root):
        shutil.rmtree(dst_root)
    shutil.copytree(src_root, dst_root)
    
    # 查找所有的 JSON 文件
    # 注意：我们主要关心 final_score.json，但如果有 output.json 包含分数也可以处理
    # 为了安全起见，我们只处理 final_score.json，因为这是汇总分数的地方
    json_files = glob.glob(os.path.join(dst_root, "**", "final_score.json"), recursive=True)
    
    fixed_count = 0
    total_files = 0
    
    for json_file in json_files:
        # 计算相对于 dst_root 的路径，以便打印
        rel_path = os.path.relpath(json_file, dst_root)
        
        # 处理文件 (原地修改，因为已经复制了)
        is_modified = process_file(json_file, json_file)
        
        if is_modified:
            fixed_count += 1
            print(f"已修复: {rel_path}")
        total_files += 1
        
    print(f"\n处理完成。")
    print(f"共扫描文件: {total_files}")
    print(f"修复包含错误分数的文件: {fixed_count}")
    print(f"修复后的结果已保存在: {dst_root}")

if __name__ == "__main__":
    main()
