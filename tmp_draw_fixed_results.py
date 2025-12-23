#!/usr/bin/env python3
"""
基于修复后的评估结果（fixed_eva_results）进行绘图的临时脚本。
大部分逻辑复用自 scripts/evaluation/draw_evaluation_results.py
"""
import json
import os
import sys
from pathlib import Path
from collections import OrderedDict
import matplotlib.pyplot as plt
import matplotlib
import numpy as np
import pandas as pd
from statistics import mean

# 设置中文字体（尝试通用字体，避免Mac缺少SimHei报错）
matplotlib.rcParams['font.sans-serif'] = ['Arial', 'DejaVu Sans', 'SimHei']
matplotlib.rcParams['axes.unicode_minus'] = False

# ==========================================
# 强制指定评估结果目录为我们刚刚生成的 fixed_eva_results
# ==========================================
EVAL_DIR = str(Path(__file__).resolve().parent / "fixed_eva_results")

if not os.path.exists(EVAL_DIR):
    print(f"Error: Directory {EVAL_DIR} does not exist.")
    sys.exit(1)

print(f"Using fixed evaluation results from: {EVAL_DIR}")

def _aggregate_from_final_score(final_score_path: str):
    """从 final_score.json 聚合平均指标。如果结构为 task -> index -> metrics。"""
    metrics = ['skill_match_score', 'entity_match_score', 'skill_with_entity_match_score', 'exact_match_score', 'total_score']
    buckets = {m: [] for m in metrics}
    try:
        with open(final_score_path, 'r') as f:
            data = json.load(f)
        # data: task_name -> sample_index -> metric dict
        for task_name, samples in data.items():
            if isinstance(samples, dict):
                for sample_idx, sample_metrics in samples.items():
                    if isinstance(sample_metrics, dict):
                        for m in metrics:
                            val = sample_metrics.get(m)
                            if isinstance(val, (int, float)):
                                buckets[m].append(float(val))
        # 计算平均
        averaged = {}
        for m in metrics:
            averaged[m] = mean(buckets[m]) if buckets[m] else 0.0
        averaged['count'] = len(buckets['total_score'])
        return averaged
    except Exception as e:
        print(f"[Warn] Failed to aggregate {final_score_path}: {e}")
        return {
            'skill_match_score': 0.0,
            'entity_match_score': 0.0,
            'skill_with_entity_match_score': 0.0,
            'exact_match_score': 0.0,
            'total_score': 0.0,
            'count': 0
        }

def load_evaluation_results():
    """加载 eva_results 下 6 维度 (目录名) 下各模型 final_score.json 聚合后的结果。"""
    results = OrderedDict()

    # 动态检测维度目录
    detected_dimensions = []
    for entry in os.listdir(EVAL_DIR):
        path = os.path.join(EVAL_DIR, entry)
        if os.path.isdir(path):
            detected_dimensions.append(entry)

    # 维度排序
    preferred_order = ['M&T', 'Semantic', 'Spatial', 'PhysicalLaw', 'Complex', 'CommonSense']
    dimensions = [d for d in preferred_order if d in detected_dimensions] + [d for d in detected_dimensions if d not in preferred_order]

    for d in dimensions:
        results[d] = OrderedDict()

    final_score_count = 0

    for dimension in dimensions:
        dim_dir = os.path.join(EVAL_DIR, dimension)
        if not os.path.isdir(dim_dir):
            continue
        # 枚举模型子目录
        model_dirs = [m for m in os.listdir(dim_dir) if os.path.isdir(os.path.join(dim_dir, m))]

        # 排序
        def model_sort_key(name):
            if name.lower() == 'baseline':
                return (0, 0)
            if name.startswith('checkpoint-'):
                try:
                    num = int(name.split('checkpoint-')[-1])
                except ValueError:
                    num = 999999
                return (1, num)
            return (2, name)
        model_dirs.sort(key=model_sort_key)

        for model in model_dirs:
            model_path = os.path.join(dim_dir, model)
            # 在修复版中，我们只关心 data 是否正确（final_score.json已经被我们修改了）
            # 我们忽略 summary.json，因为它可能包含旧的错误汇总
            final_score_path = os.path.join(model_path, 'final_score.json')

            if os.path.isfile(final_score_path):
                agg_scores = _aggregate_from_final_score(final_score_path)
                results[dimension][model] = agg_scores
                final_score_count += 1
                # print(f"✓ Aggregated(final_score): {dimension} - {model}")
            else:
                pass

    print(f"\nLoaded and aggregated {final_score_count} final_score.json files.")
    return results, dimensions

def print_summary_table(results, dimensions):
    """打印6个维度的汇总表格"""
    print("\n" + "="*120)
    print("6-Dimension Evaluation Results Summary (FIXED SCORES)".center(120))
    print("="*120)
    
    # 为每个维度创建DataFrame
    dimension_dfs = {}
    for dimension in dimensions:
        if dimension in results and results[dimension]:
            df = pd.DataFrame(results[dimension]).T
            df = df.round(4)
            dimension_dfs[dimension] = df
            
            print(f"\n{'='*60}")
            print(f"{dimension} Dimension Results".center(60))
            print(f"{'='*60}")
            print(df.to_string())
    
    # 创建总分对比表
    print(f"\n{'='*80}")
    print("Total Score Comparison Across Dimensions".center(80))
    print(f"{'='*80}")
    
    total_scores_df = pd.DataFrame()
    valid_dimensions = [d for d in dimensions if d in results and results[d]]
    
    for dimension in valid_dimensions:
        total_scores = [results[dimension][model]['total_score'] for model in results[dimension].keys()]
        total_scores_df[dimension] = total_scores
    
    if valid_dimensions:
        total_scores_df.index = list(results[valid_dimensions[0]].keys())
        total_scores_df = total_scores_df.round(4)
        print(total_scores_df.to_string())
    else:
        print("No valid data found for any dimension")
    
    print("="*120 + "\n")
    return dimension_dfs, total_scores_df

def create_subplot_layout(num_dims):
    """动态计算子图布局"""
    if num_dims == 0:
        return None, None, []
    elif num_dims == 1:
        return 1, 1, [0]
    elif num_dims <= 3:
        return 1, num_dims, list(range(num_dims))
    elif num_dims <= 6:
        return 2, 3, list(range(num_dims))
    else:
        return 3, 3, list(range(min(num_dims, 9)))

def plot_total_scores_by_dimension(results, dimensions):
    """绘制各维度总分对比图"""
    valid_dimensions = [d for d in dimensions if d in results and results[d]]
    num_dims = len(valid_dimensions)
    
    if num_dims == 0:
        return
    
    rows, cols, indices = create_subplot_layout(num_dims)
    fig, axes = plt.subplots(rows, cols, figsize=(18, 10))
    
    if num_dims == 1:
        axes = [axes]
    elif rows * cols > 1:
        axes = axes.flatten()
    else:
        axes = [axes]
    
    colors = ['#FF6B6B', '#4ECDC4', '#95E1D3', '#FFA07A', '#DDA0DD', '#F0E68C']
    
    for idx, dimension in enumerate(valid_dimensions):
        if idx >= len(axes):
            break
            
        ax = axes[indices[idx]]
        
        models = list(results[dimension].keys())
        total_scores = [results[dimension][m]['total_score'] for m in models]
        
        # 色彩标记（Baseline用不同颜色）
        bar_colors = ['#FF6B6B'] + [colors[idx % len(colors)]] * (len(models) - 1)
        
        bars = ax.bar(range(len(models)), total_scores, color=bar_colors, alpha=0.75, edgecolor='black', linewidth=0.8)
        
        for j, (bar, score) in enumerate(zip(bars, total_scores)):
            height = bar.get_height()
            ax.text(bar.get_x() + bar.get_width()/2., height + 0.5,
                    f'{score:.1f}', ha='center', va='bottom', fontsize=7, rotation=45)
        
        ax.set_xlabel('Model', fontsize=10, fontweight='bold')
        ax.set_ylabel('Total Score', fontsize=10, fontweight='bold')
        ax.set_title(f'{dimension} Dimension', fontsize=12, fontweight='bold')
        ax.set_xticks(range(len(models)))
        ax.set_xticklabels(models, rotation=30, ha='right', fontsize=8)
        ax.grid(axis='y', alpha=0.3, linestyle='--')
        
        all_scores = []
        for dim in valid_dimensions:
            all_scores.extend([results[dim][m]['total_score'] for m in results[dim].keys()])
        if all_scores:
            ax.set_ylim(0, max(all_scores) * 1.15)
    
    for i in range(num_dims, len(axes)):
        axes[i].set_visible(False)
    
    plt.tight_layout()
    plt.savefig(os.path.join(EVAL_DIR, '1_total_scores_by_dimension.png'), dpi=300, bbox_inches='tight')
    plt.close()

def plot_metric_trends_by_dimension(results, dimensions):
    """绘制各维度指标趋势图"""
    valid_dimensions = [d for d in dimensions if d in results and results[d]]
    num_dims = len(valid_dimensions)
    
    if num_dims == 0:
        return
    
    rows, cols, indices = create_subplot_layout(num_dims)
    fig, axes = plt.subplots(rows, cols, figsize=(16, 12))
    
    if num_dims == 1:
        axes = [axes]
    elif rows * cols > 1:
        axes = axes.flatten()
    else:
        axes = [axes]
    
    metrics = ['skill_match_score', 'entity_match_score', 'skill_with_entity_match_score', 'exact_match_score']
    metric_names = ['Skill Match', 'Entity Match', 'Skill with Entity', 'Exact Match']
    metric_colors = ['#FF6B6B', '#4ECDC4', '#95E1D3', '#FFA07A']
    
    for idx, dimension in enumerate(valid_dimensions):
        if idx >= len(axes):
            break
            
        ax = axes[indices[idx]]
        
        models = list(results[dimension].keys())
        x = np.arange(len(models))
        
        for j, (metric, metric_name, color) in enumerate(zip(metrics, metric_names, metric_colors)):
            scores = [results[dimension][m][metric] for m in models]
            ax.plot(x, scores, 'o-', linewidth=2, markersize=4, label=metric_name, color=color, alpha=0.8)
        
        ax.set_xlabel('Model', fontsize=10, fontweight='bold')
        ax.set_ylabel('Score', fontsize=10, fontweight='bold')
        ax.set_title(f'{dimension} Dimension - Metric Trends', fontsize=12, fontweight='bold')
        ax.set_xticks(x)
        ax.set_xticklabels(models, rotation=45, ha='right', fontsize=8)
        ax.legend(fontsize=8, loc='best')
        ax.grid(True, alpha=0.3, linestyle='--')
    
    for i in range(num_dims, len(axes)):
        axes[i].set_visible(False)
    
    plt.tight_layout()
    plt.savefig(os.path.join(EVAL_DIR, '2_metric_trends_by_dimension.png'), dpi=300, bbox_inches='tight')
    plt.close()

def plot_stacked_bar_by_dimension(results, dimensions):
    """绘制各维度堆积柱状图（四个指标的贡献）"""
    valid_dimensions = [d for d in dimensions if d in results and results[d]]
    num_dims = len(valid_dimensions)
    
    if num_dims == 0:
        return
    
    rows, cols, indices = create_subplot_layout(num_dims)
    fig, axes = plt.subplots(rows, cols, figsize=(18, 10))
    
    if num_dims == 1:
        axes = [axes]
    elif rows * cols > 1:
        axes = axes.flatten()
    else:
        axes = [axes]
    
    metrics = ['skill_match_score', 'entity_match_score', 'skill_with_entity_match_score', 'exact_match_score']
    metric_names = ['Skill Match', 'Entity Match', 'Skill with Entity', 'Exact Match']
    colors = ['#FF6B6B', '#4ECDC4', '#95E1D3', '#FFA07A']
    
    for idx, dimension in enumerate(valid_dimensions):
        if idx >= len(axes):
            break
            
        ax = axes[indices[idx]]
        
        models = list(results[dimension].keys())
        x = np.arange(len(models))
        width = 0.55
        
        # 提取数据
        metric_data = []
        for metric in metrics:
            scores = np.array([results[dimension][m][metric] for m in models])
            metric_data.append(scores)
        
        # 绘制堆积柱状图
        bottoms = np.zeros(len(models))
        for j, (data, name, color) in enumerate(zip(metric_data, metric_names, colors)):
            bars = ax.bar(x, data, width, bottom=bottoms, label=name, color=color, alpha=0.85, edgecolor='white', linewidth=0.6)
            bottoms += data
        
        # 顶部总和标签
        totals = bottoms
        for xi, total in enumerate(totals):
            ax.text(x[xi], total + 0.5, f"{total:.1f}", ha='center', va='bottom', fontsize=7, rotation=45)
        
        ax.set_xlabel('Model', fontsize=10, fontweight='bold')
        ax.set_ylabel('Score', fontsize=10, fontweight='bold')
        ax.set_title(f'{dimension} Dimension - Stacked Metrics', fontsize=12, fontweight='bold')
        ax.set_xticks(x)
        ax.set_xticklabels(models, rotation=30, ha='right', fontsize=8)
        ax.legend(fontsize=8, loc='upper left')
        ax.grid(axis='y', alpha=0.3, linestyle='--')
    
    for i in range(num_dims, len(axes)):
        axes[i].set_visible(False)
    
    plt.tight_layout()
    plt.savefig(os.path.join(EVAL_DIR, '3_stacked_metrics_by_dimension.png'), dpi=300, bbox_inches='tight')
    plt.close()

def plot_improvement_by_dimension(results, dimensions):
    """绘制各维度相对于Baseline的改进图"""
    valid_dimensions = [d for d in dimensions if d in results and results[d]]
    num_dims = len(valid_dimensions)
    
    if num_dims == 0:
        return
    
    rows, cols, indices = create_subplot_layout(num_dims)
    fig, axes = plt.subplots(rows, cols, figsize=(18, 10))
    
    if num_dims == 1:
        axes = [axes]
    elif rows * cols > 1:
        axes = axes.flatten()
    else:
        axes = [axes]
    
    for idx, dimension in enumerate(valid_dimensions):
        if idx >= len(axes):
            break
            
        ax = axes[indices[idx]]
        
        models = list(results[dimension].keys())
        if len(models) == 0:
            continue
            
        baseline_score = results[dimension][models[0]]['total_score']
        
        improvements = []
        for model in models:
            if baseline_score > 0:
                improvement = ((results[dimension][model]['total_score'] - baseline_score) / baseline_score * 100)
            else:
                improvement = 0
            improvements.append(improvement)
        
        colors = ['#FF6B6B'] + ['green' if imp > 0 else 'red' for imp in improvements[1:]]
        
        bars = ax.bar(range(len(models)), improvements, color=colors, alpha=0.75, edgecolor='black', linewidth=0.8)
        
        for bar, imp in zip(bars, improvements):
            height = bar.get_height()
            ax.text(bar.get_x() + bar.get_width()/2., height + (0.6 if height >= 0 else -1.0),
                    f'{imp:.1f}%', ha='center', va='bottom' if height >= 0 else 'top', 
                    fontsize=7, fontweight='bold', rotation=0)
        
        ax.axhline(y=0, color='black', linestyle='-', linewidth=1)
        
        ax.set_xlabel('Model', fontsize=10, fontweight='bold')
        ax.set_ylabel('Improvement vs Baseline (%)', fontsize=10, fontweight='bold')
        ax.set_title(f'{dimension} Dimension - Improvement', fontsize=12, fontweight='bold')
        ax.set_xticks(range(len(models)))
        ax.set_xticklabels(models, rotation=30, ha='right', fontsize=8)
        ax.grid(axis='y', alpha=0.3, linestyle='--')
    
    for i in range(num_dims, len(axes)):
        axes[i].set_visible(False)
    
    plt.tight_layout()
    plt.savefig(os.path.join(EVAL_DIR, '4_improvement_by_dimension.png'), dpi=300, bbox_inches='tight')
    plt.close()

def plot_dimension_comparison(results, dimensions):
    """绘制维度间对比图"""
    valid_dimensions = [d for d in dimensions if d in results and results[d]]
    if len(valid_dimensions) == 0:
        return

    avg_scores = []
    best_scores = []
    best_models = []
    for dimension in valid_dimensions:
        total_scores = [results[dimension][m]['total_score'] for m in results[dimension].keys()]
        avg_scores.append(float(np.mean(total_scores)) if total_scores else 0.0)
        if total_scores:
            max_idx = int(np.argmax(total_scores))
            best_scores.append(float(total_scores[max_idx]))
            best_models.append(list(results[dimension].keys())[max_idx])
        else:
            best_scores.append(0.0)
            best_models.append('-')

    x = np.arange(len(valid_dimensions))
    width = 0.35
    fig, ax = plt.subplots(figsize=(14, 6))
    colors_left = '#4ECDC4'
    colors_right = '#FF6B6B'

    bars_avg = ax.bar(x - width/2, avg_scores, width, label='Average', color=colors_left, alpha=0.8, edgecolor='black', linewidth=0.8)
    bars_best = ax.bar(x + width/2, best_scores, width, label='Best', color=colors_right, alpha=0.8, edgecolor='black', linewidth=0.8)

    for bar, s in zip(bars_avg, avg_scores):
        ax.text(bar.get_x() + bar.get_width()/2., bar.get_height() + 0.6, f"{s:.2f}", ha='center', va='bottom', fontsize=8, rotation=45)
    for bar, s in zip(bars_best, best_scores):
        ax.text(bar.get_x() + bar.get_width()/2., bar.get_height() + 0.6, f"{s:.2f}", ha='center', va='bottom', fontsize=8, rotation=45)

    for xi, bm in zip(x, best_models):
        ax.text(xi + width/2, max(best_scores[xi], avg_scores[xi]) * 0.02, f"{bm}", ha='center', va='bottom', fontsize=7, rotation=0)

    ax.set_xlabel('Dimension', fontsize=12, fontweight='bold')
    ax.set_ylabel('Total Score', fontsize=12, fontweight='bold')
    ax.set_title('Average vs Best Score by Dimension', fontsize=14, fontweight='bold')
    ax.set_xticks(x)
    ax.set_xticklabels(valid_dimensions, rotation=30, ha='right', fontsize=10)
    ax.legend(fontsize=10)
    ax.grid(axis='y', alpha=0.3, linestyle='--')

    ymax = max(best_scores + avg_scores) if (best_scores or avg_scores) else 100.0
    ax.set_ylim(0, ymax * 1.15)

    plt.tight_layout()
    plt.savefig(os.path.join(EVAL_DIR, '5_dimension_comparison.png'), dpi=300, bbox_inches='tight')
    plt.close()

def plot_model_radar_chart(results, dimensions):
    """绘制模型性能雷达图"""
    valid_dimensions = [d for d in dimensions if d in results and results[d]]
    
    if len(valid_dimensions) < 3:
        return
    
    try:
        from math import pi
        
        model_names = set()
        for d in valid_dimensions:
            model_names.update(results[d].keys())
        model_names = sorted(list(model_names))

        def model_sort_key(name):
            if name.lower() == 'baseline':
                return (0, 0)
            if name.startswith('checkpoint-'):
                try:
                    num = int(name.split('checkpoint-')[-1])
                except ValueError:
                    num = 999999
                return (1, num)
            return (2, name)
        model_names.sort(key=model_sort_key)

        baseline = [m for m in model_names if m.lower() == 'baseline']
        checkpoints = [m for m in model_names if m.startswith('checkpoint-')]
        picked = []
        if baseline:
            picked.append(baseline[0])
        if checkpoints:
            n = len(checkpoints)
            indices = sorted(set([0, max(0, n//4-1), max(0, n//2-1), max(0, 3*n//4-1), n-1]))
            for idx in indices:
                picked.append(checkpoints[idx])
        representative_models = picked[:6] if picked else model_names[:6]
        
        fig, ax = plt.subplots(figsize=(10, 10), subplot_kw=dict(projection='polar'))
        
        angles = [n / float(len(valid_dimensions)) * 2 * pi for n in range(len(valid_dimensions))]
        angles += angles[:1]
        
        colors = ['#FF6B6B', '#4ECDC4', '#95E1D3', '#FFA07A', '#DDA0DD', '#98D8C8']
        
        for i, model in enumerate(representative_models):
            values = []
            for dimension in valid_dimensions:
                if model in results[dimension]:
                    values.append(results[dimension][model]['total_score'])
                else:
                    values.append(0)
            
            values += values[:1]
            
            ax.plot(angles, values, 'o-', linewidth=2, markersize=4, label=model, color=colors[i % len(colors)])
            ax.fill(angles, values, alpha=0.12, color=colors[i % len(colors)])
        
        ax.set_xticks(angles[:-1])
        ax.set_xticklabels(valid_dimensions, fontsize=12, fontweight='bold')
        
        all_scores = []
        for dim in valid_dimensions:
            all_scores.extend([results[dim][m]['total_score'] for m in results[dim].keys()])
        ymax = max(all_scores) if all_scores else 100.0
        ax.set_ylim(0, ymax * 1.15)
        ax.set_title('Model Performance Radar Chart', fontsize=16, fontweight='bold', pad=20)
        ax.legend(loc='upper right', bbox_to_anchor=(1.3, 1.0), fontsize=10)
        ax.grid(True)
        
        plt.tight_layout()
        plt.savefig(os.path.join(EVAL_DIR, '6_model_radar_chart.png'), dpi=300, bbox_inches='tight')
        plt.close()
    except Exception as e:
        print(f"Error creating radar chart: {e}")

def main():
    """Main function"""
    print("Starting FIXED evaluation results analysis...\n")
    
    results, dimensions = load_evaluation_results()
    
    print_summary_table(results, dimensions)
    
    print("Generating updated visualization charts...\n")
    plot_total_scores_by_dimension(results, dimensions)
    plot_metric_trends_by_dimension(results, dimensions)
    plot_stacked_bar_by_dimension(results, dimensions)
    plot_improvement_by_dimension(results, dimensions)
    
    plot_dimension_comparison(results, dimensions)
    plot_model_radar_chart(results, dimensions)

    # Overall Scores
    try:
        overall_scores = OrderedDict()
        all_models = set()
        for d in [dim for dim in dimensions if dim in results and results[dim]]:
            all_models.update(results[d].keys())
        for model in sorted(list(all_models)):
            total_weighted_score = 0.0
            total_count = 0
            for d in [dim for dim in dimensions if dim in results and results[dim]]:
                if model in results[d]:
                    res = results[d][model]
                    cnt = res.get('count', 0)
                    score = res.get('total_score', 0.0)
                    total_weighted_score += score * cnt
                    total_count += cnt
            
            overall_scores[model] = (total_weighted_score / total_count) if total_count > 0 else 0.0

        fig, ax = plt.subplots(figsize=(12, 6))
        models = list(overall_scores.keys())
        values = [overall_scores[m] for m in models]
        colors = ['#FF6B6B'] + ['#4ECDC4'] * (len(models) - 1) if models and models[0].lower() == 'baseline' else ['#4ECDC4'] * len(models)
        bars = ax.bar(range(len(models)), values, color=colors, alpha=0.8, edgecolor='black', linewidth=0.8)
        for bar, v in zip(bars, values):
            ax.text(bar.get_x() + bar.get_width()/2., bar.get_height() + 0.6, f"{v:.2f}", ha='center', va='bottom', fontsize=8, rotation=45)
        ax.set_xlabel('Model', fontsize=11, fontweight='bold')
        ax.set_ylabel('Overall Total Score', fontsize=11, fontweight='bold')
        ax.set_title('Overall Score Across Dimensions (FIXED)', fontsize=14, fontweight='bold')
        ax.set_xticks(range(len(models)))
        ax.set_xticklabels(models, rotation=30, ha='right', fontsize=9)
        ymax = max(values) if values else 100.0
        ax.set_ylim(0, ymax * 1.15)
        ax.grid(axis='y', alpha=0.3, linestyle='--')
        plt.tight_layout()
        plt.savefig(os.path.join(EVAL_DIR, '0_overall_total_scores.png'), dpi=300, bbox_inches='tight')
        plt.close()
        
        best_model = models[int(np.argmax(values))] if values else '-'
        best_score = max(values) if values else 0.0
        print(f"Best overall model: {best_model} ({best_score:.2f})")
    except Exception as e:
        print(f"[Warn] Failed to compute/plot overall scores: {e}")
    
    print("\n✓ Visualization completed on FIXED results!")
    print(f"\nCharts saved to: {EVAL_DIR}")

if __name__ == '__main__':
    main()
