#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Blind-10 LLM-as-a-Judge visualization.

Reads the latest score CSV under results/blind_10 (blind10_score_YYYYMMDD_HHMMSS.csv)
and generates comparison charts per model: average final_score, and per-dimension
scores (completeness, logical_sequence, hallucination_redundancy, granularity).

Outputs PNGs to results/blind_10/:
- blind10_models_final_score.png
- blind10_models_dimension_scores.png
- blind10_models_improvement_vs_baseline.png
"""

import os
from pathlib import Path
import re
import time
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib

# 设置中文字体（尽量兼容）
matplotlib.rcParams['font.sans-serif'] = ['SimHei', 'DejaVu Sans']
matplotlib.rcParams['axes.unicode_minus'] = False

WORKSPACE_ROOT = Path(__file__).resolve().parents[2]
EVAL_DIR = WORKSPACE_ROOT / "results" / "blind_10"


def find_latest_score_csv() -> Path:
    candidates = []
    for p in EVAL_DIR.glob("blind10_score_*.csv"):
        try:
            ts = p.stem.split("blind10_score_")[-1]
            time.strptime(ts, "%Y%m%d_%H%M%S")
            candidates.append((ts, p))
        except Exception:
            continue
    if not candidates:
        raise FileNotFoundError(f"No blind10_score_*.csv found in {EVAL_DIR}")
    candidates.sort(key=lambda x: x[0], reverse=True)
    return candidates[0][1]


def load_scores(csv_path: Path) -> pd.DataFrame:
    df = pd.read_csv(csv_path)
    # 过滤掉错误行（judge_raw 以 ERROR 开头的）
    if 'judge_raw' in df.columns:
        df = df[~df['judge_raw'].astype(str).str.startswith('ERROR')].copy()
    # 转类型
    num_cols = ['completeness','logical_sequence','hallucination_redundancy','granularity','final_score']
    for c in num_cols:
        if c in df.columns:
            df[c] = pd.to_numeric(df[c], errors='coerce')
    return df


def sort_models(models):
    def key(name: str):
        nl = name.lower()
        if nl == 'baseline':
            return (0, 0)
        if name.startswith('checkpoint-'):
            try:
                num = int(name.split('checkpoint-')[-1])
            except Exception:
                num = 10**9
            return (1, num)
        return (2, name)
    return sorted(models, key=key)


def plot_final_score(df: pd.DataFrame, out_dir: Path):
    grp = df.groupby('model_name')['final_score'].mean().fillna(0.0)
    models = sort_models(list(grp.index))
    values = [grp[m] for m in models]

    plt.figure(figsize=(10, 5))
    colors = ['#FF6B6B'] + ['#4ECDC4'] * (len(models)-1) if models and models[0].lower()=='baseline' else ['#4ECDC4']*len(models)
    bars = plt.bar(range(len(models)), values, color=colors, alpha=0.85, edgecolor='black', linewidth=0.8)
    for i, v in enumerate(values):
        plt.text(i, v + 0.02, f"{v:.2f}", ha='center', va='bottom', fontsize=8, rotation=45)
    plt.xticks(range(len(models)), models, rotation=30, ha='right')
    plt.ylabel('Average Final Score')
    plt.title('Blind-10: Average Final Score per Model')
    ymax = max(values) if values else 1.0
    plt.ylim(0, ymax * 1.15)
    plt.grid(axis='y', alpha=0.3, linestyle='--')
    out_path = out_dir / 'blind10_models_final_score.png'
    plt.tight_layout()
    plt.savefig(out_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"✓ Saved: {out_path}")


def plot_dimensions(df: pd.DataFrame, out_dir: Path):
    dims = ['completeness','logical_sequence','hallucination_redundancy','granularity']
    grp = df.groupby('model_name')[dims].mean().fillna(0.0)
    models = sort_models(list(grp.index))
    values = grp.loc[models]

    # 叠加柱状图展示四维度平均分
    plt.figure(figsize=(12, 6))
    x = range(len(models))
    colors = ['#FF6B6B', '#4ECDC4', '#95E1D3', '#FFA07A']
    bottom = [0]*len(models)
    for j, dim in enumerate(dims):
        v = values[dim].tolist()
        plt.bar(x, v, bottom=bottom, label=dim, color=colors[j%len(colors)], alpha=0.85, edgecolor='white', linewidth=0.6)
        bottom = [bottom[k] + v[k] for k in range(len(v))]
    # 顶部总和标签
    for i, tot in enumerate(bottom):
        plt.text(i, tot + 0.05, f"{tot:.2f}", ha='center', va='bottom', fontsize=8, rotation=45)
    plt.xticks(x, models, rotation=30, ha='right')
    plt.ylabel('Average Score (stacked)')
    plt.title('Blind-10: Average Dimension Scores per Model (stacked)')
    ymax = max(bottom) if bottom else 1.0
    plt.ylim(0, ymax * 1.15)
    plt.legend(fontsize=8)
    plt.grid(axis='y', alpha=0.3, linestyle='--')
    out_path = out_dir / 'blind10_models_dimension_scores.png'
    plt.tight_layout()
    plt.savefig(out_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"✓ Saved: {out_path}")


def plot_improvement(df: pd.DataFrame, out_dir: Path):
    grp = df.groupby('model_name')['final_score'].mean().fillna(0.0)
    models = sort_models(list(grp.index))
    values = [grp[m] for m in models]
    if not values:
        return
    baseline = values[0]
    improvements = []
    for v in values:
        if baseline > 0:
            improvements.append((v - baseline) / baseline * 100)
        else:
            improvements.append(0.0)

    plt.figure(figsize=(10, 5))
    colors = ['#FF6B6B'] + ['green' if imp>0 else 'red' for imp in improvements[1:]]
    bars = plt.bar(range(len(models)), improvements, color=colors, alpha=0.85, edgecolor='black', linewidth=0.8)
    for i, imp in enumerate(improvements):
        plt.text(i, imp + (0.6 if imp>=0 else -1.0), f"{imp:.1f}%", ha='center', va='bottom' if imp>=0 else 'top', fontsize=8)
    plt.axhline(y=0, color='black', linestyle='-', linewidth=1)
    plt.xticks(range(len(models)), models, rotation=30, ha='right')
    plt.ylabel('Improvement vs Baseline (%)')
    plt.title('Blind-10: Improvement vs Baseline (Final Score)')
    ymin = min(improvements)
    ymax = max(improvements)
    plt.ylim(ymin - abs(ymin)*0.15, ymax * 1.15)
    plt.grid(axis='y', alpha=0.3, linestyle='--')
    out_path = out_dir / 'blind10_models_improvement_vs_baseline.png'
    plt.tight_layout()
    plt.savefig(out_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"✓ Saved: {out_path}")


def main():
    try:
        csv_path = find_latest_score_csv()
    except Exception as e:
        print(f"[ERROR] {e}")
        return 2
    print(f"Using score CSV: {csv_path}")
    df = load_scores(csv_path)
    if df.empty:
        print("[WARN] No valid rows (all errors or empty). No charts generated.")
        return 0
    # 生成图表
    plot_final_score(df, EVAL_DIR)
    plot_dimensions(df, EVAL_DIR)
    plot_improvement(df, EVAL_DIR)
    print(f"\n✓ Charts saved to: {EVAL_DIR}\n")
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
