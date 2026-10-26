#!/usr/bin/env python3
"""Audit Stage 2 Entity Type-Normalization contract sensitivity on VLABench.

This script evaluates stored VLABench predictions (`output.json`) against the
official VLABench ground-truth skill sequences (`dataset/vlm_evaluation_v1.0`)
under two symbolic contract regimes:
  1. Strict Unnormalized Contract (\\psi_raw): exact Python type and value comparison
     (where quoted JSON string IDs `"4"` vs. ground-truth integer IDs `4` fail
     strict Python `==` equality inside VLABench's `get_final_score`).
  2. Entity Type-Normalized Contract (\\psi_norm): canonicalizes integer/string
     entity IDs (`str(v).strip()`) before invoking VLABench's `get_final_score`,
     eliminating the serialization type-mismatch penalty while strictly preserving
     all task-graph ordering and sequence-alignment constraints.
"""

from __future__ import annotations

import argparse
import csv
import importlib.util
import json
from pathlib import Path
from typing import Any


CODE_ROOT = Path(__file__).resolve().parents[2]
VLABENCH_UTILS = CODE_ROOT / "VLABench/VLABench/evaluation/utils.py"
SEQ_INDEP_PATH = CODE_ROOT / "VLABench/VLABench/configs/evaluation/seq_independent_task.json"
GT_ROOT = CODE_ROOT / "dataset/vlm_evaluation_v1.0"
SWEEP_ROOT = CODE_ROOT / "results/evals/vlabench/leap/checkpoint_sweep_qwen3_2b"
LORA_ROOT = CODE_ROOT / "results/evals/vlabench/leap/lora_ablation/standard"

DIMENSIONS: list[tuple[str, str, str, str]] = [
    ("M&T", "M&T", "MandT", "M&T"),
    ("Semantic", "Semantic", "Semantic", "Semantic"),
    ("Spatial", "Spatial", "Spatial", "Spatial"),
    ("PhysicalLaw", "PhysicsLaw", "PhysicsLaw", "PhysicsLaw"),
    ("Complex", "Complex", "Complex", "Complex"),
    ("CommonSense", "CommonSense", "CommenSence", "CommenSence"),
]


def load_get_final_score():
    spec = importlib.util.spec_from_file_location("vlabench_eval_utils", VLABENCH_UTILS)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Unable to load VLABench utils from {VLABENCH_UTILS}")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod.get_final_score


def normalize_skill_sequence(seq: list[dict[str, Any]] | None) -> list[dict[str, Any]]:
    if not isinstance(seq, list):
        return []
    res: list[dict[str, Any]] = []
    for step in seq:
        if isinstance(step, dict):
            s = dict(step)
            if "params" in s and isinstance(s["params"], dict):
                s["params"] = {
                    k: str(v).strip() if isinstance(v, (int, str)) else v
                    for k, v in s["params"].items()
                }
            res.append(s)
    return res


def count_param_types(seq: list[dict[str, Any]] | None) -> tuple[int, int]:
    str_cnt = 0
    int_cnt = 0
    if not isinstance(seq, list):
        return str_cnt, int_cnt
    for step in seq:
        if isinstance(step, dict) and isinstance(step.get("params"), dict):
            for val in step["params"].values():
                if isinstance(val, str):
                    str_cnt += 1
                elif isinstance(val, int):
                    int_cnt += 1
    return str_cnt, int_cnt


def evaluate_model(
    model_name: str,
    model_paths: dict[str, Path],
    get_final_score_fn: Any,
    seq_indep: set[str],
) -> dict[str, Any]:
    dim_raw_unrounded: dict[str, float] = {}
    dim_norm_unrounded: dict[str, float] = {}
    dim_raw: dict[str, float] = {}
    dim_norm: dict[str, float] = {}
    dim_skill: list[float] = []
    dim_ent_raw: list[float] = []
    dim_ent_norm: list[float] = []
    dim_joint_raw: list[float] = []
    dim_joint_norm: list[float] = []
    dim_exact_raw: list[float] = []
    dim_exact_norm: list[float] = []

    total_episodes = 0
    parsed_episodes = 0
    total_str_params = 0
    total_int_params = 0

    for dim_key, _, _, gt_dir_name in DIMENSIONS:
        pred_path = model_paths[dim_key]
        pred_data = json.loads(pred_path.read_text(encoding="utf-8"))
        scores_raw: list[float] = []
        scores_norm: list[float] = []
        sk_list: list[float] = []
        er_list: list[float] = []
        en_list: list[float] = []
        jr_list: list[float] = []
        jn_list: list[float] = []
        xr_list: list[float] = []
        xn_list: list[float] = []

        for task_name, episodes in pred_data.items():
            if task_name == "benchmeatinfo" or not isinstance(episodes, dict):
                continue
            gt_dep = "Seq-independent" if task_name in seq_indep else "Sequential"
            for ep_id, pred in episodes.items():
                gt_file = GT_ROOT / gt_dir_name / task_name / f"example{ep_id}" / "output" / "operation_sequence.json"
                if not gt_file.is_file():
                    continue
                gt_obj = json.loads(gt_file.read_text(encoding="utf-8"))
                gt_seq = gt_obj["skill_sequence"]
                total_episodes += 1

                pred_seq = pred.get("skill_sequence") if isinstance(pred, dict) else None
                if not isinstance(pred_seq, list) or "format_error" in (pred if isinstance(pred, dict) else {}):
                    scores_raw.append(0.0)
                    scores_norm.append(0.0)
                    sk_list.append(0.0)
                    er_list.append(0.0)
                    en_list.append(0.0)
                    jr_list.append(0.0)
                    jn_list.append(0.0)
                    xr_list.append(0.0)
                    xn_list.append(0.0)
                    continue

                parsed_episodes += 1
                s_c, i_c = count_param_types(pred_seq)
                total_str_params += s_c
                total_int_params += i_c

                try:
                    raw_res = get_final_score_fn(gt_seq, pred_seq, dependency=gt_dep)
                except Exception:
                    raw_res = {
                        "skill_match_score": 0.0,
                        "entity_match_score": 0.0,
                        "skill_with_entity_match_score": 0.0,
                        "exact_match_score": 0.0,
                        "total_score": 0.0,
                    }

                gt_seq_norm = normalize_skill_sequence(gt_seq)
                pred_seq_norm = normalize_skill_sequence(pred_seq)
                try:
                    norm_res = get_final_score_fn(gt_seq_norm, pred_seq_norm, dependency=gt_dep)
                except Exception:
                    norm_res = {
                        "skill_match_score": 0.0,
                        "entity_match_score": 0.0,
                        "skill_with_entity_match_score": 0.0,
                        "exact_match_score": 0.0,
                        "total_score": 0.0,
                    }

                scores_raw.append(float(raw_res["total_score"]))
                scores_norm.append(float(norm_res["total_score"]))
                sk_list.append(float(norm_res["skill_match_score"]))
                er_list.append(float(raw_res["entity_match_score"]))
                en_list.append(float(norm_res["entity_match_score"]))
                jr_list.append(float(raw_res["skill_with_entity_match_score"]))
                jn_list.append(float(norm_res["skill_with_entity_match_score"]))
                xr_list.append(float(raw_res["exact_match_score"]))
                xn_list.append(float(norm_res["exact_match_score"]))

        dim_raw_unrounded[dim_key] = sum(scores_raw) / len(scores_raw)
        dim_norm_unrounded[dim_key] = sum(scores_norm) / len(scores_norm)
        dim_raw[dim_key] = round(dim_raw_unrounded[dim_key], 2)
        dim_norm[dim_key] = round(dim_norm_unrounded[dim_key], 2)
        dim_skill.append(sum(sk_list) / len(sk_list))
        dim_ent_raw.append(sum(er_list) / len(er_list))
        dim_ent_norm.append(sum(en_list) / len(en_list))
        dim_joint_raw.append(sum(jr_list) / len(jr_list))
        dim_joint_norm.append(sum(jn_list) / len(jn_list))
        dim_exact_raw.append(sum(xr_list) / len(xr_list))
        dim_exact_norm.append(sum(xn_list) / len(xn_list))

    macro_raw = round(sum(dim_raw_unrounded.values()) / len(dim_raw_unrounded), 2)
    macro_norm = round(sum(dim_norm_unrounded.values()) / len(dim_norm_unrounded), 2)

    return {
        "model": model_name,
        "total_episodes": total_episodes,
        "parsed_episodes": parsed_episodes,
        "parse_survival_pct": round(100.0 * parsed_episodes / total_episodes, 2),
        "param_type_counts": {
            "quoted_string_ids": total_str_params,
            "unquoted_int_ids": total_int_params,
        },
        "dimensions_psi_raw": dim_raw,
        "dimensions_psi_norm": dim_norm,
        "macro_psi_raw": macro_raw,
        "macro_psi_norm": macro_norm,
        "macro_delta_norm_minus_raw": round(macro_norm - macro_raw, 2),
        "macro_submetrics_pct": {
            "skill_match": round(sum(dim_skill) / len(dim_skill), 2),
            "entity_match_raw": round(sum(dim_ent_raw) / len(dim_ent_raw), 2),
            "entity_match_norm": round(sum(dim_ent_norm) / len(dim_ent_norm), 2),
            "joint_step_match_raw": round(sum(dim_joint_raw) / len(dim_joint_raw), 2),
            "joint_step_match_norm": round(sum(dim_joint_norm) / len(dim_joint_norm), 2),
            "exact_plan_match_raw": round(sum(dim_exact_raw) / len(dim_exact_raw), 2),
            "exact_plan_match_norm": round(sum(dim_exact_norm) / len(dim_exact_norm), 2),
        },
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--out-dir",
        default=str(CODE_ROOT / "results/audits/stage2_entity_contract_2026-10-26"),
        help="Directory to write summary.json and summary.csv.",
    )
    args = parser.parse_args()

    get_final_score_fn = load_get_final_score()
    seq_indep = set(json.loads(SEQ_INDEP_PATH.read_text(encoding="utf-8")))

    models = {
        "Qwen3-VL-2B (Base)": {
            dim_key: SWEEP_ROOT / sweep_dir / "Baseline" / "output.json"
            for dim_key, sweep_dir, _, _ in DIMENSIONS
        },
        "Structured Steps + Full SFT (5k)": {
            dim_key: SWEEP_ROOT / sweep_dir / "checkpoint-5000" / "output.json"
            for dim_key, sweep_dir, _, _ in DIMENSIONS
        },
        "Structured Steps + Full SFT (7k)": {
            dim_key: SWEEP_ROOT / sweep_dir / "checkpoint-7000" / "output.json"
            for dim_key, sweep_dir, _, _ in DIMENSIONS
        },
        "Structured Steps + LoRA (r=16, 5k)": {
            dim_key: LORA_ROOT / "step5000" / lora_dir / "lora_standard_step5000" / "output.json"
            for dim_key, _, lora_dir, _ in DIMENSIONS
        },
        "Structured Steps + LoRA (r=16, 6k)": {
            dim_key: LORA_ROOT / "step6000" / lora_dir / "lora_standard_step6000" / "output.json"
            for dim_key, _, lora_dir, _ in DIMENSIONS
        },
        "Structured Steps + LoRA (r=16, 7k)": {
            dim_key: LORA_ROOT / "step7000" / lora_dir / "lora_standard_step7000" / "output.json"
            for dim_key, _, lora_dir, _ in DIMENSIONS
        },
    }

    results = [
        evaluate_model(model_name, paths, get_final_score_fn, seq_indep)
        for model_name, paths in models.items()
    ]

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    json_path = out_dir / "summary.json"
    csv_path = out_dir / "summary.csv"

    json_path.write_text(json.dumps(results, ensure_ascii=False, indent=2), encoding="utf-8")

    with csv_path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow([
            "model",
            "quoted_string_ids",
            "unquoted_int_ids",
            "macro_psi_raw",
            "macro_psi_norm",
            "macro_delta",
            "M&T_norm",
            "Semantic_norm",
            "Spatial_norm",
            "PhysicalLaw_norm",
            "Complex_norm",
            "CommonSense_norm",
            "skill_match_pct",
            "entity_match_raw_pct",
            "entity_match_norm_pct",
            "joint_step_match_norm_pct",
            "exact_plan_match_norm_pct",
        ])
        for row in results:
            d = row["dimensions_psi_norm"]
            s = row["macro_submetrics_pct"]
            c = row["param_type_counts"]
            writer.writerow([
                row["model"],
                c["quoted_string_ids"],
                c["unquoted_int_ids"],
                row["macro_psi_raw"],
                row["macro_psi_norm"],
                row["macro_delta_norm_minus_raw"],
                d["M&T"],
                d["Semantic"],
                d["Spatial"],
                d["PhysicalLaw"],
                d["Complex"],
                d["CommonSense"],
                s["skill_match"],
                s["entity_match_raw"],
                s["entity_match_norm"],
                s["joint_step_match_norm"],
                s["exact_plan_match_norm"],
            ])

    print(json.dumps(results, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
