#!/usr/bin/env python3
"""Audit parser-gating sensitivity from stored VLABench output.json files.

This script is intentionally diagnostic. It reads existing model outputs,
counts strict parser failures, and checks whether failed examples can be
recovered by limited relaxed JSON extraction. It does not modify input files
or recompute benchmark scores.
"""

from __future__ import annotations

import argparse
import csv
import json
import re
import sys
from collections import Counter
from pathlib import Path
from typing import Any


CODE_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_ROOT = CODE_ROOT / "results/evals/vlabench/leap/checkpoint_sweep_qwen3_2b"


def load_json(path: Path) -> Any:
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def is_sequence_like(value: Any) -> bool:
    if not isinstance(value, list):
        return False
    for item in value:
        if not isinstance(item, dict):
            return False
        if not isinstance(item.get("name"), str):
            return False
        params = item.get("params", {})
        if params is not None and not isinstance(params, dict):
            return False
    return True


def first_balanced_array(text: str) -> tuple[str | None, str]:
    start = text.find("[")
    if start < 0:
        return None, "no_array_candidate"

    depth = 0
    in_string = False
    escape = False
    for idx, char in enumerate(text[start:], start):
        if in_string:
            if escape:
                escape = False
            elif char == "\\":
                escape = True
            elif char == '"':
                in_string = False
            continue

        if char == '"':
            in_string = True
        elif char == "[":
            depth += 1
        elif char == "]":
            depth -= 1
            if depth == 0:
                return text[start : idx + 1], "balanced_array"

    return None, "truncated_or_unclosed_json"


def parse_json_candidate(candidate: str) -> Any | None:
    try:
        value = json.loads(candidate.strip())
    except Exception:
        return None
    return value if is_sequence_like(value) else None


def relaxed_extract(text: Any) -> tuple[Any | None, str]:
    if not isinstance(text, str):
        return None, "non_string_origin_output"
    if not text.strip():
        return None, "empty_origin_output"

    fenced_blocks = re.findall(r"```(?:json)?\s*([\s\S]*?)```", text, flags=re.IGNORECASE)
    for block in fenced_blocks:
        value = parse_json_candidate(block)
        if value is not None:
            return value, "fenced_any_language"

    whole_value = parse_json_candidate(text)
    if whole_value is not None:
        return whole_value, "whole_text_json"

    candidate, reason = first_balanced_array(text)
    if candidate is None:
        return None, reason
    value = parse_json_candidate(candidate)
    if value is not None:
        return value, reason
    return None, "malformed_json"


def default_paths(root: Path) -> list[Path]:
    paths = list(root.glob("*/Baseline/output.json"))
    paths.extend(root.glob("*/checkpoint-5000/output.json"))
    return sorted(paths)


def label_for(path: Path) -> tuple[str, str]:
    dimension = path.parent.parent.name
    run = path.parent.name
    return dimension, run


def summarize(path: Path) -> dict[str, Any]:
    payload = load_json(path)
    total = 0
    strict_success = 0
    format_error = 0
    origin_missing = 0
    recoverable = 0
    reason_counter: Counter[str] = Counter()
    recovered_lengths: list[int] = []

    for task_name, examples in payload.items():
        if task_name == "benchmeatinfo" or not isinstance(examples, dict):
            continue
        for pred in examples.values():
            total += 1
            if isinstance(pred, dict) and is_sequence_like(pred.get("skill_sequence")):
                strict_success += 1
                continue
            if isinstance(pred, dict) and "format_error" in pred:
                format_error += 1
                origin = pred.get("origin_output")
                if origin is None:
                    origin_missing += 1
                value, reason = relaxed_extract(origin)
                reason_counter[reason] += 1
                if value is not None:
                    recoverable += 1
                    recovered_lengths.append(len(value))
                continue
            reason_counter["nonstandard_prediction_record"] += 1

    dimension, run = label_for(path)
    return {
        "dimension": dimension,
        "run": run,
        "output_json": str(path),
        "total_examples": total,
        "strict_success": strict_success,
        "format_error": format_error,
        "origin_missing": origin_missing,
        "recoverable_relaxed": recoverable,
        "format_error_rate": format_error / total if total else 0.0,
        "recoverable_among_format_error": recoverable / format_error if format_error else 0.0,
        "recovered_mean_length": (
            sum(recovered_lengths) / len(recovered_lengths) if recovered_lengths else 0.0
        ),
        "failure_reason_counts": dict(sorted(reason_counter.items())),
    }


def write_csv(rows: list[dict[str, Any]], out: Any) -> None:
    fieldnames = [
        "dimension",
        "run",
        "total_examples",
        "strict_success",
        "format_error",
        "format_error_rate",
        "recoverable_relaxed",
        "recoverable_among_format_error",
        "origin_missing",
        "recovered_mean_length",
        "failure_reason_counts",
        "output_json",
    ]
    writer = csv.DictWriter(out, fieldnames=fieldnames)
    writer.writeheader()
    for row in rows:
        printable = dict(row)
        printable["failure_reason_counts"] = json.dumps(
            printable["failure_reason_counts"], ensure_ascii=False, sort_keys=True
        )
        writer.writerow(printable)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "output_json",
        nargs="*",
        help="Specific output.json paths. Defaults to Baseline and checkpoint-5000 under the sweep root.",
    )
    parser.add_argument(
        "--root",
        default=str(DEFAULT_ROOT),
        help="Sweep root used when no explicit output_json paths are supplied.",
    )
    parser.add_argument("--out-json", default=None, help="Optional JSON summary path.")
    parser.add_argument("--out-csv", default=None, help="Optional CSV summary path.")
    args = parser.parse_args()

    paths = [Path(p) for p in args.output_json] if args.output_json else default_paths(Path(args.root))
    paths = [p for p in paths if p.exists()]
    if not paths:
        print("no_output_json_found", file=sys.stderr)
        return 2

    rows = [summarize(path) for path in paths]

    if args.out_json:
        out_json = Path(args.out_json)
        out_json.parent.mkdir(parents=True, exist_ok=True)
        with out_json.open("w", encoding="utf-8") as f:
            json.dump(rows, f, ensure_ascii=False, indent=2)

    if args.out_csv:
        out_csv = Path(args.out_csv)
        out_csv.parent.mkdir(parents=True, exist_ok=True)
        with out_csv.open("w", newline="", encoding="utf-8") as f:
            write_csv(rows, f)
    else:
        write_csv(rows, sys.stdout)

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
