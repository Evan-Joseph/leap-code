#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Analyze output results using LLM-as-a-Judge and export scores to CSV.

Inputs:
- Find the latest outputs CSV under eva_results/blind_10/ named like
  blind10_outputs_YYYYMMDD_HHMMSS.csv (or use --input-csv to specify).

Outputs:
- Create a score CSV alongside, named blind10_score_YYYYMMDD_HHMMSS.csv,
  containing: model_name, sample_idx, judge_raw, completeness, logical_sequence,
  hallucination_redundancy, granularity, final_score.

LLM API:
- Defaults to OpenAI-compatible endpoint at https://jeniya.top/v1/chat/completions
- Configure via env:
  - JENIYA_API_BASE (default: https://jeniya.top/v1)
  - JENIYA_API_KEY  (preferred) or OPENAI_API_KEY (fallback)
  - JENIYA_MODEL    (default: gpt-4o-mini)

Notes:
- The input CSV is expected to have columns:
  model_name, sample_idx, image_paths, prompt_text, model_output, reference_steps
- Task name is parsed from prompt_text using the "Task name:" marker.
- The first-frame image (image_paths) is embedded as a data URI in the judge request.
"""

from __future__ import annotations

import argparse
import base64
import csv
import json
import os
import re
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

try:
    import requests  # type: ignore
except Exception as e:  # pragma: no cover - helpful message if requests missing
    print("This script requires the 'requests' package. Install it via: pip install requests", file=sys.stderr)
    raise


# ---------------------------- Config & Prompts ---------------------------- #

DEFAULT_API_BASE = os.getenv("JENIYA_API_BASE", "https://jeniya.top/v1")
DEFAULT_API_URL = DEFAULT_API_BASE.rstrip("/") + "/chat/completions"
DEFAULT_MODEL = os.getenv("JENIYA_MODEL", "gpt-4o-mini")
API_KEY = os.getenv("JENIYA_API_KEY") or os.getenv("OPENAI_API_KEY")

SYSTEM_PROMPT = (
    "You are an expert evaluator for Embodied AI and Robotic Task Planning. "
    "Your goal is to objectively assess the quality of a robot's action plan compared to a golden reference.\n\n"
    "You will be provided with:\n"
    "1. **Task Instruction**: The command given to the robot.\n"
    "2. **Reference Plan (Ground Truth)**: The correct, human-annotated steps.\n"
    "3. **Generated Plan**: The plan produced by the model being evaluated.\n\n"
    "### Evaluation Mechanism\n"
    "You must evaluate the Generated Plan on 4 specific dimensions. For each dimension, assign a score from 1 to 5 based on the rubric below.\n\n"
    "### Scoring Rubric\n\n"
    "#### 1. Completeness (Critical)\n"
    "- **5**: Covers all key milestones defined in the Reference. No critical steps missing.\n"
    "- **3**: Misses minor intermediate steps but achieves the main goal.\n"
    "- **1**: Misses critical steps (e.g., \"Grasp object\" before \"Move object\"), making the task impossible.\n\n"
    "#### 2. Logical Sequence (Critical)\n"
    "- **5**: Steps follow strict physical causality (e.g., open door -> enter).\n"
    "- **3**: Minor reordering that doesn't break physics but is inefficient.\n"
    "- **1**: Severe causality errors (e.g., attempting to fold a shirt before picking it up) or dead loops.\n\n"
    "#### 3. Hallucination & Redundancy\n"
    "- **5**: Concise. No repetition. No interaction with non-existent objects.\n"
    "- **3**: Contains 1-2 redundant steps (e.g., \"Move hand\" explicitly when \"Grasp\" implies it).\n"
    "- **1**: Severe hallucination (inventing objects not in the scene) or infinite loops (e.g., repeating \"Fold sleeve\" 5 times).\n\n"
    "#### 4. Granularity Alignment\n"
    "- **5**: The level of detail matches the Reference (e.g., if Reference says \"Fold sleeve\", Generated shouldn't say \"Move effector to x,y,z\").\n"
    "- **1**: Too high-level (abstract) or too low-level (control signals) compared to Reference.\n\n"
    "### Output Format\n"
    "Output ONLY a JSON object with the following structure:\n"
    "{\n"
    "  \"reasoning\": \"A concise analysis comparing the Generated Plan vs. Reference, highlighting missing steps or errors.\",\n"
    "  \"scores\": {\n"
    "    \"completeness\": <int>,\n"
    "    \"logical_sequence\": <int>,\n"
    "    \"hallucination_redundancy\": <int>,\n"
    "    \"granularity\": <int>\n"
    "  },\n"
    "  \"final_score\": <float, average of the above>\n"
    "}\n"
)

USER_PROMPT_TEMPLATE = (
    "Please evaluate the following sample:\n\n"
    "Task Name:\n{task_name}\n\n"
    "Reference Steps (Ground Truth):\n{reference_steps}\n\n"
    "Generated Plan:\n{generated_plan}\n\n"
    "Constraint: Be strict. If the generated plan contains loops or repetitions not present in the reference, penalize the \"Hallucination & Redundancy\" score heavily.\n"
)


# ---------------------------- Data Structures ---------------------------- #

@dataclass
class JudgeResult:
    model_name: str
    sample_idx: int
    judge_raw: str
    completeness: Optional[int]
    logical_sequence: Optional[int]
    hallucination_redundancy: Optional[int]
    granularity: Optional[int]
    final_score: Optional[float]


# ---------------------------- Utility Functions ---------------------------- #

def find_latest_outputs_csv(dir_path: Path) -> Path:
    """Find the latest outputs CSV by timestamp in filename within a directory."""
    candidates = []
    for p in dir_path.glob("blind10_outputs_*.csv"):
        try:
            ts = p.stem.split("blind10_outputs_")[-1]
            # Expect YYYYMMDD_HHMMSS
            time.strptime(ts, "%Y%m%d_%H%M%S")
            candidates.append((ts, p))
        except Exception:
            continue
    if not candidates:
        raise FileNotFoundError(f"No outputs CSV found in {dir_path}")
    candidates.sort(key=lambda x: x[0], reverse=True)
    return candidates[0][1]


def derive_score_csv_path(outputs_csv: Path) -> Path:
    ts = outputs_csv.stem.split("blind10_outputs_")[-1]
    return outputs_csv.parent / f"blind10_score_{ts}.csv"


def parse_task_name(prompt_text: str) -> str:
    """Extract the task name from the prompt_text using the 'Task name:' marker."""
    # Preferred: between 'Task name:' and 'Based on'
    m = re.search(r"Task name:\s*(.*?)\s*\.\s*Based on", prompt_text, flags=re.IGNORECASE | re.DOTALL)
    if m:
        return m.group(1).strip()
    # Fallback: up to the next period
    m2 = re.search(r"Task name:\s*([^\n\.]+)\.", prompt_text, flags=re.IGNORECASE)
    if m2:
        return m2.group(1).strip()
    # Last fallback: after Task name: to end of line
    m3 = re.search(r"Task name:\s*(.*)$", prompt_text, flags=re.IGNORECASE | re.MULTILINE)
    if m3:
        return m3.group(1).strip()
    return ""


def load_image_as_data_uri(image_path: Path) -> Optional[str]:
    try:
        with open(image_path, "rb") as f:
            b64 = base64.b64encode(f.read()).decode("utf-8")
        ext = image_path.suffix.lower().lstrip(".") or "png"
        if ext not in {"png", "jpg", "jpeg", "webp"}:
            ext = "png"
        return f"data:image/{ext};base64,{b64}"
    except Exception:
        return None


def extract_json_obj(s: str) -> Dict[str, Any]:
    """Attempt to extract a JSON object from a model response string."""
    s_strip = s.strip()
    # Remove code fences if present
    if s_strip.startswith("```"):
        # Remove first fence
        s_strip = re.sub(r"^```[a-zA-Z0-9_\-]*\n", "", s_strip)
        # Remove ending fence
        s_strip = re.sub(r"\n```\s*$", "", s_strip)
    # Try direct parse first
    try:
        return json.loads(s_strip)
    except Exception:
        pass
    # Bracket matching to find first JSON object
    start = s_strip.find("{")
    end = s_strip.rfind("}")
    if start != -1 and end != -1 and end > start:
        frag = s_strip[start : end + 1]
        # Try progressively shrinking trailing content to handle extra text after JSON
        for k in range(len(frag), 1, -1):
            try:
                return json.loads(frag[:k])
            except Exception:
                continue
    # Give up, return empty structure
    return {}


def average_scores(scores: Dict[str, Any]) -> Optional[float]:
    try:
        keys = ["completeness", "logical_sequence", "hallucination_redundancy", "granularity"]
        vals = [float(scores[k]) for k in keys if k in scores]
        if len(vals) != 4:
            return None
        return sum(vals) / 4.0
    except Exception:
        return None


def to_int_or_none(x: Any) -> Optional[int]:
    try:
        return int(x)
    except Exception:
        return None


# ---------------------------- LLM Judge Caller ---------------------------- #

def call_llm_judge(
    model: str,
    api_url: str,
    api_key: str,
    task_name: str,
    reference_steps: str,
    generated_plan: str,
    image_data_uri: Optional[str] = None,
    timeout: float = 60.0,
    max_retries: int = 10,
    retry_forever: bool = False,
    retry_sleep: float = 1.0,
) -> Tuple[str, Dict[str, Any]]:
    """
    Call the OpenAI-compatible chat/completions API with system+user messages.

    Returns: (raw_text_response, parsed_json)
    """
    headers = {
        "Authorization": f"Bearer {api_key}",
        "Content-Type": "application/json",
    }

    user_text = USER_PROMPT_TEMPLATE.format(
        task_name=task_name or "(Unknown Task)",
        reference_steps=reference_steps.strip(),
        generated_plan=generated_plan.strip(),
    )

    # Compose content parts (text + optional image)
    user_content: List[Dict[str, Any]] = [{"type": "text", "text": user_text}]
    if image_data_uri:
        user_content.append({
            "type": "image_url",
            "image_url": {"url": image_data_uri}
        })

    payload = {
        "model": model,
        "messages": [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": user_content},
        ],
        "temperature": 0,
    }

    last_err: Optional[Exception] = None
    attempt = 0
    while True:
        attempt += 1
        try:
            resp = requests.post(api_url, headers=headers, data=json.dumps(payload), timeout=timeout)
            if resp.status_code >= 500:
                # Server error, retry
                last_err = RuntimeError(f"Server error {resp.status_code}: {resp.text[:200]}")
                if (not retry_forever) and (attempt >= max_retries):
                    break
                time.sleep(retry_sleep)
                continue
            if resp.status_code == 429:
                # Rate limited, retry
                last_err = RuntimeError(f"Rate limited: {resp.text[:200]}")
                if (not retry_forever) and (attempt >= max_retries):
                    break
                time.sleep(retry_sleep)
                continue
            resp.raise_for_status()
            data = resp.json()
            content = data.get("choices", [{}])[0].get("message", {}).get("content", "")
            parsed = extract_json_obj(content)
            return content, parsed
        except Exception as e:
            last_err = e
            if (not retry_forever) and (attempt >= max_retries):
                break
            time.sleep(retry_sleep)
    # If we reach here, failed
    raise RuntimeError(f"LLM judge call failed after {max_retries} attempts: {last_err}")


# ---------------------------- Main Processing ---------------------------- #

def process_csv(
    inputs_csv: Path,
    outputs_csv: Path,
    api_url: str,
    api_key: str,
    model: str,
    root_dir: Path,
    max_samples: Optional[int] = None,
    sample_filter_model: Optional[List[str]] = None,
    sleep_sec: float = 0.0,
    warn_missing_image: bool = False,
    retry_forever: bool = False,
    max_retries: int = 10,
    retry_sleep: float = 1.0,
    stop_on_error: bool = True,
    resume: bool = True,
) -> None:
    # Prepare cache for image data URI by image path to avoid repeated encoding
    image_cache: Dict[Path, Optional[str]] = {}
    warned_images: set[Path] = set()

    # Ensure output directory exists
    outputs_csv.parent.mkdir(parents=True, exist_ok=True)

    # Read inputs
    rows: List[Dict[str, str]] = []
    with open(inputs_csv, newline='', encoding='utf-8') as f:
        reader = csv.DictReader(f)
        for row in reader:
            rows.append(row)

    # Optional filtering by model_name
    if sample_filter_model:
        keep = set(sample_filter_model)
        rows = [r for r in rows if r.get("model_name") in keep]

    # Optional limit
    if max_samples is not None:
        rows = rows[:max_samples]

    # Process and write output incrementally
    fieldnames = [
        "model_name",
        "sample_idx",
        "judge_raw",
        "completeness",
        "logical_sequence",
        "hallucination_redundancy",
        "granularity",
        "final_score",
    ]

    processed_keys: set[Tuple[str, int]] = set()
    file_exists = outputs_csv.exists()
    write_header = True
    if file_exists and resume:
        # Load processed keys to support resume
        with open(outputs_csv, newline='', encoding='utf-8') as prev_f:
            prev_reader = csv.DictReader(prev_f)
            # If header matches, collect done rows
            if prev_reader.fieldnames and set(prev_reader.fieldnames) >= set(fieldnames):
                for prow in prev_reader:
                    m = (prow.get("model_name") or "").strip()
                    try:
                        sidx = int((prow.get("sample_idx") or "0").strip())
                    except Exception:
                        continue
                    if m:
                        processed_keys.add((m, sidx))
        write_header = False  # will append without header

    mode = "a" if (resume and file_exists) else "w"

    with open(outputs_csv, mode, newline='', encoding='utf-8') as out_f:
        writer = csv.DictWriter(out_f, fieldnames=fieldnames)
        if write_header:
            writer.writeheader()

        for i, r in enumerate(rows, start=1):
            model_name = (r.get("model_name") or "").strip()
            sample_idx_str = (r.get("sample_idx") or "0").strip()
            try:
                sample_idx = int(sample_idx_str)
            except Exception:
                sample_idx = i - 1

            # Skip if already processed (resume)
            if (model_name, sample_idx) in processed_keys:
                continue

            image_rel = (r.get("image_paths") or r.get("image_path") or "").strip()
            prompt_text = (r.get("prompt_text") or "").strip()
            generated_plan = (r.get("model_output") or "").strip()
            reference_steps = (r.get("reference_steps") or "").strip()

            task_name = parse_task_name(prompt_text)

            # Prepare image data
            data_uri: Optional[str] = None
            if image_rel:
                image_abs = (root_dir / image_rel).resolve()
                if image_abs not in image_cache:
                    image_cache[image_abs] = load_image_as_data_uri(image_abs)
                data_uri = image_cache[image_abs]
                if warn_missing_image and (data_uri is None) and (image_abs not in warned_images):
                    print(f"[WARN] Image not found or unreadable: {image_abs} -> fallback to text-only judging.", file=sys.stderr)
                    warned_images.add(image_abs)

            # Call judge
            try:
                raw, parsed = call_llm_judge(
                    model=model,
                    api_url=api_url,
                    api_key=api_key,
                    task_name=task_name,
                    reference_steps=reference_steps,
                    generated_plan=generated_plan,
                    image_data_uri=data_uri,
                    max_retries=max_retries,
                    retry_forever=retry_forever,
                    retry_sleep=retry_sleep,
                )
            except Exception as e:
                if stop_on_error:
                    print(f"[ERROR] Stopping on error at model={model_name} sample_idx={sample_idx}: {e}", file=sys.stderr)
                    # exit early to allow resume on next run
                    raise
                else:
                    # Optional: write error row and continue
                    writer.writerow({
                        "model_name": model_name,
                        "sample_idx": sample_idx,
                        "judge_raw": f"ERROR: {e}",
                        "completeness": None,
                        "logical_sequence": None,
                        "hallucination_redundancy": None,
                        "granularity": None,
                        "final_score": None,
                    })
                    out_f.flush()
                    if sleep_sec:
                        time.sleep(sleep_sec)
                    continue

            # Parse fields
            scores = parsed.get("scores", {}) if isinstance(parsed, dict) else {}
            completeness = to_int_or_none(scores.get("completeness"))
            logical_sequence = to_int_or_none(scores.get("logical_sequence"))
            hallucination_redundancy = to_int_or_none(scores.get("hallucination_redundancy"))
            granularity = to_int_or_none(scores.get("granularity"))

            final_score_val: Optional[float]
            if isinstance(parsed, dict) and isinstance(parsed.get("final_score"), (int, float)):
                final_score_val = float(parsed.get("final_score"))
            else:
                final_score_val = average_scores(scores)

            writer.writerow({
                "model_name": model_name,
                "sample_idx": sample_idx,
                "judge_raw": json.dumps(parsed, ensure_ascii=False),
                "completeness": completeness,
                "logical_sequence": logical_sequence,
                "hallucination_redundancy": hallucination_redundancy,
                "granularity": granularity,
                "final_score": (None if final_score_val is None else f"{final_score_val:.3f}"),
            })
            out_f.flush()
            if sleep_sec:
                time.sleep(sleep_sec)


def main(argv: Optional[List[str]] = None) -> int:
    parser = argparse.ArgumentParser(description="LLM-as-a-Judge evaluator for blind_10 outputs")
    parser.add_argument("--input-csv", type=str, default="", help="Path to blind10_outputs_*.csv; default: pick latest in eva_results/blind_10/")
    parser.add_argument("--api-base", type=str, default=DEFAULT_API_BASE, help="OpenAI-compatible API base (default: https://jeniya.top/v1)")
    parser.add_argument("--model", type=str, default=DEFAULT_MODEL, help="Model name (default: gpt-4o-mini)")
    parser.add_argument("--max-samples", type=int, default=None, help="Limit number of rows for a quick run")
    parser.add_argument("--only-models", type=str, default="", help="Comma-separated model_name filter (e.g., Baseline,checkpoint-200)")
    parser.add_argument("--sleep", type=float, default=0.0, help="Sleep seconds between samples to avoid rate limits")
    parser.add_argument("--warn-missing-image", action="store_true", help="Print warnings when image cannot be loaded; fallback to text-only if missing")
    parser.add_argument("--max-retries", type=int, default=10, help="Max retries per sample when calling judge API (default: 10)")
    parser.add_argument("--retry-forever", action="store_true", help="Ignore max retries and keep retrying until success (sleep fixed interval)")
    parser.add_argument("--retry-sleep", type=float, default=1.0, help="Seconds to sleep between judge API retries (default: 1.0)")
    parser.add_argument("--no-resume", action="store_true", help="Do not resume from existing score CSV; overwrite output file")
    parser.add_argument("--continue-on-error", action="store_true", help="Do not stop on error; write an ERROR row and continue")
    parser.add_argument("--workspace-root", type=str, default=os.getenv("PROJECT_ROOT"), help="Override workspace root (default: auto-detect from script path)")

    args = parser.parse_args(argv)

    if not API_KEY:
        print("ERROR: Missing API key. Set JENIYA_API_KEY or OPENAI_API_KEY in environment.", file=sys.stderr)
        return 2

    if args.workspace_root:
        workspace_root = Path(args.workspace_root).resolve()
    else:
        workspace_root = Path(__file__).resolve().parents[2]
    eva_dir = workspace_root / "eva_results" / "blind_10"

    input_csv_path: Path
    if args.input_csv:
        input_csv_path = Path(args.input_csv).resolve()
    else:
        input_csv_path = find_latest_outputs_csv(eva_dir)

    if not input_csv_path.exists():
        print(f"ERROR: Input CSV not found: {input_csv_path}", file=sys.stderr)
        return 2

    output_csv_path = derive_score_csv_path(input_csv_path)

    # Normalize API URL
    api_url = args.api_base.rstrip("/") + "/chat/completions"

    only_models = [s.strip() for s in args.only_models.split(",") if s.strip()] or None

    print(f"Input:  {input_csv_path}")
    print(f"Output: {output_csv_path}")
    print(f"API:     {api_url}")
    print(f"Model:   {args.model}")
    if only_models:
        print(f"Filter models: {only_models}")
    if args.max_samples:
        print(f"Max samples: {args.max_samples}")
    print(f"Retry:  {'forever' if args.retry_forever else args.max_retries} (sleep={args.retry_sleep}s)")
    print(f"Resume: {not args.no_resume}")
    print(f"Stop on error: {not args.continue_on_error}")

    process_csv(
        inputs_csv=input_csv_path,
        outputs_csv=output_csv_path,
        api_url=api_url,
        api_key=API_KEY,
        model=args.model,
        root_dir=workspace_root,
        max_samples=args.max_samples,
        sample_filter_model=only_models,
        sleep_sec=args.sleep,
        warn_missing_image=args.warn_missing_image,
        retry_forever=args.retry_forever,
        max_retries=args.max_retries,
        retry_sleep=args.retry_sleep,
        stop_on_error=(not args.continue_on_error),
        resume=(not args.no_resume),
    )

    print("Done.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
