#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Quick analysis of data/test_16812.jsonl.
Outputs distribution of task_id and task_name (if present).
"""
import json
import os
from collections import Counter
import random

DATA_PATH = os.path.join(os.path.dirname(__file__), 'data', 'test_16812.jsonl')


def load_jsonl(path):
    records = []
    bad_lines = 0
    with open(path, 'r', encoding='utf-8') as f:
        for i, line in enumerate(f, 1):
            line = line.strip()
            if not line:
                continue
            try:
                obj = json.loads(line)
                records.append(obj)
            except json.JSONDecodeError:
                bad_lines += 1
    return records, bad_lines


def percent(n, total):
    return 0.0 if total == 0 else (n / total * 100.0)


def main():
    if not os.path.exists(DATA_PATH):
        print(f"File not found: {DATA_PATH}")
        return

    records, bad_lines = load_jsonl(DATA_PATH)
    total = len(records)

    print(f"Total valid records: {total}")
    if bad_lines:
        print(f"Malformed lines skipped: {bad_lines}")

    # Collect possible fields
    task_ids = []
    task_names = []

    for r in records:
        # Accept multiple possible keys
        tid = r.get('task_id')
        if tid is None:
            tid = r.get('taskId')
        if tid is None:
            tid = r.get('task')
        if tid is not None:
            task_ids.append(str(tid))

        # Extract task_name from explicit fields or embedded text in messages
        tname = r.get('task_name') or r.get('taskName')
        if not tname:
            tv = r.get('task')
            if isinstance(tv, str):
                tname = tv
        # Try to parse from messages[0].content[].text lines like "Task name: ..."
        if not tname:
            msgs = r.get('messages')
            if isinstance(msgs, list) and msgs:
                # find any text chunk
                contents = msgs[0].get('content')
                if isinstance(contents, list):
                    for c in contents:
                        if isinstance(c, dict) and c.get('type') == 'text':
                            txt = c.get('text') or ''
                            # Find pattern
                            # English: "Task name: ..."; Chinese could be "任务名:" if appears
                            for prefix in ['Task name:', '任务名', '任务名称', 'Task:']:
                                if prefix in txt:
                                    # take line containing prefix
                                    for line in txt.splitlines():
                                        if prefix in line:
                                            # extract after colon-like separators
                                            part = line.split(':', 1)
                                            candidate = part[1].strip() if len(part) > 1 else line.replace(prefix, '').strip()
                                            if candidate:
                                                tname = candidate
                                                break
                                    if tname:
                                        break
                            if tname:
                                break
        if tname:
            task_names.append(str(tname))

    id_counter = Counter(task_ids)
    name_counter = Counter(task_names)

    def print_counter(title, counter):
        print(f"\n{title} (unique={len(counter)}):")
        if not counter:
            print("  No data found.")
            return
        # Show all sorted by count desc
        for key, cnt in counter.most_common():
            print(f"  {key}: {cnt} ({percent(cnt, total):.2f}%)")

    print_counter('Task ID distribution', id_counter)
    print_counter('Task name distribution', name_counter)

    # Cross-check if both present
    if id_counter and name_counter:
        print("\nNote: If task_id and task_name refer to the same category, their distributions should align closely.")

    # Build blind_10.jsonl: pick 1 record per task (10 tasks total)
    # Group records by parsed task name
    groups = {}
    for r in records:
        # reuse extraction logic: find tname again to ensure alignment with groups
        tname = r.get('task_name') or r.get('taskName')
        if not tname:
            tv = r.get('task')
            if isinstance(tv, str):
                tname = tv
        if not tname:
            msgs = r.get('messages')
            if isinstance(msgs, list) and msgs:
                contents = msgs[0].get('content')
                if isinstance(contents, list):
                    for c in contents:
                        if isinstance(c, dict) and c.get('type') == 'text':
                            txt = c.get('text') or ''
                            for prefix in ['Task name:', '任务名', '任务名称', 'Task:']:
                                if prefix in txt:
                                    for line in txt.splitlines():
                                        if prefix in line:
                                            part = line.split(':', 1)
                                            candidate = part[1].strip() if len(part) > 1 else line.replace(prefix, '').strip()
                                            if candidate:
                                                tname = candidate
                                                break
                                    if tname:
                                        break
                            if tname:
                                break
        if not tname:
            # skip records without recognizable task name
            continue
        groups.setdefault(tname, []).append(r)

    # Use fixed seed for reproducibility
    random.seed(42)
    sampled = []
    for tname, items in groups.items():
        if items:
            sampled.append(random.choice(items))

    # If more than 10 tasks present, select first 10 by frequency order
    if len(sampled) > 10:
        # order task names by descending frequency
        top_tasks = [name for name, _cnt in name_counter.most_common(10)]
        # map to one sample per top task
        picked = []
        seen = set()
        for name in top_tasks:
            if name in groups and groups[name]:
                choice = random.choice(groups[name])
                picked.append(choice)
                seen.add(name)
        sampled = picked
    elif len(sampled) < 10:
        print(f"\nWarning: only {len(sampled)} tasks detected; blind_10.jsonl will have {len(sampled)} rows.")

    out_path = os.path.join(os.path.dirname(__file__), 'data', 'blind_10.jsonl')
    with open(out_path, 'w', encoding='utf-8') as wf:
        for r in sampled:
            wf.write(json.dumps(r, ensure_ascii=False) + "\n")
    print(f"\nWrote {len(sampled)} samples to {out_path}")


if __name__ == '__main__':
    main()
