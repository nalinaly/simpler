#!/usr/bin/env python3
# Copyright (c) PyPTO Contributors.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------
"""Summarize L0/L1/L3 captures and export measured intervals to Perfetto."""

import argparse
import csv
import json
from pathlib import Path


def read_csv(path):
    with path.open() as stream:
        return list(csv.DictReader(stream))


def completion_events(path, pid, origin, first_execution, workload_root, data):
    events = []
    task_path = path.with_suffix(".tasks.csv")
    if task_path.exists():
        tasks = {int(t["task"]): t for t in read_csv(task_path)}
        events.append(
            {
                "name": "thread_name",
                "ph": "M",
                "pid": pid,
                "tid": 10000,
                "args": {"name": "Logical TaskDone (observer core not recorded)"},
            }
        )
        for task, row in tasks.items():
            timestamp = (int(row["completed_ns"]) - origin) / 1000
            events.append(
                {
                    "name": row["stage"],
                    "ph": "i",
                    "s": "t",
                    "pid": pid,
                    "tid": 10000,
                    "ts": timestamp,
                    "args": {"task": task, "due_ns": row["due_ns"]},
                }
            )
        if workload_root:
            edge_path = workload_root / data["flavor"] / data["case"] / "edges.tsv"
            for index, line in enumerate(edge_path.read_text().splitlines()):
                if not line or line.startswith("#"):
                    continue
                pred, succ = map(int, line.split())
                source = int(tasks[pred]["completed_ns"])
                target, tid = first_execution.get(succ, (int(tasks[succ]["completed_ns"]), 10000))
                if target < source:
                    raise ValueError(f"dependency violated: {pred}->{succ}")
                base = {
                    "name": "dependency",
                    "cat": "dependency",
                    "id": f"{pid}:{index}",
                    "pid": pid,
                    "args": {"producer": pred, "consumer": succ},
                }
                events.append({**base, "ph": "s", "tid": 10000, "ts": (source - origin) / 1000})
                events.append({**base, "ph": "f", "bp": "e", "tid": tid, "ts": (target - origin) / 1000})
    return events


def convert(path, pid, workload_root):
    data = json.loads(path.read_text())
    if not data.get("passed"):
        raise ValueError(f"run did not pass: {path}")
    level = data.get("profiling", 3)
    rows = read_csv(path.with_suffix(".samples.csv"))
    workers = read_csv(path.with_suffix(".workers.csv"))
    if len(rows) != (data["executions"] if level else 0):
        raise ValueError(f"truncated sample file: {path}")
    if level == 3:
        values = sorted(sum(int(r[k]) for k in ("search_ns", "prepare_ns", "finish_ns", "service_ns")) for r in rows)
        if values[-1] != data["scheduler_max_ns"] or sum(v > 5000 for v in values) != data["over_5us"]:
            raise ValueError(f"sample/summary mismatch: {path}")
    print(
        data["backend"],
        level,
        data["flavor"],
        data["case"],
        data["round"],
        round(data["wall_ns"] / 1e6, 6),
        *[
            round(data[k] / 1000, 3) if data[k] is not None else "-"
            for k in ("scheduler_median_ns", "scheduler_p99_ns", "scheduler_max_ns")
        ],
        data["over_5us"],
    )
    if level == 0:
        return []
    origin = min(int(w["begin_ns"]) for w in workers)
    events = [{"name": "process_name", "ph": "M", "pid": pid, "args": {"name": f"L{level} {path.stem}"}}]
    core_ids = {int(r["worker"]): r["physical_core"] for r in rows}
    for w in workers:
        worker = int(w["worker"])
        role = "AIC" if worker % 3 == 0 else "AIV"
        for lane in range(1 if level == 1 else 2):
            name = f"{role} worker={worker} core={core_ids.get(worker, 'idle')} " + (
                "kernel" if lane == 0 else "Scalar"
            )
            events.append(
                {"name": "thread_name", "ph": "M", "pid": pid, "tid": worker * 2 + lane, "args": {"name": name}}
            )
            events.append(
                {
                    "name": "thread_sort_index",
                    "ph": "M",
                    "pid": pid,
                    "tid": worker * 2 + lane,
                    "args": {"sort_index": worker * 2 + lane},
                }
            )

    def span(name, worker, lane, start, end, args):
        if end < start or start < origin:
            raise ValueError("invalid timing interval")
        events.append(
            {
                "name": name,
                "cat": "kernel" if lane == 0 else "scheduler",
                "ph": "X",
                "pid": pid,
                "tid": worker * 2 + lane,
                "ts": (start - origin) / 1000,
                "dur": (end - start) / 1000,
                "args": args,
            }
        )

    first_execution = {}
    for row in rows:
        start, end, worker, task = (int(row[k]) for k in ("begin_ns", "end_ns", "worker", "task"))
        args = {k: row[k] for k in ("task", "blocks", "block", "lane", "physical_core", "search_ns", "service_ns")}
        span(row["stage"], worker, 0, start, end, args)
        if task not in first_execution or start < first_execution[task][0]:
            first_execution[task] = (start, worker * 2)
        if level == 3:
            span("prepare", worker, 1, start - int(row["prepare_ns"]), start, args)
            span("finish", worker, 1, end, end + int(row["finish_ns"]), args)
    event_path = path.with_suffix(".events.csv")
    if level == 3 and event_path.exists():
        for row in read_csv(event_path):
            kind = int(row["kind"])
            name = {1: "CHECK / control service", 2: "ready search / claim", 3: "empty ready scan"}[kind]
            span(name, int(row["worker"]), 1, int(row["begin_ns"]), int(row["end_ns"]), {"task": row["task"]})
    events.extend(completion_events(path, pid, origin, first_execution, workload_root, data))
    return events


def summarize(directory, trace=None, split_traces=None, workloads=None):
    paths = sorted(directory.glob("*.json"))
    if not paths:
        raise ValueError("no run summaries found")
    events = []
    print("backend level flavor case round window_ms scalar_median_us scalar_p99_us scalar_max_us over_5us")
    for pid, path in enumerate(paths):
        if "traceEvents" in json.loads(path.read_text()):
            continue
        current = convert(path, pid, workloads)
        events.extend(current)
        if split_traces and current:
            split_traces.mkdir(parents=True, exist_ok=True)
            (split_traces / (path.stem + ".json")).write_text(
                json.dumps({"traceEvents": current, "displayTimeUnit": "us"})
            )
    if trace:
        trace.write_text(json.dumps({"traceEvents": events, "displayTimeUnit": "us"}))
    print("Independent runs are aligned separately. L0 window is the baseline; L1/L3 are instrumented diagnostics.")
    print("CPU clocks are not A5 evidence; trace-write and other unmeasured gaps must not be called dependency waits.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("directory", type=Path)
    parser.add_argument("--trace", type=Path)
    parser.add_argument("--split-traces", type=Path)
    parser.add_argument("--workloads", type=Path)
    options = parser.parse_args()
    summarize(options.directory, options.trace, options.split_traces, options.workloads)
