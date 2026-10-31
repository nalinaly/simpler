# Copyright (c) PyPTO Contributors.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------

"""Strict same-workload comparison of Graph benchmark ledgers and STRACE logs."""

import argparse
import importlib.util
import json
import math
import statistics
from pathlib import Path

_ROOT = Path(__file__).resolve().parents[4]
_METRICS = {
    "host_us": "chip.run",
    "host_bind_us": "chip.run.bind",
    "host_publish_us": "chip.run.publish_image",
    "host_prepare_execution_us": "chip.run.prepare_execution",
    "host_runner_us": "chip.run.runner_run",
    "device_runwall_us": "chip.run.runner_run.device_wall",
}
_WORKLOAD_FIELDS = ("case", "parameters", "platform", "device", "pto_isa_pin", "torch", "fixture_sources_sha256")


def _reader():
    path = _ROOT / "examples/a5/host_build_graph/deepseek_v4_pro_csa/analyze_perf.py"
    spec = importlib.util.spec_from_file_location("_graph_existing_hbg_timing", path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module.read_invocations


def summarize(values):
    if not values or any(not math.isfinite(value) or value < 0 for value in values):
        raise ValueError("Metrics must be finite, nonnegative and nonempty")
    ordered = sorted(values)

    def percentile(q):
        position = (len(ordered) - 1) * q
        low = math.floor(position)
        high = math.ceil(position)
        return ordered[low] + (ordered[high] - ordered[low]) * (position - low)

    return {
        "count": len(values),
        "min": ordered[0],
        "median": statistics.median(ordered),
        "p90": percentile(0.90),
        "p99": percentile(0.99),
        "max": ordered[-1],
    }


def read_batch(path, warmup=5):
    ledger = json.loads(path.read_text())
    if ledger.get("schema") != "hbg-graph-benchmark-v1" or ledger.get("all_rounds_numerically_verified") is not True:
        raise ValueError(f"Unverified benchmark ledger: {path}")
    samples = ledger["samples"]
    if warmup < 1 or len(samples) <= warmup:
        raise ValueError("Need at least one cold/warmup call and one measured call")
    if ledger["expected_backend"] not in ("resident", "legacy-graph"):
        raise ValueError("Unknown expected backend")
    invocations = _reader()(path.with_suffix(".log"), warmup=0, rounds=len(samples))
    expected = (3,) if ledger["expected_backend"] == "legacy-graph" else (1, 2)
    rows = []
    previous_publication = -1
    for index, (invocation, sample) in enumerate(zip(invocations, samples)):
        if sample["round"] != index or sample["scheduler_mode"] not in expected:
            raise ValueError("Round ordering or actual backend does not match the ledger")
        if sample["publication"] <= previous_publication:
            raise ValueError("Stale backend publication")
        previous_publication = sample["publication"]
        names = invocation.by_name()
        if not set(_METRICS.values()) <= names.keys():
            raise ValueError(f"Missing timing stages in invocation {invocation.inv}")
        if not names[_METRICS["device_runwall_us"]].is_device:
            raise ValueError("Device RunWall must use the device clock domain")
        row = {key: names[name].dur / 1000 for key, name in _METRICS.items()}
        row["python_call_us"] = sample["python_call_wall_ns"] / 1000
        row["committed_device_bytes"] = sample["committed_device_bytes"]
        if row["host_us"] <= 0 or row["device_runwall_us"] <= 0:
            raise ValueError("Missing positive full host/device timing")
        rows.append(row)
    return ledger, rows


def compare(directory, warmup=5):
    groups = {}
    signatures = {}
    versions = {}
    paths = sorted(directory.glob("b*-*.json"))
    if not paths:
        raise ValueError("No batch ledgers found")
    for path in paths:
        variant = path.stem.rsplit("-", 1)[-1]
        if variant not in ("baseline", "native"):
            raise ValueError(f"Unknown variant: {path}")
        ledger, rows = read_batch(path, warmup)
        workload = ledger["workload"]
        expected = "legacy-graph" if variant == "baseline" and workload.startswith("graph_") else "resident"
        if ledger["expected_backend"] != expected:
            raise ValueError(f"Wrong backend for comparison arm: {variant}/{workload}")
        signature = {key: ledger[key] for key in _WORKLOAD_FIELDS}
        if signatures.setdefault(workload, signature) != signature:
            raise ValueError(f"Different workload, sources or environment: {workload}")
        version = {key: ledger[key] for key in ("commit", "runtime_diff_sha256", "runtime_sha256")}
        if versions.setdefault(variant, version) != version:
            raise ValueError(f"Runtime changed between batches: {variant}")
        item = groups.setdefault(workload, {}).setdefault(variant, {"batches": [], "cold": [], "warm": []})
        item["batches"].append(
            {
                "ledger": path.name,
                "rounds": len(rows),
                "initial_committed_bytes": ledger["initial_committed_device_bytes"],
                "retained_committed_bytes": ledger["retained_committed_device_bytes"],
                "warm_device_runwall_median_us": statistics.median(row["device_runwall_us"] for row in rows[warmup:]),
            }
        )
        item["cold"].append(rows[0])
        item["warm"].extend(rows[warmup:])
    for workload, variants in groups.items():
        if set(variants) != {"baseline", "native"}:
            raise ValueError(f"Unpaired workload: {workload}")
        baseline, native = variants["baseline"], variants["native"]
        if len(baseline["batches"]) != len(native["batches"]) or len(baseline["warm"]) != len(native["warm"]):
            raise ValueError(f"Unbalanced sample counts: {workload}")
        for item in variants.values():
            for phase in ("cold", "warm"):
                item[phase] = {key: summarize([row[key] for row in item[phase]]) for key in item[phase][0]}
        variants["warm_median_change_pct"] = {
            metric: (native["warm"][metric]["median"] / baseline["warm"][metric]["median"] - 1) * 100
            for metric in ("host_us", "host_bind_us", "device_runwall_us", "committed_device_bytes")
        }
    return {
        "schema": "hbg-graph-comparison-v1",
        "warmup_per_batch": warmup,
        "quantiles": "linear interpolation at (n-1)*q; cold means first call of each fresh Worker",
        "versions": versions,
        "workload_contracts": signatures,
        "workloads": groups,
    }


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("directory", type=Path)
    parser.add_argument("--warmup", type=int, default=5)
    args = parser.parse_args()
    print(json.dumps(compare(args.directory, args.warmup), indent=2))
