# Copyright (c) PyPTO Contributors.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------

"""Graph A/B reports reject incomplete, mixed-workload or mislabeled samples."""

import importlib.util
import json
from pathlib import Path
from typing import Any

import pytest

_SOURCE = Path(__file__).resolve().parents[3] / "tests/fdwic/pa_hbg/spmd/analyze_graph_benchmark.py"


@pytest.fixture
def analyzer():
    spec = importlib.util.spec_from_file_location("_graph_benchmark_analysis", _SOURCE)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _write_batch(directory, variant, *, device_clock=True):
    mode = 3 if variant == "baseline" else 1
    ledger: dict[str, Any] = {
        "schema": "hbg-graph-benchmark-v1",
        "all_rounds_numerically_verified": True,
        "workload": "graph_2d",
        "case": "fixed",
        "parameters": {"shape": [128, 128]},
        "platform": "a5",
        "device": 0,
        "pto_isa_pin": "same-pin",
        "torch": "same-torch",
        "fixture_sources_sha256": {"same.cpp": "same-digest"},
        "commit": variant,
        "runtime_diff_sha256": "empty",
        "runtime_sha256": {"runtime.o": variant},
        "expected_backend": "legacy-graph" if mode == 3 else "resident",
        "initial_committed_device_bytes": 512,
        "retained_committed_device_bytes": 1024,
        "samples": [
            {
                "round": index,
                "scheduler_mode": mode,
                "publication": index + 1,
                "python_call_wall_ns": 12000,
                "committed_device_bytes": 1024,
            }
            for index in range(3)
        ],
    }
    path = directory / f"b0-graph_2d-{variant}.json"
    path.write_text(json.dumps(ledger))
    device_ns = 5000 if variant == "baseline" else 4500
    spans = {
        "chip.run": (10000, ""),
        "chip.run.bind": (1000, ""),
        "chip.run.publish_image": (100, ""),
        "chip.run.prepare_execution": (500, ""),
        "chip.run.runner_run": (7000, ""),
        "chip.run.runner_run.device_wall": (device_ns, "clk=dev" if device_clock else ""),
    }
    lines = []
    for invocation in range(1, 4):
        for name, (duration, attrs) in spans.items():
            depth = 0 if name == "chip.run" else 1
            lines.append(
                f"[STRACE] v=1 pid=1 tid=1 inv={invocation} hid=0 depth={depth} "
                f"name={name} ts={invocation * 100000} dur={duration} {attrs}\n"
            )
    path.with_suffix(".log").write_text("".join(lines))
    return path, ledger


def test_percentiles_are_linear_and_keep_sample_counts(analyzer):
    result = analyzer.summarize([4, 1, 3, 2])
    assert result["count"] == 4 and result["median"] == 2.5
    assert result["p90"] == pytest.approx(3.7)
    assert result["p99"] == pytest.approx(3.97)


@pytest.mark.parametrize("values", ([], [-1], [float("nan")], [float("inf")]))
def test_invalid_metrics_are_not_reported(analyzer, values):
    with pytest.raises(ValueError):
        analyzer.summarize(values)


def test_strict_comparison_uses_full_device_wall_and_cold_warm_counts(analyzer, tmp_path):
    _write_batch(tmp_path, "baseline")
    _write_batch(tmp_path, "native")
    result = analyzer.compare(tmp_path, warmup=1)["workloads"]["graph_2d"]
    assert result["baseline"]["cold"]["device_runwall_us"]["count"] == 1
    assert result["native"]["warm"]["device_runwall_us"]["count"] == 2
    assert result["native"]["warm"]["device_runwall_us"]["median"] == 4.5
    assert result["warm_median_change_pct"]["device_runwall_us"] == pytest.approx(-10)


@pytest.mark.parametrize(
    "field,value,message",
    (
        ("all_rounds_numerically_verified", False, "Unverified"),
        ("parameters", {"shape": [1, 1]}, "Different workload"),
        ("expected_backend", "legacy-graph", "Round ordering or actual backend"),
    ),
)
def test_unverified_or_mismatched_inputs_are_rejected(analyzer, tmp_path, field, value, message):
    _write_batch(tmp_path, "baseline")
    path, ledger = _write_batch(tmp_path, "native")
    ledger[field] = value
    path.write_text(json.dumps(ledger))
    with pytest.raises(ValueError, match=message):
        analyzer.compare(tmp_path, warmup=1)


def test_missing_variant_cannot_be_presented_as_a_comparison(analyzer, tmp_path):
    _write_batch(tmp_path, "native")
    with pytest.raises(ValueError, match="Unpaired"):
        analyzer.compare(tmp_path, warmup=1)


def test_host_clock_cannot_be_used_as_device_time(analyzer, tmp_path):
    path, _ = _write_batch(tmp_path, "native", device_clock=False)
    with pytest.raises(ValueError, match="device clock"):
        analyzer.read_batch(path, warmup=1)


def test_stale_publications_are_rejected(analyzer, tmp_path):
    path, ledger = _write_batch(tmp_path, "native")
    ledger["samples"][1]["publication"] = 1
    path.write_text(json.dumps(ledger))
    with pytest.raises(ValueError, match="Stale"):
        analyzer.read_batch(path, warmup=1)
