# Copyright (c) PyPTO Contributors.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------

"""The Graph benchmark cannot silently compare fallback or stale publications."""

import importlib.util
from pathlib import Path
from types import SimpleNamespace

import pytest

_SOURCE = Path(__file__).resolve().parents[3] / "tests/fdwic/pa_hbg/spmd/benchmark_graph.py"


class _FakeWorker:
    def __init__(self, **kwargs):
        del kwargs
        self.scheduler_mode_publish_count = 0
        self.scheduler_runtime_mode = 1
        self.publish = True

    def run(self, callable, args=None, config=None):
        del callable, args, config
        self.scheduler_mode_publish_count += int(self.publish)

    def committed_device_memory(self):
        return 4096


@pytest.fixture
def benchmark(monkeypatch):
    monkeypatch.setattr("simpler.worker.Worker", _FakeWorker)
    spec = importlib.util.spec_from_file_location("_graph_benchmark_under_test", _SOURCE)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    ticks = iter((100, 400))
    monkeypatch.setattr(module, "time", SimpleNamespace(perf_counter_ns=lambda: next(ticks)))
    return module


@pytest.mark.parametrize("mode,expected", [(1, (1, 2)), (2, (1, 2)), (3, (3,))])
def test_records_fresh_backend_and_real_committed_memory(benchmark, mode, expected):
    worker = benchmark._MeasuredWorker(expected)
    worker.scheduler_runtime_mode = mode
    worker.run(None, config=SimpleNamespace(enable_chip_swimlane=0))
    assert worker.samples == [
        {
            "round": 0,
            "python_call_wall_ns": 300,
            "scheduler_mode": mode,
            "publication": 1,
            "committed_device_bytes": 4096,
        }
    ]


@pytest.mark.parametrize("mode", [0, 3, 4])
def test_resident_measurement_rejects_other_backends(benchmark, mode):
    worker = benchmark._MeasuredWorker((1, 2))
    worker.scheduler_runtime_mode = mode
    with pytest.raises(AssertionError, match="unexpected backend"):
        worker.run(None, config=SimpleNamespace(enable_chip_swimlane=0))
    assert not worker.samples


def test_stale_publication_is_not_a_sample(benchmark):
    worker = benchmark._MeasuredWorker((1, 2))
    worker.publish = False
    with pytest.raises(AssertionError):
        worker.run(None, config=SimpleNamespace(enable_chip_swimlane=0))
    assert not worker.samples


def test_l4_is_not_accepted_as_a_baseline_timing_sample(benchmark):
    worker = benchmark._MeasuredWorker((1, 2))
    with pytest.raises(AssertionError):
        worker.run(None, config=SimpleNamespace(enable_chip_swimlane=4))
    assert not worker.samples
