# Copyright (c) PyPTO Contributors.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------

"""Real Scalar Graph bootstrap failure, diagnostic retention and worker retirement."""

import importlib.util
import json
import tempfile
import time
from pathlib import Path

import pytest
import torch
from simpler.task_interface import ArgDirection, CallConfig, DataType, TaskArgs, TensorArgType
from simpler.worker import Worker

from simpler_setup.scene_test import _pto_isa_compile_cache_token, compile_chip_callable_spec
from simpler_setup.tools.swimlane_converter import read_perf_data

_HERE = Path(__file__).resolve().parent
_ROOT = _HERE.parents[4]

pytestmark = [
    pytest.mark.platforms(["a5sim", "a5"]),
    pytest.mark.device_count(1),
    pytest.mark.runtime("host_build_graph"),
    pytest.mark.scheduler_mode("resident"),
    pytest.mark.timeout(120),
]


def _compile(platform):
    spec = {
        "orchestration": {
            "source": str(_HERE / "kernels/orchestration/graph_materialize_failure_orch.cpp"),
            "function_name": "aicpu_orchestration_entry",
            "signature": [ArgDirection.INOUT, ArgDirection.INOUT, ArgDirection.IN],
        },
        "incores": [
            {
                "func_id": 0,
                "name": "WRITE_CONST",
                "source": str(_HERE.parent / "predicated_dispatch/kernels/aic/kernel_write_const.cpp"),
                "core_type": "aic",
                "signature": [ArgDirection.INOUT],
            }
        ],
    }
    return compile_chip_callable_spec(
        spec, platform, "host_build_graph", (__name__, platform, _pto_isa_compile_cache_token())
    )


@pytest.fixture(scope="module")
def resident_failure_probe():
    path = _HERE.parent / "resident_failure/test_resident_failure.py"
    spec = importlib.util.spec_from_file_location("_graph_resident_failure_fixture", path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


@pytest.mark.parametrize("mask,lane", ((1, 0), (2, 1), (7, 0), (7, 1), (7, 2)))
@pytest.mark.parametrize("trailing_child", (False, True))
def test_graph_child_timeout_does_not_release_parent(
    st_platform, st_device_ids, mask, lane, trailing_child, monkeypatch, resident_failure_probe
):
    resident_failure_probe._exercise_failure(  # noqa: SLF001 -- shared real-device failure harness
        st_platform, st_device_ids, mask, lane, 2, monkeypatch, 2500, graph_tail=trailing_child
    )


def test_scalar_materialization_error_retires_before_kernel_entry(st_platform, st_device_ids):
    callable_obj = _compile(st_platform)
    output_root = _ROOT / "outputs"
    output_root.mkdir(exist_ok=True)
    artifacts = Path(tempfile.mkdtemp(prefix=f"graph-materialize-failure-{st_platform}-", dir=output_root))
    evidence = []
    for recovery in (False, True):
        worker = Worker(level=2, platform=st_platform, runtime="host_build_graph", device_id=int(st_device_ids[0]))
        buffers = []
        failed = False
        try:
            handle = worker.register(callable_obj)
            worker.init()
            values = [torch.full((16,), -1.0), torch.full((16,), -1.0), torch.ones(16, dtype=torch.int32)]
            buffers = [worker.malloc(value.nbytes) for value in values]
            for malformed in (False,) if recovery else (False, True):
                args = TaskArgs()
                for index, (buffer, value) in enumerate(zip(buffers, values, strict=True)):
                    if index < 2:
                        value.fill_(-1)
                    worker.copy_to(buffer, value)
                    args.add_tensor(
                        buffer.tensor(
                            shapes=tuple(value.shape), dtype=DataType.INT32 if index == 2 else DataType.FLOAT32
                        ),
                        TensorArgType.INPUT if index == 2 else TensorArgType.INOUT,
                    )
                args.add_scalar(int(malformed))
                prefix = artifacts / ("recovered" if recovery else "failed" if malformed else "healthy")
                prefix.mkdir()
                config = CallConfig()
                config.enable_chip_swimlane = 4
                config.output_prefix = str(prefix)
                before = worker.scheduler_mode_publish_count
                started = time.monotonic()
                failure = None
                if malformed:
                    failed = True
                    with pytest.raises(RuntimeError, match="finalize_native_run failed with code") as failure:
                        worker.run(handle, args, config)
                else:
                    worker.run(handle, args, config)
                elapsed = time.monotonic() - started
                assert elapsed < 20
                assert worker.scheduler_mode_publish_count > before, "Host rejection is not a Scalar failure"
                assert worker.scheduler_runtime_mode in (1, 2)
                paths = list(prefix.rglob("chip_swimlane_records.json"))
                assert len(paths) == 1, paths
                raw = json.loads(paths[0].read_text())
                protocol = raw["aicore_protocol"]
                assert protocol["schema"] == "hbg-native-protocol-v8"
                for buffer, value in zip(buffers[:2], values[:2], strict=True):
                    worker.copy_from(value, buffer)
                    expected = torch.full_like(value, -1)
                    if not malformed:
                        expected[0] = 42
                    torch.testing.assert_close(value, expected, rtol=0, atol=0)
                if malformed:
                    assert protocol["scheduler_error"] != 0
                    assert protocol["first_error"]["site"] == 132
                    assert protocol["first_error"]["public_task_id"] == 1
                    assert protocol["first_error"]["sub_task_id"] == 0
                    assert not protocol["executions"], "startup failure must not open the kernel entry gate"
                    assert protocol["resolved_task_count"] == 0
                    # Peers that observe the error before context READY take
                    # the early EXIT/ACK path without entering the scheduler.
                    assert all(
                        entry["finished"] == 2 or (entry["finished"] == 0 and entry["next_sequence"] == 0)
                        for entry in protocol["workers"]
                    )
                    owner = protocol["first_error"]["worker_index"]
                    assert protocol["workers"][owner]["finished"] == 2
                    graph = protocol["graph"]
                    assert graph["public_completed"] == 0
                    assert graph["remaining_instances"] >= 1
                    assert graph["instances"][1]["materialize_state"] == 3
                    assert not graph["contributions"]
                    record = graph["materializations"][1]
                    assert record["status"] == protocol["scheduler_error"]
                    assert record["publish_before"] == record["publish_after"] == [0, 0]
                    published = worker.scheduler_mode_publish_count
                    with pytest.raises(RuntimeError, match="poisoned"):
                        worker.run(handle, args, config)
                    assert worker.scheduler_mode_publish_count == published
                    for buffer in buffers:
                        with pytest.raises(RuntimeError, match="still in use by a submitted run"):
                            worker.free(buffer)
                else:
                    assert read_perf_data(str(paths[0]))["protocol_audit"]["graph_verified"]
                evidence.append(
                    {
                        "capture": str(paths[0].relative_to(artifacts)),
                        "failed": malformed,
                        "scheduler_error": protocol["scheduler_error"],
                        "first_error": protocol["first_error"],
                        "run_epoch": protocol["run_epoch"],
                        "run_seconds": elapsed,
                        "native_error": str(failure.value) if failure is not None else None,
                    }
                )
        finally:
            if not failed:
                for buffer in buffers:
                    worker.free(buffer)
            closing = time.monotonic()
            worker.close()
            close_elapsed = time.monotonic() - closing
            assert close_elapsed < 20
            if evidence:
                evidence[-1]["worker_close_seconds"] = close_elapsed
    (artifacts / "result.json").write_text(json.dumps(evidence, indent=2))
