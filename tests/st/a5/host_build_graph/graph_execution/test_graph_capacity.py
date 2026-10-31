# Copyright (c) PyPTO Contributors.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------

"""Invalid Graph body/cohort capacities fail before backend selection or device execution."""

import importlib.util
import sys
from contextlib import suppress
from pathlib import Path

import pytest
import torch
from simpler.task_interface import CallConfig, DataType, TaskArgs, TensorArgType
from simpler.worker import Worker

pytestmark = [
    pytest.mark.platforms(["a5sim", "a5"]),
    pytest.mark.device_count(1),
    pytest.mark.runtime("host_build_graph"),
    pytest.mark.scheduler_mode("resident"),
]


@pytest.fixture(scope="module")
def graph_probe():
    source = Path(__file__).resolve().parents[4] / "fdwic/pa_hbg/spmd/test_graph_wide_fanin.py"
    with pytest.MonkeyPatch.context() as patch:
        patch.syspath_prepend(str(source.parent))
        spec = importlib.util.spec_from_file_location("_graph_capacity_fixture", source)
        assert spec is not None and spec.loader is not None
        module = importlib.util.module_from_spec(spec)
        patch.setitem(sys.modules, spec.name, module)
        spec.loader.exec_module(module)
        yield module.TestHbgGraphWideFanin


@pytest.mark.parametrize(
    "shape,error_code,diagnostic",
    [pytest.param((1, 2, 1024, 0), 5, "failed to finalize asynchronous Graph", id="body1025")]
    + [
        pytest.param(
            (73 if mask in (2, 4) else 37, mask, 1, 1),
            7,
            "REQUIRE_SYNC_START_INVALID",
            id=f"sync_mask{mask}_over_capacity",
        )
        for mask in range(1, 8)
    ],
)
def test_invalid_graph_capacity_is_not_a_legacy_success(
    st_platform, st_device_ids, graph_probe, capfd, drain_host_log, shape, error_code, diagnostic
):
    blocks, mask, tasks, sync = shape
    worker = Worker(level=2, platform=st_platform, runtime="host_build_graph", device_id=int(st_device_ids[0]))
    buffers = []
    try:
        handle = worker.register(graph_probe.compile_chip_callable(st_platform))
        worker.init()
        args = TaskArgs()
        for elements in (tasks * (blocks * 3 + 1) * 16, 32):
            value = torch.zeros(elements, dtype=torch.int64)
            buffer = worker.malloc(value.nbytes)
            buffers.append(buffer)
            worker.copy_to(buffer, value)
            args.add_tensor(buffer.tensor(shapes=tuple(value.shape), dtype=DataType.INT64), TensorArgType.INOUT)
        for scalar in (blocks, mask, tasks, 0, sync):
            args.add_scalar(scalar)
        before = worker.scheduler_mode_publish_count
        with pytest.raises(RuntimeError, match=rf"prepare_native_run failed with code -{error_code}\b"):
            worker.run(handle, args, CallConfig())
        log = drain_host_log(capfd)
        assert diagnostic in log, log
        assert worker.scheduler_mode_publish_count == before, "invalid Graph reached backend publication"
    finally:
        for buffer in buffers:
            with suppress(RuntimeError):
                worker.free(buffer)
        worker.close()
