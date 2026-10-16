#!/usr/bin/env python3
# Copyright (c) PyPTO Contributors.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------
"""大参数 MIX：验证完整参数数组、尾部标量、双 AIV 身份与 sync_start。"""

import torch
from simpler.task_interface import ArgDirection as D

from simpler_setup import SceneTestCase, TaskArgsBuilder, TensorArg, scene_test


def capacity_case(runtime, count):
    signature = [D.IN] * (count - 1) + [D.INOUT]

    class CapacityCase(SceneTestCase):
        RTOL = 0
        ATOL = 0
        CALLABLE = {
            "orchestration": {
                "source": "kernels/orchestration.cpp",
                "function_name": "aicpu_orchestration_entry",
                "signature": signature,
            },
            "incores": [
                {"func_id": i, "source": f"kernels/{kind}.cpp", "core_type": kind, "signature": signature}
                for i, kind in enumerate(("aic", "aiv", "aiv"))
            ],
        }
        CASES = [{"name": f"tensors_{count}_scalars_16", "platforms": ["a2a3sim", "a2a3"], "params": {}}]

        def generate_args(self, params):
            return TaskArgsBuilder(
                *(
                    TensorArg(f"x{i}", torch.tensor([count if i == 0 else i + 1], dtype=torch.float32))
                    for i in range(count - 1)
                ),
                TensorArg("out", torch.full((24 * 3 * 16,), -1.0, dtype=torch.float32)),
            )

        def compute_golden(self, args, params):
            value = count + sum((i + 1) ** 2 for i in range(1, count - 1))
            value += sum((101 + i) * (i + 1) for i in range(16))
            for block in range(24):
                for lane in range(3):
                    args.out[(block * 3 + lane) * 16] = value + block * 1000 + lane * 100 + 24

    name = f"TestCapacity_{runtime}_{count}"
    CapacityCase.__name__ = name
    CapacityCase.__qualname__ = name
    return scene_test(level=2, runtime=runtime)(CapacityCase)


TestHbg133 = capacity_case("host_build_graph", 133)
TestHbg256 = capacity_case("host_build_graph", 256)
TestTmr133 = capacity_case("tensormap_and_ringbuffer", 133)
TestTmr256 = capacity_case("tensormap_and_ringbuffer", 256)

if __name__ == "__main__":
    SceneTestCase.run_module(__name__)
