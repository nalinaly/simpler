# Copyright (c) PyPTO Contributors.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------

"""Adjacent instance storage, odd scalar tails and aliased boundary views."""

import copy

import pytest
import torch
from simpler.task_interface import ArgDirection

from simpler_setup import Scalar, SceneTestCase, TaskArgsBuilder, TensorArg, scene_test
from simpler_setup.kernel_compiler import KernelCompiler
from simpler_setup.scene_test import _pto_isa_compile_cache_token, compile_chip_callable_spec

if __package__:
    from .test_graph_execution_resident import _ResidentGraphChecks
else:
    from test_graph_execution_resident import _ResidentGraphChecks


@scene_test(level=2, runtime="host_build_graph")
class TestGraphParameterBoundary(_ResidentGraphChecks):
    pytestmark = [pytest.mark.scheduler_mode("resident")]
    RTOL = 0
    ATOL = 0
    CALLABLE = {
        "orchestration": {
            "source": "kernels/orchestration/graph_parameter_boundary_orch.cpp",
            "function_name": "aicpu_orchestration_entry",
            "signature": [ArgDirection.INOUT],
        },
        "incores": [
            {
                "func_id": lane,
                "name": f"parameter_echo_{lane}",
                "source": "kernels/graph_parameter_echo.cpp",
                "core_type": "aic" if lane == 0 else "aiv",
                "signature": [ArgDirection.INOUT, ArgDirection.INOUT],
            }
            for lane in (0, 1)
        ],
    }
    CASES = [
        {
            "name": f"lane{lane}_instances{instances}_scalars{scalars}",
            "platforms": ["a5", "a5sim"],
            "params": {"lane": lane, "instances": instances, "scalars": scalars},
        }
        for lane in (0, 1)
        for instances in (16, 65)
        for scalars in (1, 7, 9, 15)
    ]

    @classmethod
    def compile_chip_callable(cls, platform):
        compiler = KernelCompiler(platform=platform)
        variant = "sim" if platform.endswith("sim") else "onboard"
        spec = copy.deepcopy(cls.CALLABLE)
        for kernel in spec["incores"]:
            kernel["extra_include_dirs"] = [str(compiler.platform_dir / variant / "aicore")]
        return compile_chip_callable_spec(
            spec, platform, cls._st_runtime, (__name__, platform, _pto_isa_compile_cache_token())
        )

    def generate_args(self, params):
        return TaskArgsBuilder(
            TensorArg("records", torch.zeros(params["instances"] * 64, dtype=torch.int64)),
            Scalar("instances", params["instances"]),
            Scalar("scalars", params["scalars"]),
            Scalar("lane", params["lane"]),
        )

    def compute_golden(self, args, params):
        for instance, row in enumerate(args.records.view(-1, 64)):
            scalars = [params["scalars"] + ((instance + 1) << 8)]
            scalars.extend(
                0x100000 + instance if index % 2 == 0 else 100 + index for index in range(1, params["scalars"])
            )
            for index, value in enumerate(scalars):
                row[index] = value
                row[32 + index] = value ^ 0x5555
            row[31] = instance * 64
            row[63] = instance * 64 + 32


if __name__ == "__main__":
    SceneTestCase.run_module(__name__)
