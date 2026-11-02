# Copyright (c) PyPTO Contributors.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------

"""Large Graph parameter construction, mixed instance sizes and replay on all engines."""

import copy
import json

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
class TestGraphParallelMaterialization(_ResidentGraphChecks):
    pytestmark = [pytest.mark.scheduler_mode("resident")]
    RTOL = 0
    ATOL = 0
    CALLABLE = {
        "orchestration": {
            "source": "kernels/orchestration/graph_parallel_materialize_orch.cpp",
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
        ]
        + [
            {
                "func_id": lane,
                "name": f"mix_writer_{lane}",
                "source": "kernels/aic/kernel_spmd_mix.cpp" if lane == 2 else "kernels/aiv/kernel_spmd_mix.cpp",
                "core_type": "aic" if lane == 2 else "aiv",
                "signature": [ArgDirection.INOUT],
            }
            for lane in (2, 3, 4)
        ],
    }
    CASES = [
        {
            "name": f"mode{mode}_children{children}_instances{instances}_scalars{scalars}_tail{tail}",
            "platforms": ["a5", "a5sim"],
            "params": dict(mode=mode, children=children, instances=instances, scalars=scalars, seed=seed, tail=tail),
        }
        for mode, children, instances, scalars, seed, tail in (
            (0, 64, 2, 1, 17, 0),
            (1, 1024, 1, 9, 0x10003, 0),
            (1, 64, 3, 15, 0x20005, 0),
            (2, 64, 2, 1, 0, 0),
            (1, 64, 2, 7, 0x30007, 1),
            (1, 1, 3, 7, 0x40009, 0),
        )
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
        mixed = params["mode"] == 2
        count = params["instances"] * params["children"] * (96 if mixed else 64)
        return TaskArgsBuilder(
            TensorArg("records", torch.full((count,), -1, dtype=torch.float32 if mixed else torch.int64)),
            *(Scalar(key, params[key]) for key in ("instances", "children", "scalars", "mode", "seed", "tail")),
        )

    def compute_golden(self, args, params):
        mixed = params["mode"] == 2
        width = 96 if mixed else 64
        outputs = args.records.view(params["instances"], params["children"], width)
        for instance, rows in enumerate(outputs):
            count = params["tail"] if params["tail"] and instance + 1 == params["instances"] else params["children"]
            for child, row in enumerate(rows[:count]):
                if mixed:
                    for block in range(2):
                        for lane in range(3):
                            row[(block * 3 + lane) * 16] = block
                else:
                    scalars = [params["scalars"] + ((child + 1) << 8)]
                    scalars.extend(
                        params["seed"] + instance * 0x10000 if index % 2 == 0 else 100 + index
                        for index in range(1, params["scalars"])
                    )
                    for index, value in enumerate(scalars):
                        row[index] = value
                        row[32 + index] = value ^ 0x5555
                    row[31] = (instance * params["children"] + child) * 64
                    row[63] = row[31] + 32

    def compare_outputs(self, test_args, golden_args, output_names, params):
        super().compare_outputs(test_args, golden_args, output_names, params)
        if self._graph_l4_capture is not None:
            raw = json.loads(self._graph_l4_capture.read_text())["aicore_protocol"]
            parallel = params["children"] >= 64
            assert raw["schema"] == ("hbg-native-protocol-v12" if parallel else "hbg-native-protocol-v9")
            if parallel:
                assert raw["graph"]["work_count"] > params["instances"]
                assert len(raw["graph"]["work"]) == raw["graph"]["work_count"]
                assert len(raw["graph"]["publications"]) == params["instances"]


if __name__ == "__main__":
    SceneTestCase.run_module(__name__)
