# Copyright (c) PyPTO Contributors.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------

"""A Graph's saved CSR supports wider fanin than a public task's inline payload."""

from simpler_setup import SceneTestCase, scene_test

if __package__:
    from .test_spmd_contract import TestHbgSpmdContract as _SpmdContract
else:
    from test_spmd_contract import TestHbgSpmdContract as _SpmdContract


@scene_test(level=2, runtime="host_build_graph")
class TestHbgGraphWideFanin(_SpmdContract):
    CALLABLE = {
        **_SpmdContract.CALLABLE,
        "orchestration": {
            **_SpmdContract.CALLABLE["orchestration"],
            "source": "kernels/graph_wide_fanin_orchestration.cpp",
        },
    }
    CASES = [
        {
            "name": f"mask{mask}_fanin{tasks}_body{tasks + 1}",
            "platforms": ["a5", "a5sim"],
            "params": {"mask": mask, "blocks": 1, "tasks": tasks, "tail_delay": 4096},
        }
        for mask in (1, 2)
        for tasks in (128, 129, 1023)
    ]

    def generate_args(self, params):
        args = super().generate_args(params)
        args.summary.resize_(32).zero_()
        return args

    def compute_golden(self, args, params):
        super().compute_golden(args, params)
        args.summary[16] = args.summary[0]

    def _run_and_validate(self, worker, *args, **kwargs):
        kwargs["rounds"] = max(3, kwargs.get("rounds", 1))
        return super()._run_and_validate(worker, *args, **kwargs)


if __name__ == "__main__":
    SceneTestCase.run_module(__name__)
