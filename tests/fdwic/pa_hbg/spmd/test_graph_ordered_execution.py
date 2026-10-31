# Copyright (c) PyPTO Contributors.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------

"""Graph roots and separate instances retain native block, overlap and sync contracts."""

from simpler_setup import SceneTestCase, scene_test

if __package__:
    from .test_ordered_execution import TestHbgOrderedExecution as _Ordered
    from .test_sync_ordered_execution import TestHbgSyncOrderedExecution as _SyncOrdered
else:
    from test_ordered_execution import TestHbgOrderedExecution as _Ordered
    from test_sync_ordered_execution import TestHbgSyncOrderedExecution as _SyncOrdered


_CALLABLE = {
    **_SyncOrdered.CALLABLE,
    "orchestration": {
        **_SyncOrdered.CALLABLE["orchestration"],
        "source": "kernels/graph_ordered_orchestration.cpp",
    },
}


def _grouped(case, split):
    return {
        **case,
        "name": f"{case['name']}_{'instances' if split else 'roots'}",
        "params": {**case["params"], "split_graphs": split},
    }


class _GraphRunChecks:
    def generate_args(self, params):
        args = super().generate_args(params)
        args.add_scalar("split_graphs", int(params["split_graphs"]))
        return args

    def _run_and_validate(self, worker, *args, **kwargs):
        kwargs["rounds"] = max(3, kwargs.get("rounds", 1))
        return super()._run_and_validate(worker, *args, **kwargs)


@scene_test(level=2, runtime="host_build_graph")
class TestHbgGraphOrderedExecution(_GraphRunChecks, _Ordered):
    CALLABLE = _CALLABLE
    CASES = [
        _grouped(
            {
                "name": f"mask{mask}_n{blocks}",
                "platforms": ["a5", "a5sim"],
                "params": {"scenario": "shape", "mask": mask, "blocks": blocks},
            },
            False,
        )
        for mask in range(1, 8)
        for blocks in (1, 2, 7, 8, 9, 31, 32, 33, 63, 64, 65, 512)
    ] + [
        _grouped(case, split)
        for case in _Ordered.CASES
        if case["params"]["scenario"] in ("overlap", "edge", "fanin") or "invocation_blocks" in case["params"]
        for split in (False, True)
    ]


@scene_test(level=2, runtime="host_build_graph")
class TestHbgGraphSyncOrderedExecution(_GraphRunChecks, _SyncOrdered):
    CALLABLE = _CALLABLE
    CASES = [
        _grouped(case, split)
        for case in _SyncOrdered.CASES
        if case["params"]["mask"] == case["params"]["other_mask"]
        or (case["params"]["mask"], case["params"]["other_mask"]) in ((1, 2), (3, 6), (6, 5), (7, 1))
        for split in (False, True)
    ]


if __name__ == "__main__":
    SceneTestCase.run_module(__name__)
