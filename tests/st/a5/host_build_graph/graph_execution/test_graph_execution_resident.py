#!/usr/bin/env python3
# Copyright (c) PyPTO Contributors.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------

"""Original Graph kernels and golden, with per-invocation resident provenance."""

import importlib.util
import sys
from pathlib import Path

import pytest

from simpler_setup import Scalar, SceneTestCase, TaskArgsBuilder, scene_test

if __package__:
    from .test_graph_execution import TestGraphExecutionHostBuildGraphA5 as _OriginalGraph
    from .test_graph_execution_aic_aiv import TestGraphExecutionAicAivHostBuildGraphA5 as _OriginalAicAiv
    from .test_graph_execution_mix_spmd import TestGraphExecutionMixSpmdHostBuildGraphA5 as _OriginalMix
else:
    from test_graph_execution import TestGraphExecutionHostBuildGraphA5 as _OriginalGraph
    from test_graph_execution_aic_aiv import TestGraphExecutionAicAivHostBuildGraphA5 as _OriginalAicAiv
    from test_graph_execution_mix_spmd import TestGraphExecutionMixSpmdHostBuildGraphA5 as _OriginalMix


def _original_neighbor(directory, filename, class_name):
    path = Path(__file__).resolve().parent.parent / directory / filename
    name = f"_hbg_graph_original_{directory}"
    spec = importlib.util.spec_from_file_location(name, path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    original = getattr(module, class_name)
    callable_spec = dict(original.CALLABLE)
    callable_spec["orchestration"] = {
        **original.CALLABLE["orchestration"],
        "source": str((path.parent / original.CALLABLE["orchestration"]["source"]).resolve()),
    }
    callable_spec["incores"] = [
        {**kernel, "source": str((path.parent / kernel["source"]).resolve())} for kernel in original.CALLABLE["incores"]
    ]
    return original, callable_spec


_OriginalSync, _SYNC_CALLABLE = _original_neighbor(
    "graph_sync_start_early_dispatch", "test_graph_sync_start_early_dispatch.py", "TestGraphSyncStartEarlyDispatchHbg"
)
_OriginalPredicate, _PREDICATE_CALLABLE = _original_neighbor(
    "graph_predicated_dispatch", "test_graph_predicated_dispatch.py", "TestGraphPredicatedDispatchA5"
)


class _ResidentGraphChecks(SceneTestCase):
    def _run_and_validate(self, worker, callable_obj, case, *args, **kwargs):
        assert not (kwargs.get("skip_golden", False) or case.get("skip_golden", self.SKIP_GOLDEN)), (
            "resident Graph acceptance cannot skip golden"
        )
        self._graph_worker = worker
        self._graph_publish_count = worker.scheduler_mode_publish_count
        kwargs["rounds"] = max(3, kwargs.get("rounds", 1))
        return super()._run_and_validate(worker, callable_obj, case, *args, **kwargs)

    def compare_outputs(self, test_args, golden_args, output_names, params):
        super().compare_outputs(test_args, golden_args, output_names, params)
        count = self._graph_worker.scheduler_mode_publish_count
        mode = self._graph_worker.scheduler_runtime_mode
        assert count > self._graph_publish_count, "resident Graph requires a fresh scheduler publication"
        assert mode in (1, 2), f"resident Graph required; invocation mode={mode} (3/4 are AICPU fallback)"
        self._graph_publish_count = count


@scene_test(level=2, runtime="host_build_graph")
class TestGraphExecutionResident(_ResidentGraphChecks, _OriginalGraph):
    pytestmark = [pytest.mark.scheduler_mode("resident")]


@scene_test(level=2, runtime="host_build_graph")
class TestGraphExecutionAicAivResident(_ResidentGraphChecks, _OriginalAicAiv):
    pytestmark = [pytest.mark.scheduler_mode("resident")]


@scene_test(level=2, runtime="host_build_graph")
class TestGraphMixResident(_ResidentGraphChecks, _OriginalMix):
    pytestmark = [pytest.mark.scheduler_mode("resident")]


@scene_test(level=2, runtime="host_build_graph")
class TestGraphSyncResident(_ResidentGraphChecks, _OriginalSync):
    pytestmark = [pytest.mark.scheduler_mode("resident")]
    CALLABLE = _SYNC_CALLABLE


@scene_test(level=2, runtime="host_build_graph")
class TestGraphPredicateResident(_ResidentGraphChecks, _OriginalPredicate):
    pytestmark = [pytest.mark.scheduler_mode("resident")]
    CALLABLE = _PREDICATE_CALLABLE


@scene_test(level=2, runtime="host_build_graph")
class TestGraphNoBoundaryResident(_ResidentGraphChecks):
    pytestmark = [pytest.mark.scheduler_mode("resident")]
    CALLABLE = {
        "orchestration": {
            "source": "kernels/orchestration/graph_no_boundary_orch.cpp",
            "function_name": "aicpu_orchestration_entry",
            "signature": [],
        },
        "incores": [],
    }
    CASES = [
        {
            "name": f"instances{instances}_children{children}",
            "platforms": ["a5", "a5sim"],
            "params": {"instances": instances, "children": children},
        }
        for instances, children in ((1, 2), (16, 8), (2, 1024))
    ]

    def generate_args(self, params):
        return TaskArgsBuilder(Scalar("instances", params["instances"]), Scalar("children", params["children"]))

    def compute_golden(self, args, params):
        pass


if __name__ == "__main__":
    SceneTestCase.run_module(__name__)
