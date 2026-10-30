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

import pytest

from simpler_setup import SceneTestCase, scene_test

if __package__:
    from .test_graph_execution import TestGraphExecutionHostBuildGraphA5 as _OriginalGraph
    from .test_graph_execution_aic_aiv import TestGraphExecutionAicAivHostBuildGraphA5 as _OriginalAicAiv
else:
    from test_graph_execution import TestGraphExecutionHostBuildGraphA5 as _OriginalGraph
    from test_graph_execution_aic_aiv import TestGraphExecutionAicAivHostBuildGraphA5 as _OriginalAicAiv


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


if __name__ == "__main__":
    SceneTestCase.run_module(__name__)
