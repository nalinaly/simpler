# Copyright (c) PyPTO Contributors.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------

"""Graph Definition, payload and heap ownership across prepared pipeline slots."""

from contextlib import suppress

import pytest

from simpler_setup import SceneTestCase, scene_test
from simpler_setup.scene_test import _build_chip_task_args, _compare_outputs

if __package__:
    from .test_graph_wide_fanin import TestHbgGraphWideFanin as _WideFanin
    from .test_spmd_contract import require_fresh_resident
else:
    from test_graph_wide_fanin import TestHbgGraphWideFanin as _WideFanin
    from test_spmd_contract import require_fresh_resident


@scene_test(level=2, runtime="host_build_graph")
class TestHbgGraphNativeLifecycle(_WideFanin):
    CASES = [
        {"name": f"mask{mask}_retained_slots", "platforms": ["a5", "a5sim"], "params": {"mask": mask}}
        for mask in (1, 2)
    ]

    def _run_and_validate_l2(self, worker, callable_obj, case, **kwargs):
        assert not kwargs.get("skip_golden", False)
        chip = worker._chip_worker  # noqa: SLF001 -- production native-run lifecycle seam
        assert chip is not None
        concurrent = bool(chip._impl.supports_concurrent_native_prepare)  # noqa: SLF001
        config = self._build_config(case.get("config", {}))
        handle = worker.register(callable_obj)
        live = []

        def prepare(tasks, slot):
            params = {"mask": case["params"]["mask"], "blocks": 1, "tasks": tasks, "tail_delay": 4096}
            actual = self.generate_args(params)
            args, outputs = _build_chip_task_args(actual, self.CALLABLE["orchestration"]["signature"])
            expected = actual.clone()
            self.compute_golden(expected, params)
            before = chip.scheduler_mode_publish_count
            generation = getattr(self, "_graph_lease_generation", 0) + 1
            token = chip._prepare_native_run_with_pipeline_lease(  # noqa: SLF001
                callable_id, args, slot, generation, config=config
            )
            self._graph_lease_generation = generation
            live.append(token)
            require_fresh_resident(chip, before)
            return token, actual, args, outputs, expected

        def finish(bundle):
            token, actual, _, outputs, expected = bundle
            chip._wait_native_run(token)  # noqa: SLF001
            chip._finalize_native_run(token)  # noqa: SLF001
            live.remove(token)
            _compare_outputs(actual, expected, outputs, self.RTOL, self.ATOL)

        first = None
        try:
            callable_id = worker._resolve_handle(handle).slot_id  # noqa: SLF001 -- native-run seam needs the local slot
            for round_index, (large, small) in enumerate(((129, 1), (1, 1023), (1023, 129)), 1):
                a = prepare(large, 0)
                if first is None:
                    first = a[0]
                chip._launch_native_run(a[0])  # noqa: SLF001
                if concurrent:
                    b = prepare(small, 1)
                    assert a[0].slot_id != b[0].slot_id
                    assert a[0].run_epoch != b[0].run_epoch
                    banks = (chip.arena_bank_gm_heap_base(0), chip.arena_bank_gm_heap_base(1))
                    assert all(banks) and banks[0] != banks[1]
                    with pytest.raises(RuntimeError, match="launch_native_run failed"):
                        chip._launch_native_run(b[0])  # noqa: SLF001
                    finish(a)
                else:
                    finish(a)
                    b = prepare(small, 0)
                chip._launch_native_run(b[0])  # noqa: SLF001
                finish(b)
                with pytest.raises(RuntimeError, match="stale|finalized|wrong phase"):
                    chip._poll_native_run(first)  # noqa: SLF001
                print(
                    f"[GRAPH_SLOT_REUSE] round={round_index} concurrent_prepare={concurrent} "
                    f"bodies={large + 1}/{small + 1}"
                )
        finally:
            for token in reversed(live):
                with suppress(RuntimeError):
                    chip._finalize_native_run(token)  # noqa: SLF001
            worker.unregister(handle)


if __name__ == "__main__":
    SceneTestCase.run_module(__name__)
