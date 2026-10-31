# Copyright (c) PyPTO Contributors.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------

"""The native Graph lifecycle probe owns its registration, not a fixed slot."""

import importlib.util
import sys
from pathlib import Path
from types import SimpleNamespace

import pytest


@pytest.fixture(scope="module")
def native():
    path = Path(__file__).resolve().parents[3] / "fdwic/pa_hbg/spmd/test_graph_native_lifecycle.py"
    with pytest.MonkeyPatch.context() as patch:
        patch.syspath_prepend(str(path.parent))
        spec = importlib.util.spec_from_file_location("_graph_lifecycle_native_fixture", path)
        assert spec is not None and spec.loader is not None
        module = importlib.util.module_from_spec(spec)
        patch.setitem(sys.modules, spec.name, module)
        spec.loader.exec_module(module)
        yield module


@pytest.mark.parametrize("slot", (7, 17))
@pytest.mark.parametrize("mask", (1, 2))
def test_existing_slot_zero_is_preserved_and_own_registration_is_released_on_failure(native, slot, mask):
    class PrepareStopped(RuntimeError):
        pass

    unrelated = object()
    target = object()
    handle = object()
    registered = {0: unrelated}
    releases = []

    def register_at_slot(index, callable_obj):
        if index in registered:
            raise RuntimeError(f"callable_id={index} already registered")
        registered[index] = callable_obj

    def register(callable_obj):
        assert callable_obj is target
        register_at_slot(slot, callable_obj)
        return handle

    def resolve(value):
        assert value is handle
        return SimpleNamespace(slot_id=slot)

    def unregister(value):
        assert value is handle
        releases.append(value)
        assert registered.pop(slot) is target

    chip = SimpleNamespace(
        _impl=SimpleNamespace(supports_concurrent_native_prepare=True),
        _register_callable_at_slot=register_at_slot,
    )
    worker = SimpleNamespace(_chip_worker=chip, register=register, _resolve_handle=resolve, unregister=unregister)
    probe = native.TestHbgGraphNativeLifecycle()

    def stop_before_prepare(_params):
        assert registered == {0: unrelated, slot: target}
        raise PrepareStopped("argument preparation interrupted")

    probe.generate_args = stop_before_prepare
    with pytest.raises(PrepareStopped, match="argument preparation interrupted"):
        probe._run_and_validate_l2(worker, target, {"params": {"mask": mask}})
    assert releases == [handle]
    assert registered == {0: unrelated}
