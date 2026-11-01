# Copyright (c) PyPTO Contributors.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------

"""Graph materialization recovery respects the native execution-owner lifetime."""

import importlib.util
import json
from pathlib import Path
from types import SimpleNamespace

import pytest


@pytest.fixture(scope="module")
def native(request):
    request.config.addinivalue_line("markers", "scheduler_mode(kind): required scheduler selection")
    path = Path(__file__).resolve().parents[3] / "st/a5/host_build_graph/graph_execution/test_graph_device_failure.py"
    spec = importlib.util.spec_from_file_location("_graph_materialization_failure_fixture", path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


@pytest.mark.parametrize("platform", ("a5", "a5sim"))
def test_recovery_uses_a_fresh_process_only_on_hardware(native, platform, monkeypatch, tmp_path):
    calls = []

    def run(actual_platform, device_id, artifacts, *, recovery):
        assert (actual_platform, device_id) == (platform, 0)
        assert artifacts.is_dir()
        calls.append(("local", recovery))
        return [{"failed": False}] if recovery else [{"failed": False}, {"failed": True}]

    def recover(actual_platform, device_id, artifacts):
        assert (actual_platform, device_id) == ("a5", 0)
        assert artifacts.is_dir()
        calls.append(("child", True))
        return [{"failed": False}]

    monkeypatch.setattr(native, "_ROOT", tmp_path)
    monkeypatch.setattr(native, "_run_materialization_pass", run)
    monkeypatch.setattr(native, "_recover_materialization_pass", recover)
    native.test_scalar_materialization_error_retires_before_kernel_entry(platform, [0])
    assert calls == [("local", False), ("local" if platform.endswith("sim") else "child", True)]
    paths = list((tmp_path / "outputs").glob("*/result.json"))
    assert len(paths) == 1
    assert [entry["failed"] for entry in json.loads(paths[0].read_text())] == [False, True, False]


@pytest.mark.parametrize("returncode", (0, 1))
def test_recovery_child_failure_is_not_reported_as_success(native, returncode, monkeypatch, tmp_path):
    def run(command, **kwargs):
        assert str(Path(native.__file__).resolve()) in command
        assert command[-6:] == ["--platform", "a5", "--device", "0", "--artifacts", str(tmp_path)]
        assert kwargs["timeout"] == 300
        return SimpleNamespace(
            returncode=returncode,
            stdout=native._RECOVERY_MARKER + '[{"failed": false}]',
            stderr="recovery refused" if returncode else "",
        )

    monkeypatch.setattr(native.subprocess, "run", run)
    if returncode:
        with pytest.raises(AssertionError, match="recovery refused"):
            native._recover_materialization_pass("a5", 0, tmp_path)
    else:
        assert native._recover_materialization_pass("a5", 0, tmp_path) == [{"failed": False}]
