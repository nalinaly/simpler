# Copyright (c) PyPTO Contributors.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------
"""Uniform host-runtime pipeline-symbol contract tests."""

from __future__ import annotations

import subprocess
import sys
from pathlib import Path

import pytest

_PROJECT_ROOT = Path(__file__).resolve().parents[3]
_NEWLY_REQUIRED_PIPELINE_SYMBOLS = {
    "get_arena_bank_gm_heap_base_ctx",
    "get_pipeline_contract",
    "get_retained_temp_addr_ctx",
    "set_native_run_identity_ctx",
    "set_task_accepted_state_ctx",
    "supports_concurrent_native_prepare_ctx",
}
_HBG_AICPU_CANN_ENTRY_SYMBOLS = {
    "simpler_aicpu_exec",
    "simpler_aicpu_init",
    "simpler_aicpu_l1_hbg_exec",
    "simpler_aicpu_l1_hbg_register_callable",
    "simpler_aicpu_l1_hbg_register_execution_slot",
}

_SIM_CASES = [
    pytest.param(arch, "sim", runtime, id=f"{arch}-sim-{runtime}")
    for arch in ("a2a3", "a5")
    for runtime in ("host_build_graph", "tensormap_and_ringbuffer")
]
_ONBOARD_CASES = [
    pytest.param(
        arch,
        "onboard",
        runtime,
        id=f"{arch}-onboard-{runtime}",
        marks=[pytest.mark.requires_hardware, pytest.mark.platforms([arch])],
    )
    for arch in ("a2a3", "a5")
    for runtime in ("host_build_graph", "tensormap_and_ringbuffer")
]


def _defined_external_symbols(path: Path) -> set[str]:
    if sys.platform == "darwin":
        command = ["nm", "-gU", str(path)]
    else:
        command = ["nm", "-D", "--defined-only", str(path)]
    result = subprocess.run(command, check=True, capture_output=True, text=True)
    symbols = {line.split()[-1] for line in result.stdout.splitlines() if line.split()}
    if sys.platform == "darwin":
        return {symbol.removeprefix("_") for symbol in symbols}
    return symbols


@pytest.mark.parametrize(
    ("arch", "variant", "runtime"),
    _SIM_CASES + _ONBOARD_CASES,
)
def test_host_runtime_exports_required_pipeline_symbols(arch: str, variant: str, runtime: str):
    runtime_dir = _PROJECT_ROOT / "build" / "lib" / arch / variant / runtime
    runtime_libraries = tuple(runtime_dir.glob("libhost_runtime.*"))
    assert len(runtime_libraries) == 1, runtime_dir

    runtime_path = runtime_libraries[0]
    symbols = _defined_external_symbols(runtime_path)

    assert _NEWLY_REQUIRED_PIPELINE_SYMBOLS <= symbols, sorted(_NEWLY_REQUIRED_PIPELINE_SYMBOLS - symbols)


@pytest.mark.parametrize(
    "arch",
    [
        pytest.param("a2a3", marks=[pytest.mark.requires_hardware, pytest.mark.platforms(["a2a3"])]),
        pytest.param("a5", marks=[pytest.mark.requires_hardware, pytest.mark.platforms(["a5"])]),
    ],
)
def test_hbg_onboard_aicpu_exports_only_cann_entries(arch: str):
    """HBG internals must not interpose another runtime in CANN's global namespace."""
    runtime_dir = _PROJECT_ROOT / "build" / "lib" / arch / "onboard" / "host_build_graph"
    runtime_libraries = tuple(runtime_dir.glob("libaicpu_kernel.*"))
    assert len(runtime_libraries) == 1, runtime_dir

    assert _defined_external_symbols(runtime_libraries[0]) == _HBG_AICPU_CANN_ENTRY_SYMBOLS
