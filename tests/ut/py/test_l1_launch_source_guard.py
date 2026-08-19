# Copyright (c) PyPTO Contributors.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------

"""Source-level architectural guard for the borrowed-L1 launch path."""

from pathlib import Path

_PROJECT_ROOT = Path(__file__).resolve().parents[3]
_DEVICE_RUNNER_SOURCE = _PROJECT_ROOT / "src/common/platform/onboard/host/device_runner_base.cpp"
_AICPU_LOADER_SOURCE = _PROJECT_ROOT / "src/common/aicpu_loader/host/load_aicpu_op.cpp"

_FORBIDDEN_LAUNCH_APIS = (
    # Host-blocking completion is owned by the caller, never by PyPTO L1.
    "aclrtSynchronizeStream",
    "aclrtSynchronizeDevice",
    "rtStreamSynchronize",
    "rtDeviceSynchronize",
    "rtsStreamSynchronize",
    "rtsDeviceSynchronize",
    # PyPTO stays capture-transparent and never attaches hidden work to a model.
    "aclmdlRICapture",
    "rtStreamBeginCapture",
    "rtStreamEndCapture",
    "rtStreamGetCaptureInfo",
    "rtStreamAddToModel",
    "rtModelBindStream",
    # Persistent resources must have been created during init/prepare.
    "aclrtCreateStream",
    "aclrtDestroyStream",
    "aclrtCreateEvent",
    "aclrtDestroyEvent",
    "rtStreamCreate",
    "rtStreamDestroy",
    "rtEventCreate",
    "rtEventDestroy",
    "aclrtMalloc",
    "aclrtFree",
    "rtMalloc",
    "rtFree",
    "aclrtBinaryLoad",
    "aclrtBinaryUnLoad",
    "rtBinaryLoad",
    "rtBinaryUnload",
    "rtRegisterAllKernel",
)


def _extract_function(source: str, signature: str) -> str:
    signature_offset = source.index(signature)
    opening_brace = source.index("{", signature_offset)
    depth = 0
    for offset in range(opening_brace, len(source)):
        if source[offset] == "{":
            depth += 1
        elif source[offset] == "}":
            depth -= 1
            if depth == 0:
                return source[signature_offset : offset + 1]
    raise AssertionError(f"unterminated function: {signature}")


def test_l1_launch_cannot_spell_forbidden_runtime_apis() -> None:
    body = _extract_function(
        _DEVICE_RUNNER_SOURCE.read_text(encoding="utf-8"),
        "int DeviceRunnerBase::launch_l1_callable(",
    )

    observed = [api for api in _FORBIDDEN_LAUNCH_APIS if api in body]
    assert observed == []
    assert body.count("enqueue_l1_launch_sequence(") == 1


def test_l1_aicpu_and_aicore_callbacks_only_use_their_borrowed_stream_argument() -> None:
    body = _extract_function(
        _DEVICE_RUNNER_SOURCE.read_text(encoding="utf-8"),
        "int DeviceRunnerBase::launch_l1_callable(",
    )
    aicpu_start = body.index(".launch_aicpu =")
    aicore_start = body.index(".launch_aicore =", aicpu_start)
    cancel_start = body.index(".cancel_waiting_aicore =", aicore_start)
    aicpu_callback = body[aicpu_start:aicore_start]
    aicore_callback = body[aicore_start:cancel_start]

    assert "reinterpret_cast<rtStream_t>(stream)" in aicpu_callback
    assert "hidden_aicore_stream" not in aicpu_callback
    assert "LaunchWithHostArgs" in aicpu_callback
    assert "LaunchWithMutableHostArgs" in aicpu_callback

    assert "reinterpret_cast<rtStream_t>(stream)" in aicore_callback
    assert "launch_prepared_aicore_kernel" in aicore_callback
    assert "ensure_aicore_binary_registered" not in aicore_callback


def test_l1_close_pins_binary_without_any_unload_call() -> None:
    runner_body = _extract_function(
        _DEVICE_RUNNER_SOURCE.read_text(encoding="utf-8"),
        "int DeviceRunnerBase::finalize_l1_borrowed()",
    )
    pinned_body = _extract_function(
        _AICPU_LOADER_SOURCE.read_text(encoding="utf-8"),
        "int LoadAicpuOp::FinalizeL1Pinned()",
    )

    assert "load_aicpu_op_.FinalizeL1Pinned()" in runner_body
    assert "load_aicpu_op_.Finalize()" not in runner_body
    assert "aclrtBinaryUnLoad(" not in pinned_body
    assert "rtsBinaryUnload(" not in pinned_body
