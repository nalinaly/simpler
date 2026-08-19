# Copyright (c) PyPTO Contributors.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------

"""Device-free tests for the public simpler borrowed-L1 wrapper."""

from __future__ import annotations

from types import SimpleNamespace

import pytest
from simpler.task_interface import (
    ArgDirection,
    CallConfig,
    ChipCallable,
    ChipStorageTaskArgs,
    ChipWorker,
    CoreCallable,
)


class _FakeImpl:
    def __init__(self) -> None:
        self.initialized = False
        self.init_calls = []
        self.prepare_calls = []
        self.launch_calls = []
        self.finalize_calls = 0
        self.fail_finalize_once = False
        self.fail_init_with_retained_owner = False
        self.init_exception: BaseException | None = None

    def init_l1(self, *args) -> None:
        self.init_calls.append(args)
        self.initialized = True
        if self.fail_init_with_retained_owner:
            raise RuntimeError("injected init cleanup failure; explicit finalize required")
        if self.init_exception is not None:
            raise self.init_exception

    def l1_prepare_callable(self, *args) -> None:
        self.prepare_calls.append(args)

    def l1_make_prepare_queue_call(self, callable_id, _callable):
        return ("prepare", callable_id)

    def l1_launch(self, *args) -> None:
        self.launch_calls.append(args)

    def l1_make_launch_queue_call(self, callable_id, _args):
        return ("launch", callable_id)

    def finalize(self) -> None:
        self.finalize_calls += 1
        if self.fail_finalize_once and self.finalize_calls == 1:
            raise RuntimeError("injected native close failure")
        self.initialized = False


def _callable(binary: bytes = b"core") -> ChipCallable:
    core = CoreCallable.build([ArgDirection.IN], binary)
    return ChipCallable.build([ArgDirection.IN], "orch", b"orch", [(0, core)])


def _worker(monkeypatch: pytest.MonkeyPatch) -> tuple[ChipWorker, _FakeImpl]:
    worker = ChipWorker()
    impl = _FakeImpl()
    worker._impl = impl
    monkeypatch.setattr(
        ChipWorker,
        "_bootstrap_runtime_globals",
        staticmethod(lambda _bins, _log_level: "/runtime/dispatcher.so"),
    )
    bins = SimpleNamespace(
        host_path="/runtime/host.so",
        aicpu_path="/runtime/aicpu.so",
        aicore_path="/runtime/aicore.so",
    )
    worker.init_l1(1, bins, CallConfig())
    return worker, impl


def test_l1_registry_is_append_only_and_separate_from_l2(monkeypatch: pytest.MonkeyPatch) -> None:
    worker, impl = _worker(monkeypatch)
    first = _callable()
    second = _callable(b"different")
    args = ChipStorageTaskArgs()

    with pytest.raises(KeyError, match="has not been prepared"):
        worker.l1_make_launch_queue_call(0, args)

    assert worker.l1_make_prepare_queue_call(0, first) == ("prepare", 0)
    assert worker.l1_make_prepare_queue_call(0, first) == ("prepare", 0)
    with pytest.raises(RuntimeError, match="append-only"):
        worker.l1_make_prepare_queue_call(0, second)

    assert worker.l1_make_launch_queue_call(0, args) == ("launch", 0)
    worker.l1_launch(0, args, 0x1234)
    assert impl.launch_calls == [(0, args, 0x1234)]
    assert worker._callable_registry == {}
    assert worker._l1_callable_registry == {0: first}
    worker.finalize()


def test_l1_registry_accepts_ids_beyond_legacy_l2_capacity(monkeypatch: pytest.MonkeyPatch) -> None:
    worker, _ = _worker(monkeypatch)
    callable_obj = _callable()

    assert worker.l1_make_prepare_queue_call(64, callable_obj) == ("prepare", 64)
    assert worker.l1_make_launch_queue_call(64, ChipStorageTaskArgs()) == ("launch", 64)
    worker.finalize()


def test_l1_finalize_failure_preserves_retry_owners(monkeypatch: pytest.MonkeyPatch) -> None:
    worker, impl = _worker(monkeypatch)
    callable_obj = _callable()
    worker.l1_make_prepare_queue_call(3, callable_obj)
    impl.fail_finalize_once = True

    with pytest.raises(RuntimeError, match="injected native close failure"):
        worker.finalize()

    assert worker._execution_mode == "l1"
    assert worker._l1_callable_registry == {3: callable_obj}
    worker.finalize()
    assert worker._execution_mode is None
    assert worker._l1_callable_registry == {}
    assert impl.finalize_calls == 2


def test_l1_init_cleanup_failure_preserves_l1_finalize_mode(monkeypatch: pytest.MonkeyPatch) -> None:
    worker = ChipWorker()
    impl = _FakeImpl()
    impl.fail_init_with_retained_owner = True
    worker._impl = impl
    monkeypatch.setattr(
        ChipWorker,
        "_bootstrap_runtime_globals",
        staticmethod(lambda _bins, _log_level: "/runtime/dispatcher.so"),
    )
    bins = SimpleNamespace(
        host_path="/runtime/host.so",
        aicpu_path="/runtime/aicpu.so",
        aicore_path="/runtime/aicore.so",
    )

    with pytest.raises(RuntimeError, match="explicit finalize required"):
        worker.init_l1(1, bins, CallConfig())

    assert worker.initialized
    assert worker._execution_mode == "l1"
    worker.finalize()
    assert not worker.initialized
    assert worker._execution_mode is None


def test_l1_init_keyboard_interrupt_preserves_l1_finalize_mode(monkeypatch: pytest.MonkeyPatch) -> None:
    worker = ChipWorker()
    impl = _FakeImpl()
    impl.init_exception = KeyboardInterrupt("injected interrupt after native ownership")
    worker._impl = impl
    monkeypatch.setattr(
        ChipWorker,
        "_bootstrap_runtime_globals",
        staticmethod(lambda _bins, _log_level: "/runtime/dispatcher.so"),
    )
    bins = SimpleNamespace(
        host_path="/runtime/host.so",
        aicpu_path="/runtime/aicpu.so",
        aicore_path="/runtime/aicore.so",
    )

    with pytest.raises(KeyboardInterrupt, match="injected interrupt"):
        worker.init_l1(1, bins, CallConfig())

    assert worker.initialized
    assert worker._execution_mode == "l1"
    worker.finalize()
    assert not worker.initialized
    assert worker._execution_mode is None


@pytest.mark.parametrize("prior_mode", ["l1", "l2"])
def test_l1_init_requires_fresh_worker_without_overwriting_mode(
    monkeypatch: pytest.MonkeyPatch,
    prior_mode: str,
) -> None:
    worker, impl = _worker(monkeypatch)
    worker._execution_mode = prior_mode

    with pytest.raises(RuntimeError, match="fresh, uninitialized worker"):
        worker.init_l1(1, object(), CallConfig())

    assert worker._execution_mode == prior_mode
    assert len(impl.init_calls) == 1
    worker._execution_mode = "l1"
    worker.finalize()
