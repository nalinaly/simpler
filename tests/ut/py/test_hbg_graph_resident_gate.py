# Copyright (c) PyPTO Contributors.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------

"""Resident provenance augments, rather than replaces, the original Graph oracle."""

import importlib.util
import sys
from pathlib import Path
from types import SimpleNamespace

import pytest


@pytest.fixture(scope="module")
def native(request):
    request.config.addinivalue_line("markers", "scheduler_mode(kind): expected A5 HBG scheduler backend")
    directory = Path(__file__).resolve().parents[2] / "st/a5/host_build_graph/graph_execution"
    name = "_hbg_graph_resident_native"
    spec = importlib.util.spec_from_file_location(name, directory / "test_graph_execution_resident.py")
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    sys.path.insert(0, str(directory))
    try:
        spec.loader.exec_module(module)
    finally:
        sys.path.remove(str(directory))
    return module


@pytest.mark.parametrize("mode", [0, 3, 4, 5])
def test_correct_values_cannot_hide_legacy_fallback(native, monkeypatch, mode):
    calls = []
    monkeypatch.setattr(native._OriginalGraph, "compare_outputs", lambda *args: calls.append(args))
    scene = native.TestGraphExecutionResident()
    scene._graph_worker = SimpleNamespace(scheduler_runtime_mode=mode, scheduler_mode_publish_count=12)
    scene._graph_publish_count = 11
    with pytest.raises(AssertionError, match="resident Graph required"):
        scene.compare_outputs(None, None, [], {})
    assert len(calls) == 1


@pytest.mark.parametrize("mode", [1, 2])
def test_every_invocation_requires_its_own_publication(native, monkeypatch, mode):
    monkeypatch.setattr(native._OriginalGraph, "compare_outputs", lambda *args: None)
    scene = native.TestGraphExecutionResident()
    scene._graph_worker = SimpleNamespace(scheduler_runtime_mode=mode, scheduler_mode_publish_count=12)
    scene._graph_publish_count = 11
    scene.compare_outputs(None, None, [], {})
    assert scene._graph_publish_count == 12
    with pytest.raises(AssertionError, match="fresh scheduler publication"):
        scene.compare_outputs(None, None, [], {})


def test_resident_selection_does_not_bypass_numerical_comparison(native, monkeypatch):
    def fail(*args):
        raise AssertionError("original Graph numerical oracle failed")

    monkeypatch.setattr(native._OriginalGraph, "compare_outputs", fail)
    with pytest.raises(AssertionError, match="numerical oracle failed"):
        native.TestGraphExecutionResident().compare_outputs(None, None, [], {})


@pytest.mark.parametrize("kwargs,case", [({"skip_golden": True}, {}), ({}, {"skip_golden": True})])
def test_resident_acceptance_refuses_skipped_golden(native, kwargs, case):
    with pytest.raises(AssertionError, match="cannot skip golden"):
        native.TestGraphExecutionResident()._run_and_validate(None, None, case, **kwargs)


def test_native_fixture_reuses_original_graph_unchanged(native):
    scene, original = native.TestGraphExecutionResident, native._OriginalGraph
    assert scene.CALLABLE is original.CALLABLE
    assert scene.CASES is original.CASES
    assert scene.RTOL == original.RTOL and scene.ATOL == original.ATOL
    assert scene.SKIP_GOLDEN is False
    assert [m.args for m in scene.pytestmark if m.name == "scheduler_mode"] == [("resident",)]


def test_native_aic_aiv_fixture_reuses_original_kernels_and_oracle(native):
    scene, original = native.TestGraphExecutionAicAivResident, native._OriginalAicAiv
    assert scene.CALLABLE is original.CALLABLE
    assert scene.CASES is original.CASES
    assert scene.compute_golden is original.compute_golden
    assert scene.generate_args is original.generate_args
    assert [m.args for m in scene.pytestmark if m.name == "scheduler_mode"] == [("resident",)]


def test_runner_preserves_three_rounds_and_captures_initial_publication(native, monkeypatch):
    received = []
    monkeypatch.setattr(native._OriginalGraph, "_run_and_validate", lambda *args, **kw: received.append(kw))
    worker = SimpleNamespace(scheduler_mode_publish_count=21)
    scene = native.TestGraphExecutionResident()
    scene._run_and_validate(worker, object(), {"name": "original"}, rounds=1)
    assert received == [{"rounds": 3}]
    assert scene._graph_worker is worker and scene._graph_publish_count == 21


def test_each_diagnostic_round_gets_a_distinct_output_prefix(native, monkeypatch, tmp_path):
    paths = []

    def run(scene, worker, callable_obj, case, **options):
        for _ in range(options["rounds"]):
            scene._build_config({}, enable_chip_swimlane=4, output_prefix=options["output_prefix"])

    monkeypatch.setattr(native._OriginalGraph, "_run_and_validate", run)
    monkeypatch.setattr(
        native._OriginalGraph,
        "_build_config",
        lambda self, *args, **kwargs: paths.append(Path(kwargs["output_prefix"])),
    )
    scene = native.TestGraphExecutionResident()
    scene._run_and_validate(
        SimpleNamespace(scheduler_mode_publish_count=0),
        object(),
        {},
        enable_chip_swimlane=4,
        output_prefix=str(tmp_path),
        rounds=1,
    )
    assert len(paths) == len(set(paths)) == 3
    assert paths[-1] == tmp_path
    assert all(path.parent == tmp_path for path in paths[:-1])


@pytest.mark.parametrize("audit", ({}, {"protocol_verified": True}, {"graph_verified": False}))
def test_l4_numeric_and_resident_success_cannot_replace_a_verified_graph_capture(native, monkeypatch, tmp_path, audit):
    monkeypatch.setattr(native._OriginalGraph, "compare_outputs", lambda *args: None)
    monkeypatch.setattr(native, "read_perf_data", lambda path: {"protocol_audit": audit}, raising=False)
    scene = native.TestGraphExecutionResident()
    scene._graph_worker = SimpleNamespace(scheduler_runtime_mode=2, scheduler_mode_publish_count=2)
    scene._graph_publish_count = 1
    scene._graph_l4_capture = tmp_path / "capture.json"
    with pytest.raises(AssertionError, match="native Graph L4"):
        scene.compare_outputs(None, None, [], {})


@pytest.mark.parametrize(
    "scene_name,original_name,directory",
    [
        ("TestGraphMixResident", "_OriginalMix", "graph_execution"),
        ("TestGraphSyncResident", "_OriginalSync", "graph_sync_start_early_dispatch"),
        ("TestGraphPredicateResident", "_OriginalPredicate", "graph_predicated_dispatch"),
    ],
)
def test_capability_wrappers_keep_original_sources_inputs_and_golden(native, scene_name, original_name, directory):
    scene, original = getattr(native, scene_name), getattr(native, original_name)
    root = Path(native.__file__).parent.parent / directory
    assert scene.CASES is original.CASES
    assert scene.generate_args is original.generate_args
    assert scene.compute_golden is original.compute_golden
    assert scene.RTOL == original.RTOL and scene.ATOL == original.ATOL
    original_specs = [original.CALLABLE["orchestration"], *original.CALLABLE["incores"]]
    native_specs = [scene.CALLABLE["orchestration"], *scene.CALLABLE["incores"]]
    assert len(original_specs) == len(native_specs)
    for source, target in zip(original_specs, native_specs, strict=True):
        assert (root / source["source"]).resolve() == (root / target["source"]).resolve()
        assert {k: v for k, v in source.items() if k != "source"} == {k: v for k, v in target.items() if k != "source"}
    assert [m.args for m in scene.pytestmark if m.name == "scheduler_mode"] == [("resident",)]


@pytest.mark.parametrize(
    "scene_name,original_name",
    [
        ("TestGraphMixResident", "_OriginalMix"),
        ("TestGraphSyncResident", "_OriginalSync"),
        ("TestGraphPredicateResident", "_OriginalPredicate"),
    ],
)
def test_capability_wrappers_call_original_oracle_before_resident_check(native, monkeypatch, scene_name, original_name):
    calls = []
    monkeypatch.setattr(getattr(native, original_name), "compare_outputs", lambda *args: calls.append(args))
    scene = getattr(native, scene_name)()
    scene._graph_worker = SimpleNamespace(scheduler_runtime_mode=3, scheduler_mode_publish_count=2)
    scene._graph_publish_count = 1
    with pytest.raises(AssertionError, match="resident Graph required"):
        scene.compare_outputs(None, None, [], {})
    assert len(calls) == 1
