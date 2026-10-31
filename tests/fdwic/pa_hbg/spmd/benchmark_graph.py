# Copyright (c) PyPTO Contributors.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------

"""Run unchanged HBG fixtures with per-call provenance, memory and host-wall samples.

Run from the checkout whose editable installation is active. The harness can
live outside that checkout, so a detached legacy reference needs no source
changes. Host bind and complete Device RunWall come from the accompanying
standard STRACE log, not from Python elapsed time or a kernel-only interval.
"""

from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
import subprocess
import sys
import time
from pathlib import Path
from typing import Any

import _task_interface
import torch
from simpler.worker import Worker

from simpler_setup.environment import PROJECT_ROOT

_GRAPH_DIR = "tests/st/a5/host_build_graph/graph_execution"
_FIXTURES = {
    "graph_2d": (
        f"{_GRAPH_DIR}/test_graph_execution.py",
        "TestGraphExecutionHostBuildGraphA5",
        "record_then_replay_2d",
    ),
    "graph_cv": (
        f"{_GRAPH_DIR}/test_graph_execution_aic_aiv.py",
        "TestGraphExecutionAicAivHostBuildGraphA5",
        "record_then_replay_aic_aiv",
    ),
    "graph_mix": (
        f"{_GRAPH_DIR}/test_graph_execution_mix_spmd.py",
        "TestGraphExecutionMixSpmdHostBuildGraphA5",
        "record_then_replay_mix_spmd",
    ),
    "csa": (
        "examples/a5/host_build_graph/deepseek_v4_pro_csa/test_deepseek_v4_pro_csa.py",
        "TestDeepSeekV4ProCSAHBG",
        "B4S2Seed0",
    ),
    "pa": (
        "examples/a5/host_build_graph/paged_attention_unroll_manual_scope/test_paged_attention_unroll_manual_scope.py",
        "TestPagedAttentionUnrollManualScopeHostBuildGraph",
        "Case1",
    ),
    "ordinary_mix": (
        "tests/st/a5/host_build_graph/single_block_mix/test_single_block_mix.py",
        "TestSingleBlockMixHostBuildGraphA5",
        "ordinary_mix_interleaved",
    ),
}


def _git(repo, *args):
    return subprocess.check_output(["git", "-C", str(repo), *args], text=True).strip()


def _sha256(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _fixture(repo, workload):
    relative, class_name, case_name = _FIXTURES[workload]
    path = repo / relative
    sys.path.insert(0, str(path.parent))
    spec = importlib.util.spec_from_file_location("_graph_benchmark_fixture", path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Cannot load fixture: {path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    cls = getattr(module, class_name)
    case = next(case for case in cls.CASES if case["name"] == case_name)
    sources = [Path(cls.CALLABLE["orchestration"]["source"])]
    sources.extend(Path(kernel["source"]) for kernel in cls.CALLABLE["incores"])
    hashes = {str(source.relative_to(repo)): _sha256(source) for source in sources}
    return cls, case, hashes


class _MeasuredWorker(Worker):
    def __init__(self, expected_modes, **kwargs):
        super().__init__(**kwargs)
        self.expected_modes = expected_modes
        self.samples: list[dict[str, Any]] = []

    def run(self, callable, args=None, config=None):
        assert config is not None and config.enable_chip_swimlane == 0
        before = self.scheduler_mode_publish_count
        start = time.perf_counter_ns()
        super().run(callable, args, config)
        elapsed = time.perf_counter_ns() - start
        assert self.scheduler_mode_publish_count > before
        mode = self.scheduler_runtime_mode
        assert mode in self.expected_modes, f"unexpected backend {mode}, expected {self.expected_modes}"
        self.samples.append(
            {
                "round": len(self.samples),
                "python_call_wall_ns": elapsed,
                "scheduler_mode": mode,
                "publication": self.scheduler_mode_publish_count,
                "committed_device_bytes": self.committed_device_memory(),
            }
        )


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--workload", choices=_FIXTURES, required=True)
    parser.add_argument("--expected-backend", choices=("resident", "legacy-graph"), required=True)
    parser.add_argument("--platform", choices=("a5", "a5sim"), default="a5")
    parser.add_argument("--device", type=int, default=0)
    parser.add_argument("--rounds", type=int, default=105)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.rounds < 1 or args.output.exists():
        parser.error("rounds must be positive and output must not already exist")
    repo = Path.cwd().resolve()
    if repo != Path(PROJECT_ROOT).resolve():
        raise RuntimeError("Activate this checkout's editable installation before benchmarking")
    commit = _git(repo, "rev-parse", "HEAD")
    if _task_interface.__build_commit__ != commit:
        raise RuntimeError("Bindings are stale; rebuild this checkout before benchmarking")
    torch.manual_seed(0)
    cls, case, sources = _fixture(repo, args.workload)
    if case.get("skip_golden", cls.SKIP_GOLDEN):
        raise RuntimeError("Benchmark fixtures must preserve their original numerical validation")
    callable_obj = cls.compile_chip_callable(args.platform)
    expected_modes = (1, 2) if args.expected_backend == "resident" else (3,)
    variant = "sim" if args.platform.endswith("sim") else "onboard"
    runtime = repo / "build/lib/a5" / variant / "host_build_graph"
    runtime_hashes = {path.name: _sha256(path) for path in runtime.iterdir() if path.suffix in (".o", ".so")}
    worker = _MeasuredWorker(
        expected_modes,
        level=2,
        platform=args.platform,
        runtime="host_build_graph",
        device_id=args.device,
    )
    try:
        worker.init()
        initial_memory = worker.committed_device_memory()
        # The existing SceneTestCase owns inputs, reset policy, tolerance and
        # golden; every timed call still undergoes its original comparison.
        cls()._run_and_validate_l2(worker, callable_obj, case, rounds=args.rounds)  # noqa: SLF001
        retained_memory = worker.committed_device_memory()
        assert len(worker.samples) == args.rounds
    finally:
        worker.close()
    result = {
        "schema": "hbg-graph-benchmark-v1",
        "workload": args.workload,
        "case": case["name"],
        "parameters": case.get("params", {}),
        "platform": args.platform,
        "device": args.device,
        "expected_backend": args.expected_backend,
        "commit": commit,
        "runtime_diff_sha256": hashlib.sha256(_git(repo, "diff", "HEAD", "--", "src").encode()).hexdigest(),
        "pto_isa_pin": (repo / "pto_isa.pin").read_text().strip(),
        "torch": torch.__version__,
        "fixture_sources_sha256": sources,
        "runtime_sha256": runtime_hashes,
        "initial_committed_device_bytes": initial_memory,
        "retained_committed_device_bytes": retained_memory,
        "all_rounds_numerically_verified": True,
        "samples": worker.samples,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("x") as stream:
        json.dump(result, stream, indent=2)
    print(f"[GRAPH_BENCHMARK] {args.workload}: {args.rounds} verified calls; {args.output}")


if __name__ == "__main__":
    main()
