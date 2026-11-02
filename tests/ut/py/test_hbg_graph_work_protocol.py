# Copyright (c) PyPTO Contributors.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------

import copy
import json

import pytest
from _capture_builder import Capture
from _native_graph_work_builder import parallel_graph_capture, resequence

from simpler_setup.tools.hbg.native_causality import audit_native_protocol
from simpler_setup.tools.hbg.native_protocol import (
    STAMPS,
    _event_records,
    audit_protocol_sequences,
    read_native_protocol,
)
from simpler_setup.tools.swimlane_converter import read_perf_data


@pytest.mark.parametrize("instances", (1, 2, 16))
@pytest.mark.parametrize("mixed,scalar_gap", ((False, False), (False, True), (True, True)))
def test_parallel_ranges_owners_and_unique_publication(instances, mixed, scalar_gap):
    raw, identities = parallel_graph_capture(instances, mixed=mixed, scalar_gap=scalar_gap)
    before = copy.deepcopy(raw)
    assert audit_native_protocol(raw, identities)["graph_verified"]
    assert raw == before


def test_parallel_proof_does_not_assume_synchronized_worker_clocks():
    raw, identities = parallel_graph_capture()
    for record, kind, _ in _event_records(raw):
        offset = (record["worker_index"] + 1) * 10_000_000
        for field in STAMPS[kind].split():
            if record[field][1]:
                record[field][0] += offset
        if kind == "execution":
            record["publication_before_cycles"] += offset
            record["publication_after_cycles"] += offset
    assert audit_native_protocol(raw, identities)["graph_verified"]


@pytest.mark.parametrize("schema", ("hbg-native-protocol-v9", "hbg-native-protocol-v10", "hbg-native-protocol-v11"))
def test_parallel_evidence_cannot_be_relabelled_as_an_older_schema(schema):
    raw, _ = parallel_graph_capture()
    raw["schema"] = schema
    with pytest.raises(ValueError, match="missing or unknown fields"):
        read_native_protocol(raw)


@pytest.mark.parametrize(
    "path,value",
    [
        (["work_count"], 1),
        (["next_work"], 0),
        (["next_instance"], 5),
        (["instances", 0, "work_begin"], 1),
        (["instances", 0, "work_count"], 3),
        (["instances", 0, "remaining_work"], 1),
        (["instances", 0, "scalar_arg_count"], 1),
        (["instances", 0, "control_address"], 0x400000),
        (["instances", 0, "tensor_pool_address"], 0x400000),
        (["instances", 0, "scalar_pool_address"], 0x110000),
        (["instances", 1, "scalar_pool_address"], 0x400000 + 896),
        (["instances", 0, "scalar_pool_address"], (1 << 64) - 8),
        (["work_plan", 0, "instance_index"], 1),
        (["work_plan", 0, "owner_state"], 5),
        (["work_plan", 0, "write_bytes"], 1),
        (["work_plan", 0, "first_range"], 1),
        (["work_plan", 0, "region"], 3),
        (["work_ranges", 0, "count"], 0),
        (["work", 0, "remaining_before"], 1),
        (["work", 0, "run_epoch"], 8),
        (["work", 0, "status"], 6),
        (["work", 0, "work_index"], 1),
        (["work", 0, "instance_index"], 1),
        (["work", 0, "writeback_after"], [0, 0]),
        (["publications"], []),
        (["work"], []),
    ],
)
def test_parallel_audit_rejects_missing_aliased_stale_and_early_evidence(path, value):
    raw, identities = parallel_graph_capture()
    assert audit_native_protocol(raw, identities)["graph_verified"]
    target = raw["graph"]
    for field in path[:-1]:
        target = target[field]
    target[path[-1]] = value
    with pytest.raises(ValueError, match="native protocol"):
        audit_native_protocol(raw, identities)


def test_disjoint_scalars_on_one_cache_line_cannot_have_different_writers():
    raw, identities = parallel_graph_capture(scalar_gap=True, split_scalars=True)
    assert audit_protocol_sequences(raw)["sequence_integrity_verified"]
    with pytest.raises(ValueError, match="share a cache line"):
        audit_native_protocol(raw, identities)


def test_unreferenced_parameter_holes_must_not_be_written():
    raw, identities = parallel_graph_capture(scalar_gap=True)
    raw["graph"]["work_ranges"][3]["count"] = 2
    for plan in raw["graph"]["work_plan"]:
        if plan["region"] == 2:
            plan["write_bytes"] = 24
    with pytest.raises(ValueError, match="exactly the referenced elements"):
        audit_native_protocol(raw, identities)


def test_final_contribution_tickets_cannot_reverse_a_workers_program_order():
    raw, identities = parallel_graph_capture()
    first, final = raw["graph"]["work"][0], raw["graph"]["work"][3]
    assert first["worker_index"] == final["worker_index"]
    first["remaining_before"], final["remaining_before"] = 1, 4
    assert audit_protocol_sequences(raw)["sequence_integrity_verified"]
    with pytest.raises(ValueError, match="contradictory causal cycle"):
        audit_native_protocol(raw, identities)


def test_a_different_worker_cannot_replace_the_final_contributor_as_publisher():
    raw, identities = parallel_graph_capture()
    publication = raw["graph"]["publications"][0]
    publication["worker_index"] = (publication["worker_index"] + 1) % 3
    raw["graph"]["instances"][0]["materialize_owner"] = publication["worker_index"] + 1
    resequence(raw)
    assert audit_protocol_sequences(raw)["sequence_integrity_verified"]
    with pytest.raises(ValueError, match="final work contributor"):
        audit_native_protocol(raw, identities)


def test_failed_and_incomplete_work_remains_readable_but_never_verifies():
    raw, identities = parallel_graph_capture()
    raw["scheduler_error"] = 6
    raw["error_claimed"] = 1
    raw["graph"]["work"][0].update(status=6, writeback_after=[0, 0], remaining_before=0)
    raw["graph"]["definitions"][0]["tasks"][0]["scalar_offset"] = -1
    raw["graph"]["publications"] = []
    assert read_native_protocol(raw) == raw
    with pytest.raises(ValueError, match="scheduler reported an error"):
        audit_native_protocol(raw, identities)


def test_parallel_converter_displays_each_real_writer_and_a_separate_publisher(tmp_path):
    protocol, identities = parallel_graph_capture(scalar_gap=True)
    capture = Capture(
        runtime="host_build_graph",
        level=4,
        clock_freq_hz=1_000_000_000,
        core_types=("aic", "aiv", "aiv"),
        scheduler_producer="aicore",
    )
    by_index = {row["reg_task_id"]: row for row in identities}
    for row in protocol["executions"]:
        capture.task(
            task_id=by_index[row["index"]]["task_id"],
            core_id=row["worker_index"],
            reg_task_id=row["index"],
            run_epoch=protocol["run_epoch"],
            start=row["entry"][0],
            end=row["visible"][0],
        )
    raw = capture.build()
    raw["aicore_protocol"] = protocol
    raw["aicore_execution_identities"] = dict(schema="hbg-execution-identity-v1", producer="aicore", records=identities)
    path = tmp_path / "parallel.json"
    path.write_text(json.dumps(raw))
    parsed = read_perf_data(path)
    assert parsed["protocol_audit"]["graph_verified"]
    graph = parsed["native_graph"]
    work = [row for row in graph["operations"] if row["name"] == "Graph.materialize.work"]
    assert len(work) == protocol["graph"]["work_count"]
    assert {row["worker_index"] for row in work} == {0, 1, 2}
    assert not any(row["name"] == "Graph.materialize" for row in graph["operations"])
    assert len([row for row in graph["operations"] if row["name"] == "Graph.publish"]) == 2
