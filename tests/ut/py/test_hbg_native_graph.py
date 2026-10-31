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
from _native_graph_builder import graph_capture

from simpler_setup.tools.hbg.native_causality import audit_native_protocol
from simpler_setup.tools.hbg.native_protocol import (
    STAMPS,
    _event_records,
    audit_protocol_sequences,
    has_graph_protocol,
    read_native_protocol,
    uses_resource_scoped_mix,
)
from simpler_setup.tools.swimlane_converter import generate_chrome_trace_json, read_perf_data


@pytest.mark.parametrize("instances", (1, 2, 16))
@pytest.mark.parametrize("dummy,host_prefix", ((False, False), (False, True), (True, True)))
@pytest.mark.parametrize("schema", ("hbg-native-protocol-v8", "hbg-native-protocol-v9"))
def test_graph_manifest_publication_and_parent_completion(instances, dummy, host_prefix, schema):
    raw, identities = graph_capture(instances, dummy=dummy, host_prefix=host_prefix)
    raw["schema"] = schema
    assert has_graph_protocol(raw)
    assert uses_resource_scoped_mix(raw) == (schema == "hbg-native-protocol-v9")
    before = copy.deepcopy(raw)
    result = audit_native_protocol(raw, identities)
    assert result["graph_verified"] is True
    assert result["graph_instances"] == instances
    assert result["graph_children"] == 2 * instances
    assert result["graph_definitions"] == 1
    assert raw == before


def test_graph_proof_uses_local_sequences_and_counter_tickets_not_cross_core_clocks():
    raw, identities = graph_capture()
    for record, kind, _ in _event_records(raw):
        offset = (record["worker_index"] + 1) * 100000
        for field in STAMPS[kind].split():
            if record[field][1]:
                record[field][0] += offset
        if kind == "execution":
            record["publication_before_cycles"] += offset
            record["publication_after_cycles"] += offset
    assert audit_native_protocol(raw, identities)["graph_verified"] is True


def test_counter_ticket_permutation_cannot_reverse_two_publications_on_one_worker():
    raw, identities = graph_capture(instances=4)
    first, last = raw["graph"]["materializations"][0], raw["graph"]["materializations"][3]
    assert first["worker_index"] == last["worker_index"]
    first["remaining_before"], last["remaining_before"] = last["remaining_before"], first["remaining_before"]
    assert audit_protocol_sequences(raw)["sequence_integrity_verified"] is True
    with pytest.raises(ValueError, match="contradictory causal cycle"):
        audit_native_protocol(raw, identities)


def test_child_dependency_does_not_force_its_remaining_counter_contribution_to_come_first():
    raw, identities = graph_capture(instances=1)
    predecessor, successor = raw["graph"]["contributions"]
    predecessor["remaining_before"], successor["remaining_before"] = 1, 2
    assert audit_native_protocol(raw, identities)["graph_verified"] is True


@pytest.mark.parametrize("field", ("graph", "close_seal_stride"))
@pytest.mark.parametrize("schema", ("hbg-native-protocol-v8", "hbg-native-protocol-v9"))
def test_graph_schema_requires_graph_and_close_seal_contracts(field, schema):
    raw, _ = graph_capture()
    raw["schema"] = schema
    del raw[field]
    with pytest.raises(ValueError, match="missing or unknown fields"):
        read_native_protocol(raw)


def test_v7_must_not_accept_graph_evidence():
    raw, _ = graph_capture()
    raw["schema"] = "hbg-native-protocol-v7"
    with pytest.raises(ValueError, match="missing or unknown fields"):
        read_native_protocol(raw)


def test_failed_and_partial_graph_events_remain_readable_but_never_verify():
    raw, identities = graph_capture()
    materialize = raw["graph"]["materializations"][0]
    materialize["status"] = 6
    materialize["publish_after"] = [0, 0]
    raw["scheduler_error"] = 6
    raw["error_claimed"] = 1
    raw["first_error"].update(public_task_id=1, sub_task_id=0)
    assert read_native_protocol(raw) == raw
    with pytest.raises(ValueError, match="scheduler reported an error"):
        audit_native_protocol(raw, identities)


@pytest.mark.parametrize(
    "path,replacement",
    [
        pytest.param(["public_task_count"], 5, id="public_count"),
        pytest.param(["remaining_instances"], 1, id="remaining_instances"),
        pytest.param(["public_completed"], 0, id="public_completed"),
        pytest.param(["definitions"], [], id="missing_definition"),
        pytest.param(["definitions"], lambda rows: [*rows, {**rows[0], "index": 1}], id="duplicate_definition"),
        pytest.param(["definitions", 0, "full_key"], 18, id="definition_key"),
        pytest.param(["ranges", 2, "begin"], 4, id="range_overlap"),
        pytest.param(["ranges", 1, "children"], 3, id="range_children"),
        pytest.param(["nodes", 2, "public_task_id"], 2, id="node_parent"),
        pytest.param(["nodes", 2, "sub_task_id"], 1, id="node_local"),
        pytest.param(["nodes", 2, "task_id"], 0, id="node_raw_id"),
        pytest.param(["nodes", 2, "kind"], 0, id="node_kind"),
        pytest.param(["nodes", 2, "storage_address"], lambda value: value + 320, id="node_storage"),
        pytest.param(["nodes", 2, "dependencies"], [], id="node_dependency"),
        pytest.param(["definitions", 0, "fanin_indices", 0], 1, id="csr_source"),
        pytest.param(["definitions", 0, "tasks", 0, "kernel_ids", 1], 7, id="source_kernel"),
        pytest.param(["definitions", 0, "tasks", 0, "active_mask"], 4, id="source_mask"),
        pytest.param(["definitions", 0, "tasks", 0, "task_attrs"], 2, id="source_attrs"),
        pytest.param(["instances", 0, "parent_id"], 2, id="instance_parent"),
        pytest.param(["instances", 0, "run_epoch"], 8, id="instance_epoch"),
        pytest.param(["instances", 0, "materialize_owner"], 0, id="instance_owner"),
        pytest.param(["instances", 0, "materialize_state"], 1, id="instance_state"),
        pytest.param(["instances", 0, "remaining_children"], 1, id="instance_remaining"),
        pytest.param(["instances", 0, "task_storage_bytes"], 639, id="instance_storage"),
        pytest.param(["instances", 1, "definition_address"], lambda value: value + 512, id="definition_copy"),
        pytest.param(["materializations"], lambda rows: rows[:-1], id="missing_materialization"),
        pytest.param(["materializations", 0, "remaining_before"], 1, id="publish_ticket"),
        pytest.param(["materializations", 0, "worker_index"], 1, id="materialize_owner"),
        pytest.param(["contributions"], lambda rows: rows[:-1], id="missing_contribution"),
        pytest.param(["contributions", 0, "remaining_before"], 1, id="contribution_ticket"),
        pytest.param(["contributions", 0, "instance_index"], 1, id="contribution_instance"),
        pytest.param(["contributions", 0, "worker_index"], 2, id="contribution_owner"),
        pytest.param(["contributions", 0, "task_ref"], 3, id="contribution_task"),
        pytest.param(["bootstrap"], lambda rows: rows[:-1], id="missing_waiter"),
        pytest.param(["bootstrap", 0, "remaining_observed"], 1, id="wait_remaining"),
        pytest.param(["bootstrap", 0, "poll_count"], 0, id="wait_poll_count"),
        pytest.param(["bootstrap", 0, "status"], 10, id="wait_status"),
        pytest.param(["bootstrap", 0, "instance_count"], 1, id="wait_instance_count"),
    ],
)
@pytest.mark.parametrize("schema", ("hbg-native-protocol-v8", "hbg-native-protocol-v9"))
def test_graph_audit_rejects_wrong_identity_topology_and_completion(path, replacement, schema):
    raw, identities = graph_capture()
    raw["schema"] = schema
    assert audit_native_protocol(raw, identities)["graph_verified"] is True
    target = raw["graph"]
    for field in path[:-1]:
        target = target[field]
    field = path[-1]
    target[field] = replacement(target[field]) if callable(replacement) else replacement
    with pytest.raises(ValueError, match="native protocol"):
        audit_native_protocol(raw, identities)


def test_execution_identity_cannot_replace_the_public_child_id_with_its_dense_ref():
    raw, identities = graph_capture()
    identities[1]["task_id"] = 2
    with pytest.raises(ValueError, match="identity join mismatch"):
        audit_native_protocol(raw, identities)


def test_graph_event_loss_is_not_hidden_by_an_otherwise_valid_kernel_stream():
    raw, _ = graph_capture()
    del raw["graph"]["contributions"][0]
    with pytest.raises(ValueError, match="lost or unexpected events"):
        audit_protocol_sequences(raw)


@pytest.mark.parametrize("dummy", (False, True))
@pytest.mark.parametrize("schema", ("hbg-native-protocol-v8", "hbg-native-protocol-v9"))
def test_native_graph_converter_shows_lifecycle_scalar_records_and_parent_child_flows(tmp_path, dummy, schema):
    protocol, identities = graph_capture(dummy=dummy)
    protocol["schema"] = schema
    capture = Capture(
        runtime="host_build_graph",
        level=4,
        clock_freq_hz=1000000000,
        core_types=("aic", "aiv", "aiv"),
        scheduler_producer="aicore",
    )
    identity_by_index = {row["reg_task_id"]: row for row in identities}
    for row in protocol["executions"]:
        capture.task(
            task_id=identity_by_index[row["index"]]["task_id"],
            core_id=row["worker_index"],
            reg_task_id=row["index"],
            run_epoch=protocol["run_epoch"],
            start=row["entry"][0],
            end=row["visible"][0],
        )
    raw = capture.build()
    raw["aicore_protocol"] = protocol
    raw["aicore_execution_identities"] = dict(schema="hbg-execution-identity-v1", producer="aicore", records=identities)
    path = tmp_path / "capture.json"
    path.write_text(json.dumps(raw))
    parsed = read_perf_data(path)
    assert parsed["protocol_audit"]["graph_verified"] is True
    assert all(task["func_id"] == 0 for task in parsed["tasks"])
    assert all(task["task_id_hex"] == f"0x{task['task_id']:016x}" for task in parsed["tasks"])
    trace = generate_chrome_trace_json(
        parsed["tasks"],
        None,
        runtime_name="host_build_graph",
        timeline_metadata=parsed["timeline_metadata"],
        native_graph=parsed["native_graph"],
    )
    events = trace["traceEvents"]
    envelopes = [event for event in events if event.get("cat") == "graph_execution" and event["ph"] == "X"]
    assert len(envelopes) == 2
    assert all(event["args"]["total_sub_task_count"] == 2 for event in envelopes)
    assert all(event["args"]["graph_verified"] is True for event in envelopes)
    assert {event["args"]["public_task_id"] for event in envelopes} == {1, 2}
    waits = [
        event for event in events if event.get("cat") == "graph_scalar" and event["name"] == "Graph.materialize_wait"
    ]
    assert len(waits) == 3 and all(event["args"]["poll_count"] == 1 for event in waits)
    controls = [event for event in events if event.get("cat") == "graph_scalar"]
    assert sum(event["name"] == "Graph.ENTER" for event in controls) == 2
    assert sum(event["name"] == "Graph.EXIT" for event in controls) == 2
    assert sum(event["name"] == "Graph.child_contribution" for event in controls) == 4
    assert sum(event["name"] == "Graph.Dummy" for event in controls) == (4 if dummy else 0)
    assert all(event["dur"] >= 0 and event["ts"] >= 0 for event in events if event.get("ph") == "X")
    flows = [event for event in events if event.get("cat") == "graph_causality"]
    assert len(flows) == 8
    assert len({event["id"] for event in flows}) == 4
    assert all(event["args"]["run_epoch"] == protocol["run_epoch"] for event in flows)
