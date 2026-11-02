# Copyright (c) PyPTO Contributors.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------

"""Synthetic v10 contracts complement tests of the actual C++ pool recorder."""

import copy
import json

import pytest
from _capture_builder import Capture
from _native_protocol_builder import ProtocolBuilder
from test_hbg_native_causality import _add_claim_seal_certificates, _isolated_close_seals

from simpler_setup.tools.hbg.native_causality import audit_native_protocol
from simpler_setup.tools.hbg.native_protocol import STAMPS, _event_records, read_native_protocol
from simpler_setup.tools.swimlane_converter import generate_chrome_trace_json, read_perf_data

MAX = (1 << 64) - 1


def pool_capture(shapes=((1, 3, ()),), *, clusters=2, sync_tasks=()):
    raw, identities = ProtocolBuilder(
        shapes, clusters=clusters, epoch=17, sync_tasks=sync_tasks, resource_scoped=True
    ).run()
    _add_claim_seal_certificates(raw, identities)
    _isolated_close_seals(raw)
    tasks = {task["task_ref"]: task for task in raw["tasks"]}
    pooled = {
        ref
        for ref, task in tasks.items()
        if task["block_num"] > 1 and task["active_mask"] in (1, 2, 4) and not task["flags"] & 16
    }
    old_orders = {ref: order for order, ref in enumerate(raw["admission_log"])}
    log = [ref for ref in raw["admission_log"] if ref not in pooled]
    orders = {ref: order for order, ref in enumerate(log)}

    def prefix(old_tail):
        return sum(old_orders[ref] < old_tail for ref in log)

    raw.update(
        schema="hbg-native-protocol-v10",
        admission_log=log,
        commit_tail=len(log),
        cooperative_task_count=len(log),
        member_count=len(log) * raw["worker_count"],
    )
    publications = []
    for row in raw["task_events"]:
        ref = row["index"]
        if ref in pooled:
            actor = row["admission"]
            actor.update(order_key=MAX, worker_index=0 if tasks[ref]["active_mask"] == 1 else 1)
            publications.append(dict(index=ref, publication=actor))
            row["admission"] = None
        for name in ("admission", "completion"):
            if row[name] is not None:
                row[name]["order_key"] = orders.get(ref, MAX)
    raw["ordinary_pool"] = dict(
        task_count=len(pooled),
        word_count=(raw["task_count"] + 63) // 64,
        words=[0] * (2 * ((raw["task_count"] + 63) // 64)),
        publications=publications,
    )
    _pool_membership(raw, pooled, orders, prefix)
    for block in raw["blocks"]:
        owner = block["owner"]
        ref = owner["task_ref"]
        owner["order_key"] = orders.get(ref, MAX)
        for lane in block["lanes"]:
            lane["order_key"] = orders[ref]
        if ref in pooled:
            claim = block.pop("claim")
            claim["retire_before"] = claim.pop("published_before")
            claim["retire_after"] = claim.pop("published_after")
            claim["retired_word_before"] = (1 << (ref % 64)) if claim.pop("seal_value") else 0
            block["pool_claim"] = claim
    for execution in raw["executions"]:
        ref = execution["task_ref"]
        execution["order_key"] = orders.get(ref, MAX)
        if ref in pooled:
            execution.update(
                flags=(execution["flags"] & ~4) | 128 | (16 if orders else 32),
                ready_source=4,
                permit_tail=prefix(old_orders[ref]),
            )
        elif ref not in orders:
            execution["permit_tail"] = prefix(execution["permit_tail"])
            execution["flags"] = (execution["flags"] & ~(16 | 32)) | (16 if orders else 32)
    for identity in identities:
        ref = identity["task_id"]
        identity.update(
            order_key=orders.get(ref),
            dispatch_kind="pool" if ref in pooled else "ordered" if ref in orders else "single",
            task_flags=tasks[ref]["flags"],
            active_mask=tasks[ref]["active_mask"],
        )
    streams = {worker["index"]: [] for worker in raw["workers"]}
    for record, kind, _ in _event_records(raw):
        streams[record["worker_index"]].extend(record[name] for name in STAMPS[kind].split() if record[name][1])
    for worker in raw["workers"]:
        stamps = sorted(streams[worker["index"]], key=lambda stamp: (stamp[0], stamp[1]))
        for sequence, stamp in enumerate(stamps, 1):
            stamp[1] = sequence
        worker["next_sequence"] = len(stamps)
    return raw, identities


def _pool_membership(raw, pooled, orders, prefix):
    for record in raw["admissions"]:
        ref = record["index"]
        if ref in pooled:
            record.update(
                run_epoch=0,
                task_ref=-1,
                order_key=MAX,
                block_num=0,
                member_count=0,
                active_mask=0,
                flags=0,
                member_words=[0] * len(record["member_words"]),
            )
        elif ref in orders:
            record["order_key"] = orders[ref]
    if not orders:
        raw["admissions"] = []
    raw["members"] = [member for member in raw["members"] if member["task_ref"] not in pooled]
    for member in raw["members"]:
        member.update(order_key=orders[member["task_ref"]], observed_tail=prefix(member["observed_tail"]))
        member["index"] = member["order_key"] * raw["worker_count"] + member["worker_index"]
    for control in raw["task_controls"]:
        if control["task_ref"] in pooled:
            control.update(admission_state=0, admission_claim_lock=2, claim_seal=0)


def audit(raw, identities):
    return audit_native_protocol(read_native_protocol(raw), identities)


def word16_capture(shapes=((1, 3, ()),), **kwargs):
    raw, identities = pool_capture(shapes, **kwargs)
    raw["schema"] = "hbg-native-protocol-v11"
    words = (raw["task_count"] + 15) // 16
    raw["ordinary_pool"].update(word_task_capacity=16, word_count=words, words=[0] * (2 * words))
    for block in raw["blocks"]:
        claim = block.get("pool_claim")
        if claim is not None and claim["retired_word_before"]:
            claim["retired_word_before"] = 1 << (block["owner"]["task_ref"] % 16)
    return raw, identities


@pytest.mark.parametrize("mask", (1, 2, 4))
def test_word16_protocol_accepts_real_word_boundaries(mask):
    raw, identities = word16_capture([(mask, 2, ())] + [(None, 0, ())] * 15 + [(mask, 3, (0,))])
    result = audit(raw, identities)
    assert result["protocol_verified"] and result["executions"] == 5
    assert result["ordinary_pool_tasks"] == 2 and raw["ordinary_pool"]["word_count"] == 2


@pytest.mark.parametrize("capacity", (0, 1, 8, 32, 64, True, 16.0, "16", None, 1 << 64))
def test_word16_rejects_wrong_or_untyped_mapping(capacity):
    raw, identities = word16_capture()
    raw["ordinary_pool"]["word_task_capacity"] = capacity
    with pytest.raises(ValueError, match="word_task_capacity"):
        audit(raw, identities)


@pytest.mark.parametrize("mutation", ("missing_mapping", "wrong_count", "short_words", "old_bit", "foreign_word_bit"))
def test_word16_mapping_and_returned_words_are_not_guessed(mutation):
    raw, identities = word16_capture([(1, 2, ())] + [(None, 0, ())] * 16 + [(1, 2, ())])
    pool = raw["ordinary_pool"]
    if mutation == "missing_mapping":
        del pool["word_task_capacity"]
    elif mutation == "wrong_count":
        pool["word_count"] = 1
    elif mutation == "short_words":
        pool["words"].pop()
    elif mutation == "old_bit":
        raw["blocks"][-1]["pool_claim"]["retired_word_before"] = 1 << 17
    else:
        raw["blocks"][1]["pool_claim"]["retired_word_before"] |= 1 << 1
    with pytest.raises(ValueError):
        audit(raw, identities)


def test_word16_cannot_use_the_bit_of_a_task_in_a_different_word():
    raw, identities = word16_capture([(1, 2, ())] + [(None, 0, ())] * 16 + [(1, 2, ())])
    raw["blocks"][1]["pool_claim"]["retired_word_before"] = 3
    with pytest.raises(ValueError, match="ineligible bitmap bit"):
        audit(raw, identities)


def test_word16_cannot_be_relabelled_as_a_v10_capture():
    raw, identities = word16_capture([(None, 0, ())] * 16 + [(1, 2, ())])
    raw["schema"] = "hbg-native-protocol-v10"
    del raw["ordinary_pool"]["word_task_capacity"]
    raw["ordinary_pool"].update(word_count=1, words=[0, 0])
    with pytest.raises(ValueError, match="published bit"):
        audit(raw, identities)


@pytest.mark.parametrize("mask", (1, 2, 4))
@pytest.mark.parametrize("blocks", (2, 3, 7))
@pytest.mark.parametrize("clusters", (1, 2, 23, 32))
def test_pool_proves_every_block_without_admission_members(mask, blocks, clusters):
    raw, identities = pool_capture([(mask, blocks, ())], clusters=clusters)
    result = audit(raw, identities)
    assert result["protocol_verified"] and result["executions"] == blocks
    assert result["ordinary_pool_tasks"] == 1 and result["cooperative_tasks"] == result["member_obligations"] == 0


@pytest.mark.parametrize("mixed", (3, 5, 6, 7))
def test_pool_and_sync_start_keep_resource_scoped_mix_and_dependencies(mixed):
    raw, identities = pool_capture(
        [(1, 3, ()), (1, 2, (0,)), (mixed, 3, (1,)), (4, 5, (2,)), (2, 1, (3,))], sync_tasks=(1,)
    )
    result = audit(raw, identities)
    assert result["protocol_verified"]
    assert result["ordinary_pool_tasks"] == 2 and result["cooperative_tasks"] == 2
    assert result["executions"] == 11 + 3 * mixed.bit_count()


def test_pool_bitmap_word_after_first_sixty_four_tasks_is_preserved():
    raw, identities = pool_capture([(None, 0, ())] * 64 + [(1, 2, ())])
    assert raw["ordinary_pool"]["word_count"] == 2
    assert audit(raw, identities)["protocol_verified"]


def test_pool_and_single_block_dependency_work_without_an_admission_log():
    raw, identities = pool_capture([(1, 3, ()), (2, 1, (0,))])
    assert audit(raw, identities)["protocol_verified"]


@pytest.mark.parametrize("version", (10, 11))
def test_full_converter_preserves_versioned_pool_audit_and_dispatch_kind(tmp_path, version):
    factory = pool_capture if version == 10 else word16_capture
    protocol, identities = factory([(1, 3, ()), (3, 2, (0,)), (4, 3, (1,))])
    capture = Capture(
        runtime="host_build_graph",
        level=4,
        clock_freq_hz=1000000000,
        core_types=("aic", "aiv", "aiv") * 2,
        scheduler_producer="aicore",
    )
    for row in protocol["executions"]:
        capture.task(
            task_id=row["task_ref"],
            core_id=row["worker_index"],
            reg_task_id=row["index"],
            run_epoch=protocol["run_epoch"],
            start=row["entry"][0],
            end=row["visible"][0],
        )
    raw = capture.build()
    raw["aicore_protocol"] = protocol
    raw["aicore_execution_identities"] = dict(schema="hbg-execution-identity-v2", producer="aicore", records=identities)
    path = tmp_path / "pool.json"
    path.write_text(json.dumps(raw))
    parsed = read_perf_data(path)
    assert parsed["protocol_audit"]["protocol_verified"]
    assert parsed["protocol_audit"]["ordinary_pool_tasks"] == 2
    trace = generate_chrome_trace_json(
        parsed["tasks"], None, runtime_name="host_build_graph", timeline_metadata=parsed["timeline_metadata"]
    )
    assert trace["metadata"]["hbg_protocol_audit"] == parsed["protocol_audit"]
    assert any(event.get("args", {}).get("dispatch_kind") == "pool" for event in trace["traceEvents"])


@pytest.mark.parametrize(
    "mutation",
    (
        "missing_publication",
        "duplicate_publication",
        "fake_order",
        "wrong_publisher",
        "remaining_bit",
        "wrong_count",
        "missing_claim",
        "duplicate_confirmation",
        "stale_epoch",
        "wrong_claim_owner",
        "early_retirement",
        "missing_retirement",
        "wrong_retired_word",
        "ineligible_retired_bit",
        "duplicate_completion",
        "lost_completion",
        "fake_admission_lock",
        "fake_admission_state",
        "fake_claim_seal",
        "excess_tickets",
        "fake_entry_kind",
        "fake_ready_source",
        "unknown_owner",
        "false_identity_order",
        "false_identity_kind",
        "missing_execution",
        "missing_identity",
    ),
)
@pytest.mark.parametrize("version", (10, 11))
def test_pool_mutations_cannot_pass_protocol_verification(mutation, version):
    raw, identities = pool_capture() if version == 10 else word16_capture()
    publication = raw["ordinary_pool"]["publications"][0]["publication"]
    first, last = raw["blocks"][0], raw["blocks"][-1]
    control = raw["task_controls"][0]
    changes = {
        "fake_order": (publication, "order_key", 0),
        "wrong_publisher": (publication, "worker_index", 1),
        "remaining_bit": (raw["ordinary_pool"]["words"], 0, 1),
        "wrong_count": (raw["ordinary_pool"], "task_count", 0),
        "duplicate_confirmation": (first["pool_claim"], "confirmed_count", 2),
        "stale_epoch": (first["pool_claim"], "run_epoch", raw["run_epoch"] + 1),
        "wrong_claim_owner": (first["pool_claim"], "worker_index", 3),
        "early_retirement": (first["pool_claim"], "retired_word_before", 1),
        "missing_retirement": (last["pool_claim"], "retire_after", [0, 0]),
        "wrong_retired_word": (last["pool_claim"], "retired_word_before", 0),
        "ineligible_retired_bit": (last["pool_claim"], "retired_word_before", 3),
        "duplicate_completion": (first["owner"], "completion_ticket", 1),
        "lost_completion": (control, "completed_blocks", control["completed_blocks"] - 1),
        "fake_admission_lock": (control, "admission_claim_lock", 0),
        "fake_admission_state": (control, "admission_state", 3),
        "fake_claim_seal": (control, "claim_seal", 3),
        "excess_tickets": (control, "claim_tickets", 6),
        "fake_entry_kind": (raw["executions"][0], "flags", raw["executions"][0]["flags"] & ~128),
        "fake_ready_source": (raw["executions"][0], "ready_source", 3),
        "unknown_owner": (first["owner"], "worker_index", 99),
        "false_identity_order": (identities[0], "order_key", 0),
        "false_identity_kind": (identities[0], "dispatch_kind", "ordered"),
    }
    if mutation in changes:
        target, key, value = changes[mutation]
        target[key] = value
    elif mutation == "missing_publication":
        raw["ordinary_pool"]["publications"].clear()
    elif mutation == "duplicate_publication":
        raw["ordinary_pool"]["publications"].append(copy.deepcopy(raw["ordinary_pool"]["publications"][0]))
    elif mutation == "missing_claim":
        del first["pool_claim"]
    elif mutation == "missing_execution":
        raw["executions"].pop()
    elif mutation == "missing_identity":
        identities.pop()
    with pytest.raises(ValueError):
        audit(raw, identities)


@pytest.mark.parametrize("version", (6, 7, 8, 9))
def test_pool_cannot_be_disguised_as_a_legacy_protocol(version):
    raw, identities = pool_capture()
    raw["schema"] = f"hbg-native-protocol-v{version}"
    with pytest.raises(ValueError):
        audit(raw, identities)
