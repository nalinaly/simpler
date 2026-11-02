# Copyright (c) PyPTO Contributors.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------

"""Join device-captured HBG execution identities; never infer them from row order."""

FIELDS = frozenset(
    (
        "run_epoch",
        "core_id",
        "reg_task_id",
        "task_id",
        "block_idx",
        "block_num",
        "subtask_slot",
        "pending_slot",
        "dispatch_generation",
        "order_key",
        "group_seq",
        "cluster_index",
        "cluster_lane",
        "executed",
    )
)
JOIN = ("run_epoch", "core_id", "reg_task_id")
POOL_FIELDS = FIELDS | {"dispatch_kind", "task_flags", "active_mask"}


def _validate_record(record, *, version=1):
    fields = POOL_FIELDS if version == 2 else FIELDS
    if not isinstance(record, dict) or set(record) != fields:
        raise ValueError("execution identity has missing or unknown fields")
    for field in fields - {"order_key", "group_seq", "executed", "dispatch_kind"}:
        if type(record[field]) is not int or not 0 <= record[field] < 1 << 64:
            raise ValueError(f"invalid execution identity {field}")
    for field in ("order_key", "group_seq"):
        value = record[field]
        minimum = 1 if field == "group_seq" else 0
        if value is not None and (type(value) is not int or not minimum <= value < 1 << 64):
            raise ValueError(f"invalid execution identity {field}")
    if (
        type(record["executed"]) is not bool
        or not 0 <= record["block_idx"] < record["block_num"] <= 32767
        or record["subtask_slot"] > 2
        or record["pending_slot"] > 1
        or record["cluster_lane"] > 2
        or not 0 < record["dispatch_generation"] < 1 << 32
        or (version == 1 and record["block_num"] > 1 and record["order_key"] is None)
    ):
        raise ValueError("invalid execution identity shape or generation")
    if version == 2:
        _validate_pool_identity_shape(record)


def _validate_pool_identity_shape(record):
    flags, mask, blocks = record["task_flags"], record["active_mask"], record["block_num"]
    if (
        not 0 < mask < 8
        or flags >= 128
        or not flags & 1
        or flags & 32
        or bool(flags & 4) != (mask.bit_count() > 1)
        or bool(flags & 8) != (blocks > 1)
        or not mask & (1 << record["subtask_slot"])
        or (flags & 16 and blocks == 1)
        or (flags & 64 and (blocks != 1 or mask.bit_count() != 1))
        or (not record["executed"] and not flags & 64)
    ):
        raise ValueError("execution identity task shape/flags mismatch")
    pooled = blocks > 1 and mask.bit_count() == 1 and not flags & (4 | 16 | 32 | 64)
    ordered = not pooled and (blocks > 1 or bool(flags & (4 | 16)))
    expected = "pool" if pooled else "ordered" if ordered else "single"
    if (
        record["dispatch_kind"] != expected
        or (record["order_key"] is not None) != ordered
        or (record["group_seq"] is not None) != bool(flags & 4)
        or (pooled and record["run_epoch"] == 0)
    ):
        raise ValueError("execution identity dispatch kind does not match its task contract")


def attach_execution_identities(section, tasks, runtime):
    if (
        runtime != "host_build_graph"
        or not isinstance(section, dict)
        or set(section) != {"schema", "producer", "records"}
        or section["schema"] not in ("hbg-execution-identity-v1", "hbg-execution-identity-v2")
        or section["producer"] != "aicore"
        or not isinstance(section["records"], list)
    ):
        raise ValueError("invalid execution identity envelope")
    version = 2 if section["schema"] == "hbg-execution-identity-v2" else 1
    fields = POOL_FIELDS if version == 2 else FIELDS
    rows = {tuple(task[field] for field in JOIN): task for task in tasks}
    if len(rows) != len(tasks):
        raise ValueError("duplicate execution identity row join")
    observed, executions, generations = set(), set(), set()
    positions, orders, groups, block_counts = {}, {}, {}, {}
    shapes, coverage = {}, {}
    for record in section["records"]:
        _validate_record(record, version=version)
        key = tuple(record[field] for field in JOIN)
        task = rows.get(key)
        if key in observed or task is None or task["task_id"] != record["task_id"]:
            raise ValueError("missing, duplicate or mismatched execution identity join")
        is_cube = task["core_type"] == "aic"
        if (record["subtask_slot"] == 0) != is_cube or (record["cluster_lane"] == 0) != is_cube:
            raise ValueError("execution identity lane/type mismatch")
        if record["group_seq"] is not None and (
            record["order_key"] is None or record["subtask_slot"] != record["cluster_lane"]
        ):
            raise ValueError("execution identity Mix placement mismatch")
        execution = (record["run_epoch"], record["task_id"], record["block_idx"], record["subtask_slot"])
        generation = (record["run_epoch"], record["core_id"], record["pending_slot"], record["dispatch_generation"])
        if execution in executions or generation in generations:
            raise ValueError("duplicate execution identity or slot generation")
        position_key = (record["run_epoch"], record["core_id"])
        position = (record["cluster_index"], record["cluster_lane"])
        if positions.setdefault(position_key, position) != position:
            raise ValueError("execution identity worker changed physical position")
        order_key = (record["run_epoch"], record["task_id"])
        if version == 2:
            _attach_pool_shape(record, order_key, shapes, coverage)
        if orders.setdefault(order_key, record["order_key"]) != record["order_key"]:
            raise ValueError("execution identity task has inconsistent admission order")
        group_key = execution[:3]
        group = (record["group_seq"], record["cluster_index"] if record["group_seq"] is not None else None)
        if groups.setdefault(group_key, group) != group:
            raise ValueError("execution identity Mix block spans different groups")
        block_counts[group_key] = block_counts.get(group_key, 0) + 1
        observed.add(key)
        executions.add(execution)
        generations.add(generation)
        task.update({field: record[field] for field in fields - set(JOIN) - {"task_id"}})
    if observed != rows.keys():
        raise ValueError("execution identity coverage is incomplete")
    if any(count > 1 and groups[key][0] is None for key, count in block_counts.items()):
        raise ValueError("execution identity multi-lane block has no group")
    for key, (_, mask, blocks, _) in shapes.items():
        expected = {(block, slot) for block in range(blocks) for slot in range(3) if mask & (1 << slot)}
        if coverage[key] != expected:
            raise ValueError("execution identity block/lane coverage is incomplete")


def _attach_pool_shape(record, key, shapes, coverage):
    shape = (record["task_flags"], record["active_mask"], record["block_num"], record["dispatch_kind"])
    if shapes.setdefault(key, shape) != shape:
        raise ValueError("execution identity task has inconsistent shape or dispatch kind")
    coverage.setdefault(key, set()).add((record["block_idx"], record["subtask_slot"]))
