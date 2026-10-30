# Copyright (c) PyPTO Contributors.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------

"""Synthetic Graph protocol captures for negative auditor tests, not device evidence."""

from __future__ import annotations

from typing import Any

from _native_protocol_builder import ProtocolBuilder

from simpler_setup.tools.hbg.task_id import TaskId


def graph_capture(instances=2, *, dummy=False, host_prefix=False, epoch=9):
    shapes: list[tuple[int | None, int, tuple[int, ...]]] = [(None if host_prefix else 2, 1, ())]
    ranges = [dict(public_task_id=0, begin=0, end=1, children=0, instance_index=-1)]
    for index in range(instances):
        begin = len(shapes)
        shapes.extend(
            ((0, 1, (begin - 1,)), (0 if dummy else 2, 1, (begin,)), (0 if dummy else 2, 1, (begin + 1,)), (0, 1, ()))
        )
        ranges.append(dict(public_task_id=index + 1, begin=begin, end=begin + 4, children=2, instance_index=index))
    shapes.append((2, 1, (len(shapes) - 1,)))
    ranges.append(
        dict(public_task_id=instances + 1, begin=len(shapes) - 1, end=len(shapes), children=0, instance_index=-1)
    )
    builder = ProtocolBuilder(shapes, clusters=1, epoch=epoch)
    raw = builder.raw
    raw.update(schema="hbg-native-protocol-v8", close_seal_stride=128)
    raw["first_error"].update(public_task_id=(1 << 32) - 1, sub_task_id=(1 << 32) - 1)
    for control in raw["task_controls"]:
        control.update(claim_tickets=0, admission_claim_lock=0, claim_seal=0)
    definition: dict[str, Any] = dict(
        index=0,
        full_key=17,
        image_offset=32,
        image_bytes=256,
        fanin_indices_offset=128,
        task_count=2,
        edge_count=1,
        fanin_rows=[0, 0, 1],
        fanin_indices=[0],
        tasks=[],
    )
    for local in range(2):
        definition["tasks"].append(
            dict(
                local_id=local,
                active_mask=0 if dummy else 2,
                task_attrs=0,
                block_num=1,
                predicate_slot=0,
                kernel_ids=[-1, -1 if dummy else 0, -1],
            )
        )
    graph: dict[str, Any] = dict(
        public_task_count=instances + 2,
        instance_count=instances,
        remaining_instances=0,
        next_instance=instances + 3,
        public_completed=instances + 2,
        run_epoch=epoch,
        definitions_address=0x100000,
        definitions_bytes=512,
        ranges=ranges,
        instance_definitions=[0] * instances,
        definitions=[definition],
        nodes=[],
        instances=[],
        materializations=[],
        contributions=[],
        bootstrap=[],
    )
    raw["graph"] = graph
    for group in ranges:
        parent, begin, end, instance = (group[key] for key in ("public_task_id", "begin", "end", "instance_index"))
        for ref in range(begin, end):
            kind = 0 if instance < 0 else 1 if ref == begin else 3 if ref == end - 1 else 2
            local = ref - begin - 1 if kind == 2 else (1 << 32) - 1
            task_id = TaskId.make_sub_task(parent, local) if kind == 2 else TaskId.make_global(parent)
            dependencies = raw["tasks"][ref]["dependencies"]
            fanin_kind = 2 if kind == 2 and local == 0 else 1 if kind == 2 else 0
            fanin_address = (
                0
                if fanin_kind == 2 or not dependencies
                else 0x100000 + 32 + 128
                if fanin_kind == 1
                else 0x200000 + parent * 4
            )
            graph["nodes"].append(
                dict(
                    task_ref=ref,
                    kind=kind,
                    task_id=task_id.raw,
                    public_space=int(kind == 2),
                    public_task_id=parent,
                    sub_task_id=local,
                    instance_index=instance,
                    storage_address=(
                        0x400000 + instance * 4096 + local * 320
                        if kind == 2
                        else 0x800000 + parent * 320
                        if kind == 0
                        else 0
                    ),
                    fanin_address=fanin_address,
                    fanin_kind=fanin_kind,
                    fanin_ref_base=begin if fanin_kind == 2 else begin + 1 if fanin_kind == 1 else 0,
                    dependencies=list(dependencies),
                )
            )
        if instance < 0:
            continue
        owner = instance % 3
        graph["instances"].append(
            dict(
                index=instance,
                instance_index=instance,
                parent_id=parent,
                definition_key=17,
                definition_address=0x100000,
                definition_bytes=512,
                definition_offset=32,
                task_storage_address=0x400000 + instance * 4096,
                task_storage_bytes=640,
                remaining_children=0,
                materialize_owner=owner + 1,
                materialize_state=2,
                run_epoch=epoch,
                enter_ref=begin,
                child_begin=begin + 1,
                child_count=2,
                exit_ref=end - 1,
            )
        )
        record = dict(
            index=instance,
            run_epoch=epoch,
            definition_key=17,
            remaining_before=instances - instance,
            instance_index=instance,
            parent_id=parent,
            worker_index=owner,
            child_count=2,
            status=0,
        )
        graph["materializations"].append(record)
        for field in ("materialize_before", "materialize_after", "publish_before", "publish_after"):
            builder.stamp(record, field)
    for worker in range(3):
        record = dict(
            index=worker,
            run_epoch=epoch,
            remaining_observed=0,
            poll_count=1,
            worker_index=worker,
            instance_count=instances,
            status=0,
        )
        graph["bootstrap"].append(record)
        builder.stamp(record, "before")
        builder.stamp(record, "after")
    for ref, task in enumerate(raw["tasks"]):
        if not task["flags"]:
            continue
        worker = 1 + ref % 2
        if task["flags"] & 32:
            builder.actor(ref, worker, "completion")
        else:
            builder.execute(ref, 0, 1, worker)
        node = graph["nodes"][ref]
        if node["kind"] == 2:
            record = dict(
                index=ref,
                run_epoch=epoch,
                task_ref=ref,
                remaining_before=2 - node["sub_task_id"],
                worker_index=worker,
                instance_index=node["instance_index"],
            )
            graph["contributions"].append(record)
            builder.stamp(record, "before")
            builder.stamp(record, "after")
    for identity in builder.identities:
        identity["task_id"] = graph["nodes"][identity["task_id"]]["task_id"]
    raw["executed_task_count"] = len(builder.identities)
    return raw, builder.identities
