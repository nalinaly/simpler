# Copyright (c) PyPTO Contributors.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------

"""Synthetic parallel materialization witnesses; these are not device evidence."""

from _native_graph_builder import graph_capture

from simpler_setup.tools.hbg.native_protocol import STAMPS, _event_records


def resequence(raw):
    for worker in raw["workers"]:
        worker["next_sequence"] = 0
    events = []
    for record, kind, _ in _event_records(raw):
        for field in STAMPS[kind].split():
            if record[field][1]:
                events.append((record[field][0], record["worker_index"], record, field))
    for _, worker, record, field in sorted(events, key=lambda event: event[0]):
        raw["workers"][worker]["next_sequence"] += 1
        record[field][1] = raw["workers"][worker]["next_sequence"]


def parallel_graph_capture(instances=2, *, mixed=False, scalar_gap=False, split_scalars=False):
    raw, identities = graph_capture(instances)
    for record, kind, _ in _event_records(raw):
        if kind == "graph_materialize":
            continue
        for field in STAMPS[kind].split():
            if record[field][1]:
                record[field][0] += 1_000_000
        if kind == "execution":
            record["publication_before_cycles"] += 1_000_000
            record["publication_after_cycles"] += 1_000_000
    raw["schema"] = "hbg-native-protocol-v12"
    graph = raw["graph"]
    graph.update(
        next_instance=instances,
        work_count=0,
        work_range_count=5 if scalar_gap else 3,
        next_work=0,
        scheduler_state_address=0x100000,
        scheduler_state_bytes=0x200000,
        build_control_address=0x160000,
        work_address=0x110000,
        work_ranges_address=0x120000,
        work_states_address=0x130000,
        protocol_work_address=0x140000,
        work_plan=[],
        work=[],
        publications=[],
        work_ranges=[dict(index=0, begin=0, count=1), dict(index=1, begin=1, count=1), dict(index=2, begin=0, count=2)],
    )
    if scalar_gap:
        graph["work_ranges"].extend([dict(index=3, begin=0, count=1), dict(index=4, begin=2, count=1)])
    definition = graph["definitions"][0]
    scalar_count = 4 if scalar_gap else 2
    definition.update(tensor_arg_count=2, scalar_arg_count=scalar_count)
    for index, task in enumerate(definition["tasks"]):
        task.update(tensor_offset=index, tensor_count=1, scalar_offset=index * (2 if scalar_gap else 1), scalar_count=1)
    clock = 100

    def stamp(record, field):
        nonlocal clock
        clock += 10
        record[field] = [clock, 1]

    for instance, old in zip(graph["instances"], graph.pop("materializations"), strict=True):
        index = instance["index"]
        serial = mixed and index > 0 and index == instances - 1
        pieces = [(0, 0, 1, 320), (0, 1, 1, 320), (1, 2, 1, 256)]
        if scalar_gap and split_scalars:
            pieces += [(2, 3, 1, 8), (2, 4, 1, 8)]
        else:
            pieces += [(2, 3 if scalar_gap else 2, 2 if scalar_gap else 1, 16)]
        if serial:
            pieces = [(3, 0, 0, 896 + scalar_count * 8)]
        instance.update(
            work_begin=len(graph["work_plan"]),
            work_count=len(pieces),
            remaining_work=0,
            tensor_pool_address=instance["task_storage_address"] + 640,
            scalar_pool_address=instance["task_storage_address"] + 896,
            tensor_arg_count=2,
            scalar_arg_count=scalar_count,
            control_address=0x150000 + index * 128,
        )
        for position, (region, first, count, size) in enumerate(pieces):
            work_index = len(graph["work_plan"])
            owner = (index + position) % 3
            graph["work_plan"].append(
                dict(
                    index=work_index,
                    instance_index=index,
                    region=region,
                    first_range=first,
                    range_count=count,
                    write_bytes=size,
                    owner_state=((owner + 1) << 2) | 2,
                )
            )
            record = dict(
                index=work_index,
                work_index=work_index,
                instance_index=index,
                worker_index=owner,
                run_epoch=raw["run_epoch"],
                remaining_before=len(pieces) - position,
                status=0,
            )
            for field in ("begin", "writeback_after", "complete_before", "complete_after"):
                stamp(record, field)
            graph["work"].append(record)
        publisher = {key: value for key, value in old.items() if not key.startswith("materialize_")}
        publisher["worker_index"] = graph["work"][-1]["worker_index"]
        instance["materialize_owner"] = publisher["worker_index"] + 1
        stamp(publisher, "publish_before")
        stamp(publisher, "publish_after")
        graph["publications"].append(publisher)
    graph["work_count"] = len(graph["work_plan"])
    graph["next_work"] = graph["work_count"] + raw["worker_count"]
    resequence(raw)
    return raw, identities
