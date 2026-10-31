# Copyright (c) PyPTO Contributors.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------

"""Verify Graph ownership, shared definitions and ENTER/child/EXIT causality."""

from collections import defaultdict

from .native_causality import EXECUTABLE, FANIN, INLINE, MIX, PREDICATE, SPMD, SYNC, point, require
from .native_protocol import STAMPS, _event_records
from .task_id import Space, TaskId

ORDINARY, ENTER, CHILD, EXIT = range(4)
DENSE, CSR, ENTER_GATE = range(3)
NO_CHILD = (1 << 32) - 1


def _definitions(graph):
    definitions = {row["index"]: row for row in graph["definitions"]}
    require(0 < len(definitions) <= 16, "Graph definitions are absent or exceed the run limit")
    require(
        len({row["full_key"] for row in definitions.values()}) == len(definitions), "duplicate Graph definition key"
    )
    spans = []
    for definition in definitions.values():
        count, rows, edges = (definition[key] for key in ("task_count", "fanin_rows", "fanin_indices"))
        require(0 < count <= 1024 and rows[0] == 0 and rows[-1] == len(edges), "invalid frozen Graph CSR bounds")
        require(all(left <= right for left, right in zip(rows, rows[1:], strict=False)), "nonmonotonic Graph CSR")
        for local in range(count):
            require(
                definition["tasks"][local]["local_id"] == local, "Graph source tasks are not in recorded local order"
            )
            sources = edges[rows[local] : rows[local + 1]]
            require(
                len(set(sources)) == len(sources) and all(source < local for source in sources),
                "invalid Graph CSR source",
            )
        start, size = definition["image_offset"], definition["image_bytes"]
        require(
            start >= 32 and start % 16 == 0 and size >= 104 and start + size <= graph["definitions_bytes"],
            "frozen Graph definition image outside the carrier",
        )
        spans.append((start - 32, start + size))
    spans.sort()
    require(
        all(left[1] <= right[0] for left, right in zip(spans, spans[1:], strict=False)), "overlapping Graph definitions"
    )
    require(
        set(graph["instance_definitions"]) == set(definitions), "Graph definition references are missing or foreign"
    )
    return definitions


def _source_task(task, source):
    mask, blocks, attrs = (source[key] for key in ("active_mask", "block_num", "task_attrs"))
    require(mask < 8 and attrs < 256 and 0 < blocks <= 32767, "invalid frozen Graph task source")
    require(bool(source["predicate_slot"]) == bool(attrs & 4), "Graph predicate source mismatch")
    flags = EXECUTABLE | (FANIN if task["dependencies"] else 0)
    flags |= (INLINE if mask == 0 else 0) | (MIX if mask.bit_count() > 1 else 0)
    flags |= (SPMD if blocks > 1 else 0) | (SYNC if attrs & 2 else 0) | (PREDICATE if attrs & 4 else 0)
    require(
        task["flags"] == flags and task["active_mask"] == (mask or 2) and task["block_num"] == blocks,
        "Graph task metadata differs from its frozen Definition",
    )
    require(
        task["kernel_ids"] == [65535 if kernel == -1 else kernel for kernel in source["kernel_ids"]],
        "Graph kernel mapping differs from its frozen Definition",
    )


def _instance(graph, group, instance, definition, epoch):
    index, parent, begin, end, children = (
        group[key] for key in ("instance_index", "public_task_id", "begin", "end", "children")
    )
    require(
        (
            instance["index"],
            instance["instance_index"],
            instance["parent_id"],
            instance["enter_ref"],
            instance["child_begin"],
            instance["child_count"],
            instance["exit_ref"],
        )
        == (index, index, parent, begin, begin + 1, children, end - 1),
        "Graph instance binding differs from the Host range",
    )
    require(
        children == definition["task_count"]
        and instance["definition_key"] == definition["full_key"]
        and instance["definition_offset"] == definition["image_offset"]
        and instance["definition_address"] == graph["definitions_address"]
        and instance["definition_bytes"] == graph["definitions_bytes"],
        "Graph instance does not use the shared frozen Definition",
    )
    require(
        instance["run_epoch"] == epoch and instance["materialize_state"] == 2 and instance["remaining_children"] == 0,
        "Graph instance is stale, unmaterialized or unfinished",
    )
    start, size = instance["task_storage_address"], instance["task_storage_bytes"]
    require(
        start > 0 and start % 64 == 0 and size == children * 320 and start + size <= 1 << 64,
        "invalid Graph child storage extent",
    )


def validate_graph_manifest(audit):
    graph = audit.raw["graph"]
    require(
        0 < graph["instance_count"] <= graph["public_task_count"]
        and graph["remaining_instances"] == 0
        and graph["public_completed"] == graph["public_task_count"]
        and graph["run_epoch"] == audit.raw["run_epoch"]
        and graph["next_instance"] == graph["instance_count"] + audit.raw["worker_count"],
        "Graph public/build terminal counts mismatch",
    )
    require(
        graph["definitions_address"] > 0 and graph["definitions_address"] % 16 == 0, "invalid Graph definition carrier"
    )
    definitions = _definitions(graph)
    nodes = {node["task_ref"]: node for node in graph["nodes"]}
    instances = {instance["index"]: instance for instance in graph["instances"]}
    cursor = next_instance = 0
    storage = []
    for group in sorted(graph["ranges"], key=lambda row: row["public_task_id"]):
        parent, begin, end, children, index = (
            group[key] for key in ("public_task_id", "begin", "end", "children", "instance_index")
        )
        require(
            begin == cursor and end == begin + (children + 2 if index >= 0 else 1),
            "Graph ranges overlap or leave holes",
        )
        require(
            (index == next_instance and 0 < children <= 1024) if index >= 0 else (index == -1 and children == 0),
            "Graph range has an invalid child count or instance index",
        )
        if index >= 0:
            require(index in instances, "Graph range names an absent instance")
            definition = definitions[graph["instance_definitions"][index]]
            instance = instances[index]
            _instance(graph, group, instance, definition, audit.raw["run_epoch"])
            storage.append(
                (instance["task_storage_address"], instance["task_storage_address"] + instance["task_storage_bytes"])
            )
            next_instance += 1
        for ref in range(begin, end):
            require(ref in nodes, "Graph range names an absent node")
            node, task = nodes[ref], audit.tasks[ref]
            kind = ORDINARY if index < 0 else ENTER if ref == begin else EXIT if ref == end - 1 else CHILD
            local = ref - begin - 1 if kind == CHILD else NO_CHILD
            require(
                (
                    node["kind"],
                    node["public_space"],
                    node["public_task_id"],
                    node["sub_task_id"],
                    node["instance_index"],
                )
                == (kind, int(kind == CHILD), parent, local, index),
                "Graph node public identity or kind mismatch",
            )
            public = TaskId(node["task_id"])
            require(
                public.is_canonical()
                and public.space() is (Space.SUB_TASK if kind == CHILD else Space.GLOBAL)
                and public.local_id() == (local if kind == CHILD else parent)
                and (kind != CHILD or public.parent_id() == parent),
                "Graph node raw public TaskId mismatch",
            )
            audit.public_task_ids[ref] = public.raw
            require(node["dependencies"] == task["dependencies"], "device Graph edges differ from frozen dependencies")
            if kind == CHILD:
                rows, edges = definition["fanin_rows"], definition["fanin_indices"]
                sources = edges[rows[local] : rows[local + 1]]
                expected = [begin + 1 + source for source in sources] if sources else [begin]
                require(task["dependencies"] == expected, "Graph root gate or translated internal CSR mismatch")
                require(
                    node["storage_address"] == instance["task_storage_address"] + local * 320,
                    "Graph node storage binding mismatch",
                )
                require(
                    node["fanin_kind"] == (CSR if sources else ENTER_GATE)
                    and node["fanin_ref_base"] == (begin + 1 if sources else begin)
                    and node["fanin_address"]
                    == (
                        graph["definitions_address"]
                        + definition["image_offset"]
                        + definition["fanin_indices_offset"]
                        + rows[local] * 2
                        if sources
                        else 0
                    ),
                    "Graph node CSR address or root gate encoding mismatch",
                )
                _source_task(task, definition["tasks"][local])
            else:
                require(
                    node["fanin_kind"] == DENSE and node["fanin_ref_base"] == 0 and len(task["dependencies"]) <= 128,
                    "invalid external Graph fanin encoding",
                )
                require(
                    all(nodes[source]["kind"] in (ORDINARY, EXIT) for source in task["dependencies"]),
                    "external dependency bypasses a Graph EXIT",
                )
                if kind in (ENTER, EXIT):
                    require(
                        node["storage_address"] == 0 and task["flags"] & INLINE,
                        "Graph ENTER/EXIT is not a kernel-free service",
                    )
                if kind == EXIT:
                    require(
                        not task["dependencies"] and node["fanin_address"] == 0,
                        "Graph EXIT incorrectly uses a fanin row",
                    )
        cursor = end
    require(
        cursor == len(nodes) and next_instance == graph["instance_count"], "Graph ranges do not cover the expanded run"
    )
    storage.sort()
    require(
        all(left[1] <= right[0] for left, right in zip(storage, storage[1:], strict=False)),
        "Graph instances alias mutable child storage",
    )


def _ticket_chain(audit, rows, count, name, first, last):
    require(
        sorted(row["remaining_before"] for row in rows) == list(range(1, count + 1)),
        f"{name} counter tickets are missing or duplicated",
    )
    prior = None
    for row in sorted(rows, key=lambda record: -record["remaining_before"]):
        current = audit.causal.bounded((name, row["index"]), point(row, first), point(row, last))
        if prior is not None:
            audit.causal.add(prior, current)
        prior = current
    return prior


def audit_graph_events(audit):
    graph = audit.raw["graph"]
    instances = {row["index"]: row for row in graph["instances"]}
    materializations = {row["index"]: row for row in graph["materializations"]}
    waiters = {row["index"]: row for row in graph["bootstrap"]}
    nodes = {node["task_ref"]: node for node in graph["nodes"]}
    contributions = {row["index"]: row for row in graph["contributions"]}
    require(set(materializations) == set(instances), "missing Graph materialization witness")
    require(set(waiters) == set(audit.workers), "missing Graph bootstrap observer")
    require(
        set(contributions) == {ref for ref, node in nodes.items() if node["kind"] == CHILD},
        "missing or extraneous Graph child contribution",
    )
    for index, record in materializations.items():
        instance = instances[index]
        require(
            record["instance_index"] == index
            and record["parent_id"] == instance["parent_id"]
            and record["definition_key"] == instance["definition_key"]
            and record["child_count"] == instance["child_count"]
            and record["status"] == 0
            and instance["materialize_owner"] == record["worker_index"] + 1,
            "Graph materialization identity, owner or status mismatch",
        )
        audit.before(
            point(record, "materialize_before"), point(record, "materialize_after"), "reversed Graph materialization"
        )
        audit.before(
            point(record, "materialize_after"),
            point(record, "publish_before"),
            "Graph publication before materialization",
        )
    last_publish = _ticket_chain(
        audit, list(materializations.values()), len(instances), "GraphPublish", "publish_before", "publish_after"
    )
    for worker, record in waiters.items():
        require(
            record["worker_index"] == worker
            and record["instance_count"] == len(instances)
            and record["remaining_observed"] == 0
            and record["poll_count"] > 0
            and record["status"] == 0,
            "Graph bootstrap did not observe a successful zero counter",
        )
        audit.before(point(record, "before"), point(record, "after"), "reversed Graph bootstrap wait")
        audit.causal.add(last_publish, point(record, "after"))
    for record, kind, _ in _event_records(audit.raw):
        if kind.startswith("graph_"):
            continue
        for field in STAMPS[kind].split():
            if record[field][1]:
                audit.before(
                    point(waiters[record["worker_index"]], "after"),
                    point(record, field),
                    "scheduler event before Graph bootstrap observation",
                )
    for execution in audit.executions.values():
        audit.causal.add(
            point(waiters[execution["worker_index"]], "after"), audit.payload_points[execution["index"]][0]
        )
    contributions_by_instance = defaultdict(list)
    for ref, record in contributions.items():
        require(
            record["task_ref"] == ref
            and record["instance_index"] == nodes[ref]["instance_index"]
            and record["worker_index"] == audit.actors[ref]["completion"]["worker_index"],
            "Graph contribution has the wrong child, instance or completion owner",
        )
        audit.before(
            point(audit.actors[ref]["completion"], "after"),
            point(record, "before"),
            "Graph contribution precedes child TaskDone",
        )
        contributions_by_instance[record["instance_index"]].append(record)
    for index, instance in instances.items():
        rows = contributions_by_instance[index]
        last = _ticket_chain(audit, rows, instance["child_count"], "GraphContribution", "before", "after")
        audit.causal.add(last, audit.done_points[instance["exit_ref"]])
        final = next(row for row in rows if row["remaining_before"] == 1)
        audit.before(
            point(final, "after"),
            point(audit.actors[instance["exit_ref"]]["completion"], "before"),
            "Graph EXIT precedes its final child contribution",
        )
    return dict(
        graph_verified=True,
        graph_instances=len(instances),
        graph_children=len(contributions),
        graph_definitions=len(graph["definitions"]),
    )
