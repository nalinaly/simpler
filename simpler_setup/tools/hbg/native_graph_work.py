# Copyright (c) PyPTO Contributors.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------

"""v12 Graph work ownership, physical writeback ranges and publication evidence."""

from collections import defaultdict

NUMBERS = {
    "graph_work_plan": "index instance_index region first_range range_count write_bytes owner_state",
    "graph_work_range": "index begin count",
    "graph_work": "index run_epoch remaining_before work_index instance_index worker_index status",
    "graph_publication": (
        "index run_epoch definition_key remaining_before instance_index parent_id worker_index child_count status"
    ),
}
STAMPS = {
    "graph_work": "begin writeback_after complete_before complete_after",
    "graph_publication": "publish_before publish_after",
}
ROOT_NUMBERS = frozenset(
    (
        "work_count work_range_count next_work build_control_address work_address work_ranges_address "
        "work_states_address protocol_work_address scheduler_state_address scheduler_state_bytes"
    ).split()
)
ROOT_TABLES = frozenset("work_plan work_ranges work publications".split())
INSTANCE_NUMBERS = (
    "work_begin work_count remaining_work tensor_pool_address scalar_pool_address "
    "tensor_arg_count scalar_arg_count control_address"
).split()
DEFINITION_NUMBERS = ("tensor_arg_count", "scalar_arg_count")
TASK_NUMBERS = ("tensor_offset", "tensor_count", "scalar_offset", "scalar_count")


def read_work_tables(graph):
    from .native_protocol import _indices, _record  # noqa: PLC0415

    for table, kind, count, dense in (
        ("work_plan", "graph_work_plan", graph["work_count"], True),
        ("work_ranges", "graph_work_range", graph["work_range_count"], True),
        ("work", "graph_work", graph["work_count"], False),
    ):
        _indices(graph[table], count, f"graph.{table}", dense=dense)
        for row in graph[table]:
            _record(row, kind, f"graph.{table}[{row['index']}]")


def _merged(intervals):
    result = []
    for begin, end in sorted(intervals):
        if result and begin <= result[-1][1]:
            result[-1] = result[-1][0], max(result[-1][1], end)
        else:
            result.append((begin, end))
    return result


def _parameter_ranges(definition, name):
    from .native_causality import require  # noqa: PLC0415

    capacity = definition[f"{name}_arg_count"]
    require(capacity >= 0, "negative Graph parameter pool capacity")
    result = []
    for task in definition["tasks"]:
        begin, count = task[f"{name}_offset"], task[f"{name}_count"]
        require(
            0 <= begin <= capacity and 0 <= count <= capacity - begin, "Graph parameter reference is outside its pool"
        )
        if count:
            result.append((begin, begin + count))
    return _merged(result)


def _line_span(base, begin, count, stride):
    from .native_causality import require  # noqa: PLC0415

    require(base > 0 and base % min(stride, 64) == 0, "Graph work region has invalid alignment")
    first, end = base + begin * stride, base + (begin + count) * stride
    require(count > 0 and end <= 1 << 64, "Graph work writeback address overflows")
    return first // 64, (end - 1) // 64


def _exclusive_lines(lines):
    from .native_causality import require  # noqa: PLC0415

    active_end, active_owner = -1, -1
    for first, last, owner in sorted(lines):
        if first <= active_end:
            require(owner == active_owner, "Graph work writers share a cache line or overlap scheduler storage")
            active_end = max(active_end, last)
        else:
            active_end, active_owner = last, owner


def _work_ranges(graph, instance, definition, work, parts, covered, used, lines):
    from .native_causality import require  # noqa: PLC0415

    counts = (instance["child_count"], instance["tensor_arg_count"], instance["scalar_arg_count"])
    bases = (instance["task_storage_address"], instance["tensor_pool_address"], instance["scalar_pool_address"])
    strides = (320, 128, 8)
    region = work["region"]
    require(region in (0, 1, 2, 3), "unknown Graph work region")
    if region == 3:
        require(
            instance["work_count"] == 1 and work["range_count"] == 0 and work["first_range"] == 0,
            "serial Graph work is not the instance's sole writer",
        )
        require(
            work["write_bytes"] == sum(n * s for n, s in zip(counts, strides, strict=True)),
            "serial Graph work byte count differs from its regions",
        )
        covered[0].append((0, counts[0]))
        covered[1].extend(_parameter_ranges(definition, "tensor"))
        covered[2].extend(_parameter_ranges(definition, "scalar"))
        for base, count, stride in zip(bases, counts, strides, strict=True):
            if count:
                lines.append((*_line_span(base, 0, count, stride), work["index"]))
        return
    first, count = work["first_range"], work["range_count"]
    require(count > 0 and first + count <= graph["work_range_count"], "Graph work range list is out of bounds")
    previous_end = 0
    write_bytes = 0
    for index in range(first, first + count):
        part = parts[index]
        begin, size = part["begin"], part["count"]
        require(
            size > 0 and begin >= previous_end and begin + size <= counts[region],
            "Graph work ranges overlap or exceed their region",
        )
        previous_end = begin + size
        used.add(index)
        covered[region].append((begin, begin + size))
        write_bytes += size * strides[region]
        lines.append((*_line_span(bases[region], begin, size, strides[region]), work["index"]))
    require(write_bytes == work["write_bytes"], "Graph work byte count differs from its ranges")


def audit_work_plan(audit):
    from .native_causality import require  # noqa: PLC0415

    graph = audit.raw["graph"]
    require(
        0 < graph["work_count"] <= (1 << 32) - 1
        and graph["next_work"] == graph["work_count"] + audit.raw["worker_count"],
        "Graph work claim count mismatch",
    )
    definitions = {row["index"]: row for row in graph["definitions"]}
    work = {row["index"]: row for row in graph["work_plan"]}
    parts = {row["index"]: row for row in graph["work_ranges"]}
    state_begin, state_size = graph["scheduler_state_address"], graph["scheduler_state_bytes"]
    require(
        state_begin > 0 and state_begin % 128 == 0 and state_size > 0 and state_begin + state_size <= 1 << 64,
        "invalid Graph scheduler storage extent",
    )
    lines = [(*_line_span(state_begin, 0, state_size, 1), -1)]
    for field, count, stride, alignment in (
        ("build_control_address", 1, 256, 128),
        ("work_address", graph["work_count"], 24, 8),
        ("work_ranges_address", graph["work_range_count"], 8, 4),
        ("work_states_address", graph["work_count"], 8, 128),
        ("protocol_work_address", graph["work_count"], 128, 128),
    ):
        address = graph[field]
        require(
            address % alignment == 0
            and state_begin <= address
            and address + count * stride <= state_begin + state_size,
            "Graph work allocation is outside scheduler storage",
        )
    cursor, used = 0, set()
    for instance in sorted(graph["instances"], key=lambda row: row["index"]):
        definition = definitions[graph["instance_definitions"][instance["index"]]]
        require(
            instance["work_begin"] == cursor
            and instance["work_count"] > 0
            and cursor + instance["work_count"] <= graph["work_count"]
            and instance["remaining_work"] == 0,
            "Graph instance work coverage or completion mismatch",
        )
        require(
            all(instance[key] == definition[key] for key in DEFINITION_NUMBERS),
            "Graph parameter capacities differ from the frozen Definition",
        )
        control = instance["control_address"]
        require(
            control % 128 == 0 and state_begin <= control and control + 128 <= state_begin + state_size,
            "Graph work counter is outside scheduler storage",
        )
        expected = [
            [(0, instance["child_count"])],
            _parameter_ranges(definition, "tensor"),
            _parameter_ranges(definition, "scalar"),
        ]
        covered = defaultdict(list)
        for index in range(cursor, cursor + instance["work_count"]):
            item = work[index]
            require(item["instance_index"] == instance["index"], "Graph work belongs to another instance")
            _work_ranges(graph, instance, definition, item, parts, covered, used, lines)
        for region in range(3):
            actual = sorted(covered[region])
            require(
                all(left[1] <= right[0] for left, right in zip(actual, actual[1:], strict=False)),
                "Graph work writes a referenced element more than once",
            )
            require(_merged(actual) == expected[region], "Graph work does not cover exactly the referenced elements")
        cursor += instance["work_count"]
    require(cursor == graph["work_count"] and used == set(parts), "Graph work plan contains orphaned entries")
    _exclusive_lines(lines)


def audit_work_events(audit, instances, publications):
    from .native_causality import point, require  # noqa: PLC0415
    from .native_graph_audit import _ticket_chain  # noqa: PLC0415

    graph = audit.raw["graph"]
    plan = {row["index"]: row for row in graph["work_plan"]}
    records = {row["index"]: row for row in graph["work"]}
    require(set(records) == set(plan), "missing Graph work completion witness")
    by_instance = defaultdict(list)
    for index, record in records.items():
        work = plan[index]
        require(
            record["work_index"] == index
            and record["instance_index"] == work["instance_index"]
            and work["owner_state"] == ((record["worker_index"] + 1) << 2) | 2
            and record["status"] == 0,
            "Graph work identity, unique owner or success state mismatch",
        )
        for before, after in (
            ("begin", "writeback_after"),
            ("writeback_after", "complete_before"),
            ("complete_before", "complete_after"),
        ):
            audit.before(point(record, before), point(record, after), "Graph work contributes before writeback")
        by_instance[work["instance_index"]].append(record)
    for index, instance in instances.items():
        rows = by_instance[index]
        last = _ticket_chain(audit, rows, instance["work_count"], "GraphWork", "complete_before", "complete_after")
        publisher = publications[index]
        final = next(row for row in rows if row["remaining_before"] == 1)
        require(final["worker_index"] == publisher["worker_index"], "Graph publisher is not its final work contributor")
        audit.causal.add(last, point(publisher, "publish_before"))
        audit.before(
            point(final, "complete_after"),
            point(publisher, "publish_before"),
            "Graph instance publishes before its final work completion",
        )
