# Copyright (c) PyPTO Contributors.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------

"""Render verified native Graph lifecycle intervals; control services are not kernels."""

from collections import defaultdict

from .graph_lane import assign_instance_lanes, graph_lane_metadata
from .native_graph_audit import CHILD, ENTER_GATE


def decode_graph_lane(section, tasks, to_us):
    graph = section["graph"]
    actors = {row["index"]: row["completion"] for row in section["task_events"]}
    materializations = {row["index"]: row for row in graph["materializations"]}
    nodes = {row["task_ref"]: row for row in graph["nodes"]}
    executions = {row["index"]: row for row in section["executions"]}
    specifications = {row["task_ref"]: row for row in section["tasks"]}
    task_rows = defaultdict(list)
    for task in tasks:
        execution = executions[task["reg_task_id"]]
        task["func_id"] = specifications[execution["task_ref"]]["kernel_ids"][execution["subtask_slot"]]
        node = nodes[execution["task_ref"]]
        task.update(
            task_ref=execution["task_ref"],
            public_task_id=node["public_task_id"],
            sub_task_id=node["sub_task_id"] if node["kind"] == CHILD else None,
            task_id_hex=f"0x{node['task_id']:016x}",
        )
        task_rows[task["task_id"]].append(task)
    result = dict(
        run_epoch=section["run_epoch"], verified=True, instances=[], operations=[], flows=[], workers=section["workers"]
    )

    def operation(name, record, before, after, *, instance=-1, ref=None, **args):
        start, end = to_us(record[before][0]), to_us(record[after][0])
        if end < start:
            raise ValueError("native Graph display has a reversed local interval")
        args.update(run_epoch=section["run_epoch"], worker_index=record["worker_index"], instance_index=instance)
        if ref is not None:
            node = nodes[ref]
            args.update(
                task_ref=ref,
                public_task_id=node["public_task_id"],
                sub_task_id=node["sub_task_id"] if node["kind"] == CHILD else None,
            )
        row = dict(name=name, worker_index=record["worker_index"], start_time_us=start, end_time_us=end, args=args)
        result["operations"].append(row)
        return row

    for instance in graph["instances"]:
        index, parent = instance["index"], instance["parent_id"]
        materialize = materializations[index]
        enter, exit_record = actors[instance["enter_ref"]], actors[instance["exit_ref"]]
        prepared = operation(
            "Graph.materialize",
            materialize,
            "materialize_before",
            "materialize_after",
            instance=index,
            public_task_id=parent,
            definition_key=materialize["definition_key"],
            child_count=instance["child_count"],
        )
        operation(
            "Graph.publish",
            materialize,
            "publish_before",
            "publish_after",
            instance=index,
            public_task_id=parent,
            remaining_before=materialize["remaining_before"],
        )
        enter_op = operation("Graph.ENTER", enter, "before", "after", instance=index, ref=instance["enter_ref"])
        exit_op = operation("Graph.EXIT", exit_record, "before", "after", instance=index, ref=instance["exit_ref"])
        start, end = prepared["start_time_us"], exit_op["end_time_us"]
        if end < start:
            raise ValueError("native Graph lifecycle display needs aligned device clocks")
        result["instances"].append(
            dict(
                instance_index=index,
                outer_task_id=parent,
                run_epoch=section["run_epoch"],
                prepare_start_us=start,
                execution_end_us=end,
                child_count=instance["child_count"],
                stages=[prepared, enter_op, exit_op],
            )
        )
    for wait in graph["bootstrap"]:
        operation(
            "Graph.materialize_wait",
            wait,
            "before",
            "after",
            poll_count=wait["poll_count"],
            remaining_observed=wait["remaining_observed"],
            instance_count=wait["instance_count"],
        )
    for ref, node in nodes.items():
        if node["kind"] != CHILD:
            continue
        rows = task_rows[node["task_id"]]
        if not rows:
            operation("Graph.Dummy", actors[ref], "before", "after", instance=node["instance_index"], ref=ref)
        elif not any(row.get("executed", True) for row in rows):
            operation("Graph.predicate_skip", actors[ref], "before", "after", instance=node["instance_index"], ref=ref)
        if node["fanin_kind"] == ENTER_GATE:
            endpoint = min(rows, key=lambda row: row["start_time_us"]) if rows else None
            result["flows"].append(
                dict(
                    kind="ENTER→root",
                    instance_index=node["instance_index"],
                    target_ref=ref,
                    endpoint=endpoint,
                    worker_index=actors[ref]["worker_index"],
                    target_time_us=to_us(actors[ref]["before"][0]),
                )
            )
    for record in graph["contributions"]:
        ref, index = record["task_ref"], record["instance_index"]
        row = operation(
            "Graph.child_contribution",
            record,
            "before",
            "after",
            instance=index,
            ref=ref,
            remaining_before=record["remaining_before"],
        )
        if record["remaining_before"] == 1:
            result["flows"].append(
                dict(
                    kind="last child→EXIT",
                    instance_index=index,
                    target_ref=ref,
                    worker_index=record["worker_index"],
                    source_time_us=row["end_time_us"],
                )
            )
    assign_instance_lanes(result["instances"])
    return result


def contribute_graph_events(graph, ctx, *, scalar_pid, scalar_tid):
    if not graph.get("verified"):
        raise ValueError("native Graph rendering requires a verified protocol")
    events = graph_lane_metadata(graph["instances"], ctx)
    events.append(
        dict(name="process_name", cat="__metadata", ph="M", pid=scalar_pid, args={"name": "AICore Scheduler"})
    )
    for worker in graph["workers"]:
        index = worker["index"]
        events.append(
            dict(
                name="thread_name",
                cat="__metadata",
                ph="M",
                pid=scalar_pid,
                tid=scalar_tid(index),
                args={"name": f"Scheduler_{index} / Graph detail (same Scalar)"},
            )
        )
    for instance in graph["instances"]:
        parent = instance["outer_task_id"]
        events.append(
            dict(
                name=f"GraphExecution({ctx.display(parent)}, {instance['child_count']} sub-tasks)",
                cat="graph_execution",
                ph="X",
                pid=ctx.pid,
                tid=ctx.tid_base + instance["lane_idx"],
                ts=instance["prepare_start_us"],
                dur=instance["execution_end_us"] - instance["prepare_start_us"],
                cname="rail_animation",
                args=dict(
                    public_task_id=parent,
                    total_sub_task_count=instance["child_count"],
                    graph_verified=True,
                    run_epoch=graph["run_epoch"],
                ),
            )
        )
        for stage in instance["stages"]:
            events.append(
                dict(
                    name=stage["name"],
                    cat="graph_lifecycle",
                    ph="X",
                    pid=ctx.pid,
                    tid=ctx.tid_base + instance["lane_idx"],
                    ts=stage["start_time_us"],
                    dur=stage["end_time_us"] - stage["start_time_us"],
                    args=stage["args"],
                )
            )
    for row in graph["operations"]:
        events.append(
            dict(
                name=row["name"],
                cat="graph_scalar",
                ph="X",
                pid=scalar_pid,
                tid=scalar_tid(row["worker_index"]),
                ts=row["start_time_us"],
                dur=row["end_time_us"] - row["start_time_us"],
                args=row["args"],
            )
        )
    return events


def contribute_graph_flows(graph, ctx, *, scalar_pid, scalar_tid, kernel_pid, kernel_tid, flow_id):
    instances = {row["instance_index"]: row for row in graph["instances"]}
    events = []
    for flow in graph["flows"]:
        instance = instances[flow["instance_index"]]
        graph_tid = ctx.tid_base + instance["lane_idx"]
        if flow["kind"] == "ENTER→root":
            source = (ctx.pid, graph_tid, instance["stages"][1]["start_time_us"])
            endpoint = flow["endpoint"]
            target = (
                (kernel_pid, kernel_tid[endpoint["core_id"]], endpoint["start_time_us"])
                if endpoint
                else (scalar_pid, scalar_tid(flow["worker_index"]), flow["target_time_us"])
            )
        else:
            source = (scalar_pid, scalar_tid(flow["worker_index"]), flow["source_time_us"])
            target = (ctx.pid, graph_tid, instance["stages"][2]["start_time_us"])
        args = dict(
            run_epoch=graph["run_epoch"],
            public_task_id=instance["outer_task_id"],
            task_ref=flow["target_ref"],
            relation=flow["kind"],
            source_anchor="publication interval lower bound"
            if flow["kind"] == "ENTER→root"
            else "contribution returned",
        )
        for phase, (pid, tid, timestamp) in (("s", source), ("f", target)):
            events.append(
                dict(
                    name=flow["kind"],
                    cat="graph_causality",
                    ph=phase,
                    pid=pid,
                    tid=tid,
                    ts=timestamp,
                    id=flow_id,
                    args=args,
                    **({"bp": "e"} if phase == "f" else {}),
                )
            )
        flow_id += 1
    return events, flow_id
