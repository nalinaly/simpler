# Copyright (c) PyPTO Contributors.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------

"""The v8 Graph extension: frozen expectations, actual bindings, and bounded device records."""

NUMBERS = {
    "graph_range": "public_task_id begin end children instance_index",
    "graph_definition": "index full_key image_offset image_bytes fanin_indices_offset task_count edge_count",
    "graph_definition_task": "local_id active_mask task_attrs block_num predicate_slot",
    "graph_node": (
        "task_ref kind task_id public_space public_task_id sub_task_id instance_index storage_address "
        "fanin_address fanin_kind fanin_ref_base"
    ),
    "graph_instance": (
        "index instance_index parent_id definition_key definition_address definition_bytes definition_offset "
        "task_storage_address task_storage_bytes remaining_children materialize_owner materialize_state "
        "run_epoch enter_ref child_begin child_count exit_ref"
    ),
    "graph_materialize": (
        "index run_epoch definition_key remaining_before instance_index parent_id worker_index child_count status"
    ),
    "graph_contribution": "index run_epoch task_ref remaining_before worker_index instance_index",
    "graph_bootstrap": "index run_epoch remaining_observed poll_count worker_index instance_count status",
}
STAMPS = {
    "graph_materialize": "materialize_before materialize_after publish_before publish_after",
    "graph_contribution": "before after",
    "graph_bootstrap": "before after",
}
ARRAYS = {
    "graph_definition": {"fanin_rows": None, "fanin_indices": None},
    "graph_definition_task": {"kernel_ids": 3},
    "graph_node": {"dependencies": None},
}
ROOT_NUMBERS = frozenset(
    (
        "public_task_count instance_count remaining_instances next_instance public_completed run_epoch "
        "definitions_address definitions_bytes"
    ).split()
)
ROOT_TABLES = frozenset(
    "ranges definitions instance_definitions nodes instances materializations contributions bootstrap".split()
)
EVENT_TABLES = (
    ("materializations", "graph_materialize"),
    ("contributions", "graph_contribution"),
    ("bootstrap", "graph_bootstrap"),
)


def read_graph_protocol(section):
    from .native_protocol import _array, _error, _indices, _number, _object, _record  # noqa: PLC0415

    graph = section["graph"]
    _object(graph, ROOT_NUMBERS | ROOT_TABLES, "graph")
    for field in ROOT_NUMBERS:
        _number(graph[field], f"graph.{field}")
    _array(graph["instance_definitions"], "graph.instance_definitions", graph["instance_count"])
    for table, kind, count, key, dense in (
        ("ranges", "graph_range", graph["public_task_count"], "public_task_id", True),
        ("nodes", "graph_node", section["task_count"], "task_ref", True),
        ("instances", "graph_instance", graph["instance_count"], "index", True),
        ("materializations", "graph_materialize", graph["instance_count"], "index", False),
        ("contributions", "graph_contribution", section["task_count"], "index", False),
        ("bootstrap", "graph_bootstrap", section["worker_count"], "index", False),
    ):
        _indices(graph[table], count, f"graph.{table}", key=key, dense=dense)
        for row in graph[table]:
            _record(row, kind, f"graph.{table}[{row[key]}]")
    definitions = graph["definitions"]
    if not isinstance(definitions, list):
        _error("graph.definitions must be a table")
    _indices(definitions, len(definitions), "graph.definitions", dense=True)
    for definition in definitions:
        header = {key: value for key, value in definition.items() if key != "tasks"}
        _record(header, "graph_definition", "graph.definition")
        _array(header["fanin_rows"], "graph.definition.fanin_rows", header["task_count"] + 1)
        _array(header["fanin_indices"], "graph.definition.fanin_indices", header["edge_count"])
        tasks = definition.get("tasks")
        _indices(tasks, header["task_count"], "graph.definition.tasks", key="local_id", dense=True)
        for task in tasks:
            _record(task, "graph_definition_task", "graph.definition.task")


def graph_event_records(section):
    for table, kind in EVENT_TABLES:
        for row in section["graph"][table]:
            yield row, kind, f"graph.{table}[{row['index']}]"
