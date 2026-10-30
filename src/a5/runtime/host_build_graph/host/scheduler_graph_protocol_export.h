/*
 * Copyright (c) PyPTO Contributors.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 * -----------------------------------------------------------------------------------------------------------
 */

#pragma once

#include "scheduler_graph_capture.h"
#include "scheduler_protocol_export_common.h"
#include "scheduler/scheduler_graph_protocol_types.h"

namespace scheduler_protocol_export {

template <class T>
inline const T *graph_array(
    const Image &image, uint64_t address, uint64_t device, uint64_t region, uint64_t region_bytes, uint64_t count
) {
    image.array<uint8_t>(region, region_bytes);
    if (address < device || address - device < region || address - device - region > region_bytes ||
        count > (region_bytes - (address - device - region)) / sizeof(T))
        throw std::invalid_argument("Graph protocol address outside its Host planned region");
    return image.array<T>(address - device, count);
}

inline void graph_definitions(JsonObject &root, const SchedulerGraphManifest &manifest) {
    auto &out = root.field("definitions");
    out << '[';
    for (size_t index = 0; index < manifest.definitions.size(); ++index) {
        if (index != 0) out << ',';
        const auto &saved = manifest.definitions[index];
        JsonObject row(out);
        row.number("index", index);
        row.number("full_key", saved.header.full_key);
        row.number("image_offset", saved.image_offset);
        row.number("image_bytes", saved.header.total_bytes);
        row.number("fanin_indices_offset", saved.header.off_fanin_indices);
        row.number("task_count", saved.header.task_count);
        row.number("edge_count", saved.header.edge_count);
        row.numbers("fanin_rows", saved.fanin_rows.data(), saved.fanin_rows.size());
        row.numbers("fanin_indices", saved.fanin_indices.data(), saved.fanin_indices.size());
        row.field("tasks") << '[';
        for (size_t local = 0; local < saved.tasks.size(); ++local) {
            if (local != 0) out << ',';
            const auto &source = saved.tasks[local];
            JsonObject task(out);
            task.number("local_id", local);
            task.number("active_mask", source.active_mask);
            task.number("task_attrs", source.task_attrs);
            task.number("block_num", source.logical_block_num);
            task.number("predicate_slot", source.predicate_slot);
            task.numbers("kernel_ids", source.kernel_id, 3);
        }
        out << ']';
    }
    out << ']';
}

inline void graph_nodes(JsonObject &root, const Image &image, const AicoreSchedulerLayout &layout, uint64_t device) {
    const auto &plan = layout.graph;
    const auto *nodes = image.array<SchedulerGraphNode>(plan.nodes_offset, layout.task_count);
    auto &out = root.field("nodes");
    out << '[';
    for (uint64_t ref = 0; ref < layout.task_count; ++ref) {
        const auto &node = nodes[ref];
        if (node.fanin_count < 0 || node.fanin_count > SCHEDULER_GRAPH_MAX_BODY_FANIN)
            throw std::invalid_argument("Graph protocol fanin count out of bounds");
        std::vector<int64_t> dependencies;
        dependencies.reserve(node.fanin_count);
        if (node.fanin_count != 0) {
            if (node.fanin_kind == SchedulerGraphFaninKind::DENSE) {
                const auto *edges = graph_array<int32_t>(
                    image, node.fanin_address, device, plan.external_fanins_offset,
                    plan.external_fanin_count * sizeof(int32_t), node.fanin_count
                );
                dependencies.assign(edges, edges + node.fanin_count);
            } else if (node.fanin_kind == SchedulerGraphFaninKind::DEFINITION_CSR) {
                const auto *edges = graph_array<uint16_t>(
                    image, node.fanin_address, device, plan.definitions_offset, plan.definition_bytes, node.fanin_count
                );
                for (int32_t edge = 0; edge < node.fanin_count; ++edge)
                    dependencies.push_back(static_cast<int64_t>(node.fanin_ref_base) + edges[edge]);
            } else if (node.fanin_kind == SchedulerGraphFaninKind::ENTER_GATE && node.fanin_count == 1) {
                dependencies.push_back(node.fanin_ref_base);
            } else {
                throw std::invalid_argument("invalid Graph protocol fanin encoding");
            }
        }
        int64_t instance_index = -1;
        if (node.instance_control_address != 0) {
            graph_array<SchedulerGraphInstanceControl>(
                image, node.instance_control_address, device, plan.instance_controls_offset,
                plan.instance_count * sizeof(SchedulerGraphInstanceControl), 1
            );
            instance_index = static_cast<int64_t>(
                (node.instance_control_address - device - plan.instance_controls_offset) /
                sizeof(SchedulerGraphInstanceControl)
            );
        }
        if (ref != 0) out << ',';
        JsonObject row(out);
        const auto space = TaskId::space_of(node.public_task_id);
        row.number("task_ref", ref);
        row.number("kind", static_cast<uint32_t>(node.kind));
        row.number("task_id", TaskId::to_uint64(node.public_task_id));
        row.number("public_space", static_cast<uint32_t>(space));
        row.number(
            "public_task_id", space == TaskId::Space::SUB_TASK ? TaskId::parent_id_of(node.public_task_id) :
                                                                 TaskId::local_id_of(node.public_task_id)
        );
        row.number(
            "sub_task_id", space == TaskId::Space::SUB_TASK ?
                               static_cast<uint32_t>(TaskId::local_id_of(node.public_task_id)) :
                               UINT32_MAX
        );
        row.number("instance_index", instance_index);
        row.number("storage_address", node.storage_address);
        row.number("fanin_address", node.fanin_address);
        row.number("fanin_kind", static_cast<uint32_t>(node.fanin_kind));
        row.number("fanin_ref_base", node.fanin_ref_base);
        row.numbers("dependencies", dependencies.data(), dependencies.size());
    }
    out << ']';
}

inline void graph_instances(JsonObject &root, const Image &image, const AicoreSchedulerLayout &layout) {
    const auto &plan = layout.graph;
    const auto *instances = image.array<SchedulerGraphInstance>(plan.instances_offset, plan.instance_count);
    const auto *controls =
        image.array<SchedulerGraphInstanceControl>(plan.instance_controls_offset, plan.instance_count);
    auto &out = root.field("instances");
    out << '[';
    for (uint64_t index = 0; index < plan.instance_count; ++index) {
        if (index != 0) out << ',';
        const auto &source = instances[index];
        const auto &control = controls[index];
        JsonObject row(out);
        row.number("index", index);
        row.number("instance_index", control.instance_index);
        row.number("parent_id", source.parent_id);
        row.number("definition_key", source.definition_key);
        row.number("definition_address", source.definition_section_address);
        row.number("definition_bytes", source.definition_section_bytes);
        row.number("definition_offset", source.definition_offset);
        row.number("task_storage_address", source.task_storage_address);
        row.number("task_storage_bytes", source.task_storage_bytes);
        row.number("remaining_children", control.remaining_children);
        row.number("materialize_owner", control.materialize_owner);
        row.number("materialize_state", control.materialize_state);
        row.number("run_epoch", control.run_epoch);
        row.number("enter_ref", control.enter_ref);
        row.number("child_begin", control.child_begin);
        row.number("child_count", control.child_count);
        row.number("exit_ref", control.exit_ref);
    }
    out << ']';
}

inline void graph_protocol(
    JsonObject &root, const Image &image, const AicoreSchedulerLayout &layout, const SchedulerGraphManifest &manifest,
    uint64_t device
) {
    const auto &plan = layout.graph;
    if (device == 0 || device > UINT64_MAX - layout.total_size || plan.instance_count == 0 ||
        plan.protocol_instances_offset == 0 || plan.protocol_contributions_offset == 0 ||
        plan.protocol_workers_offset == 0 || manifest.ranges.size() != plan.public_task_count ||
        manifest.instance_definitions.size() != plan.instance_count)
        throw std::invalid_argument("Graph protocol manifest or storage mismatch");
    const auto &build = *image.array<SchedulerGraphBuildControl>(plan.build_control_offset, 1);
    if (build.protocol_instances_address != device + plan.protocol_instances_offset ||
        build.protocol_contributions_address != device + plan.protocol_contributions_offset ||
        build.protocol_workers_address != device + plan.protocol_workers_offset)
        throw std::invalid_argument("Graph protocol addresses disagree with the Host layout");
    JsonObject graph(root.field("graph"));
    graph.number("public_task_count", build.public_task_count);
    graph.number("instance_count", build.instance_count);
    graph.number("remaining_instances", build.remaining_instances);
    graph.number("next_instance", build.next_instance);
    graph.number("public_completed", build.public_completed);
    graph.number("run_epoch", build.run_epoch);
    graph.number("definitions_address", device + plan.definitions_offset);
    graph.number("definitions_bytes", plan.definition_bytes);
    auto &ranges = graph.field("ranges");
    ranges << '[';
    for (size_t parent = 0; parent < manifest.ranges.size(); ++parent) {
        if (parent != 0) ranges << ',';
        const auto &range = manifest.ranges[parent];
        JsonObject row(ranges);
        row.number("public_task_id", parent);
        row.number("begin", range.begin);
        row.number("end", range.end);
        row.number("children", range.children);
        row.number("instance_index", range.instance_index);
    }
    ranges << ']';
    graph.numbers("instance_definitions", manifest.instance_definitions.data(), manifest.instance_definitions.size());
    graph_definitions(graph, manifest);
    graph_nodes(graph, image, layout, device);
    graph_instances(graph, image, layout);
    sparse(
        graph, "materializations",
        image.array<SchedulerGraphProtocolInstance>(plan.protocol_instances_offset, plan.instance_count),
        plan.instance_count, [](JsonObject &row, const auto &record) {
            row.stamp("materialize_before", record.materialize_before);
            row.stamp("materialize_after", record.materialize_after);
            row.stamp("publish_before", record.publish_before);
            row.stamp("publish_after", record.publish_after);
            row.number("run_epoch", record.run_epoch);
            row.number("definition_key", record.definition_key);
            row.number("remaining_before", record.remaining_before);
            row.number("instance_index", record.instance_index);
            row.number("parent_id", record.parent_id);
            row.number("worker_index", record.worker_index);
            row.number("child_count", record.child_count);
            row.number("status", record.status);
        }
    );
    sparse(
        graph, "contributions",
        image.array<SchedulerGraphProtocolContribution>(plan.protocol_contributions_offset, layout.task_count),
        layout.task_count, [](JsonObject &row, const auto &record) {
            row.stamp("before", record.before);
            row.stamp("after", record.after);
            row.number("run_epoch", record.run_epoch);
            row.number("task_ref", record.task_ref);
            row.number("remaining_before", record.remaining_before);
            row.number("worker_index", record.worker_index);
            row.number("instance_index", record.instance_index);
        }
    );
    sparse(
        graph, "bootstrap",
        image.array<SchedulerGraphProtocolWorker>(plan.protocol_workers_offset, layout.protocol.worker_count),
        layout.protocol.worker_count, [](JsonObject &row, const auto &record) {
            row.stamp("before", record.before);
            row.stamp("after", record.after);
            row.number("run_epoch", record.run_epoch);
            row.number("remaining_observed", record.remaining_observed);
            row.number("poll_count", record.poll_count);
            row.number("worker_index", record.worker_index);
            row.number("instance_count", record.instance_count);
            row.number("status", record.status);
        }
    );
}

}  // namespace scheduler_protocol_export
