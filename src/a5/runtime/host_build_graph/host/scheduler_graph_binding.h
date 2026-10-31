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

#include <cstring>
#include <optional>
#include <unordered_map>
#include <unordered_set>
#include <vector>

#include "host_build_graph/graph_image_view.h"
#include "host_build_graph/shared_memory.h"
#include "scheduler_graph_capture.h"
#include "scheduler/scheduler_graph_materialize.h"
#include "scheduler/scheduler_graph_protocol_types.h"

static_assert(sizeof(simpler::hbg::TensorData) == SCHEDULER_GRAPH_TENSOR_BYTES);
static_assert(offsetof(simpler::hbg::TensorData, owner_task_id) == SCHEDULER_GRAPH_TENSOR_OWNER_OFFSET);
static_assert(offsetof(simpler::hbg::TensorData, start_offset) == SCHEDULER_GRAPH_TENSOR_ORIGIN_OFFSET);
static_assert(offsetof(simpler::hbg::TensorData, address_space) == SCHEDULER_GRAPH_TENSOR_SPACE_OFFSET);
static_assert(offsetof(simpler::hbg::TensorData, extent_elem_cache) == SCHEDULER_GRAPH_TENSOR_EXTENT_OFFSET);
static_assert(sizeof(GraphPredicate) == SCHEDULER_GRAPH_PREDICATE_BYTES);
static_assert(offsetof(GraphPredicate, operand) == 0);
static_assert(offsetof(GraphPredicate, elem_offset) == SCHEDULER_GRAPH_PREDICATE_ELEMENT_OFFSET);
static_assert(offsetof(GraphPredicate, target) == SCHEDULER_GRAPH_PREDICATE_TARGET_OFFSET);
static_assert(offsetof(GraphPredicate, elem_size) == SCHEDULER_GRAPH_PREDICATE_WIDTH_OFFSET);
static_assert(offsetof(GraphPredicate, op) == SCHEDULER_GRAPH_PREDICATE_OP_OFFSET);
static_assert(offsetof(TaskDescriptor, packed_buffer_base) == SCHEDULER_GRAPH_PACKED_BASE_OFFSET);
static_assert(offsetof(TaskDescriptor, packed_buffer_end) == SCHEDULER_GRAPH_PACKED_END_OFFSET);
static_assert(offsetof(TaskPayload, dump_metadata) == SCHEDULER_GRAPH_DUMP_METADATA_OFFSET);

struct SchedulerGraphSubmitState {
    ActiveMask active_mask{};
    TaskAttrs task_attrs{};
    int32_t logical_block_num{1};
    int32_t total_required_subtasks{0};
    bool completed{false};
    int32_t fanin_count{0};
    SchedulerTaskShape shape{};
    SchedulerGraphResult status{SchedulerGraphResult::UNSUPPORTED_SHAPE};
    SchedulerDispatchPredicate predicate{};
};

class SchedulerGraphBinding {
public:
    static std::optional<SchedulerGraphBinding>
    prepare(SharedMemoryTaskHeader &tasks, int32_t public_count, const std::byte *section, uint64_t bytes) {
        if (public_count < 0 || section == nullptr || bytes == 0 || bytes > UINT32_MAX) return std::nullopt;
        SchedulerGraphBinding result;
        result.tasks_ = &tasks;
        result.section_ = section;
        result.bytes_ = bytes;
        result.definitions_.resize(public_count, nullptr);
        std::vector<SchedulerGraphPlanShape> shapes(public_count);
        std::unordered_set<const GraphDefinition *> checked;
        GraphImageView image{section, static_cast<uint32_t>(bytes)};
        for (int32_t parent = 0; parent < public_count; ++parent) {
            const auto &slot = tasks.get_slot_state_by_task_id(parent);
            const auto &payload = slot.to_payload();
            if (payload.fanin_count < 0 || payload.fanin_count > SCHEDULER_GRAPH_MAX_FANIN) return std::nullopt;
            const auto *fanin = payload.fanin_data();
            if (payload.fanin_count != 0 && fanin == nullptr) return std::nullopt;
            for (int32_t edge = 0; edge < payload.fanin_count; ++edge) {
                if (!scheduler_task_ref_precedes(fanin[edge], parent)) return std::nullopt;
                for (int32_t prior = 0; prior < edge; ++prior)
                    if (fanin[prior] == fanin[edge]) return std::nullopt;
            }
            result.external_fanins_ += payload.fanin_count;
            if (slot.task_kind != TaskKind::GRAPH) continue;
            GraphDefinitionValue value{};
            if (!graph_definition_decode_framed(image, slot.to_descriptor().graph_definition_offset, &value))
                return std::nullopt;
            const auto *definition = reinterpret_cast<const GraphDefinition *>(section + value.image_offset);
            if (definition->task_count <= 0 || definition->task_count > SUB_TASK_MAX_NUM ||
                payload.tensor_count != definition->boundary_tensor_count ||
                payload.scalar_count != definition->boundary_scalar_count)
                return std::nullopt;
            if (checked.insert(definition).second && !result.validate_definition(*definition)) return std::nullopt;
            result.definitions_[parent] = definition;
            shapes[parent] = {true, definition->task_count};
        }
        auto plan = SchedulerGraphPlan::build(shapes);
        if (!plan || plan->instance_count() == 0) return std::nullopt;
        result.plan_ = std::move(*plan);
        return result;
    }

    const SchedulerGraphPlan &plan() const { return plan_; }
    bool supported() const { return supported_; }

    SchedulerGraphCapture freeze_protocol(const std::vector<SchedulerTaskMetadata> &metadata) const {
        if (metadata.size() != static_cast<size_t>(plan_.node_count()))
            throw std::invalid_argument("expanded protocol graph task count mismatch");
        SchedulerGraphCapture capture;
        capture.manifest.ranges = plan_.ranges();
        capture.manifest.instance_definitions.resize(plan_.instance_count());
        std::unordered_map<const GraphDefinition *, uint32_t> distinct;
        for (size_t parent = 0; parent < definitions_.size(); ++parent) {
            const auto *definition = definitions_[parent];
            if (definition == nullptr) continue;
            const auto [found, inserted] = distinct.emplace(definition, static_cast<uint32_t>(distinct.size()));
            capture.manifest.instance_definitions[plan_.ranges()[parent].instance_index] = found->second;
            if (inserted) {
                SchedulerGraphDefinitionCapture saved;
                saved.header = *definition;
                saved.image_offset = static_cast<uint32_t>(reinterpret_cast<const std::byte *>(definition) - section_);
                saved.tasks =
                    freeze_array<SubTaskDefinition>(*definition, definition->off_sub_tasks, definition->task_count);
                saved.fanin_rows =
                    freeze_array<int32_t>(*definition, definition->off_fanin_offsets, definition->task_count + 1);
                saved.fanin_indices =
                    freeze_array<uint16_t>(*definition, definition->off_fanin_indices, definition->edge_count);
                capture.manifest.definitions.push_back(std::move(saved));
            }
        }
        capture.tasks.reserve(metadata.size());
        for (int32_t ref = 0; ref < plan_.node_count(); ++ref) {
            const auto node = plan_.node_at(ref);
            if (!node) throw std::invalid_argument("invalid expanded protocol reference");
            SchedulerProtocolTaskSpec spec{metadata[ref], {}};
            auto append = [&](int32_t source) {
                const auto dependency = plan_.dependency_ref(ref, source);
                if (!dependency) throw std::invalid_argument("invalid expanded protocol dependency");
                spec.dependencies.push_back(*dependency);
            };
            if (node->kind == SchedulerGraphNodeKind::ORDINARY || node->kind == SchedulerGraphNodeKind::ENTER) {
                const auto &payload = tasks_->get_slot_state_by_task_id(node->outer_id).to_payload();
                spec.dependencies.reserve(payload.fanin_count);
                for (int32_t edge = 0; edge < payload.fanin_count; ++edge)
                    append(payload.fanin_data()[edge]);
            } else if (node->kind == SchedulerGraphNodeKind::CHILD) {
                const auto &saved =
                    capture.manifest.definitions[capture.manifest.instance_definitions[node->instance_index]];
                const int32_t begin = saved.fanin_rows[node->child_index];
                const int32_t end = saved.fanin_rows[node->child_index + 1];
                if (begin == end) {
                    spec.dependencies.push_back(plan_.ranges()[node->outer_id].begin);
                } else {
                    spec.dependencies.reserve(end - begin);
                    for (int32_t edge = begin; edge < end; ++edge)
                        append(saved.fanin_indices[edge]);
                }
            }
            capture.tasks.push_back(std::move(spec));
        }
        return capture;
    }

    SchedulerGraphSubmitState submission(int64_t ref, const SchedulerGraphView &ordinary) const {
        SchedulerGraphSubmitState result;
        const auto planned = plan_.node_at(ref);
        if (!planned) {
            result.status = SchedulerGraphResult::INVALID_TASK_ID;
            return result;
        }
        const auto &node = *planned;
        const auto &outer = tasks_->get_slot_state_by_task_id(node.outer_id);
        if (node.kind == SchedulerGraphNodeKind::ORDINARY) {
            result.active_mask = outer.active_mask;
            result.task_attrs = outer.task_attrs;
            result.logical_block_num = outer.logical_block_num;
            result.total_required_subtasks = outer.total_required_subtasks;
            result.completed = tasks_->is_completed(node.outer_id);
            result.fanin_count = outer.to_payload().fanin_count;
            result.status = scheduler_classify_task_shape(ordinary, node.outer_id, &result.shape);
            const auto &predicate = outer.to_payload().predicate;
            result.predicate = {
                predicate.addr, predicate.target, predicate.elem_size, static_cast<uint8_t>(predicate.op)
            };
        } else if (node.kind == SchedulerGraphNodeKind::ENTER) {
            result.fanin_count = outer.to_payload().fanin_count;
        } else if (node.kind == SchedulerGraphNodeKind::CHILD) {
            const auto &definition = *definitions_[node.outer_id];
            const auto *sources =
                graph_definition_array<SubTaskDefinition>(definition, definition.off_sub_tasks, definition.task_count);
            const auto *rows =
                graph_definition_array<int32_t>(definition, definition.off_fanin_offsets, definition.task_count + 1);
            const auto &source = sources[node.child_index];
            result.active_mask = ActiveMask(source.active_mask);
            result.task_attrs = TaskAttrs(source.task_attrs);
            result.logical_block_num = source.logical_block_num;
            result.total_required_subtasks = source.total_required_subtasks;
            result.fanin_count = std::max(1, rows[node.child_index + 1] - rows[node.child_index]);
            result.shape.active_mask = source.active_mask;
            for (uint32_t lane = 0; lane < 3; ++lane)
                result.shape.kernel_ids[lane] = source.kernel_id[lane];
            result.status =
                source.active_mask == 0 ? SchedulerGraphResult::UNSUPPORTED_SHAPE : SchedulerGraphResult::OK;
        }
        if (node.kind == SchedulerGraphNodeKind::ENTER || node.kind == SchedulerGraphNodeKind::EXIT)
            for (auto &kernel : result.shape.kernel_ids)
                kernel = INVALID_KERNEL_ID;
        result.shape.task_ref = ref;
        return result;
    }

    bool reserve(AicoreSchedulerLayout *layout) const {
        if (layout == nullptr) return false;
        auto &out = layout->graph;
        out = {};
        out.instance_count = plan_.instance_count();
        out.public_task_count = plan_.ranges().size();
        out.definition_bytes = bytes_;
        out.external_fanin_count = external_fanins_;
        uint64_t cursor = layout->total_size;
        auto array = [&](uint64_t count, uint64_t stride, uint64_t alignment, uint64_t *offset) {
            uint64_t bytes = 0;
            return scheduler_layout_checked_mul(count, stride, &bytes) &&
                   scheduler_layout_reserve(&cursor, bytes, alignment, offset);
        };
        if (!array(plan_.node_count(), sizeof(SchedulerGraphNode), alignof(SchedulerGraphNode), &out.nodes_offset) ||
            !array(
                out.instance_count, sizeof(SchedulerGraphInstance), alignof(SchedulerGraphInstance),
                &out.instances_offset
            ) ||
            !array(
                out.instance_count, sizeof(SchedulerGraphInstanceControl), alignof(SchedulerGraphInstanceControl),
                &out.instance_controls_offset
            ) ||
            !array(
                1, sizeof(SchedulerGraphBuildControl), alignof(SchedulerGraphBuildControl), &out.build_control_offset
            ) ||
            !array(bytes_, 1, GRAPH_DEFINITION_OBJECT_ALIGN, &out.definitions_offset) ||
            !array(external_fanins_, sizeof(int32_t), alignof(int32_t), &out.external_fanins_offset))
            return false;
        if (layout->protocol_layout_offset != 0 &&
            (!array(
                 out.instance_count, sizeof(SchedulerGraphProtocolInstance), alignof(SchedulerGraphProtocolInstance),
                 &out.protocol_instances_offset
             ) ||
             !array(
                 plan_.node_count(), sizeof(SchedulerGraphProtocolContribution),
                 alignof(SchedulerGraphProtocolContribution), &out.protocol_contributions_offset
             ) ||
             !array(
                 layout->protocol.worker_count, sizeof(SchedulerGraphProtocolWorker),
                 alignof(SchedulerGraphProtocolWorker), &out.protocol_workers_offset
             )))
            return false;
        return scheduler_layout_checked_align(cursor, SCHEDULER_STATE_ALIGNMENT, &layout->total_size);
    }

    bool bind(
        void *host, uint64_t device, const AicoreSchedulerLayout &layout, const void *packed_image, uint64_t device_sm,
        const sm_layout::SegmentOffsets &segments, uint64_t epoch, uint64_t already_complete
    ) const {
        const auto &out = layout.graph;
        auto *nodes = scheduler_state_at<SchedulerGraphNode>(host, out.nodes_offset);
        auto *instances = scheduler_state_at<SchedulerGraphInstance>(host, out.instances_offset);
        auto *controls = scheduler_state_at<SchedulerGraphInstanceControl>(host, out.instance_controls_offset);
        auto *build = scheduler_state_at<SchedulerGraphBuildControl>(host, out.build_control_offset);
        auto *fanins = scheduler_state_at<int32_t>(host, out.external_fanins_offset);
        std::memcpy(scheduler_state_at<std::byte>(host, out.definitions_offset), section_, bytes_);
        build->remaining_instances = out.instance_count;
        build->instance_count = out.instance_count;
        build->public_task_count = out.public_task_count;
        build->public_completed = already_complete;
        build->run_epoch = epoch;
        if (layout.protocol_layout_offset != 0) {
            build->protocol_instances_address = device + out.protocol_instances_offset;
            build->protocol_contributions_address = device + out.protocol_contributions_offset;
            build->protocol_workers_address = device + out.protocol_workers_offset;
        }
        const auto *image = static_cast<const uint8_t *>(packed_image);
        const auto *packed_tasks = reinterpret_cast<const ChipTaskStorage *>(image + segments.storage);
        uint64_t external_cursor = 0;
        for (int32_t ref = 0; ref < plan_.node_count(); ++ref) {
            const auto planned = plan_.node_at(ref);
            if (!planned) return false;
            const auto &source = *planned;
            auto &node = nodes[ref];
            node = {};
            node.kind = source.kind;
            node.public_task_id = source.task_id;
            const auto &outer = packed_tasks[source.outer_id];
            if (source.kind == SchedulerGraphNodeKind::ORDINARY)
                node.storage_address =
                    device_sm + segments.storage + static_cast<uint64_t>(source.outer_id) * sizeof(ChipTaskStorage);
            if (source.instance_index >= 0)
                node.instance_control_address =
                    device + out.instance_controls_offset +
                    static_cast<uint64_t>(source.instance_index) * sizeof(SchedulerGraphInstanceControl);
            if (source.kind == SchedulerGraphNodeKind::ORDINARY || source.kind == SchedulerGraphNodeKind::ENTER) {
                const auto &payload = tasks_->get_slot_state_by_task_id(source.outer_id).to_payload();
                node.fanin_count = payload.fanin_count;
                if (payload.fanin_count != 0)
                    node.fanin_address = device + out.external_fanins_offset + external_cursor * sizeof(int32_t);
                for (int32_t i = 0; i < payload.fanin_count; ++i) {
                    const auto dependency = plan_.dependency_ref(ref, payload.fanin_data()[i]);
                    if (!dependency) return false;
                    fanins[external_cursor++] = *dependency;
                }
            }
            if (source.kind == SchedulerGraphNodeKind::ENTER) {
                const auto &definition = *definitions_[source.outer_id];
                const auto range = plan_.ranges()[source.outer_id];
                GraphExecutionStorageLayout storage{};
                if (!graph_execution_storage_layout(
                        definition.task_count, definition.tensor_arg_count, definition.scalar_arg_count,
                        definition.edge_count, &storage
                    ))
                    return false;
                auto &control = controls[source.instance_index];
                control.remaining_children = definition.task_count;
                control.run_epoch = epoch;
                control.enter_ref = range.begin;
                control.child_begin = range.begin + 1;
                control.child_count = definition.task_count;
                control.exit_ref = range.end - 1;
                control.instance_index = static_cast<uint32_t>(source.instance_index);
                auto &instance = instances[source.instance_index];
                instance.definition_section_address = device + out.definitions_offset;
                instance.definition_section_bytes = bytes_;
                instance.definition_key = definition.full_key;
                instance.definition_offset = outer.task.graph_definition_offset;
                instance.parent_id = source.outer_id;
                const uint64_t heap = reinterpret_cast<uint64_t>(outer.task.packed_buffer_base);
                const uint64_t end = reinterpret_cast<uint64_t>(outer.task.packed_buffer_end);
                if (end < heap || definition.required_heap > end - heap ||
                    storage.total_bytes > end - heap - definition.required_heap)
                    return false;
                const uint64_t tail = heap + definition.required_heap;
                instance.task_storage_address = tail + storage.tasks_offset;
                instance.task_storage_bytes = static_cast<uint64_t>(definition.task_count) * sizeof(ChipTaskStorage);
                instance.tensor_pool_address = tail + storage.tensors_offset;
                instance.scalar_pool_address = tail + storage.scalars_offset;
                instance.heap_address = heap;
                instance.heap_bytes = definition.required_heap;
                instance.control_address = node.instance_control_address;
                instance.boundary_tensor_count = definition.boundary_tensor_count;
                instance.boundary_scalar_count = definition.boundary_scalar_count;
                instance.tensor_arg_count = definition.tensor_arg_count;
                instance.scalar_arg_count = definition.scalar_arg_count;
                if (instance.boundary_tensor_count != 0)
                    instance.boundary_tensors_address =
                        device_sm + (reinterpret_cast<const uint8_t *>(outer.payload.tensor_data()) - image);
                if (instance.boundary_scalar_count != 0)
                    instance.boundary_scalars_address =
                        device_sm + (reinterpret_cast<const uint8_t *>(outer.payload.scalar_data()) - image);
            } else if (source.kind == SchedulerGraphNodeKind::CHILD) {
                const auto &definition = *definitions_[source.outer_id];
                const auto *rows = graph_definition_array<int32_t>(
                    definition, definition.off_fanin_offsets, definition.task_count + 1
                );
                const auto range = plan_.ranges()[source.outer_id];
                const auto &instance = instances[source.instance_index];
                node.storage_address =
                    instance.task_storage_address + static_cast<uint64_t>(source.child_index) * sizeof(ChipTaskStorage);
                const int32_t begin = rows[source.child_index];
                const int32_t count = rows[source.child_index + 1] - begin;
                node.fanin_count = std::max(1, count);
                node.fanin_kind =
                    count == 0 ? SchedulerGraphFaninKind::ENTER_GATE : SchedulerGraphFaninKind::DEFINITION_CSR;
                node.fanin_ref_base = count == 0 ? range.begin : range.begin + 1;
                if (count != 0)
                    node.fanin_address = instance.definition_section_address + instance.definition_offset +
                                         definition.off_fanin_indices + static_cast<uint64_t>(begin) * sizeof(uint16_t);
            }
        }
        return external_cursor == external_fanins_;
    }

private:
    template <class T>
    static std::vector<T> freeze_array(const GraphDefinition &definition, uint32_t offset, int32_t count) {
        if (count == 0) return {};
        const auto *array = graph_definition_array<T>(definition, offset, count);
        if (array == nullptr) throw std::invalid_argument("invalid frozen Graph Definition section");
        return {array, array + count};
    }

    static bool validate_predicate(const GraphDefinition &definition, int32_t child, uint16_t slot) {
        const auto *predicates =
            graph_definition_array<GraphPredicate>(definition, definition.off_predicates, definition.predicate_count);
        if (predicates == nullptr || slot == 0 || slot > definition.predicate_count) return false;
        const auto &predicate = predicates[slot - 1];
        const uint64_t width = get_element_size(predicate.operand.dtype);
        if ((width != 1 && width != 2 && width != 4 && width != 8) || predicate.elem_size != width ||
            predicate.op < static_cast<uint8_t>(PredicateOp::EQ) ||
            predicate.op > static_cast<uint8_t>(PredicateOp::LE) ||
            predicate.elem_offset >= predicate.operand.extent_elem_cache)
            return false;
        const auto owner = predicate.operand.owner_task_id;
        if (owner.space() == TaskId::Space::PARAM)
            return owner.local_id() >= 0 && owner.local_id() < definition.boundary_tensor_count;
        return owner.space() == TaskId::Space::SUB_TASK && owner.local_id() >= 0 && owner.local_id() < child;
    }

    bool validate_definition(const GraphDefinition &definition) {
        GraphExecutionStorageLayout storage{};
        if (!graph_execution_storage_layout(
                definition.task_count, definition.tensor_arg_count, definition.scalar_arg_count, definition.edge_count,
                &storage
            ) ||
            storage.total_bytes != definition.execution_storage_bytes)
            return false;
        const auto *tasks =
            graph_definition_array<SubTaskDefinition>(definition, definition.off_sub_tasks, definition.task_count);
        const auto *rows =
            graph_definition_array<int32_t>(definition, definition.off_fanin_offsets, definition.task_count + 1);
        const auto *edges =
            graph_definition_array<uint16_t>(definition, definition.off_fanin_indices, definition.edge_count);
        if (tasks == nullptr || rows == nullptr || (definition.edge_count != 0 && edges == nullptr) || rows[0] != 0 ||
            rows[definition.task_count] != definition.edge_count)
            return false;
        std::vector<int32_t> seen(definition.task_count, -1);
        for (int32_t child = 0; child < definition.task_count; ++child) {
            if (rows[child] < 0 || rows[child] > rows[child + 1] || rows[child + 1] > definition.edge_count)
                return false;
            for (int32_t edge = rows[child]; edge < rows[child + 1]; ++edge) {
                if (edges[edge] >= child || seen[edges[edge]] == child) return false;
                seen[edges[edge]] = child;
            }
            const auto &task = tasks[child];
            if (task.active_mask > 7 || task.logical_block_num <= 0 || task.reserved != 0 || task.tensor_count < 0 ||
                task.tensor_count > MAX_TENSOR_ARGS || task.scalar_count < 0 || task.scalar_count > MAX_SCALAR_ARGS)
                return false;
            for (uint32_t lane = 0; lane < 3; ++lane)
                if (task.kernel_id[lane] < INVALID_KERNEL_ID ||
                    ((task.active_mask & (1U << lane)) != 0) != (task.kernel_id[lane] != INVALID_KERNEL_ID))
                    return false;
            const TaskAttrs attrs(task.task_attrs);
            if (attrs.has_predicate() != (task.predicate_slot != 0) || definition.predicate_count < 0 ||
                task.predicate_slot > definition.predicate_count)
                return false;
            if (task.predicate_slot != 0) {
                if (!validate_predicate(definition, child, task.predicate_slot)) return false;
                if (task.logical_block_num != 1 || __builtin_popcount(task.active_mask) != 1) supported_ = false;
            }
        }
        return true;
    }

    SharedMemoryTaskHeader *tasks_{nullptr};
    const std::byte *section_{nullptr};
    uint64_t bytes_{0};
    uint64_t external_fanins_{0};
    bool supported_{true};
    SchedulerGraphPlan plan_;
    std::vector<const GraphDefinition *> definitions_;
};
