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

#include "scheduler_graph_definition.h"
#include "scheduler_memory.h"

// The Host binding checks these against the shared TensorData/TaskPayload
// declarations. Scalar operates on their existing wire layout, not CPU objects.
inline constexpr uint32_t SCHEDULER_GRAPH_TENSOR_BYTES = 96;
inline constexpr uint32_t SCHEDULER_GRAPH_TENSOR_OWNER_OFFSET = 16;
inline constexpr uint32_t SCHEDULER_GRAPH_TENSOR_ORIGIN_OFFSET = 24;
inline constexpr uint32_t SCHEDULER_GRAPH_TENSOR_SPACE_OFFSET = 40;
inline constexpr uint32_t SCHEDULER_GRAPH_TENSOR_EXTENT_OFFSET = 64;
inline constexpr uint32_t SCHEDULER_GRAPH_PREDICATE_BYTES = 120;
inline constexpr uint32_t SCHEDULER_GRAPH_PREDICATE_ELEMENT_OFFSET = 96;
inline constexpr uint32_t SCHEDULER_GRAPH_PREDICATE_TARGET_OFFSET = 104;
inline constexpr uint32_t SCHEDULER_GRAPH_PREDICATE_WIDTH_OFFSET = 112;
inline constexpr uint32_t SCHEDULER_GRAPH_PREDICATE_OP_OFFSET = 113;
inline constexpr uint32_t SCHEDULER_GRAPH_PACKED_BASE_OFFSET = 24;
inline constexpr uint32_t SCHEDULER_GRAPH_PACKED_END_OFFSET = 32;
inline constexpr uint32_t SCHEDULER_GRAPH_DUMP_METADATA_OFFSET = 152;

inline __aicore__ bool scheduler_graph_bind_region(__gm__ uint8_t *field, uint64_t target, int32_t count) {
    int32_t delta = 0;
    if (count != 0) {
        const uint64_t address = reinterpret_cast<uint64_t>(field);
        if (target == 0) return false;
        if (target >= address) {
            if (target - address > INT32_MAX) return false;
            delta = static_cast<int32_t>(target - address);
        } else {
            if (address - target > UINT64_C(2147483648)) return false;
            delta = static_cast<int32_t>(-static_cast<int64_t>(address - target));
        }
        if (delta == 0) return false;
    }
    *reinterpret_cast<__gm__ int32_t *>(field) = delta;
    return true;
}

struct SchedulerGraphTensorBinding {
    uint64_t address;
    uint64_t size;
    uint64_t start;
    uint8_t address_space;
};

inline __aicore__ bool scheduler_graph_tensor_binding(
    const __gm__ SchedulerGraphInstance &instance, const __gm__ uint8_t *record, SchedulerGraphTensorBinding *out
) {
    const auto *source_words = reinterpret_cast<const __gm__ uint64_t *>(record);
    out->address = source_words[0];
    out->size = source_words[1];
    out->start = *reinterpret_cast<const __gm__ uint64_t *>(record + SCHEDULER_GRAPH_TENSOR_ORIGIN_OFFSET);
    out->address_space = record[SCHEDULER_GRAPH_TENSOR_SPACE_OFFSET];
    const auto *owner = reinterpret_cast<const __gm__ TaskId *>(record + SCHEDULER_GRAPH_TENSOR_OWNER_OFFSET);
    const auto space = TaskId::space_of(*owner);
    if (space == TaskId::Space::PARAM) {
        const int32_t param = TaskId::local_id_of(*owner);
        if (param < 0 || param >= instance.boundary_tensor_count || instance.boundary_tensors_address == 0)
            return false;
        const auto *boundary = reinterpret_cast<const __gm__ uint8_t *>(instance.boundary_tensors_address) +
                               static_cast<uint64_t>(param) * TASKPAYLOAD_TENSOR_STRIDE;
        const auto *boundary_words = reinterpret_cast<const __gm__ uint64_t *>(boundary);
        out->address = boundary_words[0];
        out->size = boundary_words[1];
        const uint64_t start =
            *reinterpret_cast<const __gm__ uint64_t *>(boundary + SCHEDULER_GRAPH_TENSOR_ORIGIN_OFFSET);
        if (out->start > UINT64_MAX - start) return false;
        out->start += start;
        out->address_space = boundary[SCHEDULER_GRAPH_TENSOR_SPACE_OFFSET];
        return true;
    }
    if (space != TaskId::Space::SUB_TASK || out->address > instance.heap_bytes ||
        out->size > instance.heap_bytes - out->address || instance.heap_address > UINT64_MAX - out->address)
        return false;
    out->address += instance.heap_address;
    return true;
}

inline __aicore__ bool scheduler_graph_rebind_tensor(
    const __gm__ SchedulerGraphInstance &instance, const __gm__ uint8_t *record, __gm__ uint8_t *target
) {
    SchedulerGraphTensorBinding binding{};
    if (!scheduler_graph_tensor_binding(instance, record, &binding)) return false;
    const auto *source_words = reinterpret_cast<const __gm__ uint64_t *>(record);
    auto *target_words = reinterpret_cast<__gm__ uint64_t *>(target);
    for (uint32_t word = 0; word < TASKPAYLOAD_TENSOR_STRIDE / sizeof(uint64_t); ++word)
        target_words[word] = word < SCHEDULER_GRAPH_TENSOR_BYTES / sizeof(uint64_t) ? source_words[word] : 0;
    target_words[0] = binding.address;
    target_words[1] = binding.size;
    *reinterpret_cast<__gm__ uint64_t *>(target + SCHEDULER_GRAPH_TENSOR_ORIGIN_OFFSET) = binding.start;
    target[SCHEDULER_GRAPH_TENSOR_SPACE_OFFSET] = binding.address_space;
    return true;
}

inline __aicore__ bool scheduler_graph_materialize_predicate(
    const __gm__ SchedulerGraphInstance &instance, const SchedulerGraphDefinitionView &view, uint16_t slot,
    int32_t child, __gm__ SchedulerDispatchPredicate *target
) {
    const auto *definition = reinterpret_cast<const __gm__ GraphDefinition *>(view.image_address);
    if (slot == 0 || definition->predicate_count < 0 || slot > definition->predicate_count) return false;
    const uint64_t offset = static_cast<uint64_t>(definition->off_predicates) +
                            static_cast<uint64_t>(slot - 1) * SCHEDULER_GRAPH_PREDICATE_BYTES;
    if (offset > UINT32_MAX) return false;
    const auto *words = scheduler_graph_definition_array<uint64_t>(
        view, static_cast<uint32_t>(offset), SCHEDULER_GRAPH_PREDICATE_BYTES / sizeof(uint64_t)
    );
    if (words == nullptr) return false;
    const auto *record = reinterpret_cast<const __gm__ uint8_t *>(words);
    const auto *owner = reinterpret_cast<const __gm__ TaskId *>(record + SCHEDULER_GRAPH_TENSOR_OWNER_OFFSET);
    if (TaskId::space_of(*owner) == TaskId::Space::SUB_TASK &&
        (TaskId::local_id_of(*owner) < 0 || TaskId::local_id_of(*owner) >= child))
        return false;
    const uint64_t element =
        *reinterpret_cast<const __gm__ uint64_t *>(record + SCHEDULER_GRAPH_PREDICATE_ELEMENT_OFFSET);
    const uint64_t extent = *reinterpret_cast<const __gm__ uint64_t *>(record + SCHEDULER_GRAPH_TENSOR_EXTENT_OFFSET);
    const uint8_t width = record[SCHEDULER_GRAPH_PREDICATE_WIDTH_OFFSET];
    const uint8_t op = record[SCHEDULER_GRAPH_PREDICATE_OP_OFFSET];
    SchedulerGraphTensorBinding binding{};
    if ((width != 1 && width != 2 && width != 4 && width != 8) || op == 0 || op > 6 || element >= extent ||
        !scheduler_graph_tensor_binding(instance, record, &binding) || binding.start > UINT64_MAX - element ||
        binding.size < width || binding.start + element > (binding.size - width) / width)
        return false;
    const uint64_t byte_offset = (binding.start + element) * width;
    if (binding.address > UINT64_MAX - byte_offset ||
        !scheduler_dispatch_predicate_metadata_valid(binding.address + byte_offset, width, op))
        return false;
    target->addr = binding.address + byte_offset;
    target->target = *reinterpret_cast<const __gm__ int64_t *>(record + SCHEDULER_GRAPH_PREDICATE_TARGET_OFFSET);
    target->elem_size = width;
    target->op = op;
    return true;
}

inline __aicore__ void scheduler_graph_writeback_range(uint64_t address, uint64_t bytes) {
    for (uint64_t offset = 0; offset < bytes; offset += 64)
        scheduler_writeback_cache_line(reinterpret_cast<__gm__ uint8_t *>(address) + offset);
}

inline __aicore__ SchedulerGraphResult
scheduler_graph_materialize(const __gm__ SchedulerGraphInstance &instance, const SchedulerGraphView &graph) {
    SchedulerGraphDefinitionView view{};
    if (!scheduler_graph_definition_open(
            instance.definition_section_address, instance.definition_section_bytes, instance.definition_offset,
            instance.definition_key, &view
        ) ||
        instance.control_address == 0 || instance.control_address % alignof(SchedulerGraphInstanceControl) != 0)
        return SchedulerGraphResult::INVALID_ARGUMENTS;
    const auto *definition = reinterpret_cast<const __gm__ GraphDefinition *>(view.image_address);
    const auto *control = reinterpret_cast<const __gm__ SchedulerGraphInstanceControl *>(instance.control_address);
    const int32_t tasks = definition->task_count;
    if (tasks <= 0 || tasks > SUB_TASK_MAX_NUM || tasks != control->child_count || control->child_begin < 0 ||
        static_cast<uint64_t>(control->child_begin) + tasks > graph.task_count || instance.reserved != 0 ||
        instance.task_storage_address == 0 || instance.task_storage_address % 64 != 0 ||
        instance.task_storage_bytes < static_cast<uint64_t>(tasks) * SCHEDULER_GRAPH_TASK_STORAGE_STRIDE ||
        instance.boundary_tensor_count != definition->boundary_tensor_count ||
        instance.boundary_scalar_count != definition->boundary_scalar_count ||
        instance.tensor_arg_count != definition->tensor_arg_count ||
        instance.scalar_arg_count != definition->scalar_arg_count || instance.tensor_arg_count < 0 ||
        instance.tensor_arg_count > SUB_TASK_MAX_NUM * MAX_TENSOR_ARGS || instance.scalar_arg_count < 0 ||
        instance.scalar_arg_count > SUB_TASK_MAX_NUM * MAX_SCALAR_ARGS ||
        definition->required_heap > instance.heap_bytes || instance.heap_address > UINT64_MAX - instance.heap_bytes ||
        (instance.tensor_arg_count != 0 &&
         (instance.tensor_pool_address == 0 || instance.tensor_pool_address % 64 != 0)) ||
        (instance.scalar_arg_count != 0 &&
         (instance.scalar_pool_address == 0 || instance.scalar_pool_address % 8 != 0)))
        return SchedulerGraphResult::INVALID_ARGUMENTS;
    for (int32_t child = 0; child < tasks; ++child) {
        SubTaskDefinition source{};
        uint64_t output_offset = 0;
        if (!scheduler_graph_definition_read(view, definition->off_sub_tasks, tasks, child, &source) ||
            !scheduler_graph_definition_read(view, definition->off_sub_task_offsets, tasks, child, &output_offset) ||
            source.tensor_count < 0 || source.tensor_count > MAX_TENSOR_ARGS || source.scalar_count < 0 ||
            source.scalar_count > MAX_SCALAR_ARGS || source.tensor_offset < 0 || source.scalar_offset < 0 ||
            source.tensor_offset > instance.tensor_arg_count ||
            source.tensor_count > instance.tensor_arg_count - source.tensor_offset ||
            source.scalar_offset > instance.scalar_arg_count ||
            source.scalar_count > instance.scalar_arg_count - source.scalar_offset || source.total_output_size < 0)
            return SchedulerGraphResult::INVALID_ARGUMENTS;
        const uint64_t output_bytes = (static_cast<uint64_t>(source.total_output_size) + 63) & ~UINT64_C(63);
        if (output_offset > instance.heap_bytes || output_bytes > instance.heap_bytes - output_offset)
            return SchedulerGraphResult::INVALID_ARGUMENTS;
        const int64_t ref = static_cast<int64_t>(control->child_begin) + child;
        const auto *node = scheduler_graph_node_at(graph, ref);
        const uint64_t storage_address =
            instance.task_storage_address + static_cast<uint64_t>(child) * SCHEDULER_GRAPH_TASK_STORAGE_STRIDE;
        if (node == nullptr || node->kind != SchedulerGraphNodeKind::CHILD || node->storage_address != storage_address)
            return SchedulerGraphResult::INVALID_ARGUMENTS;
        auto *storage = reinterpret_cast<__gm__ uint8_t *>(storage_address);
        auto *words = reinterpret_cast<__gm__ uint64_t *>(storage);
        for (uint32_t word = 0; word < SCHEDULER_GRAPH_TASK_STORAGE_STRIDE / sizeof(uint64_t); ++word)
            words[word] = 0;
        words[0] = TaskId::to_uint64(node->public_task_id);
        auto *kernels = reinterpret_cast<__gm__ int32_t *>(storage + SCHEDULER_GRAPH_KERNEL_IDS_OFFSET);
        for (uint32_t lane = 0; lane < 3; ++lane)
            kernels[lane] = source.kernel_id[lane];
        *reinterpret_cast<__gm__ uint64_t *>(storage + SCHEDULER_GRAPH_PACKED_BASE_OFFSET) =
            instance.heap_address + output_offset;
        *reinterpret_cast<__gm__ uint64_t *>(storage + SCHEDULER_GRAPH_PACKED_END_OFFSET) =
            instance.heap_address + output_offset + output_bytes;
        auto *payload = storage + SCHEDULER_GRAPH_PAYLOAD_OFFSET;
        if (source.predicate_slot != 0 &&
            !scheduler_graph_materialize_predicate(
                instance, view, source.predicate_slot, child,
                reinterpret_cast<__gm__ SchedulerDispatchPredicate *>(payload + SCHEDULER_GRAPH_PREDICATE_OFFSET)
            ))
            return SchedulerGraphResult::INVALID_ARGUMENTS;
        *reinterpret_cast<__gm__ int32_t *>(payload + TASKPAYLOAD_TENSOR_COUNT_OFFSET) = source.tensor_count;
        *reinterpret_cast<__gm__ int32_t *>(payload + TASKPAYLOAD_SCALAR_COUNT_OFFSET) = source.scalar_count;
        const uint64_t tensors =
            instance.tensor_pool_address + static_cast<uint64_t>(source.tensor_offset) * TASKPAYLOAD_TENSOR_STRIDE;
        const uint64_t scalars =
            instance.scalar_pool_address + static_cast<uint64_t>(source.scalar_offset) * sizeof(uint64_t);
        if (!scheduler_graph_bind_region(payload + TASKPAYLOAD_TENSORS_DELTA_OFFSET, tensors, source.tensor_count) ||
            !scheduler_graph_bind_region(payload + TASKPAYLOAD_SCALARS_DELTA_OFFSET, scalars, source.scalar_count))
            return SchedulerGraphResult::INVALID_ARGUMENTS;
        for (int32_t index = 0; index < source.tensor_count; ++index) {
            const uint64_t offset = static_cast<uint64_t>(definition->off_tensors) +
                                    static_cast<uint64_t>(source.tensor_offset + index) * SCHEDULER_GRAPH_TENSOR_BYTES;
            if (offset > UINT32_MAX) return SchedulerGraphResult::INVALID_ARGUMENTS;
            const auto *record = scheduler_graph_definition_array<uint64_t>(
                view, static_cast<uint32_t>(offset), SCHEDULER_GRAPH_TENSOR_BYTES / sizeof(uint64_t)
            );
            auto *target =
                reinterpret_cast<__gm__ uint8_t *>(tensors) + static_cast<uint64_t>(index) * TASKPAYLOAD_TENSOR_STRIDE;
            if (record == nullptr ||
                !scheduler_graph_rebind_tensor(instance, reinterpret_cast<const __gm__ uint8_t *>(record), target))
                return SchedulerGraphResult::INVALID_ARGUMENTS;
        }
        for (int32_t index = 0; index < source.scalar_count; ++index) {
            GraphScalarInheritance inheritance{};
            const int32_t element = source.scalar_offset + index;
            uint64_t value = 0;
            if (!scheduler_graph_definition_read(
                    view, definition->off_scalar_inheritance, definition->scalar_arg_count, element, &inheritance
                ))
                return SchedulerGraphResult::INVALID_ARGUMENTS;
            if (inheritance.inherited()) {
                if (inheritance.boundary_index() >= instance.boundary_scalar_count ||
                    instance.boundary_scalars_address == 0)
                    return SchedulerGraphResult::INVALID_ARGUMENTS;
                value = reinterpret_cast<const __gm__ uint64_t *>(
                    instance.boundary_scalars_address
                )[inheritance.boundary_index()];
            } else if (!scheduler_graph_definition_read(
                           view, definition->off_scalars, definition->scalar_arg_count, element, &value
                       )) {
                return SchedulerGraphResult::INVALID_ARGUMENTS;
            }
            reinterpret_cast<__gm__ uint64_t *>(scalars)[index] = value;
        }
        const auto *dump = reinterpret_cast<const uint8_t *>(&source.dump_metadata);
        for (uint32_t byte = 0; byte < sizeof(source.dump_metadata); ++byte)
            payload[SCHEDULER_GRAPH_DUMP_METADATA_OFFSET + byte] = dump[byte];
    }
    scheduler_graph_writeback_range(
        instance.task_storage_address, static_cast<uint64_t>(tasks) * SCHEDULER_GRAPH_TASK_STORAGE_STRIDE
    );
    scheduler_graph_writeback_range(
        instance.tensor_pool_address, static_cast<uint64_t>(instance.tensor_arg_count) * TASKPAYLOAD_TENSOR_STRIDE
    );
    scheduler_graph_writeback_range(
        instance.scalar_pool_address, static_cast<uint64_t>(instance.scalar_arg_count) * sizeof(uint64_t)
    );
    scheduler_cache_barrier();
    return SchedulerGraphResult::OK;
}
