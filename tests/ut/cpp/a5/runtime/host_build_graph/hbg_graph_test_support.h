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

#include <array>
#include <cstddef>
#include <cstdint>

#include "host_build_graph/graph_execution.h"
#include "scheduler/scheduler_graph_materialize.h"

struct MaterializeFixture {
    struct Body {
        GraphDefinition definition{};
        std::array<SubTaskDefinition, 2> tasks{};
        std::array<uint64_t, 2> task_offsets{0, 64};
        std::array<int32_t, 3> fanin_rows{0, 0, 1};
        std::array<uint16_t, 1> fanin_indices{0};
        std::array<int32_t, 3> fanout_rows{0, 1, 1};
        std::array<uint16_t, 1> fanout_indices{1};
        std::array<uint16_t, 1> roots{0};
        std::array<simpler::hbg::TensorData, 2> tensors{};
        std::array<uint64_t, 2> scalars{13, 29};
        std::array<GraphScalarInheritance, 2> sources{
            GraphScalarInheritance::from_boundary(0), GraphScalarInheritance::self_value()
        };
        std::array<GraphPredicate, 2> predicates{};
    };
    struct Object {
        GraphDefinitionHeader header{};
        Body body{};
    } object;
    struct Storage {
        std::array<ChipTaskStorage, 2> tasks{};
        std::array<simpler::hbg::Tensor, 2> tensors{};
        std::array<uint64_t, 2> scalars{};
    } output;
    std::array<simpler::hbg::Tensor, 1> boundary{};
    std::array<uint64_t, 1> scalar{97};
    std::array<uint8_t, 128> heap{};
    std::array<SchedulerGraphNode, 4> nodes{};
    SchedulerGraphInstanceControl control{};
    SchedulerGraphInstance instance{};

    MaterializeFixture() {
        object.header.magic = GRAPH_DEFINITION_OBJECT_MAGIC;
        object.header.full_key = 17;
        object.header.definition_bytes = sizeof(Body);
        auto &def = object.body.definition;
        def.full_key = 17;
        def.total_bytes = sizeof(Body);
        def.required_heap = heap.size();
        def.task_count = 2;
        def.edge_count = 1;
        def.root_count = 1;
        def.boundary_tensor_count = def.boundary_scalar_count = 1;
        def.tensor_arg_count = def.scalar_arg_count = 2;
        def.off_sub_tasks = offsetof(Body, tasks);
        def.off_sub_task_offsets = offsetof(Body, task_offsets);
        def.off_fanin_offsets = offsetof(Body, fanin_rows);
        def.off_fanin_indices = offsetof(Body, fanin_indices);
        def.off_fanout_offsets = offsetof(Body, fanout_rows);
        def.off_fanout_indices = offsetof(Body, fanout_indices);
        def.off_root_indices = offsetof(Body, roots);
        def.off_tensors = offsetof(Body, tensors);
        def.off_scalars = offsetof(Body, scalars);
        def.off_scalar_inheritance = offsetof(Body, sources);
        size_t storage_bytes = 0;
        graph_execution_storage_bytes(2, 2, 2, 1, &storage_bytes);
        def.execution_storage_bytes = static_cast<uint32_t>(storage_bytes);
        for (int32_t i = 0; i < 2; ++i) {
            auto &task = object.body.tasks[i];
            task.kernel_id[0] = task.kernel_id[2] = INVALID_KERNEL_ID;
            task.kernel_id[1] = 7;
            task.active_mask = 2;
            task.logical_block_num = task.total_required_subtasks = 1;
            task.total_output_size = 64;
            task.tensor_count = task.scalar_count = 1;
            task.tensor_offset = task.scalar_offset = i;
            auto &tensor = object.body.tensors[i];
            tensor.owner_task_id = i == 0 ? TaskId::make_param(0) : TaskId::make_sub_task(0, 0);
            tensor.buffer = {static_cast<uint64_t>(i * 16), 32};
            tensor.ndims = 1;
            tensor.dtype = DataType::FLOAT32;
            tensor.shapes[0] = 4;
            tensor.strides[0] = 1;
            tensor.extent_elem_cache = 4;
            tensor.start_offset = 2;
            nodes[i + 1].kind = SchedulerGraphNodeKind::CHILD;
            nodes[i + 1].public_task_id = TaskId::make_sub_task(5, i);
            nodes[i + 1].storage_address = reinterpret_cast<uint64_t>(&output.tasks[i]);
        }
        boundary[0].buffer = {0x100000, 1024};
        boundary[0].start_offset = 5;
        boundary[0].address_space = AddressSpace::DEVICE;
        control.child_begin = 1;
        control.child_count = 2;
        control.enter_ref = 0;
        control.exit_ref = 3;
        control.remaining_children = 2;
        instance.definition_section_address = reinterpret_cast<uint64_t>(&object);
        instance.definition_section_bytes = sizeof(object);
        instance.definition_key = 17;
        instance.definition_offset = offsetof(Object, body);
        instance.parent_id = 5;
        instance.task_storage_address = reinterpret_cast<uint64_t>(output.tasks.data());
        instance.task_storage_bytes = sizeof(output.tasks);
        instance.tensor_pool_address = reinterpret_cast<uint64_t>(output.tensors.data());
        instance.scalar_pool_address = reinterpret_cast<uint64_t>(output.scalars.data());
        instance.boundary_tensors_address = reinterpret_cast<uint64_t>(boundary.data());
        instance.boundary_scalars_address = reinterpret_cast<uint64_t>(scalar.data());
        instance.boundary_tensor_count = instance.boundary_scalar_count = 1;
        instance.tensor_arg_count = instance.scalar_arg_count = 2;
        instance.heap_address = reinterpret_cast<uint64_t>(heap.data());
        instance.heap_bytes = heap.size();
        instance.control_address = reinterpret_cast<uint64_t>(&control);
    }

    SchedulerGraphResult materialize() {
        SchedulerGraphView graph{0, 0, nodes.size(), nodes.size() - 1, reinterpret_cast<uint64_t>(nodes.data())};
        return scheduler_graph_materialize(instance, graph);
    }

    void enable_predicates() {
        object.body.definition.predicate_count = 2;
        object.body.definition.off_predicates = offsetof(Body, predicates);
        TaskAttrs attrs{};
        attrs.set_predicate();
        for (int32_t i = 0; i < 2; ++i) {
            object.body.tasks[i].predicate_slot = i + 1;
            object.body.tasks[i].task_attrs = attrs.raw();
            auto &predicate = object.body.predicates[i];
            predicate.operand = object.body.tensors[i];
            predicate.elem_offset = 1;
            predicate.elem_size = 4;
            predicate.op = static_cast<uint8_t>(PredicateOp::EQ);
            predicate.target = 13 + i;
        }
    }
};
