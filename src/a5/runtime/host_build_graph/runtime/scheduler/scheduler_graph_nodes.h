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

#include <cstdint>
#include <type_traits>

#include "host_build_graph/task_id.h"

enum class SchedulerGraphNodeKind : uint32_t { ORDINARY, ENTER, CHILD, EXIT };

enum class SchedulerGraphFaninKind : uint32_t { DENSE, DEFINITION_CSR, ENTER_GATE };

// A run-owned, immutable address table. Addresses name device allocations,
// never Host objects. Definition CSR rows stay shared across instances; the
// reader adds fanin_ref_base to each uint16 local child index.
// ENTER/EXIT have no kernel payload and may carry a zero storage address.
struct SchedulerGraphNode {
    uint64_t storage_address;
    TaskId public_task_id;
    uint64_t fanin_address;
    uint64_t instance_control_address;
    int32_t fanin_count;
    SchedulerGraphNodeKind kind;
    int32_t fanin_ref_base{0};
    SchedulerGraphFaninKind fanin_kind{SchedulerGraphFaninKind::DENSE};
};

static_assert(std::is_standard_layout_v<SchedulerGraphNode> && std::is_trivially_copyable_v<SchedulerGraphNode>);
static_assert(sizeof(SchedulerGraphNode) == 48, "resident Graph address entry layout changed");
static_assert(alignof(SchedulerGraphNode) == 8, "resident Graph address entry alignment changed");

inline constexpr int32_t SCHEDULER_GRAPH_MAX_BODY_FANIN = 1023;

// Host initializes the whole line before launch. Device mutations use atomics
// exclusively; no cached writeback may overlap these counters afterwards.
struct alignas(128) SchedulerGraphInstanceControl {
    volatile uint64_t remaining_children;
    volatile uint64_t materialize_owner;
    volatile uint64_t materialize_state;
    uint64_t run_epoch;
    int32_t enter_ref;
    int32_t child_begin;
    int32_t child_count;
    int32_t exit_ref;
    uint32_t instance_index;
};

static_assert(
    std::is_standard_layout_v<SchedulerGraphInstanceControl> &&
    std::is_trivially_copyable_v<SchedulerGraphInstanceControl>
);
static_assert(sizeof(SchedulerGraphInstanceControl) == 128, "Graph instance counters occupy one isolated line");

struct alignas(128) SchedulerGraphInstance {
    uint64_t definition_section_address;
    uint64_t definition_section_bytes;
    uint64_t definition_key;
    uint32_t definition_offset;
    int32_t parent_id;
    uint64_t task_storage_address;
    uint64_t tensor_pool_address;
    uint64_t scalar_pool_address;
    uint64_t boundary_tensors_address;
    uint64_t boundary_scalars_address;
    uint64_t heap_address;
    uint64_t control_address;
    uint64_t task_storage_bytes;
    uint64_t heap_bytes;
    int32_t boundary_tensor_count;
    int32_t boundary_scalar_count;
    int32_t tensor_arg_count;
    int32_t scalar_arg_count;
    uint64_t reserved;
};

static_assert(
    std::is_standard_layout_v<SchedulerGraphInstance> && std::is_trivially_copyable_v<SchedulerGraphInstance>
);
static_assert(sizeof(SchedulerGraphInstance) == 128, "Graph instance description occupies one immutable line");

struct alignas(128) SchedulerGraphBuildControl {
    volatile uint64_t next_instance;
    volatile uint64_t remaining_instances;
    volatile uint64_t public_completed;
    uint64_t instance_count;
    uint64_t public_task_count;
    uint64_t run_epoch;
    uint64_t protocol_instances_address;
    uint64_t protocol_contributions_address;
    uint64_t protocol_workers_address;
};
static_assert(
    std::is_standard_layout_v<SchedulerGraphBuildControl> && std::is_trivially_copyable_v<SchedulerGraphBuildControl>
);
static_assert(sizeof(SchedulerGraphBuildControl) == 128);

struct SchedulerGraphStorageLayout {
    uint64_t nodes_offset;
    uint64_t instances_offset;
    uint64_t instance_controls_offset;
    uint64_t build_control_offset;
    uint64_t definitions_offset;
    uint64_t definition_bytes;
    uint64_t external_fanins_offset;
    uint64_t external_fanin_count;
    uint64_t instance_count;
    uint64_t public_task_count;
    uint64_t protocol_instances_offset;
    uint64_t protocol_contributions_offset;
    uint64_t protocol_workers_offset;
};
