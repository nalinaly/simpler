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

#include "scheduler_graph.h"
#include "scheduler_graph_protocol_types.h"
#include "scheduler_protocol.h"

template <class Record, class Local>
inline __aicore__ __gm__ Record *scheduler_graph_protocol_record_at(
    __gm__ void *base, const Local *local, uint64_t address, uint64_t index, uint64_t count
) {
    auto *layout = scheduler_protocol_layout_at(base, local);
    if (layout == nullptr) return nullptr;
    if (address == 0 || address % alignof(Record) != 0 || index >= count ||
        count > (UINT64_MAX - address) / sizeof(Record)) {
        scheduler_protocol_fail(base, local, layout, SCHEDULER_PROTOCOL_INVALID_IDENTITY);
        return nullptr;
    }
    return reinterpret_cast<__gm__ Record *>(address) + index;
}

template <class Record, class Local>
inline __aicore__ __gm__ SchedulerProtocolLayout *
scheduler_graph_protocol_owned(__gm__ void *base, const Local *local, __gm__ const Record *record) {
    auto *layout = scheduler_protocol_layout_at(base, local);
    if (layout == nullptr || record == nullptr) return nullptr;
    if (record->run_epoch != layout->run_epoch || record->worker_index != local->worker_id()) {
        scheduler_protocol_fail(base, local, layout, SCHEDULER_PROTOCOL_INVALID_IDENTITY);
        return nullptr;
    }
    return layout;
}

template <class Local>
inline __aicore__ __gm__ SchedulerGraphProtocolInstance *scheduler_graph_protocol_materialize_begin(
    __gm__ void *base, const Local *local, __gm__ const SchedulerGraphBuildControl *build, uint64_t index,
    __gm__ const SchedulerGraphInstance &instance, uint32_t children
) {
    auto *layout = scheduler_protocol_layout_at(base, local);
    if (layout == nullptr) return nullptr;
    auto *record = scheduler_graph_protocol_record_at<SchedulerGraphProtocolInstance>(
        base, local, build->protocol_instances_address, index, build->instance_count
    );
    if (record == nullptr) return nullptr;
    scheduler_protocol_observe_record(record);
    if (!scheduler_protocol_stamp(base, local, layout, &record->materialize_before)) return nullptr;
    record->run_epoch = layout->run_epoch;
    record->worker_index = static_cast<uint32_t>(local->worker_id());
    record->instance_index = static_cast<uint32_t>(index);
    record->parent_id = static_cast<uint32_t>(instance.parent_id);
    record->definition_key = instance.definition_key;
    record->child_count = children;
    scheduler_protocol_publish_record(record);
    return record;
}

template <class Local>
inline __aicore__ void scheduler_graph_protocol_materialize_end(
    __gm__ void *base, const Local *local, __gm__ SchedulerGraphProtocolInstance *record, SchedulerGraphResult status
) {
    auto *layout = scheduler_graph_protocol_owned(base, local, record);
    if (layout == nullptr) return;
    record->status = static_cast<uint32_t>(status);
    scheduler_protocol_stamp(base, local, layout, &record->materialize_after);
    scheduler_protocol_publish_record(record);
}

template <class Local>
inline __aicore__ void scheduler_graph_protocol_publish(
    __gm__ void *base, const Local *local, __gm__ SchedulerGraphProtocolInstance *record, bool before,
    uint64_t remaining_before = 0
) {
    auto *layout = scheduler_graph_protocol_owned(base, local, record);
    if (layout == nullptr) return;
    if (before) {
        scheduler_protocol_stamp(base, local, layout, &record->publish_before);
    } else {
        record->remaining_before = remaining_before;
        scheduler_protocol_stamp(base, local, layout, &record->publish_after);
    }
    scheduler_protocol_publish_record(record);
}

template <class Local>
inline __aicore__ __gm__ SchedulerGraphProtocolWorker *scheduler_graph_protocol_wait_begin(
    __gm__ void *base, const Local *local, __gm__ const SchedulerGraphBuildControl *build
) {
    auto *layout = scheduler_protocol_layout_at(base, local);
    if (layout == nullptr) return nullptr;
    auto *record = scheduler_graph_protocol_record_at<SchedulerGraphProtocolWorker>(
        base, local, build->protocol_workers_address, local->worker_id(), layout->worker_count
    );
    if (record == nullptr) return nullptr;
    scheduler_protocol_observe_record(record);
    if (!scheduler_protocol_stamp(base, local, layout, &record->before)) return nullptr;
    record->run_epoch = layout->run_epoch;
    record->worker_index = static_cast<uint32_t>(local->worker_id());
    record->instance_count = static_cast<uint32_t>(build->instance_count);
    scheduler_protocol_publish_record(record);
    return record;
}

template <class Local>
inline __aicore__ void scheduler_graph_protocol_wait_end(
    __gm__ void *base, const Local *local, __gm__ SchedulerGraphProtocolWorker *record, uint64_t polls,
    uint64_t remaining, uint32_t status
) {
    auto *layout = scheduler_graph_protocol_owned(base, local, record);
    if (layout == nullptr) return;
    record->remaining_observed = remaining;
    record->poll_count = polls;
    record->status = status;
    scheduler_protocol_stamp(base, local, layout, &record->after);
    scheduler_protocol_publish_record(record);
}

template <class Local>
inline __aicore__ __gm__ SchedulerGraphProtocolContribution *scheduler_graph_protocol_contribution_begin(
    __gm__ void *base, const Local *local, const SchedulerGraphView &graph,
    __gm__ const SchedulerGraphInstanceControl *instance, int64_t task_ref
) {
    auto *layout = scheduler_protocol_layout_at(base, local);
    if (layout == nullptr) return nullptr;
    if (graph.build_control_address == 0 || task_ref < 0) {
        scheduler_protocol_fail(base, local, layout, SCHEDULER_PROTOCOL_INVALID_IDENTITY);
        return nullptr;
    }
    const auto *build = reinterpret_cast<__gm__ const SchedulerGraphBuildControl *>(graph.build_control_address);
    if (instance->instance_index >= build->instance_count) {
        scheduler_protocol_fail(base, local, layout, SCHEDULER_PROTOCOL_INVALID_IDENTITY);
        return nullptr;
    }
    auto *record = scheduler_graph_protocol_record_at<SchedulerGraphProtocolContribution>(
        base, local, build->protocol_contributions_address, static_cast<uint64_t>(task_ref), graph.task_count
    );
    if (record == nullptr) return nullptr;
    scheduler_protocol_observe_record(record);
    if (!scheduler_protocol_stamp(base, local, layout, &record->before)) return nullptr;
    record->run_epoch = layout->run_epoch;
    record->worker_index = static_cast<uint32_t>(local->worker_id());
    record->instance_index = instance->instance_index;
    record->task_ref = task_ref;
    scheduler_protocol_publish_record(record);
    return record;
}

template <class Local>
inline __aicore__ void scheduler_graph_protocol_contribution_end(
    __gm__ void *base, const Local *local, __gm__ SchedulerGraphProtocolContribution *record, uint64_t prior
) {
    auto *layout = scheduler_graph_protocol_owned(base, local, record);
    if (layout == nullptr) return;
    record->remaining_before = prior;
    scheduler_protocol_stamp(base, local, layout, &record->after);
    scheduler_protocol_publish_record(record);
}
