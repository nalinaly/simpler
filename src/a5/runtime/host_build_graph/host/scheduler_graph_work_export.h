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

inline void graph_work_plan(
    JsonObject &root, const Image &image, const SchedulerGraphStorageLayout &plan,
    const SchedulerGraphManifest &manifest
) {
    if (manifest.materialize_work.size() != plan.work_count ||
        manifest.materialize_ranges.size() != plan.work_range_count)
        throw std::invalid_argument("Graph work plan differs from its frozen size");
    const auto *work = image.array<SchedulerGraphWork>(plan.work_offset, plan.work_count);
    const auto *ranges = image.array<SchedulerGraphWorkRange>(plan.work_ranges_offset, plan.work_range_count);
    const auto *states = image.array<SchedulerGraphWorkState>(plan.work_states_offset, plan.work_count);
    auto &out = root.field("work_plan");
    out << '[';
    for (uint64_t index = 0; index < plan.work_count; ++index) {
        const auto &source = work[index];
        const auto &expected = manifest.materialize_work[index];
        if (source.instance_index != expected.instance_index || source.region != expected.region ||
            source.first_range != expected.first_range || source.range_count != expected.range_count ||
            source.write_bytes != expected.write_bytes)
            throw std::invalid_argument("Graph work descriptor changed after the protocol snapshot");
        if (index != 0) out << ',';
        JsonObject row(out);
        row.number("index", index);
        row.number("instance_index", source.instance_index);
        row.number("region", static_cast<uint32_t>(source.region));
        row.number("first_range", source.first_range);
        row.number("range_count", source.range_count);
        row.number("write_bytes", source.write_bytes);
        row.number("owner_state", states[index].owner_state);
    }
    out << ']';
    auto &parts = root.field("work_ranges");
    parts << '[';
    for (uint64_t index = 0; index < plan.work_range_count; ++index) {
        const auto &source = ranges[index];
        const auto &expected = manifest.materialize_ranges[index];
        if (source.begin != expected.begin || source.count != expected.count)
            throw std::invalid_argument("Graph work range changed after the protocol snapshot");
        if (index != 0) parts << ',';
        JsonObject row(parts);
        row.number("index", index);
        row.number("begin", source.begin);
        row.number("count", source.count);
    }
    parts << ']';
    sparse(
        root, "work", image.array<SchedulerGraphProtocolWork>(plan.protocol_work_offset, plan.work_count),
        plan.work_count, [](JsonObject &row, const auto &record) {
            row.stamp("begin", record.begin);
            row.stamp("writeback_after", record.writeback_after);
            row.stamp("complete_before", record.complete_before);
            row.stamp("complete_after", record.complete_after);
            row.number("run_epoch", record.run_epoch);
            row.number("remaining_before", record.remaining_before);
            row.number("work_index", record.work_index);
            row.number("instance_index", record.instance_index);
            row.number("worker_index", record.worker_index);
            row.number("status", record.status);
        }
    );
}

}  // namespace scheduler_protocol_export
