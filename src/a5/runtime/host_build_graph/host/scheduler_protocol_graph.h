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

#include <stdexcept>
#include <utility>
#include <vector>

#include "scheduler/scheduler_graph.h"
#include "scheduler/scheduler_types.h"

struct SchedulerProtocolTaskSpec {
    SchedulerTaskMetadata metadata;
    std::vector<int32_t> dependencies;
};

inline std::vector<SchedulerProtocolTaskSpec>
scheduler_protocol_freeze_graph(const SchedulerGraphView &graph, const std::vector<SchedulerTaskMetadata> &metadata) {
    if (graph.task_count != metadata.size()) throw std::invalid_argument("protocol graph task count mismatch");
    std::vector<SchedulerProtocolTaskSpec> result;
    result.reserve(metadata.size());
    for (size_t task = 0; task < metadata.size(); ++task) {
        SchedulerTaskShape shape{};
        const auto status = scheduler_classify_task_shape(graph, static_cast<int64_t>(task), &shape);
        if (status != SchedulerGraphResult::OK &&
            !(status == SchedulerGraphResult::UNSUPPORTED_SHAPE &&
              (metadata[task].active_mask == 0 || scheduler_task_is_inline(metadata[task].flags))))
            throw std::invalid_argument("invalid frozen protocol graph");
        const auto count = scheduler_graph_fanin_count(graph, static_cast<int64_t>(task));
        SchedulerProtocolTaskSpec spec{metadata[task], {}};
        spec.dependencies.reserve(static_cast<size_t>(count));
        for (int32_t index = 0; index < count; ++index)
            spec.dependencies.push_back(scheduler_graph_fanin_id(graph, static_cast<int64_t>(task), index));
        result.push_back(std::move(spec));
    }
    return result;
}
