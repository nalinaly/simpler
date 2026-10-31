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

#include <algorithm>
#include <cstddef>
#include <cstdint>
#include <iterator>
#include <limits>
#include <optional>
#include <vector>

#include "host_build_graph/graph_execution.h"
#include "host_build_graph/task_id.h"
#include "scheduler/scheduler_graph_nodes.h"

using SchedulerGraphPlanNodeKind = SchedulerGraphNodeKind;
static_assert(SCHEDULER_GRAPH_MAX_BODY_FANIN == SUB_TASK_MAX_NUM - 1);

struct SchedulerGraphPlanShape {
    bool graph{false};
    int32_t children{0};
};

struct SchedulerGraphPlanRange {
    int32_t begin;
    int32_t end;
    int32_t children;
    int32_t instance_index;
};

struct SchedulerGraphPlanNode {
    TaskId task_id;
    SchedulerGraphPlanNodeKind kind;
    int32_t outer_id;
    int32_t child_index;
    int32_t instance_index;
};

// Host-only numbering. Ranges are sized by public submissions, not expanded
// nodes, and ENTER/EXIT share their public parent's identity without sharing
// an internal reference. Only CHILD/ORDINARY nodes can name kernel executions.
class SchedulerGraphPlan {
public:
    static std::optional<SchedulerGraphPlan> build(const std::vector<SchedulerGraphPlanShape> &shapes) {
        if (shapes.size() > static_cast<size_t>(TaskId::GLOBAL_TASK_MAX_NUM)) return std::nullopt;
        uint64_t nodes = 0;
        int32_t instances = 0;
        for (const auto &shape : shapes) {
            if (shape.graph) {
                if (shape.children <= 0 || shape.children > SUB_TASK_MAX_NUM) return std::nullopt;
                nodes += static_cast<uint64_t>(shape.children) + 2;
                ++instances;
            } else {
                if (shape.children != 0) return std::nullopt;
                ++nodes;
            }
            if (nodes > static_cast<uint64_t>(std::numeric_limits<int32_t>::max())) return std::nullopt;
        }
        SchedulerGraphPlan plan;
        plan.ranges_.reserve(shapes.size());
        plan.node_count_ = static_cast<int32_t>(nodes);
        plan.instance_count_ = instances;
        int32_t cursor = 0;
        int32_t instance = 0;
        for (const auto &shape : shapes) {
            const int32_t width = shape.graph ? shape.children + 2 : 1;
            plan.ranges_.push_back({cursor, cursor + width, shape.children, shape.graph ? instance++ : -1});
            cursor += width;
        }
        return plan;
    }

    int32_t node_count() const { return node_count_; }
    int32_t instance_count() const { return instance_count_; }
    const std::vector<SchedulerGraphPlanRange> &ranges() const { return ranges_; }

    std::optional<SchedulerGraphPlanNode> node_at(int64_t ref) const {
        if (ref < 0 || ref >= node_count_) return std::nullopt;
        auto found = std::upper_bound(
            ranges_.begin(), ranges_.end(), ref, [](int64_t value, const SchedulerGraphPlanRange &range) {
                return value < range.end;
            }
        );
        if (found == ranges_.end()) return std::nullopt;
        const int32_t outer = static_cast<int32_t>(std::distance(ranges_.begin(), found));
        SchedulerGraphPlanNode result{
            TaskId::make_global(outer), SchedulerGraphPlanNodeKind::ORDINARY, outer, -1, found->instance_index
        };
        if (found->instance_index < 0) return result;
        if (ref == found->begin) {
            result.kind = SchedulerGraphPlanNodeKind::ENTER;
        } else if (ref == found->end - 1) {
            result.kind = SchedulerGraphPlanNodeKind::EXIT;
        } else {
            result.kind = SchedulerGraphPlanNodeKind::CHILD;
            result.child_index = static_cast<int32_t>(ref) - found->begin - 1;
            result.task_id = TaskId::make_sub_task(outer, result.child_index);
        }
        return result;
    }

    std::optional<int32_t> completion_ref(int64_t outer_id) const {
        if (outer_id < 0 || static_cast<uint64_t>(outer_id) >= ranges_.size()) return std::nullopt;
        return ranges_[static_cast<size_t>(outer_id)].end - 1;
    }

    // The caller uses this edge only for a CHILD whose saved CSR row is empty.
    std::optional<int32_t> root_gate(int64_t child_ref) const {
        const auto node = node_at(child_ref);
        if (!node || node->kind != SchedulerGraphPlanNodeKind::CHILD) return std::nullopt;
        return ranges_[node->outer_id].begin;
    }

    // source_id is a GLOBAL fanin for ORDINARY/ENTER, or a recorded local CSR
    // index for CHILD. EXIT has no fanin row: the final child makes it ready.
    std::optional<int32_t> dependency_ref(int64_t consumer_ref, int32_t source_id) const {
        const auto node = node_at(consumer_ref);
        if (!node || source_id < 0 || node->kind == SchedulerGraphPlanNodeKind::EXIT) return std::nullopt;
        if (node->kind == SchedulerGraphPlanNodeKind::CHILD) {
            if (source_id >= node->child_index) return std::nullopt;
            return ranges_[node->outer_id].begin + 1 + source_id;
        }
        if (source_id >= node->outer_id) return std::nullopt;
        return completion_ref(source_id);
    }

private:
    std::vector<SchedulerGraphPlanRange> ranges_;
    int32_t node_count_{0};
    int32_t instance_count_{0};
};
