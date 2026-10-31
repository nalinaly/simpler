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

#include <vector>

#include "cla_plan.h"
#include "orchestration_api.h"

namespace {

struct GraphPlan : simpler::hbg::ConfigBase {
    std::vector<int64_t> rows;
    std::vector<int64_t> sync;
    int32_t begin{0};
    int32_t end{0};
    bool wait_for_peer{false};

    uint64_t hash() const {
        uint64_t result = simpler::hbg::config_hash(begin, end, wait_for_peer);
        for (auto value : rows)
            result = simpler::hbg::config_hash_field(result, value);
        for (auto value : sync)
            result = simpler::hbg::config_hash_field(result, value);
        return result;
    }
    bool operator==(const GraphPlan &other) const {
        return begin == other.begin && end == other.end && wait_for_peer == other.wait_for_peer && rows == other.rows &&
               sync == other.sync;
    }
};

void graph_body(const GraphTaskArgs &boundary, const GraphPlan &plan) {
    auto records = boundary.tensor(0).ref();
    records.manual_dep = true;
    std::vector<TaskId> handles;
    handles.reserve(plan.end - plan.begin);
    for (int32_t task = plan.begin; task < plan.end; ++task) {
        const auto *row = plan.rows.data() + task * cla_contract::kPlanWidth;
        const auto blocks = static_cast<int16_t>(row[static_cast<uint32_t>(cla_contract::Field::Blocks)]);
        const auto mask = static_cast<uint32_t>(row[static_cast<uint32_t>(cla_contract::Field::Mask)]);
        const int64_t count = row[static_cast<uint32_t>(cla_contract::Field::DependencyCount)];
        std::vector<TaskId> fanins;
        for (int64_t edge = 0; edge < count; ++edge) {
            const int64_t producer = row[cla_contract::kDependencyOffset + edge];
            if (producer >= plan.begin) fanins.push_back(handles[producer - plan.begin]);
        }
        CoreTaskArgs args;
        args.set_dependencies(fanins.data(), static_cast<uint32_t>(fanins.size()));
        const uint32_t shape[1] = {(static_cast<uint32_t>(blocks) * 3 + 1) * 16};
        const uint32_t offset[1] = {static_cast<uint32_t>(row[static_cast<uint32_t>(cla_contract::Field::Base)]) * 16};
        auto state = records.view(shape, offset, true);
        args.add_inout(state);
        args.add_input(records, boundary.tensor(1).ref());
        args.add_scalar(static_cast<int64_t>(task));
        args.add_scalar(plan.sync[task]);
        args.add_scalar(plan.wait_for_peer ? int64_t{1 - task} : int64_t{-1});
        args.launch_spec.set_block_num(blocks);
        args.launch_spec.set_require_sync_start(plan.sync[task] != 0);
        MixedKernels kernels;
        kernels.aic_kernel_id = (mask & 1U) ? 0 : INVALID_KERNEL_ID;
        kernels.aiv0_kernel_id = (mask & 2U) ? 1 : INVALID_KERNEL_ID;
        kernels.aiv1_kernel_id = (mask & 4U) ? 2 : INVALID_KERNEL_ID;
        handles.push_back(rt_submit_task(kernels, args).task_id());
    }
}

}  // namespace

extern "C" __attribute__((visibility("default"))) OrchestrationConfig aicpu_orchestration_config(const ChipTaskArgs &) {
    return OrchestrationConfig{.expected_arg_count = 6};
}

extern "C" __attribute__((visibility("default"))) void aicpu_orchestration_entry(const ChipTaskArgs &args) {
    const auto &host = args.tensor(2).ref();
    const auto &flags = args.tensor(3).ref();
    if (host.ndims != 2 || host.dtype != DataType::INT64 || host.shapes[1] != cla_contract::kPlanWidth ||
        flags.ndims != 1 || flags.dtype != DataType::INT64 || flags.shapes[0] != host.shapes[0] ||
        host.shapes[0] == 0 || host.shapes[0] > 16) {
        rt_report_fatal(SIMPLER_ERROR_INVALID_ARGS, "Graph ordered probe: invalid plan");
        return;
    }
    const auto count = static_cast<int32_t>(host.shapes[0]);
    const auto *rows = reinterpret_cast<const int64_t *>(host.buffer.addr) + host.start_offset;
    const auto *sync = reinterpret_cast<const int64_t *>(flags.buffer.addr) + flags.start_offset;
    GraphPlan plan;
    plan.rows.assign(rows, rows + count * cla_contract::kPlanWidth);
    plan.sync.assign(sync, sync + count);
    plan.wait_for_peer = args.scalar<int64_t>(0) != 0;
    if (plan.wait_for_peer && count != 2) {
        rt_report_fatal(SIMPLER_ERROR_INVALID_ARGS, "Graph ordered probe: peer wait requires two tasks");
        return;
    }
    for (int32_t task = 0; task < count; ++task) {
        const auto *row = rows + task * cla_contract::kPlanWidth;
        const auto blocks = row[static_cast<uint32_t>(cla_contract::Field::Blocks)];
        const auto mask = row[static_cast<uint32_t>(cla_contract::Field::Mask)];
        const auto deps = row[static_cast<uint32_t>(cla_contract::Field::DependencyCount)];
        if (blocks <= 0 || blocks > INT16_MAX || mask <= 0 || mask > 7 || deps < 0 ||
            deps > cla_contract::kMaxDependencies || sync[task] < 0 || sync[task] > 1) {
            rt_report_fatal(SIMPLER_ERROR_INVALID_ARGS, "Graph ordered probe: invalid task shape");
            return;
        }
        for (int64_t edge = 0; edge < deps; ++edge) {
            if (row[cla_contract::kDependencyOffset + edge] < 0 ||
                row[cla_contract::kDependencyOffset + edge] >= task) {
                rt_report_fatal(SIMPLER_ERROR_INVALID_ARGS, "Graph ordered probe: invalid predecessor");
                return;
            }
        }
    }
    const bool split = args.scalar<int64_t>(1) != 0;
    auto records = args.tensor(0).ref();
    records.manual_dep = true;
    for (int32_t first = 0; first < count; first = plan.end) {
        plan.begin = first;
        plan.end = split ? first + 1 : count;
        GraphTaskArgs boundary;
        boundary.add_inout(records);
        boundary.add_input(args.tensor(1).ref());
        const auto *first_row = rows + first * cla_contract::kPlanWidth;
        const auto *last_row = rows + (plan.end - 1) * cla_contract::kPlanWidth;
        const auto first_base = first_row[static_cast<uint32_t>(cla_contract::Field::Base)];
        const auto last_end = last_row[static_cast<uint32_t>(cla_contract::Field::Base)] +
                              3 * last_row[static_cast<uint32_t>(cla_contract::Field::Blocks)] + 1;
        const uint32_t output_shape[1] = {static_cast<uint32_t>(last_end - first_base) * 16};
        const uint32_t output_offset[1] = {static_cast<uint32_t>(first_base) * 16};
        auto output = records.view(output_shape, output_offset, false);
        boundary.add_output(output);
        std::vector<simpler::hbg::Tensor> predecessors;
        if (split) {
            const auto count = first_row[static_cast<uint32_t>(cla_contract::Field::DependencyCount)];
            predecessors.reserve(count);
            for (int64_t edge = 0; edge < count; ++edge) {
                const auto *producer =
                    rows + first_row[cla_contract::kDependencyOffset + edge] * cla_contract::kPlanWidth;
                const uint32_t shape[1] = {
                    static_cast<uint32_t>(producer[static_cast<uint32_t>(cla_contract::Field::Blocks)] * 3 + 1) * 16
                };
                const uint32_t offset[1] = {
                    static_cast<uint32_t>(producer[static_cast<uint32_t>(cla_contract::Field::Base)]) * 16
                };
                predecessors.push_back(records.view(shape, offset, false));
                boundary.add_input(predecessors.back());
            }
        }
        rt_submit_graph(&graph_body, boundary, plan);
    }
}
