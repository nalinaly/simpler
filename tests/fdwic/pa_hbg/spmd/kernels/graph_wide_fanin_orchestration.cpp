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

#include "orchestration_api.h"

namespace {

struct WideConfig : simpler::hbg::ConfigBase {
    int64_t blocks{1};
    int64_t mask{2};
    int64_t tasks{1};
    int64_t delay{0};
    int64_t sync{0};

    uint64_t hash() const { return simpler::hbg::config_hash(blocks, mask, tasks, delay, sync); }
    bool operator==(const WideConfig &other) const {
        return blocks == other.blocks && mask == other.mask && tasks == other.tasks && delay == other.delay &&
               sync == other.sync;
    }
};

void verify_records(
    const simpler::hbg::Tensor &records, const simpler::hbg::Tensor &summary, const WideConfig &config,
    const std::vector<TaskId> &dependencies
) {
    CoreTaskArgs verify;
    verify.set_dependencies(dependencies.data(), static_cast<uint32_t>(dependencies.size()));
    verify.add_input(records);
    verify.add_inout(summary);
    verify.add_scalar(config.blocks);
    verify.add_scalar(config.mask);
    verify.add_scalar(config.tasks);
    rt_submit_aiv_task(3, verify);
}

void wide_body(const GraphTaskArgs &boundary, const WideConfig &config) {
    auto records = boundary.tensor(0).ref();
    records.manual_dep = true;
    const uint32_t shape[1] = {(static_cast<uint32_t>(config.blocks) * 3 + 1) * 16};
    MixedKernels kernels;
    kernels.aic_kernel_id = (config.mask & 1) ? 0 : INVALID_KERNEL_ID;
    kernels.aiv0_kernel_id = (config.mask & 2) ? 1 : INVALID_KERNEL_ID;
    kernels.aiv1_kernel_id = (config.mask & 4) ? 2 : INVALID_KERNEL_ID;
    std::vector<TaskId> predecessors;
    predecessors.reserve(config.tasks);
    for (int64_t task = 0; task < config.tasks; ++task) {
        const uint32_t offset[1] = {static_cast<uint32_t>(task) * shape[0]};
        auto state = records.view(shape, offset, true);
        CoreTaskArgs args;
        args.add_inout(state);
        args.add_scalar(config.blocks);
        args.add_scalar(config.mask);
        args.add_scalar(task);
        args.add_scalar(config.delay);
        args.add_scalar(config.sync);
        args.launch_spec.set_block_num(static_cast<int16_t>(config.blocks));
        args.launch_spec.set_require_sync_start(config.sync != 0);
        predecessors.push_back(rt_submit_task(kernels, args).task_id());
    }
    const uint32_t summary_shape[1] = {16};
    const uint32_t summary_offset[1] = {0};
    auto summary = boundary.tensor(1).ref().view(summary_shape, summary_offset);
    verify_records(records, summary, config, predecessors);
}

}  // namespace

extern "C" __attribute__((visibility("default"))) OrchestrationConfig aicpu_orchestration_config(const ChipTaskArgs &) {
    return OrchestrationConfig{.expected_arg_count = 7};
}

extern "C" __attribute__((visibility("default"))) void aicpu_orchestration_entry(const ChipTaskArgs &args) {
    WideConfig config;
    config.blocks = args.scalar<int64_t>(0);
    config.mask = args.scalar<int64_t>(1);
    config.tasks = args.scalar<int64_t>(2);
    config.delay = args.scalar<int64_t>(3);
    config.sync = args.scalar<int64_t>(4);
    GraphTaskArgs boundary;
    boundary.add_inout(args.tensor(0).ref(), args.tensor(1).ref());
    rt_submit_graph(&wide_body, boundary, config);
    const uint32_t shape[1] = {16};
    const uint32_t offset[1] = {16};
    auto summary = args.tensor(1).ref().view(shape, offset);
    verify_records(args.tensor(0).ref(), summary, config, {});
}
