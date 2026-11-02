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

#include <cstdint>

#include "orchestration_api.h"  // NOLINT(build/include_subdir)

namespace {

struct ParallelConfig : simpler::hbg::ConfigBase {
    int64_t children;
    int64_t scalars;
    int64_t mode;
    ParallelConfig(int64_t count, int64_t parameters, int64_t engine) :
        children(count),
        scalars(parameters),
        mode(engine) {}
    uint64_t hash() const { return simpler::hbg::config_hash(children, scalars, mode); }
    bool operator==(const ParallelConfig &other) const {
        return children == other.children && scalars == other.scalars && mode == other.mode;
    }
};

void body(const GraphTaskArgs &boundary, const ParallelConfig &config) {
    for (int64_t child = 0; child < config.children; ++child) {
        CoreTaskArgs task;
        if (config.mode == 2) {
            MixedKernels kernels;
            kernels.aic_kernel_id = 2;
            kernels.aiv0_kernel_id = 3;
            kernels.aiv1_kernel_id = 4;
            task.add_inout(boundary.tensor(0).ref());
            task.add_scalar(child * 6);
            task.launch_spec.set_block_num(2);
            task.launch_spec.set_require_sync_start(true);
            rt_submit_task(kernels, task);
        } else {
            const uint32_t shape[1] = {32};
            const uint32_t left_offset[1] = {static_cast<uint32_t>(child * 64)};
            const uint32_t right_offset[1] = {left_offset[0] + 32};
            const auto left = boundary.tensor(0).ref().view(shape, left_offset);
            const auto right = boundary.tensor(0).ref().view(shape, right_offset);
            task.add_inout(left, right);
            task.add_scalar(config.scalars + ((child + 1) << 8));
            for (int64_t index = 1; index < config.scalars; ++index) {
                if (index % 2 == 0) task.add_scalar(boundary.scalar(0));
                else task.add_scalar(100 + index);
            }
            if (config.mode == 0) rt_submit_aic_task(0, task);
            else rt_submit_aiv_task(1, task);
        }
    }
}

}  // namespace

extern "C" {

__attribute__((visibility("default"))) OrchestrationConfig aicpu_orchestration_config(const ChipTaskArgs &) {
    return OrchestrationConfig{.expected_arg_count = 7};
}

__attribute__((visibility("default"))) void aicpu_orchestration_entry(const ChipTaskArgs &args) {
    const int64_t instances = args.scalar<int64_t>(0);
    const int64_t children = args.scalar<int64_t>(1);
    const int64_t scalars = args.scalar<int64_t>(2);
    const int64_t mode = args.scalar<int64_t>(3);
    const int64_t seed = args.scalar<int64_t>(4);
    const int64_t tail_children = args.scalar<int64_t>(5);
    const int64_t span = children * (mode == 2 ? 96 : 64);
    const uint32_t shape[1] = {static_cast<uint32_t>(span)};
    for (int64_t instance = 0; instance < instances; ++instance) {
        const uint32_t offset[1] = {static_cast<uint32_t>(instance * span)};
        const auto output = args.tensor(0).ref().view(shape, offset);
        GraphTaskArgs boundary;
        boundary.add_inout(output);
        boundary.add_scalar(seed + instance * 0x10000);
        const int64_t count = tail_children > 0 && instance + 1 == instances ? tail_children : children;
        rt_submit_graph(&body, boundary, ParallelConfig(count, scalars, mode));
    }
}

}  // extern "C"
