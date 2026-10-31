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

struct FailureConfig : simpler::hbg::ConfigBase {
    int64_t blocks;
    int64_t mask;
    bool tail;
    FailureConfig(int64_t count, int64_t active, bool trailing) :
        blocks(count),
        mask(active),
        tail(trailing) {}
    uint64_t hash() const { return simpler::hbg::config_hash(blocks, mask, tail); }
    bool operator==(const FailureConfig &other) const {
        return blocks == other.blocks && mask == other.mask && tail == other.tail;
    }
};

void body(const GraphTaskArgs &boundary, const FailureConfig &config) {
    CoreTaskArgs task;
    task.add_inout(boundary.tensor(0).ref());
    for (int32_t scalar = 0; scalar < 4; ++scalar)
        task.add_scalar(boundary.scalar(scalar));
    task.launch_spec.set_block_num(static_cast<int16_t>(config.blocks));
    const MixedKernels kernels{
        (config.mask & 1) != 0 ? 0 : INVALID_KERNEL_ID,
        (config.mask & 2) != 0 ? 1 : INVALID_KERNEL_ID,
        (config.mask & 4) != 0 ? 2 : INVALID_KERNEL_ID,
    };
    const TaskId child = rt_submit_task(kernels, task).task_id();
    if (config.tail) {
        CoreTaskArgs tail;
        tail.set_dependencies(&child, 1);
        rt_submit_dummy_task(tail);
    }
}

}  // namespace

extern "C" {

__attribute__((visibility("default"))) OrchestrationConfig aicpu_orchestration_config(const ChipTaskArgs &) {
    return OrchestrationConfig{.expected_arg_count = 7};
}

__attribute__((visibility("default"))) void aicpu_orchestration_entry(const ChipTaskArgs &args) {
    const FailureConfig config(args.scalar<int64_t>(0), args.scalar<int64_t>(1), args.scalar<int64_t>(4) != 0);
    GraphTaskArgs boundary;
    boundary.add_inout(args.tensor(0).ref());
    for (int32_t scalar = 0; scalar < 4; ++scalar)
        boundary.add_scalar(args.scalar<int64_t>(scalar));
    rt_submit_graph(&body, boundary, config);
    CoreTaskArgs verify;
    verify.add_input(args.tensor(0).ref());
    verify.add_inout(args.tensor(1).ref());
    verify.add_scalar(config.blocks);
    verify.add_scalar(config.mask);
    verify.add_scalar(INT64_C(1));
    rt_submit_aiv_task(3, verify);
}

}  // extern "C"
