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

struct ParameterConfig : simpler::hbg::ConfigBase {
    int64_t scalars;
    int64_t lane;
    ParameterConfig(int64_t count, int64_t core) :
        scalars(count),
        lane(core) {}
    uint64_t hash() const { return simpler::hbg::config_hash(scalars, lane); }
    bool operator==(const ParameterConfig &other) const { return scalars == other.scalars && lane == other.lane; }
};

void body(const GraphTaskArgs &boundary, const ParameterConfig &config) {
    CoreTaskArgs args;
    args.add_inout(boundary.tensor(0).ref(), boundary.tensor(1).ref());
    args.add_scalar(boundary.scalar(0));
    for (int64_t index = 1; index < config.scalars; ++index) {
        if (index % 2 == 0) args.add_scalar(boundary.scalar(1));
        else args.add_scalar(100 + index);
    }
    if (config.lane == 0) rt_submit_aic_task(0, args);
    else rt_submit_aiv_task(1, args);
}

}  // namespace

extern "C" {

__attribute__((visibility("default"))) OrchestrationConfig aicpu_orchestration_config(const ChipTaskArgs &) {
    return OrchestrationConfig{.expected_arg_count = 4};
}

__attribute__((visibility("default"))) void aicpu_orchestration_entry(const ChipTaskArgs &args) {
    const int64_t instances = args.scalar<int64_t>(0);
    const ParameterConfig config(args.scalar<int64_t>(1), args.scalar<int64_t>(2));
    const uint32_t shape[1] = {32};
    for (int64_t instance = 0; instance < instances; ++instance) {
        const uint32_t left_offset[1] = {static_cast<uint32_t>(instance * 64)};
        const uint32_t right_offset[1] = {left_offset[0] + 32};
        const auto left = args.tensor(0).ref().view(shape, left_offset);
        const auto right = args.tensor(0).ref().view(shape, right_offset);
        GraphTaskArgs boundary;
        boundary.add_inout(left, right);
        boundary.add_scalar(config.scalars + ((instance + 1) << 8));
        boundary.add_scalar(0x100000 + instance);
        rt_submit_graph(&body, boundary, config);
    }
}

}  // extern "C"
