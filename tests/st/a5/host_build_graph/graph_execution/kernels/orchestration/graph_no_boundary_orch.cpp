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

#include "orchestration_api.h"

namespace {

struct ServiceConfig : simpler::hbg::ConfigBase {
    explicit ServiceConfig(int64_t count) :
        children(count) {}
    int64_t children;
    uint64_t hash() const { return simpler::hbg::config_hash(children); }
    bool operator==(const ServiceConfig &other) const { return children == other.children; }
};

void service_body(const GraphTaskArgs &, const ServiceConfig &config) {
    TaskId previous = TaskId::invalid();
    for (int64_t child = 0; child < config.children; ++child) {
        CoreTaskArgs args;
        if (child != 0) args.set_dependencies(&previous, 1);
        previous = rt_submit_dummy_task(args).task_id();
    }
}

}  // namespace

extern "C" __attribute__((visibility("default"))) OrchestrationConfig aicpu_orchestration_config(const ChipTaskArgs &) {
    return OrchestrationConfig{.expected_arg_count = 2};
}

extern "C" __attribute__((visibility("default"))) void aicpu_orchestration_entry(const ChipTaskArgs &args) {
    const int64_t instances = args.scalar<int64_t>(0);
    const ServiceConfig config(args.scalar<int64_t>(1));
    for (int64_t index = 0; index < instances; ++index) {
        GraphTaskArgs boundary;
        rt_submit_graph(&service_body, boundary, config);
    }
}
