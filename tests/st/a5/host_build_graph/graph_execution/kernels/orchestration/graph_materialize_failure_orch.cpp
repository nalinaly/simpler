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

void body(const GraphTaskArgs &args) {
    CoreTaskPredicate predicate;
    predicate.operand.tensor = &args.tensor(1).ref();
    predicate.operand.ndims = 1;
    predicate.operand.indices[0] = 0;
    predicate.op = PredicateOp::GT;
    predicate.target = 0;
    CoreTaskArgs task;
    task.add_inout(args.tensor(0).ref());
    task.set_predicate(predicate);
    rt_submit_aic_task(0, task);
}

}  // namespace

extern "C" {

__attribute__((visibility("default"))) OrchestrationConfig aicpu_orchestration_config(const ChipTaskArgs &args) {
    (void)args;
    return OrchestrationConfig{.expected_arg_count = 4};
}

__attribute__((visibility("default"))) void aicpu_orchestration_entry(const ChipTaskArgs &args) {
    for (int32_t instance = 0; instance < 2; ++instance) {
        simpler::hbg::Tensor gate = args.tensor(2).ref();
        gate.manual_dep = true;
        // Deliberately malformed replay metadata. The recorded body is valid,
        // and no kernel is allowed to dereference this out-of-buffer operand.
        // Scalar materialization must reject it before its startup gate opens.
        if (instance == 1 && args.scalar<int64_t>(0) != 0) gate.start_offset = 32;
        GraphTaskArgs boundary;
        boundary.add_inout(args.tensor(instance).ref());
        boundary.add_input(gate);
        rt_submit_graph(&body, boundary);
    }
}

}  // extern "C"
