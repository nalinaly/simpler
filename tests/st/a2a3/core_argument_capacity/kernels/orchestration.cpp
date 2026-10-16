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

extern "C" __attribute__((visibility("default"))) OrchestrationConfig
aicpu_orchestration_config(const ChipTaskArgs &args) {
    return OrchestrationConfig{.expected_arg_count = args.tensor_count()};
}

extern "C" __attribute__((visibility("default"))) void aicpu_orchestration_entry(const ChipTaskArgs &entry) {
    // 第二次提交依赖同一输出；同时覆盖任务复用与 sync_start 上下文。
    for (int repeat = 0; repeat < 2; ++repeat) {
        CoreTaskArgs args;
        for (int i = 0; i < entry.tensor_count() - 1; ++i)
            args.add_input(entry.tensor(i).ref());
        args.add_inout(entry.tensor(entry.tensor_count() - 1).ref());
        int64_t scalars[CORE_MAX_SCALAR_ARGS];
        for (int i = 0; i < CORE_MAX_SCALAR_ARGS; ++i) {
            scalars[i] = 101 + i;
            args.add_scalar(scalars[i]);
        }
        // 选择跨 64/128 位边界的 Tensor，以及最后一个标量。
        args.dump(
            entry.tensor(63).ref(), entry.tensor(64).ref(), entry.tensor(128).ref(),
            entry.tensor(entry.tensor_count() - 1).ref(), scalars[15]
        );
        args.launch_spec.set_block_num(24);
        args.launch_spec.set_require_sync_start(true);
        args.set_allow_early_resolve(true);
        MixedKernels kernels;
        kernels.aic_kernel_id = 0;
        kernels.aiv0_kernel_id = 1;
        kernels.aiv1_kernel_id = 2;
        rt_submit_task(kernels, args);
    }
}
