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

#include <cstdint>
#include <memory>

#include "common/host_api.h"
#include "hbg_graph_plan.h"
#include "task_args.h"

class Runtime;

/**
 * Runtime-specific host-build hook used only by HBG L1.
 *
 * The caller supplies a trusted frozen destination binding and a validated
 * identity whose host_total_tasks field is zero. The HBG implementation runs
 * host orchestration over borrowed device tensor addresses, fills the actual
 * task count, and publishes an immutable owning plan transactionally. TRB has
 * no strong definition and remains on its fixed L1 invocation ABI.
 */
extern "C" int build_l1_hbg_graph_plan_impl(
    Runtime *runtime, const HostApi *api, const ChipStorageTaskArgs *orch_args, void *host_orch_func_ptr,
    const simpler::hbg::HbgExecutionBinding *binding, const simpler::hbg::HbgInvocationIdentity *identity,
    uint64_t plan_generation, const uint64_t *ring_task_window, const uint64_t *ring_heap,
    const uint64_t *ring_dep_pool, std::unique_ptr<const simpler::hbg::HbgGraphPlan> *out
);
