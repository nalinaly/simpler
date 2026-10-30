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

#include <vector>

#include "scheduler_graph_plan.h"
#include "scheduler_protocol_graph.h"

struct SchedulerGraphDefinitionCapture {
    GraphDefinition header{};
    uint32_t image_offset{0};
    std::vector<SubTaskDefinition> tasks;
    std::vector<int32_t> fanin_rows;
    std::vector<uint16_t> fanin_indices;
};

struct SchedulerGraphManifest {
    std::vector<SchedulerGraphPlanRange> ranges;
    std::vector<uint32_t> instance_definitions;
    std::vector<SchedulerGraphDefinitionCapture> definitions;
};

struct SchedulerGraphCapture {
    std::vector<SchedulerProtocolTaskSpec> tasks;
    SchedulerGraphManifest manifest;
};
