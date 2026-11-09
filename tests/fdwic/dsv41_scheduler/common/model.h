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

#ifndef DSV_DEVICE
#define DSV_DEVICE
#define DSV_GM
#endif

namespace dsv41 {
constexpr uint32_t kMaxTasks = 4096;
constexpr uint32_t kMaxEdges = 131072;
constexpr uint32_t kMaxExecutions = 32768;
constexpr uint32_t kMaxWorkers = 96;
constexpr uint32_t kTraceCapacity = 16384;
struct Event {
    uint64_t begin, end;
    uint32_t task, kind;
};
constexpr uint32_t kTileElements = 256;
constexpr uint64_t kTimeoutNs = 5000000000ULL;
constexpr uint32_t kNone = UINT32_MAX;
enum : uint32_t { kCube = 1, kVector = 2, kMix = 3 };
enum : uint64_t { kWaiting = 0, kReady = 1, kPending = 2, kDone = 3 };

struct Task {
    uint32_t lanes{0};
    uint32_t blocks{0};
    uint32_t sample_base{0};
    uint32_t predecessor_begin{0};
    uint32_t predecessor_count{0};
    uint32_t successor_begin{0};
    uint32_t successor_count{0};
    uint32_t stage{0};
    uint64_t defer_ns{0};
};

// Shared control contains only atomic words. Payload and diagnostic writes never
// clean a line containing another worker's atomic state.
struct alignas(128) Control {
    uint64_t status{0};
    uint64_t remaining{0};
    uint64_t next[3]{};
    uint64_t returned{0};
    uint64_t due{0};
    uint64_t completed_ns{0};
    uint64_t checksum{0};
};
struct Sample {
    uint64_t begin{0};
    uint64_t end{0};
    uint64_t search_ns{0};
    uint64_t prepare_ns{0};
    uint64_t finish_ns{0};
    uint64_t service_ns{0};
    uint64_t value{0};
    uint32_t worker{kNone};
    uint32_t physical_core{kNone};
    uint32_t task{kNone};
    uint32_t block{kNone};
    uint32_t lane{kNone};
    uint32_t reserved[13]{};
};
struct Worker {
    uint64_t begin{0};
    uint64_t end{0};
    uint64_t scan_ns{0};
    uint64_t check_ns{0};
    uint64_t empty_scan_ns{0};
    uint64_t max_scan_ns{0};
    uint64_t scans_over_budget{0};
    uint64_t max_check_ns{0};
    uint64_t checks_over_budget{0};
    uint64_t checks{0};
    uint64_t executed{0};
    uint64_t record_ns{0};
    uint64_t trace_count{0};
    uint64_t trace_dropped{0};
    uint64_t reserved[2]{};
};
static_assert(sizeof(Sample) == 128 && sizeof(Worker) == 128);
struct alignas(128) Run {
    uint64_t arrived{0};
    uint64_t done{0};
    uint64_t error{0};
    uint64_t start{0};
};
struct Image {
    uint32_t task_count{0};
    uint32_t execution_count{0};
    uint32_t clusters{0};
    uint32_t repeats{1};
    uint64_t budget_ns{5000};
    uint64_t empty_kernel{0};
    uint64_t profiling{3};
    Run run{};
    Task tasks[kMaxTasks]{};
    uint32_t predecessors[kMaxEdges]{};
    uint32_t successors[kMaxEdges]{};
    Control controls[kMaxTasks]{};
    alignas(128) Sample samples[kMaxExecutions]{};
    alignas(128) Worker workers[kMaxWorkers]{};
    // Independent per-worker engine workspaces: input A, input B, output.
    alignas(128) float tiles[kMaxWorkers][3][kTileElements]{};
    alignas(128) Event events[kMaxWorkers][kTraceCapacity]{};
};

DSV_DEVICE inline constexpr uint32_t lane_count(uint32_t mask) {
    return ((mask & 1) != 0) + ((mask & 2) != 0) + ((mask & 4) != 0);
}
DSV_DEVICE inline constexpr uint32_t sample_index(const DSV_GM Task &task, uint32_t lane, uint32_t block) {
    return task.sample_base + block * lane_count(task.lanes) + lane_count(task.lanes & ((1U << lane) - 1));
}
DSV_DEVICE inline constexpr uint64_t output_value(uint32_t task, uint32_t block, uint32_t lane) {
    return (static_cast<uint64_t>(task + 1) << 32) + (block + 1) * 4 + lane;
}
}  // namespace dsv41
