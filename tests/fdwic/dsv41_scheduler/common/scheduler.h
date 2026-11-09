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

#include "model.h"

#ifndef DSV_DEVICE
#define DSV_DEVICE
#define DSV_GM
#endif

namespace dsv41 {
template <typename Ops, uint32_t Level>
DSV_DEVICE static uint64_t complete(DSV_GM Image *image, uint32_t id) {
    auto *control = &image->controls[id];
    const DSV_GM Task &task = image->tasks[id];
    if (Level != 0 || task.defer_ns != 0) Ops::store(&control->completed_ns, Ops::now());
    Ops::store(&control->status, kDone);
    for (uint32_t edge = 0; edge < task.successor_count; ++edge) {
        uint32_t succ = image->successors[task.successor_begin + edge];
        if (Ops::add(&image->controls[succ].remaining, UINT64_MAX) == 1)
            Ops::store(&image->controls[succ].status, kReady);
    }
    return Ops::add(&image->run.done, 1);
}

// A compact experimental scanner, shared by CPU and CCEC. It is not a copy of
// the production resident scheduler. All failed scans and CHECK service remain
// visible in the report, including work spent while dependencies are pending.
template <typename Ops, uint32_t Level>
DSV_DEVICE static void
event(DSV_GM Image *image, Worker &stats, uint32_t worker, uint32_t kind, uint32_t task, uint64_t begin, uint64_t end) {
    if constexpr (Level == 3) {
        if (stats.trace_count >= kTraceCapacity) {
            ++stats.trace_dropped;
            return;
        }
        auto &record = image->events[worker][stats.trace_count++];
        record.begin = begin;
        record.end = end;
        record.task = task;
        record.kind = kind;
    }
}

template <typename Ops, uint32_t Level>
DSV_DEVICE static void execute(DSV_GM Image *image, uint32_t worker, uint32_t physical_core) {
    const uint32_t worker_lane = worker % 3;
    Worker stats{};
    stats.begin = Ops::now();
    Ops::add(&image->run.arrived, 1);
    while (Ops::load(&image->run.arrived) != image->clusters * 3) {
        if (Ops::now() - stats.begin > kTimeoutNs) {
            Ops::store(&image->run.error, 1);
            return;
        }
    }
    uint32_t cursor = worker % image->task_count;
    uint32_t check_cursor = cursor;
    uint64_t accumulated_search = 0;
    uint64_t accumulated_service = 0;
    while (Ops::load(&image->run.done) != image->task_count && Ops::load(&image->run.error) == 0) {
        if (Ops::now() - stats.begin > kTimeoutNs) {
            Ops::store(&image->run.error, 2);
            break;
        }
        uint64_t check_start = Level == 3 ? Ops::now() : 0;
        const uint32_t checked_task = check_cursor;
        auto *check = &image->controls[check_cursor];
        uint64_t check_status = Ops::load(&check->status);
        uint64_t check_stamp = check_status;
        if (image->tasks[check_cursor].lanes == 0 && check_status == kReady) {
            if (Ops::cas(&check->status, kReady, 4) == kReady) check_stamp = complete<Ops, Level>(image, check_cursor);
        } else if (check_status == kPending) {
            ++stats.checks;
            if (Ops::now() >= Ops::load(&check->due) && Ops::cas(&check->status, kPending, 4) == kPending)
                check_stamp = complete<Ops, Level>(image, check_cursor);
        }
        if (++check_cursor == image->task_count) check_cursor = 0;
        uint64_t check_end = Level == 3 ? Ops::now_after(check_stamp) : 0;
        uint64_t check_ns = check_end - check_start;
        event<Ops, Level>(image, stats, worker, 1, checked_task, check_start, check_end);
        stats.check_ns += check_ns;
        accumulated_service += check_ns;
        if (check_ns > stats.max_check_ns) stats.max_check_ns = check_ns;
        stats.checks_over_budget += check_ns > image->budget_ns;

        const uint64_t start = Level == 3 ? Ops::now() : 0;
        uint32_t selected = kNone, block = 0, lane = worker_lane;
        // A full pass is measured as one operation; it is never split into
        // artificial sub-budget samples to conceal a slow ready search.
        for (uint32_t step = 0; step < image->task_count; ++step) {
            uint32_t id = cursor;
            if (++cursor == image->task_count) cursor = 0;
            const DSV_GM Task &task = image->tasks[id];
            lane = worker_lane == 0 ? 0 : ((task.lanes & 4) ? worker_lane : 1);
            if ((task.lanes & (1U << lane)) == 0 || Ops::load(&image->controls[id].status) != kReady) continue;
            auto *next = &image->controls[id].next[lane];
            uint64_t candidate = Ops::load(next);
            if (candidate >= task.blocks || Ops::cas(next, candidate, candidate + 1) != candidate) continue;
            selected = id;
            block = candidate;
            break;
        }
        uint64_t scan_end = Level == 3 ? Ops::now_after(static_cast<uint64_t>(selected) + block) : 0;
        uint64_t scan_ns = scan_end - start;
        event<Ops, Level>(image, stats, worker, selected == kNone ? 3 : 2, selected, start, scan_end);
        stats.scan_ns += scan_ns;
        if (scan_ns > stats.max_scan_ns) stats.max_scan_ns = scan_ns;
        stats.scans_over_budget += scan_ns > image->budget_ns;
        accumulated_search += scan_ns;
        if (selected == kNone) {
            stats.empty_scan_ns += scan_ns;
            continue;
        }
        const uint64_t prepare_start = Level == 3 ? Ops::now() : 0;
        const DSV_GM Task &task = image->tasks[selected];
        uint64_t predecessor_sum = 0;
        for (uint32_t i = 0; i < task.predecessor_count; ++i) {
            uint32_t pred = image->predecessors[task.predecessor_begin + i];
            if (Ops::load(&image->controls[pred].status) != kDone) Ops::store(&image->run.error, 3);
            predecessor_sum += Ops::load(&image->controls[pred].checksum);
        }
        Sample sample{};
        sample.worker = worker;
        sample.physical_core = physical_core;
        sample.task = selected;
        sample.block = block;
        sample.lane = lane;
        sample.search_ns = accumulated_search;
        sample.service_ns = accumulated_service;
        accumulated_search = 0;
        accumulated_service = 0;
        sample.begin = Level != 0 ? Ops::now_after(predecessor_sum) : 0;
        sample.prepare_ns = Level == 3 ? sample.begin - prepare_start : 0;
        if (!image->empty_kernel) Ops::compute(image, worker);
        sample.end = Level != 0 ? Ops::now() : 0;
        sample.value = output_value(selected, block, lane) + predecessor_sum;
        auto *control = &image->controls[selected];
        Ops::add(&control->checksum, sample.value);
        uint64_t completion_stamp = Ops::add(&control->returned, 1);
        if (completion_stamp + 1 == task.blocks * lane_count(task.lanes)) {
            if (task.defer_ns != 0) {
                Ops::store(&control->due, Ops::now() + task.defer_ns);
                completion_stamp = Ops::store(&control->status, kPending);
            } else completion_stamp = complete<Ops, Level>(image, selected);
        }
        const uint64_t record_start = Level == 3 ? Ops::now_after(completion_stamp) : 0;
        sample.finish_ns = Level == 3 ? record_start - sample.end : 0;
        if constexpr (Level != 0) Ops::save_sample(&image->samples[sample_index(task, lane, block)], sample);
        if constexpr (Level == 3) stats.record_ns += Ops::now() - record_start;
        ++stats.executed;
    }
    stats.end = Ops::now();
    if constexpr (Level == 3) Ops::flush_events(image->events[worker], stats.trace_count);
    Ops::save_worker(&image->workers[worker], stats);
}
template <typename Ops>
DSV_DEVICE static void dispatch(DSV_GM Image *image, uint32_t worker, uint32_t physical) {
    if (image->profiling == 0) execute<Ops, 0>(image, worker, physical);
    else if (image->profiling == 1) execute<Ops, 1>(image, worker, physical);
    else execute<Ops, 3>(image, worker, physical);
}
}  // namespace dsv41
