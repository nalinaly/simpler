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

#include "workloads.h"

#include <fstream>
#include <limits>

namespace dsv41 {
inline void require(bool valid, const char *message) {
    if (!valid) throw std::runtime_error(message);
}

inline void validate(const Workload &graph) {
    const Image &image = *graph.image;
    require(image.run.error == 0, "device scheduler reported an error");
    require(image.run.done == image.task_count, "not all tasks completed");
    std::vector<uint64_t> expected(image.task_count);
    uint64_t execution_count = 0;
    for (uint32_t id = 0; id < image.task_count; ++id) {
        const Task &task = image.tasks[id];
        const Control &control = image.controls[id];
        uint64_t input = 0, ready = 0, last_return = 0;
        for (uint32_t pred : graph.dependencies[id]) {
            input += expected[pred];
            ready = std::max(ready, image.controls[pred].completed_ns);
        }
        for (uint32_t lane = 0; lane < 3; ++lane) {
            require(control.next[lane] == ((task.lanes & (1U << lane)) ? task.blocks : 0), "wrong block claim count");
            if ((task.lanes & (1U << lane)) == 0) continue;
            for (uint32_t block = 0; block < task.blocks; ++block) {
                uint64_t value = output_value(id, block, lane) + input;
                expected[id] += value;
                ++execution_count;
                if (image.profiling == 0) continue;
                const Sample &s = image.samples[sample_index(task, lane, block)];
                require(s.task == id && s.block == block && s.lane == lane, "missing or incorrect execution identity");
                require(
                    s.worker < image.clusters * 3 && (lane == 0 ? s.worker % 3 == 0 : s.worker % 3 != 0),
                    "wrong worker role"
                );
                if (task.lanes & 4) require(s.worker % 3 == lane, "wrong dual-AIV subtask slot");
                require(s.begin >= ready && s.end >= s.begin, "consumer ran before its dependency");
                require(s.value == value, "wrong dependency payload");
                last_return = std::max(last_return, s.end);
            }
        }
        require(control.returned == task.blocks * lane_count(task.lanes), "wrong SPMD/group return count");
        require(control.status == kDone && control.remaining == 0, "task did not join correctly");
        require(control.completed_ns >= last_return, "logical completion preceded a block return");
        require(control.completed_ns >= control.due, "deferred task completed before notification");
        require(control.checksum == expected[id], "incorrect task checksum");
    }
    require(execution_count == image.execution_count, "execution count mismatch");
    uint64_t worker_count = 0;
    for (uint32_t w = 0; w < image.clusters * 3; ++w) {
        worker_count += image.workers[w].executed;
        require(image.workers[w].trace_dropped == 0, "L3 event buffer overflow");
        require(image.workers[w].trace_count <= kTraceCapacity, "invalid trace count");
        for (uint64_t j = 0; j < image.workers[w].trace_count; ++j) {
            const Event &event = image.events[w][j];
            require(event.end >= event.begin && event.kind >= 1 && event.kind <= 3, "invalid L3 interval");
        }
        if (!image.empty_kernel && image.workers[w].executed) {
            float expected_value = w % 3 == 0 ? 96 : 5;
            for (float value : image.tiles[w][2])
                require(value == expected_value, "engine output mismatch");
        }
    }
    require(worker_count == execution_count, "worker accounting mismatch");
}

inline void write_results(
    const Workload &graph, const std::string &prefix, const std::string &backend, const std::string &flavor,
    const std::string &name, uint32_t round
) {
    const Image &image = *graph.image;
    std::ofstream samples(prefix + ".samples.csv"), workers(prefix + ".workers.csv"), summary(prefix + ".json");
    require(samples.good() && workers.good() && summary.good(), "cannot create output files");
    if (image.profiling != 0) {
        std::ofstream tasks(prefix + ".tasks.csv");
        require(tasks.good(), "cannot create task completion file");
        tasks << "task,stage,completed_ns,due_ns\n";
        for (uint32_t t = 0; t < image.task_count; ++t)
            tasks << t << ',' << graph.names[t] << ',' << image.controls[t].completed_ns << ',' << image.controls[t].due
                  << '\n';
    }
    samples << "task,stage,blocks,block,lane,worker,physical_core,begin_ns,end_ns,search_ns,prepare_ns,finish_ns,"
               "service_ns\n";
    std::vector<uint64_t> durations;
    for (uint32_t i = 0; image.profiling != 0 && i < image.execution_count; ++i) {
        const Sample &s = image.samples[i];
        durations.push_back(s.search_ns + s.prepare_ns + s.finish_ns + s.service_ns);
        samples << s.task << ',' << graph.names[s.task] << ',' << image.tasks[s.task].blocks << ',' << s.block << ','
                << s.lane << ',' << s.worker << ',' << s.physical_core << ',' << s.begin << ',' << s.end << ','
                << s.search_ns << ',' << s.prepare_ns << ',' << s.finish_ns << ',' << s.service_ns << '\n';
    }
    workers << "worker,begin_ns,end_ns,scan_ns,check_ns,empty_scan_ns,max_scan_ns,scans_over_budget,max_check_ns,"
               "checks_over_budget,checks,executed,record_ns\n";
    uint64_t first = UINT64_MAX, last = 0, check_ns = 0, empty_ns = 0;
    for (uint32_t w = 0; w < image.clusters * 3; ++w) {
        const Worker &s = image.workers[w];
        first = std::min(first, s.begin);
        last = std::max(last, s.end);
        check_ns += s.check_ns;
        empty_ns += s.empty_scan_ns;
        workers << w << ',' << s.begin << ',' << s.end << ',' << s.scan_ns << ',' << s.check_ns << ','
                << s.empty_scan_ns << ',' << s.max_scan_ns << ',' << s.scans_over_budget << ',' << s.max_check_ns << ','
                << s.checks_over_budget << ',' << s.checks << ',' << s.executed << ',' << s.record_ns << '\n';
    }
    if (image.profiling == 3) {
        std::ofstream events(prefix + ".events.csv");
        require(events.good(), "cannot create event file");
        events << "worker,kind,task,begin_ns,end_ns\n";
        for (uint32_t w = 0; w < image.clusters * 3; ++w)
            for (uint64_t i = 0; i < image.workers[w].trace_count; ++i) {
                const Event &e = image.events[w][i];
                events << w << ',' << e.kind << ',' << e.task << ',' << e.begin << ',' << e.end << '\n';
            }
    }
    std::sort(durations.begin(), durations.end());
    uint64_t over = std::count_if(durations.begin(), durations.end(), [&](uint64_t n) {
        return n > image.budget_ns;
    });
    summary << "{\n  \"backend\": \"" << backend << "\",\n  \"flavor\": \"" << flavor << "\",\n  \"case\": \"" << name
            << "\",\n  \"round\": " << round << ",\n  \"passed\": true,\n  \"tasks\": " << image.task_count
            << ",\n  \"executions\": " << image.execution_count << ",\n  \"clusters\": " << image.clusters
            << ",\n  \"profiling\": " << image.profiling << ",\n  \"repeats\": " << image.repeats
            << ",\n  \"wall_ns\": " << last - first << ",\n  \"scheduler_median_ns\": "
            << (image.profiling == 3 ? std::to_string(durations[durations.size() / 2]) : "null")
            << ",\n  \"scheduler_p99_ns\": "
            << (image.profiling == 3 ? std::to_string(durations[(durations.size() - 1) * 99 / 100]) : "null")
            << ",\n  \"scheduler_max_ns\": " << (image.profiling == 3 ? std::to_string(durations.back()) : "null")
            << ",\n  \"over_5us\": " << (image.profiling == 3 ? std::to_string(over) : "null")
            << ",\n  \"check_core_ns\": " << check_ns << ",\n  \"empty_scan_core_ns\": " << empty_ns
            << ",\n  \"clock\": \""
            << (backend == "a5" ? "AICore SYS_CNT 1GHz" : "CPU steady_clock; NOT A5 performance")
            << "\",\n  \"scheduler\": \"standalone reference scanner; NOT production resident\",\n"
               "  \"workload\": \"captured task/block/edge slice; replacement compute and boundary inputs\",\n"
               "  \"async\": \"device deadline notification model; NO inter-device transfer\"\n}\n";
}
}  // namespace dsv41
