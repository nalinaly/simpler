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

#include <algorithm>
#include <fstream>
#include <memory>
#include <sstream>
#include <stdexcept>
#include <string>
#include <vector>

namespace dsv41 {
using Frontier = std::vector<uint32_t>;
inline const std::vector<std::string> kCases{"swa",       "c2a_full", "c2a_reuse", "c1a_full",    "c1a_reindex",
                                             "c1a_reuse", "moe",      "engram",    "layers_19_21"};

struct Workload {
    std::unique_ptr<Image> image{new Image{}};
    std::vector<std::string> names;
    std::vector<Frontier> dependencies;
};

inline Workload
make_workload(const std::string &directory, uint32_t clusters, uint32_t repeats, uint64_t defer_ns, bool empty) {
    if (clusters == 0 || clusters > 32) throw std::runtime_error("clusters must be 1..32");
    Workload graph;
    auto &image = *graph.image;
    image.clusters = clusters;
    image.repeats = repeats;
    image.empty_kernel = empty;
    std::ifstream tasks(directory + "/tasks.tsv"), edges(directory + "/edges.tsv");
    if (!tasks || !edges) throw std::runtime_error("missing captured workload: " + directory);
    std::string line;
    while (std::getline(tasks, line)) {
        if (line.empty() || line[0] == '#') continue;
        if (image.task_count == kMaxTasks) throw std::runtime_error("task capacity exceeded");
        std::istringstream row(line);
        uint64_t source_id;
        uint32_t lanes, blocks, deferred;
        std::string name, extra;
        if (!(row >> source_id >> name >> lanes >> blocks >> deferred) || (row >> extra) || lanes > 7 || blocks == 0 ||
            blocks > kMaxExecutions || deferred > 1)
            throw std::runtime_error("invalid task row");
        uint32_t id = image.task_count++;
        auto &task = image.tasks[id];
        task.lanes = lanes;
        task.blocks = blocks;
        task.sample_base = image.execution_count;
        task.stage = id;
        task.defer_ns = deferred ? defer_ns : 0;
        image.execution_count += blocks * lane_count(lanes);
        if (image.execution_count > kMaxExecutions) throw std::runtime_error("execution capacity exceeded");
        graph.names.push_back(name);
        graph.dependencies.emplace_back();
    }
    if (image.task_count == 0 || image.execution_count == 0) throw std::runtime_error("empty task graph");
    while (std::getline(edges, line)) {
        if (line.empty() || line[0] == '#') continue;
        std::istringstream row(line);
        uint32_t pred, succ;
        std::string extra;
        if (!(row >> pred >> succ) || (row >> extra) || pred >= succ || succ >= image.task_count)
            throw std::runtime_error("invalid or non-topological edge");
        graph.dependencies[succ].push_back(pred);
    }
    std::vector<Frontier> successors(image.task_count);
    uint32_t pred_cursor = 0, succ_cursor = 0;
    for (uint32_t id = 0; id < image.task_count; ++id) {
        auto &deps = graph.dependencies[id];
        std::sort(deps.begin(), deps.end());
        if (std::adjacent_find(deps.begin(), deps.end()) != deps.end()) throw std::runtime_error("duplicate edge");
        auto &task = image.tasks[id];
        task.predecessor_begin = pred_cursor;
        task.predecessor_count = deps.size();
        for (uint32_t pred : deps) {
            if (pred_cursor >= kMaxEdges) throw std::runtime_error("edge capacity exceeded");
            image.predecessors[pred_cursor++] = pred;
            successors[pred].push_back(id);
        }
        image.controls[id].remaining = task.predecessor_count;
        image.controls[id].status = task.predecessor_count ? kWaiting : kReady;
    }
    for (uint32_t id = 0; id < image.task_count; ++id) {
        auto &task = image.tasks[id];
        task.successor_begin = succ_cursor;
        task.successor_count = successors[id].size();
        for (uint32_t succ : successors[id])
            image.successors[succ_cursor++] = succ;
    }
    for (uint32_t worker = 0; worker < clusters * 3; ++worker)
        for (uint32_t i = 0; i < kTileElements; ++i) {
            image.tiles[worker][0][i] = 2;
            image.tiles[worker][1][i] = 3;
        }
    return graph;
}
}  // namespace dsv41
