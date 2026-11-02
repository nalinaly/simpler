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

#include <algorithm>
#include <cstddef>
#include <cstdint>
#include <optional>
#include <utility>
#include <vector>

#include "scheduler/scheduler_graph_work.h"

struct SchedulerGraphMaterializeRegion {
    SchedulerGraphWorkRegion kind;
    uint64_t address;
    uint32_t element_count;
    uint32_t element_bytes;
    std::vector<SchedulerGraphWorkRange> referenced;
};

class SchedulerGraphMaterializePlan {
public:
    std::vector<SchedulerGraphWork> work;
    std::vector<SchedulerGraphWorkRange> ranges;

    static std::optional<SchedulerGraphMaterializePlan>
    build(std::vector<SchedulerGraphMaterializeRegion> regions, uint32_t instance_index, uint32_t target_bytes) {
        if (target_bytes < kWritebackBytes) return std::nullopt;
        std::sort(regions.begin(), regions.end(), [](const auto &lhs, const auto &rhs) {
            return static_cast<uint32_t>(lhs.kind) < static_cast<uint32_t>(rhs.kind);
        });
        std::vector<LineRange> lines;
        for (size_t index = 0; index < regions.size(); ++index) {
            auto &region = regions[index];
            if (region.kind >= SchedulerGraphWorkRegion::SERIAL_INSTANCE ||
                (index != 0 && region.kind == regions[index - 1].kind) || !normalize(&region, &lines))
                return std::nullopt;
        }
        if (!regions_are_disjoint(std::move(lines))) return std::nullopt;
        SchedulerGraphMaterializePlan plan;
        for (const auto &region : regions) {
            if (!plan.append_region(region, instance_index, target_bytes)) return std::nullopt;
        }
        return plan;
    }

private:
    static constexpr uint32_t kWritebackBytes = 64;

    struct LineRange {
        uint64_t first;
        uint64_t last;
        SchedulerGraphWorkRegion region;
    };

    static bool normalize(SchedulerGraphMaterializeRegion *region, std::vector<LineRange> *lines) {
        const uint32_t stride = region->element_bytes;
        if (stride == 0 || (stride < kWritebackBytes ? kWritebackBytes % stride != 0 : stride % kWritebackBytes != 0))
            return false;
        const uint32_t alignment = std::min(stride, kWritebackBytes);
        const uint64_t bytes = static_cast<uint64_t>(region->element_count) * stride;
        if (region->address % alignment != 0 || bytes > UINT64_MAX - region->address) return false;
        auto &input = region->referenced;
        std::sort(input.begin(), input.end(), [](const auto &lhs, const auto &rhs) {
            return lhs.begin < rhs.begin;
        });
        std::vector<SchedulerGraphWorkRange> merged;
        for (const auto &range : input) {
            if (range.begin > region->element_count || range.count > region->element_count - range.begin) return false;
            if (range.count == 0) continue;
            const uint32_t end = range.begin + range.count;
            if (!merged.empty() && range.begin <= merged.back().begin + merged.back().count) {
                auto &previous = merged.back();
                previous.count = std::max(previous.begin + previous.count, end) - previous.begin;
            } else {
                merged.push_back(range);
            }
        }
        input = std::move(merged);
        for (const auto &range : input) {
            const uint64_t first = region->address + static_cast<uint64_t>(range.begin) * stride;
            const uint64_t end = first + static_cast<uint64_t>(range.count) * stride;
            lines->push_back({first / kWritebackBytes, (end - 1) / kWritebackBytes, region->kind});
        }
        return true;
    }

    static bool regions_are_disjoint(std::vector<LineRange> lines) {
        if (lines.empty()) return true;
        std::sort(lines.begin(), lines.end(), [](const auto &lhs, const auto &rhs) {
            return lhs.first < rhs.first;
        });
        auto active = lines.front();
        for (size_t index = 1; index < lines.size(); ++index) {
            const auto &next = lines[index];
            if (next.first <= active.last) {
                if (next.region != active.region) return false;
                active.last = std::max(active.last, next.last);
            } else {
                active = next;
            }
        }
        return true;
    }

    bool append_region(const SchedulerGraphMaterializeRegion &region, uint32_t instance_index, uint32_t target_bytes) {
        SchedulerGraphWork current{instance_index, region.kind, 0, 0, 0};
        uint64_t last_line = 0;
        for (const auto &range : region.referenced) {
            uint32_t cursor = range.begin;
            const uint32_t end = range.begin + range.count;
            while (cursor < end) {
                const uint64_t address = region.address + static_cast<uint64_t>(cursor) * region.element_bytes;
                const uint32_t line_elements = region.element_bytes >= kWritebackBytes ?
                                                   1 :
                                                   (kWritebackBytes - address % kWritebackBytes) / region.element_bytes;
                const uint32_t count = std::min(end - cursor, line_elements);
                const uint64_t bytes = static_cast<uint64_t>(count) * region.element_bytes;
                const uint64_t first_line = address / kWritebackBytes;
                // Disjoint referenced intervals in one physical line share the
                // same work item even when the byte target has been reached.
                if (current.range_count != 0 && current.write_bytes + bytes > target_bytes && first_line > last_line) {
                    if (work.size() == UINT32_MAX) return false;
                    work.push_back(current);
                    current.range_count = 0;
                    current.write_bytes = 0;
                }
                if (current.range_count == 0) {
                    if (ranges.size() == UINT32_MAX) return false;
                    current.first_range = static_cast<uint32_t>(ranges.size());
                }
                if (current.range_count != 0 && ranges.back().begin + ranges.back().count == cursor) {
                    ranges.back().count += count;
                } else {
                    if (ranges.size() == UINT32_MAX || current.range_count == UINT32_MAX) return false;
                    ranges.push_back({cursor, count});
                    ++current.range_count;
                }
                current.write_bytes += bytes;
                last_line = (address + bytes - 1) / kWritebackBytes;
                cursor += count;
            }
        }
        if (current.range_count != 0) {
            if (work.size() == UINT32_MAX) return false;
            work.push_back(current);
        }
        return true;
    }
};
