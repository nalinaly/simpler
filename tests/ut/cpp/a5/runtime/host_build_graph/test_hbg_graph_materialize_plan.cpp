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

#include "inner_kernel.h"

#include <gtest/gtest.h>

#include <algorithm>
#include <array>
#include <cstdint>
#include <map>
#include <numeric>
#include <random>
#include <vector>

#include "hbg_graph_test_support.h"
#include "scheduler_graph_materialize_plan.h"

namespace {

using Region = SchedulerGraphMaterializeRegion;
using Kind = SchedulerGraphWorkRegion;
using Plan = SchedulerGraphMaterializePlan;

void check_coverage(const std::vector<Region> &regions, const Plan &plan) {
    std::array<std::vector<uint32_t>, 3> counts;
    std::array<std::vector<uint32_t>, 3> expected;
    for (const auto &region : regions) {
        const auto kind = static_cast<size_t>(region.kind);
        counts[kind].resize(region.element_count);
        expected[kind].resize(region.element_count);
        for (const auto &range : region.referenced) {
            for (uint32_t i = 0; i < range.count; ++i)
                expected[kind][range.begin + i] = 1;
        }
    }
    std::map<uint64_t, size_t> writers;
    for (size_t work_index = 0; work_index < plan.work.size(); ++work_index) {
        const auto &work = plan.work[work_index];
        ASSERT_EQ(work.instance_index, 7u);
        ASSERT_NE(work.range_count, 0u);
        ASSERT_LE(static_cast<uint64_t>(work.first_range) + work.range_count, plan.ranges.size());
        auto found = std::find_if(regions.begin(), regions.end(), [&](const auto &region) {
            return region.kind == work.region;
        });
        ASSERT_NE(found, regions.end());
        const auto &region = *found;
        uint64_t bytes = 0;
        for (uint32_t part = 0; part < work.range_count; ++part) {
            const auto &range = plan.ranges[work.first_range + part];
            ASSERT_NE(range.count, 0u);
            for (uint32_t i = 0; i < range.count; ++i) {
                const uint32_t element = range.begin + i;
                ASSERT_LT(element, region.element_count);
                ++counts[static_cast<size_t>(work.region)][element];
                const uint64_t start = region.address + static_cast<uint64_t>(element) * region.element_bytes;
                const uint64_t last = (start + region.element_bytes - 1) / 64;
                for (uint64_t line = start / 64; line <= last; ++line) {
                    auto [owner, inserted] = writers.emplace(line, work_index);
                    EXPECT_TRUE(inserted || owner->second == work_index) << "shared writeback line " << line;
                }
                bytes += region.element_bytes;
            }
        }
        EXPECT_EQ(work.write_bytes, bytes);
    }
    EXPECT_EQ(counts, expected);
}

TEST(HbgGraphMaterializePlan, LargeSingleInstanceHasIndependentWorkInEveryRegion) {
    const std::vector<Region> input{
        {Kind::TASK_STORAGE, 0x10000, 1024, 320, {{0, 1024}}},
        {Kind::TENSOR_POOL, 0x100000, 1024, 128, {{0, 1024}}},
        {Kind::SCALAR_POOL, 0x200000, 1024, 8, {{0, 1024}}},
    };
    auto plan = Plan::build(input, 7, 4096);
    ASSERT_TRUE(plan);
    for (const auto &region : input) {
        EXPECT_GT(
            std::count_if(
                plan->work.begin(), plan->work.end(),
                [&](const auto &work) {
                    return work.region == region.kind;
                }
            ),
            1
        );
    }
    check_coverage(input, *plan);
}

TEST(HbgGraphMaterializePlan, AliasedRangesAreDeduplicatedAndHolesRemainUnwritten) {
    const std::vector<Region> input{
        {Kind::TENSOR_POOL, 0x10000, 20, 128, {{8, 5}, {1, 5}, {3, 3}, {8, 5}, {20, 0}}},
        {Kind::SCALAR_POOL, 0x20000, 40, 8, {{0, 2}, {3, 3}, {9, 1}, {15, 10}, {28, 8}}},
    };
    auto plan = Plan::build(input, 7, 64);
    ASSERT_TRUE(plan);
    check_coverage(input, *plan);
}

TEST(HbgGraphMaterializePlan, UnalignedScalarPoolKeepsEveryPhysicalLineWithOneWriter) {
    const std::vector<Region> input{{Kind::SCALAR_POOL, 0x10038, 33, 8, {{0, 33}}}};
    auto plan = Plan::build(input, 7, 64);
    ASSERT_TRUE(plan);
    EXPECT_GT(plan->work.size(), 1u);
    check_coverage(input, *plan);
}

TEST(HbgGraphMaterializePlan, DifferentRegionsCannotOwnTheSameWritebackLine) {
    EXPECT_FALSE(
        Plan::build(
            {{Kind::TENSOR_POOL, 0x10000, 1, 128, {{0, 1}}}, {Kind::SCALAR_POOL, 0x10078, 1, 8, {{0, 1}}}}, 7, 64
        )
    );
}

TEST(HbgGraphMaterializePlan, EmptyRegionsCreateNoWorkOrRangeStorage) {
    auto plan = Plan::build({{Kind::TENSOR_POOL, 0, 0, 128, {}}, {Kind::SCALAR_POOL, 0, 0, 8, {}}}, 7, 4096);
    ASSERT_TRUE(plan);
    EXPECT_TRUE(plan->work.empty());
    EXPECT_TRUE(plan->ranges.empty());
}

TEST(HbgGraphMaterializePlan, RejectsOverflowMisalignmentAndInvalidRanges) {
    EXPECT_FALSE(Plan::build({{Kind::SCALAR_POOL, UINT64_MAX - 7, 2, 8, {{0, 2}}}}, 7, 64));
    EXPECT_FALSE(Plan::build({{Kind::SCALAR_POOL, 0x10001, 1, 8, {{0, 1}}}}, 7, 64));
    EXPECT_FALSE(Plan::build({{Kind::TENSOR_POOL, 0x10008, 1, 128, {{0, 1}}}}, 7, 64));
    EXPECT_FALSE(Plan::build({{Kind::SCALAR_POOL, 0x10000, 3, 8, {{2, 2}}}}, 7, 64));
    EXPECT_FALSE(Plan::build({{Kind::SCALAR_POOL, 0x10000, 3, 8, {{UINT32_MAX, 2}}}}, 7, 64));
    EXPECT_FALSE(Plan::build({{Kind::SCALAR_POOL, 0, 0, 0, {}}}, 7, 64));
    EXPECT_FALSE(Plan::build({{Kind::SCALAR_POOL, 0x10000, 1, 8, {{0, 1}}}}, 7, 63));
    EXPECT_FALSE(Plan::build({{Kind::SERIAL_INSTANCE, 0, 0, 8, {}}}, 7, 64));
    EXPECT_FALSE(Plan::build({{Kind::SCALAR_POOL, 0, 0, 8, {}}, {Kind::SCALAR_POOL, 0, 0, 8, {}}}, 7, 64));
}

TEST(HbgGraphMaterializePlan, PermutedReferencesPreserveTheSamePlan) {
    Region region{Kind::SCALAR_POOL, 0x10000, 32, 8, {{9, 8}, {0, 5}, {3, 8}, {24, 8}}};
    auto first = Plan::build({region}, 7, 64);
    std::reverse(region.referenced.begin(), region.referenced.end());
    auto second = Plan::build({region}, 7, 64);
    ASSERT_TRUE(first);
    ASSERT_TRUE(second);
    ASSERT_EQ(first->work.size(), second->work.size());
    ASSERT_EQ(first->ranges.size(), second->ranges.size());
    for (size_t index = 0; index < first->work.size(); ++index) {
        EXPECT_EQ(first->work[index].first_range, second->work[index].first_range);
        EXPECT_EQ(first->work[index].range_count, second->work[index].range_count);
        EXPECT_EQ(first->work[index].write_bytes, second->work[index].write_bytes);
    }
    for (size_t index = 0; index < first->ranges.size(); ++index) {
        EXPECT_EQ(first->ranges[index].begin, second->ranges[index].begin);
        EXPECT_EQ(first->ranges[index].count, second->ranges[index].count);
    }
}

std::vector<uint8_t> delayed_writeback(const Plan &plan, const Region &region, uint32_t seed) {
    std::vector<uint8_t> memory(512, 0xcc);
    using Cache = std::map<size_t, std::array<uint8_t, 64>>;
    std::vector<Cache> caches(plan.work.size());
    for (size_t owner = 0; owner < plan.work.size(); ++owner) {
        const auto &work = plan.work[owner];
        for (uint32_t part = 0; part < work.range_count; ++part) {
            const auto &range = plan.ranges[work.first_range + part];
            for (uint32_t index = range.begin; index < range.begin + range.count; ++index) {
                for (uint32_t byte = 0; byte < region.element_bytes; ++byte) {
                    const size_t address = region.address + index * region.element_bytes + byte;
                    const size_t line = address / 64;
                    auto [entry, fresh] = caches[owner].try_emplace(line);
                    if (fresh) std::copy_n(memory.begin() + line * 64, 64, entry->second.begin());
                    entry->second[address % 64] = static_cast<uint8_t>(0x10 + index);
                }
            }
        }
    }
    std::vector<size_t> order(caches.size());
    std::iota(order.begin(), order.end(), 0);
    std::shuffle(order.begin(), order.end(), std::mt19937(seed));
    for (size_t owner : order) {
        for (const auto &[line, bytes] : caches[owner])
            std::copy(bytes.begin(), bytes.end(), memory.begin() + line * 64);
    }
    return memory;
}

TEST(HbgGraphMaterializePlan, DelayedWholeLineWritebackPreservesAllScalarsInAnyCompletionOrder) {
    const Region region{Kind::SCALAR_POOL, 56, 33, 8, {{0, 5}, {7, 14}, {23, 10}}};
    auto plan = Plan::build({region}, 7, 64);
    ASSERT_TRUE(plan);
    ASSERT_GT(plan->work.size(), 1u);
    check_coverage({region}, *plan);
    std::vector<uint8_t> expected(512, 0xcc);
    Plan per_element;
    for (const auto &range : region.referenced) {
        for (uint32_t index = range.begin; index < range.begin + range.count; ++index) {
            std::fill_n(expected.begin() + region.address + index * 8, 8, static_cast<uint8_t>(0x10 + index));
            per_element.work.push_back({7, Kind::SCALAR_POOL, static_cast<uint32_t>(per_element.ranges.size()), 1, 8});
            per_element.ranges.push_back({index, 1});
        }
    }
    for (uint32_t seed = 0; seed < 32; ++seed) {
        SCOPED_TRACE(seed);
        EXPECT_NE(delayed_writeback(per_element, region, seed), expected);
        EXPECT_EQ(delayed_writeback(*plan, region, seed), expected);
    }
}

TEST(HbgGraphMaterializePlan, RandomSharedReferencesHaveExactCoverageAndExclusiveLines) {
    std::mt19937 random(71);
    for (uint32_t sample = 0; sample < 64; ++sample) {
        Region region{Kind::SCALAR_POOL, 0x10000 + (sample % 8) * 8, 65, 8, {}};
        for (uint32_t entry = 0; entry < 20; ++entry) {
            const uint32_t begin = static_cast<uint32_t>(random() % 65);
            region.referenced.push_back({begin, static_cast<uint32_t>(random() % (66 - begin))});
        }
        auto plan = Plan::build({region}, 7, 64 + (sample % 4) * 16);
        ASSERT_TRUE(plan);
        check_coverage({region}, *plan);
    }
}

TEST(HbgGraphMaterializePlan, DefinitionAdapterDeduplicatesOnlyReferencedPoolSlots) {
    MaterializeFixture f;
    f.object.body.tasks[1].tensor_offset = 0;
    f.object.body.tasks[1].scalar_offset = 0;
    f.object.body.tensors[1].owner_task_id = TaskId::make_param(99);
    f.object.body.sources[1] = GraphScalarInheritance::from_boundary(99);
    const auto &definition = f.object.body.definition;
    auto plan = Plan::from_definition(definition, 7, 64);
    ASSERT_TRUE(plan);
    GraphExecutionStorageLayout layout{};
    ASSERT_TRUE(graph_execution_storage_layout(2, 2, 2, 1, &layout));
    check_coverage(
        {{Kind::TASK_STORAGE, layout.tasks_offset, 2, sizeof(ChipTaskStorage), {{0, 2}}},
         {Kind::TENSOR_POOL, layout.tensors_offset, 2, sizeof(simpler::hbg::Tensor), {{0, 1}}},
         {Kind::SCALAR_POOL, layout.scalars_offset, 2, sizeof(uint64_t), {{0, 1}}}},
        *plan
    );
}

TEST(HbgGraphMaterializePlan, DefinitionAdapterLeavesScalarBindingErrorsToDeviceValidation) {
    MaterializeFixture f;
    f.control.instance_index = 7;
    f.object.body.sources[0] = GraphScalarInheritance::from_boundary(99);
    auto plan = Plan::from_definition(f.object.body.definition, 7, 4096);
    ASSERT_TRUE(plan);
    const SchedulerGraphView graph{
        0, 0, f.nodes.size(), f.nodes.size() - 1, reinterpret_cast<uint64_t>(f.nodes.data())
    };
    bool checked = false;
    for (const auto &work : plan->work) {
        if (work.region != Kind::SCALAR_POOL) continue;
        int32_t failed_child = -1;
        EXPECT_EQ(
            scheduler_graph_materialize_work(
                f.instance, graph, work, plan->ranges.data(), plan->ranges.size(), &failed_child
            ),
            SchedulerGraphResult::INVALID_ARGUMENTS
        );
        EXPECT_EQ(failed_child, 0);
        checked = true;
    }
    EXPECT_TRUE(checked);
}

TEST(HbgGraphMaterializePlan, UnplannableMetadataDoesNotRequireReadingInvalidPoolSlots) {
    MaterializeFixture f;
    f.object.body.tasks[0].scalar_offset = -1;
    EXPECT_FALSE(Plan::from_definition(f.object.body.definition, 7, 4096));
    f.object.body.tasks[0].scalar_offset = 0;
    f.object.body.tasks[0].tensor_count = 3;
    EXPECT_FALSE(Plan::from_definition(f.object.body.definition, 7, 4096));
    f.object.body.tasks[0].tensor_count = 1;
    f.object.body.definition.execution_storage_bytes = 0;
    EXPECT_FALSE(Plan::from_definition(f.object.body.definition, 7, 4096));
}

}  // namespace
