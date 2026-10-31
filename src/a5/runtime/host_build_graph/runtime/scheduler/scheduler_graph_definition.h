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

#include "host_build_graph/graph_definition.h"
#include "scheduler_graph.h"

// The GM carrier is run-owned and aligned, unlike thread-local AICPU launch
// arguments. Array addresses stay inside one framed, immutable Definition.
struct SchedulerGraphDefinitionView {
    uint64_t image_address{0};
    uint64_t image_bytes{0};
};

inline __aicore__ bool scheduler_graph_definition_open(
    uint64_t section_address, uint64_t section_bytes, uint32_t image_offset, uint64_t expected_key,
    SchedulerGraphDefinitionView *out
) {
    if (out == nullptr) return false;
    *out = {};
    if (section_address == 0 || section_address % GRAPH_DEFINITION_OBJECT_ALIGN != 0 || section_bytes > UINT32_MAX ||
        section_address > UINT64_MAX - section_bytes || image_offset < sizeof(GraphDefinitionHeader) ||
        image_offset % GRAPH_DEFINITION_OBJECT_ALIGN != 0 || image_offset > section_bytes ||
        sizeof(GraphDefinition) > section_bytes - image_offset)
        return false;
    const auto *header = reinterpret_cast<const __gm__ GraphDefinitionHeader *>(
        section_address + image_offset - sizeof(GraphDefinitionHeader)
    );
    if (header->magic != GRAPH_DEFINITION_OBJECT_MAGIC || header->full_key != expected_key ||
        header->definition_bytes < sizeof(GraphDefinition) || header->definition_bytes > section_bytes - image_offset)
        return false;
    const auto *definition = reinterpret_cast<const __gm__ GraphDefinition *>(section_address + image_offset);
    if (definition->full_key != expected_key || definition->total_bytes != header->definition_bytes) return false;
    out->image_address = section_address + image_offset;
    out->image_bytes = definition->total_bytes;
    return true;
}

template <typename T>
inline __aicore__ const __gm__ T *
scheduler_graph_definition_array(const SchedulerGraphDefinitionView &view, uint32_t offset, int32_t count) {
    static_assert(std::is_trivially_copyable_v<T>, "a Definition section holds wire values");
    if (view.image_address == 0 || view.image_bytes < sizeof(GraphDefinition) ||
        view.image_address > UINT64_MAX - view.image_bytes || count < 0 || offset < sizeof(GraphDefinition) ||
        offset > view.image_bytes || (view.image_address + offset) % alignof(T) != 0 ||
        static_cast<uint64_t>(count) > (view.image_bytes - offset) / sizeof(T))
        return nullptr;
    return reinterpret_cast<const __gm__ T *>(view.image_address + offset);
}

template <typename T>
inline __aicore__ bool scheduler_graph_definition_read(
    const SchedulerGraphDefinitionView &view, uint32_t offset, int32_t count, int32_t index, T *out
) {
    if (out == nullptr || index < 0 || index >= count) return false;
    const auto *array = scheduler_graph_definition_array<T>(view, offset, count);
    if (array == nullptr) return false;
    const auto *source = reinterpret_cast<const __gm__ uint8_t *>(&array[index]);
    auto *target = reinterpret_cast<uint8_t *>(out);
    for (size_t byte = 0; byte < sizeof(T); ++byte)
        target[byte] = source[byte];
    return true;
}
