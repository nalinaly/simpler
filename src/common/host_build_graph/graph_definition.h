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

#include <cstddef>
#include <cstdint>
#include <type_traits>

#include "common/args_dump_task_metadata.h"
#include "data_type.h"
#include "host_build_graph/submit_types.h"

inline constexpr int32_t SUB_TASK_MAX_NUM = 1024;

// Where one sub-task scalar slot takes its value from: the Definition's own
// scalars[] entry, or the boundary parameter named by boundary_index(). It is the wire
// form of the two things recording knows about a slot -- whether it inherits, and which
// parameter it inherits -- so inherited() is the same predicate as Arg::scalar_inherited.
//
// Only a parameter of the replaying Graph's boundary can be refreshed; the index reaches
// that boundary and nothing else, and means nothing while inherited() is false. The
// fields are private so the pair can only be set together, through a factory that decides
// both: an entry carrying an index while claiming not to inherit, or the reverse, cannot
// be spelled. What the index means is still a claim about a boundary this entry cannot
// see, so the readers bound it against the boundary they do have.
class GraphScalarInheritance {
public:
    // The image's scalar section is allocated as an array, so a default-constructible
    // slot is required; the factories below are what a caller fills one with.
    GraphScalarInheritance() = default;

    static GraphScalarInheritance self_value() { return {0, false}; }
    static GraphScalarInheritance from_boundary(uint16_t index) { return {index, true}; }

    PTO_DEVICE_FUNC bool inherited() const { return inherited_ != 0; }
    PTO_DEVICE_FUNC uint16_t boundary_index() const { return boundary_index_; }

private:
    GraphScalarInheritance(uint16_t index, bool inherits) :
        boundary_index_(index),
        inherited_(inherits ? 1 : 0) {}

    // One access level for every field, which is what keeps this standard-layout and so
    // safe to memcpy to the device. inherited_ is a uint16_t rather than a bool so the
    // two fields fill the size alignof(uint16_t) rounds this type up to: there is no
    // padding byte, and so no indeterminate byte in the image's scalar section, which the
    // static_assert below pins the size of because that is what the section indexes by.
    uint16_t boundary_index_;
    uint16_t inherited_;
};

struct SubTaskDefinition {
    int32_t kernel_id[SUBTASK_SLOT_COUNT];
    uint8_t active_mask;
    uint8_t task_attrs;
    // Early-dispatch verdicts (ED_FLAG_CANDIDATE / ED_FLAG_TRACKED), decided once
    // per recorded shape rather than per execution: a body's structure is fixed by
    // its Definition, so every execution of it qualifies the same tasks. Carries
    // the same meaning as ChipTaskSlotState::ed_flags, against the Definition's
    // fanin CSR instead of a payload fanin region.
    uint8_t ed_flags;
    uint8_t reserved;
    int16_t logical_block_num;
    int16_t total_required_subtasks;
    // One-based index into the Definition's predicate array; 0 means the task
    // carries no dispatch predicate. Biased so that a zeroed SubTaskDefinition
    // is a valid predicate-free task. Predicated tasks are rare, so the
    // predicates live in their own array rather than inline.
    uint16_t predicate_slot;
    int32_t tensor_count;
    int32_t scalar_count;
    int32_t total_output_size;
    // Element indices into the Definition's argument pools, not byte offsets: each
    // pairs with the count above it to form [offset, offset + count), so it carries
    // that count's type.
    int32_t tensor_offset;
    int32_t scalar_offset;
    ArgsDumpTaskMetadata dump_metadata;
};

inline constexpr uint64_t GRAPH_DEFINITION_OBJECT_MAGIC = 0x4752415048455844ULL;

// Object framing has a fixed wire alignment across Host, AICPU and AICore.
// max_align_t is target-specific (16 on Host/AICPU, 8 in CCEC), so it cannot
// determine the size of a header another target reads immediately before an image.
inline constexpr size_t GRAPH_DEFINITION_OBJECT_ALIGN = 16;

// Header prefixing each device-resident Definition object. The definition buffer
// uploaded by the host is [GraphDefinitionHeader][GraphDefinition image], and these
// three fields are the framing the device checks before it reads a section offset
// out of the image: the object is one of ours, it is the size the upload recorded,
// and it holds the Graph the outer task was bound to.
//
// alignas, not a pad field: the image starts immediately behind the header and its
// sections are offsets from that base, so the header's own size has to be a multiple
// of the object alignment.
struct alignas(GRAPH_DEFINITION_OBJECT_ALIGN) GraphDefinitionHeader {
    uint64_t magic;
    uint64_t full_key;
    uint32_t definition_bytes;
};

static_assert(std::is_trivially_copyable_v<GraphDefinitionHeader>);
static_assert(std::is_standard_layout_v<GraphDefinitionHeader>);
static_assert(sizeof(GraphDefinitionHeader) % GRAPH_DEFINITION_OBJECT_ALIGN == 0);
static_assert(sizeof(GraphDefinitionHeader) == 32, "Definition framing must match on every compilation target");

struct GraphDefinition {
    uint64_t full_key;
    uint64_t required_heap;
    // The header splits by range, not by name. Everything that counts *things* is
    // signed: each is capped by SUB_TASK_MAX_NUM times a per-task constant, so the
    // largest of them (edge_count, at 1024 x 1023) still clears INT32_MAX by three
    // orders of magnitude, and a signed count makes a corrupt wire value testable
    // with `< 0` instead of turning it into a huge index. Everything that counts
    // *bytes* stays unsigned because graph_layout_section dimensions the image
    // against UINT32_MAX: total_bytes and the off_ fields below reach 4 GB by design,
    // and halving that range would be a functional change.
    uint32_t total_bytes;
    int32_t task_count;
    int32_t edge_count;
    int32_t root_count;
    int32_t boundary_tensor_count;
    int32_t boundary_scalar_count;
    int32_t tensor_arg_count;
    int32_t scalar_arg_count;
    int32_t predicate_count;
    // Bytes the GraphExecution header, sub-task array and sub-task
    // argument pools need in the outer Graph task's heap tail. Invocation
    // boundaries live in the outer task payload's compact argument-pool regions
    // instead.
    uint32_t execution_storage_bytes;
    uint32_t off_fanout_offsets;
    uint32_t off_fanout_indices;
    uint32_t off_fanin_offsets;
    uint32_t off_fanin_indices;
    uint32_t off_root_indices;
    uint32_t off_sub_task_offsets;
    uint32_t off_sub_tasks;
    // Every sub-task's tensor arguments, concatenated; a task names its own run
    // through tensor_offset / tensor_count.
    //
    // These are **relocation records**, not tensors. Every other simpler::hbg::Tensor in
    // this runtime -- a boundary parameter, an invocation's argument, a materialized task
    // argument -- holds values that mean something on their own. One here does not: the
    // two fields replay has a base for are stored relative, and which base to add follows
    // the tensor's own owner_task_id, stamped by the recording.
    //
    //   field           owner PARAM                           owner SUB_TASK
    //   buffer_addr     0 -- replay takes the whole buffer     offset into the graph heap,
    //                   from this call's argument              whose base the execution has
    //   start_offset    the view's offset inside that          the view's own origin,
    //                   parameter; replay adds the             absolute in its own buffer
    //                   argument's origin
    //
    // Geometry, dtype and flags travel absolute, because graph_require_boundary_match pins
    // those equal before a Definition may be reused.
    //
    // So start_offset here does not mean what it means on the boundary tensors this is
    // matched against: there it is the caller's own origin, here an offset from it. A
    // parameter's offset is relative to its own parameter rather than to its alias
    // partition's, so a tensor is rebuilt against the argument it came from; two tensors
    // over one buffer keep their recorded distance only while the arguments keep theirs,
    // which is what graph_require_boundary_match checks and what the recorded WAR/WAW edges
    // were inferred from.
    uint32_t off_tensors;
    uint32_t off_scalars;
    uint32_t off_scalar_inheritance;
    uint32_t off_predicates;
};

static_assert(std::is_trivially_copyable_v<GraphScalarInheritance>);
static_assert(std::is_standard_layout_v<GraphScalarInheritance>);
static_assert(sizeof(GraphScalarInheritance) == 4, "the image's scalar section assumes this layout");
static_assert(std::is_trivially_copyable_v<SubTaskDefinition>);
static_assert(std::is_standard_layout_v<SubTaskDefinition>);
// graph_fill_definition assigns this struct field by field, so its interior padding
// is the one part of the sub-task section no writer reaches. Nothing reads it
// either, but pinning the size makes a new field's padding cost visible in the diff
// that adds it rather than silently.
static_assert(sizeof(SubTaskDefinition) == 80, "a SubTaskDefinition field changed the wire layout");
static_assert(std::is_trivially_copyable_v<GraphDefinition>);
static_assert(std::is_standard_layout_v<GraphDefinition>);
static_assert(sizeof(GraphDefinition) == 104, "Graph Definition header layout changed");

// A section's length is the same int32 every counting field of the header carries.
// A negative one is rejected outright rather than left to wrap through the size_t
// comparison, so a corrupt header cannot widen a section by either route.
template <typename T>
inline const T *graph_definition_array(const GraphDefinition &definition, uint32_t offset, int32_t count) {
    if (count < 0 || offset == 0 || offset > definition.total_bytes || offset % alignof(T) != 0) return nullptr;
    const size_t remaining = static_cast<size_t>(definition.total_bytes - offset);
    if (static_cast<size_t>(count) > remaining / sizeof(T)) return nullptr;
    return reinterpret_cast<const T *>(reinterpret_cast<const uint8_t *>(&definition) + offset);
}

template <typename T>
inline const T *graph_definition_ptr(const GraphDefinition &definition, uint32_t offset) {
    return graph_definition_array<T>(definition, offset, 1);
}
