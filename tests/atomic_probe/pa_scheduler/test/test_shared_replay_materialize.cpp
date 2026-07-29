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

#include <array>
#include <cstring>
#include <cstdint>
#include <cstdio>
#include <new>
#include <sys/mman.h>
#include <vector>

#define PA_DEVICE inline
#define PA_GM
#include "pa_scheduler_core.h"

namespace {

using namespace pa_scheduler;

constexpr uint32_t kAllocTaskId = 0;
constexpr uint32_t kQkTaskId = 1;
constexpr uint32_t kTestTaskCount = 2;

int g_failures = 0;

struct ReplayMaterializeOps {
    static constexpr bool kAtomicReturnReadyObserved = false;

    static int64_t Load(volatile int64_t *address) {
        return __atomic_load_n(address, __ATOMIC_ACQUIRE);
    }

    static int64_t FetchAdd(
        volatile int64_t *address, int64_t value
    ) {
        return __atomic_fetch_add(address, value, __ATOMIC_ACQ_REL);
    }

    static int64_t Exchange(
        volatile int64_t *address, int64_t value
    ) {
        return __atomic_exchange_n(address, value, __ATOMIC_ACQ_REL);
    }

    static int64_t FetchMax(
        volatile int64_t *address, int64_t value, uint64_t &retries
    ) {
        int64_t current =
            __atomic_load_n(address, __ATOMIC_ACQUIRE);
        retries = 0;
        while (value > current) {
            if (__atomic_compare_exchange_n(
                    address, &current, value, true,
                    __ATOMIC_ACQ_REL, __ATOMIC_ACQUIRE
                )) {
                break;
            }
            ++retries;
        }
        return current;
    }

    static uint64_t Now() { return 0; }

    template <typename T>
    static uint64_t NowAfterAtomicResult(T) {
        return 0;
    }
};

struct TaskSpec {
    uint32_t task_id;
    TaskKind kind;
    uint32_t output_count;
    TaskArgs args;
    TensorCreateInfo create_infos[kSharedOutputMaxPerTask];
};

struct Fixture {
    SchedulerState *state;
    TaskSpec tasks[kTestTaskCount];
    SubmitContext contexts[kWorkers][kTestTaskCount];
};

void Check(bool condition, const char *message) {
    if (condition) {
        return;
    }
    std::fprintf(stderr, "[FAIL] %s\n", message);
    ++g_failures;
}

template <typename T>
T *MapSparseObject() {
    int flags = MAP_PRIVATE | MAP_ANONYMOUS;
#ifdef MAP_NORESERVE
    flags |= MAP_NORESERVE;
#endif
    void *memory =
        mmap(nullptr, sizeof(T), PROT_READ | PROT_WRITE, flags, -1, 0);
    if (memory == MAP_FAILED) {
        return nullptr;
    }
    return ::new (memory) T;
}

void InitializeTaskSpec(
    TaskSpec &spec, uint32_t task_id, TaskKind kind,
    const std::array<uint32_t, kSharedOutputMaxPerTask> &elements
) {
    spec.task_id = task_id;
    spec.kind = kind;
    spec.output_count = FrontendTaskOutputCount(kind);
    ConstructTaskArgs(spec.args);

    for (uint32_t slot = 0; slot < spec.output_count; ++slot) {
        ClearCreateInfo(spec.create_infos[slot]);
        const uint32_t shape[kMaxTensorDims] = {
            elements[slot], 1, 0, 0, 0
        };
        InitCreateInfo(
            spec.create_infos[slot], shape, 2, DataType::Float32
        );
        AddOutput(spec.args, spec.create_infos[slot]);
    }
}

void InitializeFixture(Fixture &fixture) {
    InitializeTaskSpec(
        fixture.tasks[0], kAllocTaskId, TaskKind::Alloc,
        {16, 32, 48}
    );
    InitializeTaskSpec(
        fixture.tasks[1], kQkTaskId, TaskKind::Qk,
        {64, 0, 0}
    );

    for (uint32_t shard = 0; shard < kCursorShards; ++shard) {
        fixture.state->cube_cursor[shard].value = -1;
        fixture.state->vector_cursor[shard].value = -1;
        fixture.state->alloc_cursor[shard].value = -1;
    }
    for (
        uint32_t shard = 0;
        shard < kSharedVectorCursorCapacity; ++shard
    ) {
        fixture.state->shared_map.shared_vector_cursor[shard].value =
            -1;
    }
    fixture.state->fatal.value = 0;
    fixture.state->tasks[kAllocTaskId].deps_prepared = -1;
    fixture.state->tasks[kQkTaskId].deps_prepared = -1;

    for (uint32_t worker_id = 0; worker_id < kWorkers; ++worker_id) {
        WorkerState &worker = fixture.state->workers[worker_id];
        worker.role =
            worker_id < kAicWorkers ? CoreRole::Aic : CoreRole::Aiv;
        worker.heap_next = 0;

        for (
            uint32_t task_index = 0;
            task_index < kTestTaskCount; ++task_index
        ) {
            const TaskSpec &spec = fixture.tasks[task_index];
            SubmitContext &context =
                fixture.contexts[worker_id][task_index];
            context = {};
            context.self = &worker;
            context.payload = &worker.payloads[task_index];
            context.task_id = static_cast<int32_t>(spec.task_id);
            context.result.task_id = spec.task_id;
        }
    }
}

bool MaterializeEveryActor(Fixture &fixture) {
    bool all_materialized = true;
    for (uint32_t worker_id = 0; worker_id < kWorkers; ++worker_id) {
        WorkerState &worker = fixture.state->workers[worker_id];
        for (
            uint32_t task_index = 0;
            task_index < kTestTaskCount; ++task_index
        ) {
            const TaskSpec &spec = fixture.tasks[task_index];
            SubmitContext &context =
                fixture.contexts[worker_id][task_index];
            const bool materialized = MaterializeTask(
                worker, spec.task_id, spec.args, context,
                kSyntheticHeapBase, kHeapBytes
            );
            all_materialized &=
                materialized &&
                context.result.task_id == spec.task_id &&
                context.result.count == spec.output_count;
        }
    }
    return all_materialized;
}

bool EquivalentDescriptor(
    const TensorDesc &lhs, const TensorDesc &rhs
) {
    if (lhs.buffer_addr != rhs.buffer_addr ||
        lhs.buffer_size != rhs.buffer_size ||
        lhs.owner_task_id != rhs.owner_task_id ||
        lhs.start_offset != rhs.start_offset ||
        lhs.version != rhs.version || lhs.ndims != rhs.ndims ||
        lhs.dtype != rhs.dtype || lhs.manual_dep != rhs.manual_dep ||
        lhs.is_contiguous != rhs.is_contiguous ||
        lhs.child_memory != rhs.child_memory ||
        lhs.extent_elem_cache != rhs.extent_elem_cache) {
        return false;
    }
    for (uint32_t dimension = 0; dimension < kMaxTensorDims;
         ++dimension) {
        if (lhs.shapes[dimension] != rhs.shapes[dimension] ||
            lhs.strides[dimension] != rhs.strides[dimension]) {
            return false;
        }
    }
    return true;
}

bool AllActorsHaveEquivalentPrivateDescriptors(
    const Fixture &fixture
) {
    bool exact = true;
    for (
        uint32_t task_index = 0;
        task_index < kTestTaskCount; ++task_index
    ) {
        const TaskSpec &spec = fixture.tasks[task_index];
        const TaskOutputs &reference =
            fixture.contexts[0][task_index].result;
        for (uint32_t slot = 0; slot < spec.output_count; ++slot) {
            const TensorDesc *reference_desc =
                reference.tensors[slot];
            exact &= reference_desc != nullptr;
            if (reference_desc == nullptr) {
                continue;
            }
            for (
                uint32_t worker_id = 1;
                worker_id < kWorkers; ++worker_id
            ) {
                const TensorDesc *candidate =
                    fixture.contexts[worker_id][task_index]
                        .result.tensors[slot];
                // 每个 actor 保存自己的 descriptor 对象，不能把 winner
                // payload 指针广播给 loser 或 not_attempted actor。
                exact &= candidate != nullptr &&
                         candidate != reference_desc;
                if (candidate != nullptr) {
                    // descriptor 对象虽然私有，但同一逻辑输出必须解析到
                    // 同一真实 buffer，否则后续 consumer winner 会读错地址。
                    exact &= EquivalentDescriptor(
                        *candidate, *reference_desc
                    );
                }
            }
        }
    }

    const uint64_t reference_heap_next =
        fixture.state->workers[0].heap_next;
    for (uint32_t worker_id = 1; worker_id < kWorkers; ++worker_id) {
        exact &= fixture.state->workers[worker_id].heap_next ==
                 reference_heap_next;
    }
    return exact;
}

bool RangesOverlap(
    const TensorDesc &lhs, const TensorDesc &rhs
) {
    if (lhs.buffer_size == 0 || rhs.buffer_size == 0 ||
        lhs.buffer_addr > UINT64_MAX - lhs.buffer_size ||
        rhs.buffer_addr > UINT64_MAX - rhs.buffer_size) {
        return true;
    }
    const uint64_t lhs_end = lhs.buffer_addr + lhs.buffer_size;
    const uint64_t rhs_end = rhs.buffer_addr + rhs.buffer_size;
    return lhs.buffer_addr < rhs_end && rhs.buffer_addr < lhs_end;
}

bool EachActorOwnsNonOverlappingLogicalOutputs(
    const Fixture &fixture
) {
    bool exact = true;
    for (uint32_t worker_id = 0; worker_id < kWorkers; ++worker_id) {
        std::array<const TensorDesc *, 4> outputs{};
        uint32_t count = 0;
        for (
            uint32_t task_index = 0;
            task_index < kTestTaskCount; ++task_index
        ) {
            const TaskOutputs &task_outputs =
                fixture.contexts[worker_id][task_index].result;
            for (
                uint32_t slot = 0;
                slot < task_outputs.count; ++slot
            ) {
                if (count < outputs.size()) {
                    outputs[count++] = task_outputs.tensors[slot];
                } else {
                    exact = false;
                }
            }
        }
        exact &= count == outputs.size();
        for (uint32_t lhs = 0; lhs < count; ++lhs) {
            exact &= outputs[lhs] != nullptr;
            for (uint32_t rhs = lhs + 1; rhs < count; ++rhs) {
                exact &= outputs[rhs] != nullptr;
                if (outputs[lhs] != nullptr &&
                    outputs[rhs] != nullptr) {
                    exact &=
                        !RangesOverlap(*outputs[lhs], *outputs[rhs]);
                }
            }
        }
    }
    return exact;
}

bool CrossWorkerProducerConsumerUsesSamePhysicalBytes(
    const Fixture &fixture
) {
    const TensorDesc *producer =
        fixture.contexts[0][0].result.tensors[0];
    const TensorDesc *consumer =
        fixture.contexts[kWorkers - 1U][0].result.tensors[0];
    if (producer == nullptr || consumer == nullptr ||
        producer->buffer_addr != consumer->buffer_addr ||
        producer->buffer_addr < kSyntheticHeapBase ||
        producer->buffer_size < sizeof(uint64_t)) {
        return false;
    }
    const uint64_t offset =
        producer->buffer_addr - kSyntheticHeapBase;
    if (offset > UINT64_MAX - producer->buffer_size ||
        offset + producer->buffer_size > 4096) {
        return false;
    }

    // CPU standalone 的 heap 地址是 synthetic identity，不能直接解引用。
    // 用同一 backing 的 offset 模拟实际 GM：producer winner 按自己的
    // descriptor 写入，另一 worker 只用本核 descriptor 读取。
    std::vector<uint8_t> backing(4096, 0);
    constexpr uint64_t kSentinel = 0xA55A1234FEDC9876ULL;
    std::memcpy(
        backing.data() + offset, &kSentinel, sizeof(kSentinel)
    );
    uint64_t observed = 0;
    std::memcpy(
        &observed,
        backing.data() +
            (consumer->buffer_addr - kSyntheticHeapBase),
        sizeof(observed)
    );
    return observed == kSentinel;
}

bool ClaimAfterEveryActorHasTaskOutputs(Fixture &fixture) {
    uint32_t winners = 0;
    uint32_t losers = 0;
    uint32_t not_attempted = 0;
    bool exact = true;

    // 先检查所有 actor 的本核输出，再允许首个 actor 进入 Claim。
    for (uint32_t worker_id = 0; worker_id < kWorkers; ++worker_id) {
        const TaskOutputs &outputs =
            fixture.contexts[worker_id][1].result;
        exact &= outputs.task_id == kQkTaskId &&
                 outputs.count == 1 &&
                 outputs.tensors[0] != nullptr;
    }

    for (uint32_t worker_id = 0; worker_id < kWorkers; ++worker_id) {
        WorkerState &worker = fixture.state->workers[worker_id];
        LocalStats stats{};
        const ClaimOutcome outcome =
            Claim<ReplayMaterializeOps>(
                fixture.state, worker, kQkTaskId, TaskKind::Qk,
                stats
            );
        if (!outcome.attempted) {
            ++not_attempted;
            exact &= !outcome.won && outcome.function_id == -1;
        } else if (outcome.won) {
            ++winners;
            exact &= outcome.function_id ==
                     FunctionId(TaskKind::Qk);
        } else {
            ++losers;
            exact &= outcome.function_id == -1;
        }

        // Claim 的角色结果不能撤销或替换此前的本核 TaskOutputs。
        const TaskOutputs &outputs =
            fixture.contexts[worker_id][1].result;
        exact &= outputs.count == 1 &&
                 outputs.tensors[0] != nullptr;
    }

    return exact && winners == 1 &&
           losers == kAicWorkers - 1 &&
           not_attempted == kAivWorkers &&
           fixture.state->tasks[kQkTaskId].deps_prepared == -1 &&
           fixture.state->fatal.value == 0;
}

void TestSharedReplayActorLocalMaterializeContract() {
    Fixture fixture{};
    fixture.state = MapSparseObject<SchedulerState>();
    Check(
        fixture.state != nullptr,
        "sparse scheduler state maps for replay materialize test"
    );
    if (fixture.state == nullptr) {
        return;
    }

    InitializeFixture(fixture);
    Check(
        MaterializeEveryActor(fixture),
        "winner, loser, and not_attempted actors all materialize outputs"
    );
    Check(
        AllActorsHaveEquivalentPrivateDescriptors(fixture),
        "actor-local descriptors are distinct objects with identical "
        "logical output addresses and contents"
    );
    Check(
        EachActorOwnsNonOverlappingLogicalOutputs(fixture),
        "different task/output slots do not overlap within one actor"
    );
    Check(
        CrossWorkerProducerConsumerUsesSamePhysicalBytes(fixture),
        "producer winner and cross-worker consumer address the same backing bytes"
    );
    Check(
        ClaimAfterEveryActorHasTaskOutputs(fixture),
        "QK Claim classifies one winner, AIC losers, and AIV "
        "not_attempted only after all 96 actors own TaskOutputs"
    );

    (void)munmap(fixture.state, sizeof(*fixture.state));
}

}  // namespace

int main() {
    TestSharedReplayActorLocalMaterializeContract();
    if (g_failures != 0) {
        std::fprintf(
            stderr,
            "[FAIL] shared replay materialize target contract: %d\n",
            g_failures
        );
        return 1;
    }
    std::printf("[PASS] shared replay materialize target contract\n");
    return 0;
}
