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

#define PA_DEVICE inline
#define PA_GM
#include "../common/pa_shared_tensormap.h"
#undef PA_GM
#undef PA_DEVICE
#include "shared_protocol_litmus_shared.h"

#include "acl/acl.h"
#include "runtime/rt.h"

#include <cerrno>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <fstream>
#include <new>
#include <string>
#include <sys/mman.h>
#include <vector>

namespace {

using pa_scheduler::SchedulerState;
using pa_scheduler::WorkerResult;
using pa_scheduler::shared_protocol_litmus::Control;
using pa_scheduler::shared_protocol_litmus::Direction;
using pa_scheduler::shared_protocol_litmus::ReaderOrdering;
using pa_scheduler::shared_protocol_litmus::ReaderReclaimChain;
using pa_scheduler::shared_protocol_litmus::Scenario;
using pa_scheduler::shared_protocol_litmus::kControlMagic;
using pa_scheduler::shared_protocol_litmus::kControlVersion;
using pa_scheduler::shared_protocol_litmus::kReaderReclaimActiveWorkers;
using pa_scheduler::shared_protocol_litmus::kReaderReclaimAddress;
using pa_scheduler::shared_protocol_litmus::kReaderReclaimAicToAiv;
using pa_scheduler::shared_protocol_litmus::kReaderReclaimAivToAic;
using pa_scheduler::shared_protocol_litmus::kReaderReclaimClosedDone;
using pa_scheduler::shared_protocol_litmus::kReaderReclaimHeapWindow;
using pa_scheduler::shared_protocol_litmus::kReaderReclaimHi;
using pa_scheduler::shared_protocol_litmus::kReaderReclaimInitialDone;
using pa_scheduler::shared_protocol_litmus::kReaderReclaimLo;
using pa_scheduler::shared_protocol_litmus::kReaderReclaimReaderStatus;
using pa_scheduler::shared_protocol_litmus::kReaderReclaimReplacementAddress;
using pa_scheduler::shared_protocol_litmus::kReaderReclaimReplacementHi;
using pa_scheduler::shared_protocol_litmus::kReaderReclaimReplacementLo;
using pa_scheduler::shared_protocol_litmus::kReaderReclaimReclaimerStatus;
using pa_scheduler::shared_protocol_litmus::kReaderReclaimTask;
using pa_scheduler::shared_protocol_litmus::kReaderReclaimWriterTask;
using pa_scheduler::shared_protocol_litmus::kResultMagic;

constexpr size_t kStatePrefixBytes =
    offsetof(SchedulerState, workers);
constexpr size_t kResultBytes =
    sizeof(WorkerResult) * pa_scheduler::kWorkers;
constexpr size_t kSharedSidecarBytes =
    sizeof(pa_scheduler::SharedTensorMapSidecar);

bool CheckAcl(aclError error, const char *label) {
    if (error == ACL_SUCCESS) {
        return true;
    }
    std::fprintf(
        stderr, "ACL error %d: %s\n",
        static_cast<int>(error), label
    );
    return false;
}

bool CheckRt(rtError_t error, const char *label) {
    if (error == RT_ERROR_NONE) {
        return true;
    }
    std::fprintf(
        stderr, "RT error %d: %s\n",
        static_cast<int>(error), label
    );
    return false;
}

std::vector<char> ReadBinary(const std::string &path) {
    std::ifstream file(path, std::ios::binary | std::ios::ate);
    if (!file) {
        return {};
    }
    const std::streamsize size = file.tellg();
    if (size <= 0) {
        return {};
    }
    std::vector<char> data(static_cast<size_t>(size));
    file.seekg(0, std::ios::beg);
    if (!file.read(data.data(), size)) {
        return {};
    }
    return data;
}

SchedulerState *MapSparseState() {
    int flags = MAP_PRIVATE | MAP_ANONYMOUS;
#ifdef MAP_NORESERVE
    flags |= MAP_NORESERVE;
#endif
    void *memory = mmap(
        nullptr, sizeof(SchedulerState),
        PROT_READ | PROT_WRITE, flags, -1, 0
    );
    if (memory == MAP_FAILED) {
        std::perror("mmap SchedulerState");
        return nullptr;
    }
    return ::new (memory) SchedulerState;
}

bool ParseDevice(const char *text, int32_t *device) {
    errno = 0;
    char *end = nullptr;
    const long parsed = std::strtol(text, &end, 10);
    if (errno != 0 || end == text || *end != '\0' ||
        parsed < 0 || parsed > INT32_MAX) {
        return false;
    }
    *device = static_cast<int32_t>(parsed);
    return true;
}

bool ParseScenario(const char *text, Scenario *scenario) {
    const std::string name = text;
    if (name == "reader-reclaim") {
        *scenario = Scenario::ReaderReclaim;
        return true;
    }
    return false;
}

bool ParseDirection(const char *text, Direction *direction) {
    const std::string name = text;
    if (name == "aic-to-aiv") {
        *direction = Direction::AicToAiv;
        return true;
    }
    if (name == "aiv-to-aic") {
        *direction = Direction::AivToAic;
        return true;
    }
    return false;
}

bool ParseReaderOrdering(
    const char *text, ReaderOrdering *ordering
) {
    const std::string name = text;
    if (name == "compiler-clobber") {
        *ordering = ReaderOrdering::CompilerClobber;
        return true;
    }
    if (name == "payload-dependency") {
        *ordering = ReaderOrdering::PayloadDependency;
        return true;
    }
    if (name == "dsb-all") {
        *ordering = ReaderOrdering::DsbAll;
        return true;
    }
    return false;
}

void SetInsertTurnsAfterTasks(
    SchedulerState *state, uint32_t completed_tasks
) {
    for (uint32_t lane = 0;
         lane < pa_scheduler::kSharedInsertTurnCapacity;
         ++lane) {
        pa_scheduler::SharedInsertTurnLine(
            state->shared_map, lane
        ).value =
            pa_scheduler::SharedInsertTurnTokenAfterTasks(
                completed_tasks, lane
            );
    }
}

bool InsertTurnsMatch(
    const SchedulerState &state, uint32_t completed_tasks
) {
    for (uint32_t lane = 0;
         lane < pa_scheduler::kSharedInsertTurnCapacity;
         ++lane) {
        const int64_t observed =
            lane == 0
                ? state.shared_map.committed_tasks.value
                : state.shared_map
                      .insert_turn_extra[lane - 1U].value;
        if (observed !=
            pa_scheduler::SharedInsertTurnTokenAfterTasks(
                completed_tasks, lane
            )) {
            return false;
        }
    }
    return true;
}

void ResetTaskGate(SchedulerState *state, int32_t task_id) {
    pa_scheduler::TaskCell &task =
        state->tasks[static_cast<uint32_t>(task_id)];
    task.flag = 0;
    task.deps_prepared = -1;
}

int32_t ReaderFillProducer(uint32_t cursor) {
    if (cursor == 0) {
        return 0;
    }
    if (cursor <= 32) {
        return 1;
    }
    if (cursor <= 64) {
        return 2;
    }
    if (cursor <= 96) {
        return 3;
    }
    return 4;
}

void InitializeReaderReclaimState(
    SchedulerState *state,
    const ReaderReclaimChain &chain
) {
    // 该合成镜像严格对应一条生产可达 append 序列：task0 写 1 条，
    // task1/2/3 各写 32 条，task4 写 31 条，总计填满 CAP=128。
    std::memset(state, 0, kStatePrefixBytes);
    std::memset(state->results, 0, kResultBytes);
    std::memset(
        &state->shared_map, 0, kSharedSidecarBytes
    );
    SetInsertTurnsAfterTasks(
        state, kReaderReclaimWriterTask
    );
    state->fatal.value = 0;
    state->heap_window = kReaderReclaimHeapWindow;
    state->shared_map.reclaim_upto.value = -1;

    for (uint32_t slot = 0;
         slot < pa_scheduler::kMapCapacity; ++slot) {
        state->shared_map.slots[slot].seq.value = -1;
    }
    for (uint32_t worker = 0;
         worker < pa_scheduler::kWorkers; ++worker) {
        state->shared_map.reader_done[worker].value =
            kReaderReclaimClosedDone;
    }
    state->shared_map.reader_done[
        chain.reader_worker
    ].value = kReaderReclaimInitialDone;

    const uint32_t bucket =
        pa_scheduler::TensorMapHash(kReaderReclaimAddress);
    state->shared_map.buckets[bucket].head.value = 0;
    state->shared_map.buckets[bucket].tail.value =
        pa_scheduler::kMapBucketCapacity;
    for (uint32_t cursor = 0;
         cursor < pa_scheduler::kMapBucketCapacity;
         ++cursor) {
        pa_scheduler::SharedRegionSlot &slot =
            state->shared_map.slots[
                pa_scheduler::SharedTensorMapSlotIndex(
                    bucket, cursor
                )
            ];
        slot.payload.value.buffer_addr =
            kReaderReclaimAddress;
        slot.payload.value.lo =
            cursor == 0
                ? kReaderReclaimLo
                : 0x10000ULL +
                      static_cast<uint64_t>(cursor) * 64U;
        slot.payload.value.hi =
            cursor == 0
                ? kReaderReclaimHi
                : slot.payload.value.lo + 16U;
        slot.payload.value.producer =
            ReaderFillProducer(cursor);
        slot.payload.value.reserved = 0;
        slot.seq.value = cursor;
    }

    ResetTaskGate(state, kReaderReclaimAicToAiv.blocked_signal);
    ResetTaskGate(
        state, kReaderReclaimAicToAiv.reuse_done_signal
    );
    ResetTaskGate(state, kReaderReclaimAivToAic.blocked_signal);
    ResetTaskGate(
        state, kReaderReclaimAivToAic.reuse_done_signal
    );
}

bool Expect(bool condition, const char *label) {
    std::printf(
        "[ASSERT] %-62s %s\n",
        label, condition ? "PASS" : "FAIL"
    );
    return condition;
}

bool ResultPrefixIsZero(const WorkerResult &result) {
    return result.submit_begin == 0 &&
           result.submit_end == 0 &&
           result.finish_cycle == 0 &&
           result.checksum == 0 &&
           result.submits == 0 &&
           result.claim_attempts == 0 &&
           result.claim_wins == 0 &&
           result.heap_guards == 0;
}

bool ReaderResultMatches(
    const WorkerResult &result,
    const ReaderReclaimChain &chain,
    ReaderOrdering ordering
) {
    return
        result.submit_begin ==
            (kResultMagic | chain.result_tag | 1U) &&
        result.submit_end ==
            kReaderReclaimReaderStatus &&
        result.finish_cycle == kReaderReclaimAddress &&
        result.checksum == kReaderReclaimLo &&
        result.submits == kReaderReclaimHi &&
        static_cast<int64_t>(result.claim_attempts) == 0 &&
        result.claim_wins == 0 &&
        result.heap_guards ==
            (static_cast<uint64_t>(
                 static_cast<uint32_t>(ordering)
             ) << 32 |
             static_cast<uint32_t>(
                 kReaderReclaimClosedDone
             ));
}

bool ReclaimerResultMatches(
    const WorkerResult &result,
    const ReaderReclaimChain &chain
) {
    return
        result.submit_begin ==
            (kResultMagic | chain.result_tag | 2U) &&
        result.submit_end ==
            kReaderReclaimReclaimerStatus &&
        static_cast<int64_t>(result.finish_cycle) == -1 &&
        result.checksum ==
            static_cast<uint64_t>(
                pa_scheduler::SharedAppendCheck::CapacityBlocked
            ) &&
        static_cast<int64_t>(result.submits) == 0 &&
        static_cast<int64_t>(result.claim_attempts) ==
            pa_scheduler::kMapBucketCapacity &&
        static_cast<int64_t>(result.claim_wins) == 0 &&
        result.heap_guards ==
            (static_cast<uint64_t>(1) << 32 |
             (pa_scheduler::kMapBucketCapacity + 1U));
}

bool ValidateReaderReclaim(
    const SchedulerState &state,
    const ReaderReclaimChain &chain,
    const ReaderReclaimChain &inactive,
    ReaderOrdering ordering
) {
    bool passed = true;
    const uint32_t bucket =
        pa_scheduler::TensorMapHash(kReaderReclaimAddress);
    const uint32_t replacement_bucket =
        pa_scheduler::TensorMapHash(
            kReaderReclaimReplacementAddress
        );
    passed &= Expect(
        bucket == replacement_bucket,
        "old and replacement addresses share the target bucket"
    );
    passed &= Expect(
        state.fatal.value == 0,
        "reader-reclaim protocol leaves fatal clear"
    );
    passed &= Expect(
        ReaderResultMatches(
            state.results[chain.reader_worker],
            chain, ordering
        ),
        "reader exports the complete old snapshot after reuse"
    );
    passed &= Expect(
        ReclaimerResultMatches(
            state.results[chain.reclaimer_worker],
            chain
        ),
        "reclaimer records blocked then allowed state transitions"
    );

    bool progress_ok = true;
    for (uint32_t worker = 0;
         worker < kReaderReclaimActiveWorkers; ++worker) {
        progress_ok &=
            state.shared_map.reader_done[worker].value ==
            kReaderReclaimClosedDone;
    }
    passed &= Expect(
        progress_ok,
        "selected reader advances 1->2; other 95 remain at preset task 2"
    );
    passed &= Expect(
        state.shared_map.reclaim_upto.value == 0 &&
            InsertTurnsMatch(
                state,
                static_cast<uint32_t>(
                    kReaderReclaimWriterTask + 1
                )
            ),
        "reader frontier publishes reclaim/commit and preserves inactive turns"
    );
    passed &= Expect(
        state.shared_map.buckets[bucket].head.value == 1 &&
            state.shared_map.buckets[bucket].tail.value ==
                pa_scheduler::kMapBucketCapacity + 1,
        "one safe retirement admits exactly one wrapped append"
    );

    const pa_scheduler::SharedRegionSlot &first_slot =
        state.shared_map.slots[
            pa_scheduler::SharedTensorMapSlotIndex(bucket, 0)
        ];
    passed &= Expect(
        first_slot.seq.value ==
                pa_scheduler::kMapBucketCapacity &&
            first_slot.payload.value.buffer_addr ==
                kReaderReclaimReplacementAddress &&
            first_slot.payload.value.lo ==
                kReaderReclaimReplacementLo &&
            first_slot.payload.value.hi ==
                kReaderReclaimReplacementHi &&
            first_slot.payload.value.producer ==
                kReaderReclaimWriterTask &&
            first_slot.payload.value.reserved == 0,
        "cursor 128 replaces every mutable cursor-0 payload field"
    );

    bool surviving_slots_ok = true;
    uint32_t producer_counts[5] = {};
    for (uint32_t cursor = 1;
         cursor < pa_scheduler::kMapBucketCapacity;
         ++cursor) {
        const pa_scheduler::SharedRegionSlot &slot =
            state.shared_map.slots[
                pa_scheduler::SharedTensorMapSlotIndex(
                    bucket, cursor
                )
            ];
        const int32_t producer =
            ReaderFillProducer(cursor);
        surviving_slots_ok &=
            slot.seq.value == cursor &&
            slot.payload.value.buffer_addr ==
                kReaderReclaimAddress &&
            slot.payload.value.lo ==
                0x10000ULL +
                    static_cast<uint64_t>(cursor) * 64U &&
            slot.payload.value.hi ==
                slot.payload.value.lo + 16U &&
            slot.payload.value.producer == producer &&
            slot.payload.value.reserved == 0;
        if (producer >= 0 && producer < 5) {
            ++producer_counts[
                static_cast<uint32_t>(producer)
            ];
        }
    }
    surviving_slots_ok &=
        producer_counts[1] == 32 &&
        producer_counts[2] == 32 &&
        producer_counts[3] == 32 &&
        producer_counts[4] == 31;
    passed &= Expect(
        surviving_slots_ok,
        "cursor 1..127 preserve the reachable 32/32/32/31 history"
    );

    bool other_buckets_ok = true;
    for (uint32_t index = 0;
         index < pa_scheduler::kMapBuckets; ++index) {
        if (index == bucket) {
            continue;
        }
        other_buckets_ok &=
            state.shared_map.buckets[index].head.value == 0 &&
            state.shared_map.buckets[index].tail.value == 0;
    }
    passed &= Expect(
        other_buckets_ok,
        "all non-target bucket controls remain untouched"
    );

    bool other_slots_ok = true;
    const uint32_t target_slot_begin =
        bucket * pa_scheduler::kMapBucketCapacity;
    const uint32_t target_slot_end =
        target_slot_begin + pa_scheduler::kMapBucketCapacity;
    for (uint32_t index = 0;
         index < pa_scheduler::kMapCapacity; ++index) {
        if (index >= target_slot_begin &&
            index < target_slot_end) {
            continue;
        }
        const pa_scheduler::SharedRegionSlot &slot =
            state.shared_map.slots[index];
        other_slots_ok &=
            slot.seq.value == -1 &&
            slot.payload.value.buffer_addr == 0 &&
            slot.payload.value.lo == 0 &&
            slot.payload.value.hi == 0 &&
            slot.payload.value.producer == 0 &&
            slot.payload.value.reserved == 0;
    }
    passed &= Expect(
        other_slots_ok,
        "all non-target physical ring slots remain untouched"
    );

    passed &= Expect(
        state.tasks[chain.blocked_signal].deps_prepared ==
                chain.blocked_signal &&
            state.tasks[
                chain.reuse_done_signal
            ].deps_prepared ==
                chain.reuse_done_signal &&
            state.tasks[
                inactive.blocked_signal
            ].deps_prepared == -1 &&
            state.tasks[
                inactive.reuse_done_signal
            ].deps_prepared == -1 &&
            ResultPrefixIsZero(
                state.results[inactive.reader_worker]
            ) &&
            ResultPrefixIsZero(
                state.results[inactive.reclaimer_worker]
            ),
        "opposite reader-reclaim direction remains untouched"
    );
    return passed;
}

}  // namespace

int main(int argc, char **argv) {
    if (argc != 6) {
        std::fprintf(
            stderr,
            "Usage: %s <shared_protocol_litmus_kernel.o> "
            "reader-reclaim "
            "aic-to-aiv|aiv-to-aic "
            "compiler-clobber|payload-dependency|dsb-all "
            "<device>\n",
            argv[0]
        );
        return EXIT_FAILURE;
    }

    Scenario scenario{};
    if (!ParseScenario(argv[2], &scenario)) {
        std::fprintf(
            stderr, "Invalid shared protocol scenario: %s\n",
            argv[2]
        );
        return EXIT_FAILURE;
    }
    Direction direction{};
    if (!ParseDirection(argv[3], &direction)) {
        std::fprintf(
            stderr, "Invalid shared protocol direction: %s\n",
            argv[3]
        );
        return EXIT_FAILURE;
    }
    ReaderOrdering ordering{};
    if (!ParseReaderOrdering(
            argv[4], &ordering
        )) {
        std::fprintf(
            stderr,
            "Invalid ordering '%s' for scenario '%s'.\n",
            argv[4], argv[2]
        );
        return EXIT_FAILURE;
    }
    int32_t device = 0;
    if (!ParseDevice(argv[5], &device)) {
        std::fprintf(stderr, "Invalid device id: %s\n", argv[5]);
        return EXIT_FAILURE;
    }
    const ReaderReclaimChain &reader_chain =
        direction == Direction::AicToAiv
            ? kReaderReclaimAicToAiv
            : kReaderReclaimAivToAic;
    const ReaderReclaimChain &inactive_reader =
        direction == Direction::AicToAiv
            ? kReaderReclaimAivToAic
            : kReaderReclaimAicToAiv;
    const std::vector<char> binary = ReadBinary(argv[1]);
    if (binary.empty()) {
        std::fprintf(
            stderr, "Cannot read mixed AICore ELF: %s\n", argv[1]
        );
        return EXIT_FAILURE;
    }

    SchedulerState *host_state = MapSparseState();
    if (host_state == nullptr) {
        return EXIT_FAILURE;
    }
    InitializeReaderReclaimState(
        host_state, reader_chain
    );
    Control host_control{};
    host_control.magic = kControlMagic;
    host_control.version = kControlVersion;
    host_control.scenario = static_cast<uint32_t>(scenario);
    host_control.direction = static_cast<uint32_t>(direction);
    host_control.reader_ordering =
        static_cast<uint32_t>(ordering);
    host_control.launch_nonce =
        static_cast<uint64_t>(
            reader_chain.reader_worker
        ) << 32 |
        reader_chain.reclaimer_worker;

    bool acl_initialized = false;
    bool device_set = false;
    aclrtStream stream = nullptr;
    void *kernel_handle = nullptr;
    bool registered_all = false;
    void *device_state = nullptr;
    void *device_control = nullptr;
    bool execution_ok = false;

    do {
        if (!CheckAcl(aclInit(nullptr), "aclInit")) {
            break;
        }
        acl_initialized = true;
        if (!CheckAcl(aclrtSetDevice(device), "aclrtSetDevice")) {
            break;
        }
        device_set = true;
        if (!CheckAcl(
                aclrtCreateStream(&stream), "aclrtCreateStream"
            )) {
            break;
        }

        rtDevBinary_t device_binary{
            RT_DEV_BINARY_MAGIC_ELF, 0,
            binary.data(), binary.size()
        };
        rtError_t load_error =
            rtRegisterAllKernel(&device_binary, &kernel_handle);
        if (load_error == RT_ERROR_NONE &&
            kernel_handle != nullptr) {
            registered_all = true;
        } else {
            registered_all = false;
            kernel_handle = nullptr;
            load_error = rtBinaryLoadWithoutTilingKey(
                binary.data(), binary.size(), &kernel_handle
            );
        }
        if (!CheckRt(
                load_error,
                "load shared protocol mixed AICore ELF"
            ) ||
            kernel_handle == nullptr) {
            break;
        }

        if (!CheckAcl(
                aclrtMalloc(
                    &device_state, sizeof(SchedulerState),
                    ACL_MEM_MALLOC_HUGE_FIRST
                ),
                "aclrtMalloc(shared protocol SchedulerState)"
            ) ||
            !CheckAcl(
                aclrtMalloc(
                    &device_control, sizeof(Control),
                    ACL_MEM_MALLOC_NORMAL_ONLY
                ),
                "aclrtMalloc(shared protocol control)"
            )) {
            break;
        }
        if ((reinterpret_cast<uintptr_t>(device_state) & 63U) !=
                0 ||
            (reinterpret_cast<uintptr_t>(device_control) & 63U) !=
                0) {
            std::fprintf(
                stderr,
                "Shared protocol allocations must be 64-byte aligned: "
                "state=%p control=%p\n",
                device_state, device_control
            );
            break;
        }

        if (!CheckAcl(
                aclrtMemcpy(
                    device_state, kStatePrefixBytes,
                    host_state, kStatePrefixBytes,
                    ACL_MEMCPY_HOST_TO_DEVICE
                ),
                "H2D shared protocol state prefix"
            ) ||
            !CheckAcl(
                aclrtMemcpy(
                    &static_cast<SchedulerState *>(
                         device_state
                     )->results[0],
                    kResultBytes, host_state->results,
                    kResultBytes, ACL_MEMCPY_HOST_TO_DEVICE
                ),
                "H2D zero shared protocol results"
            ) ||
            !CheckAcl(
                aclrtMemcpy(
                    &static_cast<SchedulerState *>(
                         device_state
                     )->shared_map,
                    kSharedSidecarBytes,
                    &host_state->shared_map,
                    kSharedSidecarBytes,
                    ACL_MEMCPY_HOST_TO_DEVICE
                ),
                "H2D shared protocol sidecar"
            ) ||
            !CheckAcl(
                aclrtMemcpy(
                    device_control, sizeof(Control),
                    &host_control, sizeof(Control),
                    ACL_MEMCPY_HOST_TO_DEVICE
                ),
                "H2D shared protocol control"
            )) {
            break;
        }

        void *kernel_args[] = {
            device_state, device_control
        };
        rtArgsEx_t args_info{};
        args_info.args = kernel_args;
        args_info.argsSize = sizeof(kernel_args);
        rtTaskCfgInfo_t task_config{};
        if (!CheckRt(
                rtKernelLaunchWithHandleV2(
                    kernel_handle, 0,
                    pa_scheduler::kAicWorkers,
                    &args_info, nullptr, stream,
                    &task_config
                ),
                "launch shared protocol mixed AICore kernel"
            ) ||
            !CheckAcl(
                aclrtSynchronizeStream(stream),
                "synchronize shared protocol mixed AICore kernel"
            )) {
            break;
        }

        if (!CheckAcl(
                aclrtMemcpy(
                    host_state, kStatePrefixBytes,
                    device_state, kStatePrefixBytes,
                    ACL_MEMCPY_DEVICE_TO_HOST
                ),
                "D2H shared protocol state prefix"
            ) ||
            !CheckAcl(
                aclrtMemcpy(
                    host_state->results, kResultBytes,
                    &static_cast<SchedulerState *>(
                         device_state
                     )->results[0],
                    kResultBytes, ACL_MEMCPY_DEVICE_TO_HOST
                ),
                "D2H shared protocol results"
            ) ||
            !CheckAcl(
                aclrtMemcpy(
                    &host_state->shared_map,
                    kSharedSidecarBytes,
                    &static_cast<SchedulerState *>(
                         device_state
                     )->shared_map,
                    kSharedSidecarBytes,
                    ACL_MEMCPY_DEVICE_TO_HOST
                ),
                "D2H shared protocol sidecar"
            )) {
            break;
        }
        execution_ok = ValidateReaderReclaim(
            *host_state, reader_chain,
            inactive_reader, ordering
        );
    } while (false);

    bool cleanup_ok = true;
    if (device_control != nullptr) {
        cleanup_ok &=
            CheckAcl(
                aclrtFree(device_control),
                "aclrtFree(shared protocol control)"
            );
    }
    if (device_state != nullptr) {
        cleanup_ok &=
            CheckAcl(
                aclrtFree(device_state),
                "aclrtFree(shared protocol SchedulerState)"
            );
    }
    if (kernel_handle != nullptr) {
        const rtError_t unload_error = registered_all
            ? rtDevBinaryUnRegister(kernel_handle)
            : rtBinaryUnLoad(kernel_handle);
        cleanup_ok &=
            CheckRt(
                unload_error,
                "unload shared protocol mixed AICore ELF"
            );
    }
    if (stream != nullptr) {
        cleanup_ok &=
            CheckAcl(
                aclrtDestroyStream(stream),
                "aclrtDestroyStream"
            );
    }
    if (device_set) {
        cleanup_ok &=
            CheckAcl(
                aclrtResetDevice(device), "aclrtResetDevice"
            );
    }
    if (acl_initialized) {
        cleanup_ok &=
            CheckAcl(aclFinalize(), "aclFinalize");
    }
    (void)munmap(host_state, sizeof(SchedulerState));

    std::printf(
        "[SHARED-PROTOCOL-LITMUS] scenario=%s direction=%s "
        "ordering=%s device=%d semantic=%s cleanup=%s\n",
        argv[2], argv[3], argv[4], device,
        execution_ok ? "PASS" : "FAIL",
        cleanup_ok ? "PASS" : "FAIL"
    );
    return execution_ok && cleanup_ok
        ? EXIT_SUCCESS
        : EXIT_FAILURE;
}
