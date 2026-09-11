/* Copyright (c) PyPTO Contributors.
 * Licensed under CANN Open Software License Agreement Version 2.0.
 * See LICENSE in the root of the software repository.
 */
#include <array>
#include <cstdint>
#include <cstdio>
#include <vector>

#define PA_DEVICE inline
#define PA_GM
#include "common/shared_exec_protocol.h"

using namespace pa_scheduler::cross_core;

struct ReadySource {
    std::array<bool, 128> ready{};
    mutable std::vector<int32_t> calls;
    bool IsReady(int32_t task) const {
        calls.push_back(task);
        return ready.at(static_cast<size_t>(task));
    }
};

// Frozen pre-optimization algorithm: compare return, prefix and exact poll
// sequence, including malformed bounds, not just the all-ready happy path.
bool ReferenceReady(ExecutionToken &token, const ReadySource &source) {
    if (token.control.phase != ExecTokenPhase::WaitingFanin) return false;
    const uint32_t count = ExecutionTokenFaninCount(token);
    const uint32_t task = token.control.task_id;
    if (token.control.fanin_ready_prefix > count) return false;
    for (uint32_t edge = token.control.fanin_ready_prefix; edge < count; ++edge) {
        int32_t producer = -1;
        if (!ExecutionTokenFanin(token, edge, producer) || producer < 0 ||
            static_cast<uint32_t>(producer) >= task || !source.IsReady(producer)) return false;
        token.control.fanin_ready_prefix = edge + 1U;
    }
    return true;
}

int main() {
    uint32_t checks = 0;
    for (uint32_t count = 0; count <= 16; ++count) {
        for (uint32_t prefix = 0; prefix <= count + 1; ++prefix) {
            for (uint32_t scenario = 0; scenario < 8; ++scenario) {
                ExecPayloadStorage payload{};
                ExecutionToken original{};
                original.control.phase = scenario == 7 ? ExecTokenPhase::Idle : ExecTokenPhase::WaitingFanin;
                original.control.task_id = 97;
                original.control.fanin_ready_prefix = prefix;
                original.control.payload_address = scenario == 1 ? 0 : reinterpret_cast<uintptr_t>(&payload);
                original.control.payload_lines = scenario == 2 ? 1 : 4;
                original.control.shape_and_scalar_offset =
                    (uint64_t{8} << 48U) | (static_cast<uint64_t>(count) << 32U) | (uint64_t{3} << 16U);
                for (uint32_t edge = 0; edge < count; ++edge) {
                    int32_t producer = static_cast<int32_t>(edge);
                    if (edge == prefix && scenario == 3) producer = -1;
                    if (edge == prefix && scenario == 4) producer = 97;
                    const uint32_t shift = (edge % 2) * 32;
                    payload.words[11 + edge / 2] |= static_cast<uint64_t>(static_cast<uint32_t>(producer)) << shift;
                }
                ExecutionToken reference = original, candidate = original;
                for (uint32_t round = 0; round < 3; ++round) {
                    ReadySource a, b;
                    for (uint32_t i = 0; i < 128; ++i) {
                        a.ready[i] = b.ready[i] = scenario != 5 || round == 2 || i < prefix + round;
                    }
                    const bool expected = ReferenceReady(reference, a);
                    const bool actual = ExecutionTokenFaninReady(candidate, b);
                    if (expected != actual || a.calls != b.calls ||
                        reference.control.fanin_ready_prefix != candidate.control.fanin_ready_prefix ||
                        reference.control.phase != candidate.control.phase) {
                        std::fprintf(stderr, "fanin mismatch count=%u prefix=%u scenario=%u round=%u\n",
                                     count, prefix, scenario, round);
                        return 1;
                    }
                    ++checks;
                }
            }
        }
    }
    std::printf("[PASS] fanin snapshot: %u reference-equivalence checks\n", checks);
}
