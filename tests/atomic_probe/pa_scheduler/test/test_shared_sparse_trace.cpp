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

#include <cstdio>

#include "host_support.h"

namespace {

using pa_scheduler::AtomicOp;
using pa_scheduler::AtomicSite;
using pa_scheduler::TaskKind;
using pa_scheduler::TracePhase;
using pa_scheduler::TraceRecord;
using pa_scheduler::host::AtomicRecordSchemaValid;
using pa_scheduler::host::SharedSparseTraceValidator;
using pa_scheduler::kAtomicOpMask;
using pa_scheduler::kAtomicPollBatch;
using pa_scheduler::kAtomicPollCountShift;
using pa_scheduler::kAtomicResultUsed;
using pa_scheduler::kAtomicReturnReady;

int g_failures = 0;

void Check(bool condition, const char *message) {
    if (condition) return;
    std::fprintf(stderr, "[FAIL] shared sparse trace: %s\n", message);
    ++g_failures;
}

TraceRecord MakeRecord(
    TracePhase phase, int32_t task_id, int32_t function_id,
    uint64_t begin, uint64_t end, uint32_t flags = 0,
    uint32_t auxiliary = 0
) {
    TraceRecord record{};
    record.phase = static_cast<int32_t>(phase);
    record.task_id = task_id;
    record.function_id = function_id;
    record.start_cycle = begin;
    record.end_cycle = end;
    record.flags = flags;
    record.auxiliary = auxiliary;
    return record;
}

int32_t FunctionId(TaskKind kind) {
    return kind == TaskKind::Alloc
        ? -1
        : static_cast<int32_t>(
              static_cast<uint32_t>(kind) - 1U
          );
}

uint32_t IsAlloc(TaskKind kind) {
    return kind == TaskKind::Alloc ? 1U : 0U;
}

enum class ActorOutcome {
    Winner,
    Loser,
    NotAttempted,
};

struct TaskTraceBuilder {
    SharedSparseTraceValidator &validator;
    uint64_t tick = 10;
    uint64_t submit_begin = 0;

    bool Begin(
        uint32_t task_id, TaskKind kind, ActorOutcome outcome
    ) {
        submit_begin = tick;
        const uint64_t efdrain_end = tick + 2;
        bool ok = validator.Observe(
            MakeRecord(
                TracePhase::EfDrain,
                static_cast<int32_t>(task_id), -1,
                submit_begin, efdrain_end
            )
        );

        // BuildCallbackSubmitArgs 没有独立 raw phase，因此允许它占用
        // EfDrain.end 与 Materialize.start 之间的 Submit residual。
        const uint64_t materialize_begin = efdrain_end + 1;
        const uint64_t materialize_end = materialize_begin + 3;
        ok &= validator.Observe(
            MakeRecord(
                TracePhase::Materialize,
                static_cast<int32_t>(task_id), -1,
                materialize_begin, materialize_end, 0,
                IsAlloc(kind)
            )
        );

        const bool winner = outcome == ActorOutcome::Winner;
        const bool attempted =
            outcome != ActorOutcome::NotAttempted;
        const uint32_t flags =
            (winner ? pa_scheduler::kClaimWon : 0U) |
            (attempted ? pa_scheduler::kClaimAttempted : 0U);
        ok &= validator.Observe(
            MakeRecord(
                TracePhase::Claim,
                static_cast<int32_t>(task_id),
                winner ? FunctionId(kind) : -1,
                materialize_end, materialize_end + 1,
                flags, IsAlloc(kind)
            )
        );
        tick = materialize_end + 1;
        return ok;
    }

    bool FinishWinner(uint32_t task_id, TaskKind kind) {
        const int32_t function_id = FunctionId(kind);

        // writer delta 在 Register 计时前预计算，故 Claim 与 Register
        // 之间允许有真实 residual，不能伪造为首尾相接。
        const uint64_t register_begin = tick + 2;
        const uint64_t register_end = register_begin + 3;
        bool ok = validator.Observe(
            MakeRecord(
                TracePhase::Register,
                static_cast<int32_t>(task_id), function_id,
                register_begin, register_end, 0,
                kind == TaskKind::Alloc ? 0U : 1U
            )
        );
        ok &= validator.Observe(
            MakeRecord(
                TracePhase::SharedRegisterPublishMetadata,
                static_cast<int32_t>(task_id), function_id,
                register_begin + 1, register_end - 1
            )
        );

        uint64_t previous_end = register_end;
        if (kind != TaskKind::Alloc) {
            ok &= validator.Observe(
                MakeRecord(
                    TracePhase::Fanin,
                    static_cast<int32_t>(task_id), function_id,
                    previous_end, previous_end + 2, 0, 3
                )
            );
            previous_end += 2;
        }
        ok &= validator.Observe(
            MakeRecord(
                kind == TaskKind::Alloc
                    ? TracePhase::AllocComplete
                    : TracePhase::WinnerBuild,
                static_cast<int32_t>(task_id), function_id,
                previous_end, previous_end + 4
            )
        );
        previous_end += 4;
        ok &= validator.Observe(
            MakeRecord(
                TracePhase::Submit,
                static_cast<int32_t>(task_id), function_id,
                submit_begin, previous_end + 1,
                pa_scheduler::kClaimWon, IsAlloc(kind)
            )
        );
        tick = previous_end + 2;
        return ok;
    }

    bool FinishNonWinner(uint32_t task_id, TaskKind kind) {
        const uint64_t submit_end = tick + 1;
        const bool ok = validator.Observe(
            MakeRecord(
                TracePhase::Submit,
                static_cast<int32_t>(task_id), -1,
                submit_begin, submit_end, 0, IsAlloc(kind)
            )
        );
        tick = submit_end + 1;
        return ok;
    }
};

bool OpenAllocWinnerRegister(
    SharedSparseTraceValidator &validator,
    uint64_t register_begin = 18,
    uint64_t register_end = 24
) {
    return validator.Observe(
               MakeRecord(TracePhase::EfDrain, 0, -1, 10, 12)
           ) &&
           validator.Observe(
               MakeRecord(
                   TracePhase::Materialize, 0, -1,
                   13, 16, 0, 1
               )
           ) &&
           validator.Observe(
               MakeRecord(
                   TracePhase::Claim, 0, -1, 16, 17,
                   pa_scheduler::kClaimWon |
                       pa_scheduler::kClaimAttempted,
                   1
               )
           ) &&
           validator.Observe(
               MakeRecord(
                   TracePhase::Register, 0, -1,
                   register_begin, register_end
               )
           );
}

void TestAcceptsAllActorMaterializeFlow() {
    SharedSparseTraceValidator validator;
    TaskTraceBuilder trace{validator};
    Check(
        trace.Begin(
            0, TaskKind::Alloc, ActorOutcome::Winner
        ) &&
            trace.FinishWinner(0, TaskKind::Alloc),
        "Alloc winner closes through all-actor Materialize and winner tail"
    );
    Check(
        trace.Begin(
            1, TaskKind::Qk, ActorOutcome::Loser
        ) &&
            trace.FinishNonWinner(1, TaskKind::Qk),
        "attempted loser retains its pre-Claim Materialize"
    );
    Check(
        trace.Begin(
            2, TaskKind::Sf, ActorOutcome::NotAttempted
        ) &&
            trace.FinishNonWinner(2, TaskKind::Sf),
        "not_attempted actor also retains its pre-Claim Materialize"
    );
    Check(
        trace.Begin(
            3, TaskKind::Pv, ActorOutcome::Winner
        ) &&
            trace.FinishWinner(3, TaskKind::Pv),
        "ordinary winner includes Register, metadata, Fanin and WinnerBuild"
    );
    Check(
        trace.Begin(
            4, TaskKind::Up, ActorOutcome::Loser
        ) &&
            trace.FinishNonWinner(4, TaskKind::Up),
        "final logical task may close through attempted loser"
    );
    Check(validator.Closed(), "mixed all-actor flow is fully closed");
    Check(
        validator.EfDrainCount() == 5 &&
            validator.MaterializeCount() == 5 &&
            validator.ClaimCount() == 5 &&
            validator.WinnerCount() == 2 &&
            validator.RegisterCount() == 2 &&
            validator.RegisterMetadataCount() == 2 &&
            validator.FaninCount() == 1 &&
            validator.WinnerTailCount() == 2 &&
            validator.SubmitCount() == 5,
        "all actors have four base records and only winners have tail records"
    );
}

void TestRejectsReplayPrefixAndBoundaryDrift() {
    {
        SharedSparseTraceValidator validator;
        Check(
            !validator.Observe(
                MakeRecord(TracePhase::Claim, 0, -1, 10, 11)
            ),
            "Claim cannot appear before EfDrain and Materialize"
        );
        Check(
            !validator.Observe(
                MakeRecord(
                    TracePhase::Materialize, 0, -1,
                    10, 11, 0, 1
                )
            ),
            "Materialize cannot appear before EfDrain"
        );
    }
    {
        SharedSparseTraceValidator validator;
        Check(
            !validator.Observe(
                MakeRecord(TracePhase::EfDrain, 1, -1, 10, 12)
            ),
            "the first per-core task must be task 0"
        );
    }
    {
        SharedSparseTraceValidator validator;
        Check(
            validator.Observe(
                MakeRecord(TracePhase::EfDrain, 0, -1, 10, 12)
            ),
            "EfDrain opens the missing-Materialize test"
        );
        Check(
            !validator.Observe(
                MakeRecord(
                    TracePhase::Claim, 0, -1, 12, 13,
                    pa_scheduler::kClaimAttempted, 1
                )
            ),
            "Claim cannot skip the all-actor Materialize"
        );
    }
    {
        SharedSparseTraceValidator validator;
        Check(
            validator.Observe(
                MakeRecord(TracePhase::EfDrain, 0, -1, 10, 12)
            ),
            "EfDrain opens the early-Materialize test"
        );
        Check(
            !validator.Observe(
                MakeRecord(
                    TracePhase::Materialize, 0, -1,
                    11, 14, 0, 1
                )
            ),
            "Materialize cannot begin before EfDrain ends"
        );
    }
    {
        SharedSparseTraceValidator validator;
        Check(
            validator.Observe(
                MakeRecord(TracePhase::EfDrain, 0, -1, 10, 12)
            ) &&
                validator.Observe(
                    MakeRecord(
                        TracePhase::Materialize, 0, -1,
                        13, 16, 0, 1
                    )
                ),
            "Materialize may begin after the untraced args-build gap"
        );
        Check(
            !validator.Observe(
                MakeRecord(
                    TracePhase::Claim, 0, -1, 17, 18,
                    pa_scheduler::kClaimAttempted, 1
                )
            ),
            "Claim must start exactly at Materialize.end"
        );
    }
    {
        SharedSparseTraceValidator validator;
        TaskTraceBuilder trace{validator};
        Check(
            trace.Begin(
                0, TaskKind::Alloc, ActorOutcome::Loser
            ) &&
                trace.FinishNonWinner(0, TaskKind::Alloc),
            "task 0 loser establishes the sequence-gap test"
        );
        Check(
            !validator.Observe(
                MakeRecord(TracePhase::EfDrain, 2, -1, 30, 32)
            ),
            "a skipped task_id is rejected after a loser"
        );
    }
}

void TestNonWinnerStopsAfterClaim() {
    constexpr TracePhase forbidden[] = {
        TracePhase::Materialize,
        TracePhase::PrepareMap,
        TracePhase::Fanin,
        TracePhase::Register,
        TracePhase::SharedRegisterPublishMetadata,
        TracePhase::SharedMaterializePublishTaskOutputs,
        TracePhase::SharedMaterializePublishTaskOutputsCopy,
        TracePhase::SharedMaterializePublishTaskOutputsFlush,
        TracePhase::WinnerBuild,
        TracePhase::AllocComplete,
    };
    for (TracePhase phase : forbidden) {
        SharedSparseTraceValidator validator;
        TaskTraceBuilder trace{validator};
        Check(
            trace.Begin(
                0, TaskKind::Alloc, ActorOutcome::Loser
            ),
            "loser prefix includes Materialize before Claim"
        );
        Check(
            !validator.Observe(
                MakeRecord(phase, 0, -1, 17, 18)
            ),
            "loser rejects every phase after Claim except Submit"
        );
    }
    {
        SharedSparseTraceValidator validator;
        TaskTraceBuilder trace{validator};
        Check(
            trace.Begin(
                0, TaskKind::Alloc,
                ActorOutcome::NotAttempted
            ) &&
                trace.FinishNonWinner(0, TaskKind::Alloc),
            "not_attempted actor closes through Submit"
        );
        Check(
            !validator.Observe(
                MakeRecord(
                    TracePhase::Submit, 0, -1,
                    trace.submit_begin, trace.tick, 0, 1
                )
            ),
            "duplicate non-winner Submit is rejected"
        );
    }
}

void TestRejectsRetiredSharedOutputDetailsAndPrepareMap() {
    constexpr TracePhase retired[] = {
        TracePhase::SharedMaterializePublishTaskOutputs,
        TracePhase::SharedMaterializePublishTaskOutputsCopy,
        TracePhase::SharedMaterializePublishTaskOutputsFlush,
    };
    for (TracePhase phase : retired) {
        SharedSparseTraceValidator validator;
        Check(
            !validator.Observe(
                MakeRecord(phase, -1, -1, 1, 1)
            ),
            "retired shared-output detail is rejected globally"
        );
    }
    SharedSparseTraceValidator validator;
    Check(
        !validator.Observe(
            MakeRecord(TracePhase::PrepareMap, -1, -1, 1, 1)
        ),
        "shared rejects private PrepareMap globally"
    );
}

void TestMaterializeIdentityAndClaimRoles() {
    {
        SharedSparseTraceValidator validator;
        Check(
            validator.Observe(
                MakeRecord(TracePhase::EfDrain, 0, -1, 10, 12)
            ),
            "EfDrain opens Materialize identity test"
        );
        Check(
            !validator.Observe(
                MakeRecord(
                    TracePhase::Materialize, 0, 0,
                    12, 14, 0, 1
                )
            ),
            "pre-Claim Materialize cannot claim a winner function"
        );
    }
    {
        SharedSparseTraceValidator validator;
        Check(
            validator.Observe(
                MakeRecord(TracePhase::EfDrain, 0, -1, 10, 12)
            ),
            "EfDrain opens Materialize payload test"
        );
        Check(
            !validator.Observe(
                MakeRecord(
                    TracePhase::Materialize, 0, -1,
                    12, 14, 1, 1
                )
            ),
            "Materialize flags must remain zero"
        );
    }
    {
        SharedSparseTraceValidator validator;
        Check(
            validator.Observe(
                MakeRecord(TracePhase::EfDrain, 0, -1, 10, 12)
            ) &&
                validator.Observe(
                    MakeRecord(
                        TracePhase::Materialize, 0, -1,
                        12, 14, 0, 1
                    )
                ),
            "valid Alloc prefix reaches Claim role test"
        );
        Check(
            !validator.Observe(
                MakeRecord(
                    TracePhase::Claim, 0, -1, 14, 15,
                    pa_scheduler::kClaimWon, 1
                )
            ),
            "winner always implies attempted"
        );
    }
}

void TestRejectsWinnerShapeDrift() {
    {
        SharedSparseTraceValidator validator;
        Check(
            OpenAllocWinnerRegister(validator),
            "Alloc winner reaches Register"
        );
        Check(
            validator.Observe(
                MakeRecord(
                    TracePhase::SharedRegisterPublishMetadata,
                    0, -1, 20, 22
                )
            ),
            "Alloc Register metadata is accepted"
        );
        Check(
            !validator.Observe(
                MakeRecord(TracePhase::Fanin, 0, -1, 24, 25)
            ),
            "Alloc winner cannot emit Fanin"
        );
        Check(
            !validator.Observe(
                MakeRecord(
                    TracePhase::WinnerBuild, 0, -1, 24, 28
                )
            ),
            "Alloc winner requires AllocComplete"
        );
    }
    {
        SharedSparseTraceValidator validator;
        TaskTraceBuilder trace{validator};
        Check(
            trace.Begin(
                0, TaskKind::Alloc, ActorOutcome::Loser
            ) &&
                trace.FinishNonWinner(0, TaskKind::Alloc) &&
                trace.Begin(
                    1, TaskKind::Qk, ActorOutcome::Winner
                ),
            "ordinary winner prefix is valid"
        );
        const int32_t function_id = FunctionId(TaskKind::Qk);
        const uint64_t register_begin = trace.tick + 2;
        const uint64_t register_end = register_begin + 3;
        Check(
            validator.Observe(
                MakeRecord(
                    TracePhase::Register, 1, function_id,
                    register_begin, register_end, 0, 1
                )
            ) &&
                validator.Observe(
                    MakeRecord(
                        TracePhase::SharedRegisterPublishMetadata,
                        1, function_id,
                        register_begin + 1, register_end - 1
                    )
                ),
            "ordinary winner reaches Fanin"
        );
        Check(
            !validator.Observe(
                MakeRecord(
                    TracePhase::WinnerBuild, 1, function_id,
                    register_end, register_end + 1
                )
            ),
            "ordinary winner cannot skip Fanin"
        );
    }
    {
        SharedSparseTraceValidator validator;
        TaskTraceBuilder trace{validator};
        Check(
            trace.Begin(
                0, TaskKind::Alloc, ActorOutcome::Winner
            ),
            "incomplete winner opens closure test"
        );
        Check(
            !validator.Closed(),
            "winner cannot close before Register, tail and Submit"
        );
    }
}

void TestRegisterMetadataDetailContract() {
    {
        SharedSparseTraceValidator validator;
        Check(
            OpenAllocWinnerRegister(validator),
            "Register opens missing-detail test"
        );
        Check(
            !validator.Observe(
                MakeRecord(
                    TracePhase::AllocComplete, 0, -1, 24, 28
                )
            ),
            "winner cannot omit Register metadata detail"
        );
    }
    {
        SharedSparseTraceValidator validator;
        Check(
            OpenAllocWinnerRegister(validator) &&
                validator.Observe(
                    MakeRecord(
                        TracePhase::SharedRegisterPublishMetadata,
                        0, -1, 20, 22
                    )
                ),
            "one contained Register metadata detail is accepted"
        );
        Check(
            !validator.Observe(
                MakeRecord(
                    TracePhase::SharedRegisterPublishMetadata,
                    0, -1, 20, 22
                )
            ),
            "duplicate Register metadata detail is rejected"
        );
    }
    for (int variant = 0; variant < 5; ++variant) {
        SharedSparseTraceValidator validator;
        Check(
            OpenAllocWinnerRegister(validator),
            "Register opens metadata boundary test"
        );
        TraceRecord detail = MakeRecord(
            TracePhase::SharedRegisterPublishMetadata,
            0, -1, 20, 22
        );
        if (variant == 0) detail.start_cycle = 17;
        if (variant == 1) detail.end_cycle = 25;
        if (variant == 2) detail.task_id = 1;
        if (variant == 3) detail.function_id = 0;
        if (variant == 4) {
            detail.flags = 1;
            detail.auxiliary = 1;
        }
        Check(
            !validator.Observe(detail),
            "Register metadata rejects bad containment, identity or payload"
        );
    }
}

void TestSubmitParentAndCrossTaskBoundaries() {
    {
        SharedSparseTraceValidator validator;
        TaskTraceBuilder trace{validator};
        Check(
            trace.Begin(
                0, TaskKind::Alloc, ActorOutcome::Loser
            ),
            "loser prefix opens Submit-start test"
        );
        Check(
            !validator.Observe(
                MakeRecord(
                    TracePhase::Submit, 0, -1,
                    trace.submit_begin + 1, trace.tick + 1, 0, 1
                )
            ),
            "Submit parent must begin at EfDrain.start"
        );
    }
    {
        SharedSparseTraceValidator validator;
        TaskTraceBuilder trace{validator};
        Check(
            trace.Begin(
                0, TaskKind::Alloc, ActorOutcome::Loser
            ),
            "loser prefix opens Submit-end test"
        );
        Check(
            !validator.Observe(
                MakeRecord(
                    TracePhase::Submit, 0, -1,
                    trace.submit_begin, trace.tick - 1, 0, 1
                )
            ),
            "Submit parent must contain Claim.end"
        );
    }
    {
        SharedSparseTraceValidator validator;
        TaskTraceBuilder trace{validator};
        Check(
            trace.Begin(
                0, TaskKind::Alloc, ActorOutcome::Loser
            ) &&
                trace.FinishNonWinner(0, TaskKind::Alloc),
            "task 0 closes before cross-task overlap test"
        );
        const uint64_t previous_submit_end = trace.tick - 1;
        Check(
            !validator.Observe(
                MakeRecord(
                    TracePhase::EfDrain, 1, -1,
                    previous_submit_end - 1,
                    previous_submit_end + 1
                )
            ),
            "next task EfDrain cannot overlap the previous Submit parent"
        );
    }
}

void TestSharedInsertTurnAtomicSchema() {
    TraceRecord poll = MakeRecord(
        TracePhase::Atomic, -1, -1, 100, 200,
        static_cast<uint32_t>(AtomicOp::Load) |
            kAtomicResultUsed | kAtomicPollBatch |
            kAtomicReturnReady |
            (17U << kAtomicPollCountShift),
        static_cast<uint32_t>(
            AtomicSite::SharedInsertTurnPoll
        )
    );
    Check(
        AtomicRecordSchemaValid(poll, true),
        "shared insert-turn aggregate PollBatch schema is accepted"
    );
    TraceRecord direct_poll = poll;
    direct_poll.flags =
        static_cast<uint32_t>(AtomicOp::Load) |
        kAtomicResultUsed | kAtomicReturnReady;
    direct_poll.task_id = 3;
    Check(
        !AtomicRecordSchemaValid(direct_poll, true),
        "shared insert-turn poll cannot masquerade as a direct Load"
    );

    TraceRecord handoff = MakeRecord(
        TracePhase::Atomic, 3, -1, 200, 230,
        static_cast<uint32_t>(
            AtomicOp::CompareExchange
        ) |
            kAtomicResultUsed | kAtomicReturnReady,
        static_cast<uint32_t>(
            AtomicSite::SharedInsertTurnHandoff
        )
    );
    Check(
        AtomicRecordSchemaValid(handoff, true),
        "shared insert-turn handoff CAS schema is accepted"
    );
    TraceRecord anonymous_handoff = handoff;
    anonymous_handoff.task_id = -1;
    Check(
        !AtomicRecordSchemaValid(anonymous_handoff, true),
        "handoff CAS requires its shared winner task identity"
    );
    TraceRecord wrong_handoff_op = handoff;
    wrong_handoff_op.flags =
        (wrong_handoff_op.flags & ~kAtomicOpMask) |
        static_cast<uint32_t>(AtomicOp::Exchange);
    Check(
        !AtomicRecordSchemaValid(wrong_handoff_op, true),
        "handoff site rejects a non-CAS atomic op"
    );
}

void TestPlanClosesAllLogicalTasks() {
    // SchedulerState 约 1 GiB，静态零初始化只映射本测试触碰的页面。
    static pa_scheduler::SchedulerState state{};
    state.config.batches = 2;
    state.context_lens[0] = 0;
    state.context_lens[1] = 0;
    pa_scheduler::host::SharedHostTaskPlan plan;
    Check(
        pa_scheduler::host::BuildSharedHostTaskPlan(
            state, &plan
        ),
        "two-batch zero-context host plan is valid"
    );
    Check(
        plan.total_tasks == 2 &&
            plan.TaskAt(0)->kind == TaskKind::Alloc &&
            plan.TaskAt(1)->kind == TaskKind::Alloc,
        "authoritative plan contains one Alloc per empty batch"
    );

    SharedSparseTraceValidator validator(&plan);
    TaskTraceBuilder trace{validator};
    Check(
        trace.Begin(
            0, TaskKind::Alloc, ActorOutcome::Winner
        ) &&
            trace.FinishWinner(0, TaskKind::Alloc),
        "first planned Alloc winner closes"
    );
    Check(
        !validator.Closed(),
        "plan-aware validator rejects truncated replay"
    );
    Check(
        trace.Begin(
            1, TaskKind::Alloc, ActorOutcome::Loser
        ) &&
            trace.FinishNonWinner(1, TaskKind::Alloc),
        "second planned Alloc loser still Materializes before Claim"
    );
    Check(
        validator.Closed(),
        "plan-aware validator closes after every logical task"
    );
}

}  // namespace

int main() {
    TestAcceptsAllActorMaterializeFlow();
    TestRejectsReplayPrefixAndBoundaryDrift();
    TestNonWinnerStopsAfterClaim();
    TestRejectsRetiredSharedOutputDetailsAndPrepareMap();
    TestMaterializeIdentityAndClaimRoles();
    TestRejectsWinnerShapeDrift();
    TestRegisterMetadataDetailContract();
    TestSubmitParentAndCrossTaskBoundaries();
    TestSharedInsertTurnAtomicSchema();
    TestPlanClosesAllLogicalTasks();
    if (g_failures != 0) {
        std::fprintf(
            stderr, "[FAIL] shared sparse trace tests: %d\n",
            g_failures
        );
        return 1;
    }
    std::printf("[PASS] shared sparse trace tests\n");
    return 0;
}
