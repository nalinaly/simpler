namespace {
void TestRouteCacheLifetime() {
    constexpr const char *test = "owner-route-cache-lifetime";
    MappedSchedulerState mapping;
    SchedulerState *state = mapping.Get();
    Check(state != nullptr, test, "state mapping");
    if (state == nullptr) return;
    LimitDefaultPlanToBatches(*state, 2);
    WorkerState &worker = PrepareWorker(*state, 0, CoreRole::Aic);
    LocalStats stats{};
    InitLocalStats(stats, 0, CoreRole::Aic);
    ExecScanTestOps::ResetObservations();
    ExecutionToken &token = state->exec_tokens[0][0];
    Check(PublishKernelCell(*state, 3, kAivBuildOwner, TaskKind::Pv, {1}),
          test, "publish dependency-blocked PV");
    Check(BindSharedExecTicketToToken(token, 3, 0, ExecEngineClass::Aic),
          test, "bind ticket");
    bool completed = false;
    Check(ProgressCrossCoreActiveToken<ExecScanTestOps>(
              state, worker, 0, DrainPlace::FinalDrain, stats, completed) &&
              !completed && token.control.phase == ExecTokenPhase::WaitingFanin &&
              stats.exec_validated_routes == 1U && NoFatal(*state),
          test, "first claim validates the immutable route once");
    for (uint32_t round = 0; round < 10; ++round) {
        Check(ProgressCrossCoreActiveToken<ExecScanTestOps>(
                  state, worker, 0, DrainPlace::FinalDrain, stats, completed) &&
                  !completed && stats.exec_validated_routes == 1U && NoFatal(*state),
              test, "blocked polls preserve the route proof and do not execute");
    }
    state->tasks[1].flag = 1;
    Check(ProgressCrossCoreActiveToken<ExecScanTestOps>(
              state, worker, 0, DrainPlace::FinalDrain, stats, completed) && completed &&
              token.control.phase == ExecTokenPhase::Idle && stats.exec_validated_routes == 0U,
          test, "completion clears cached proof");
    Check(PublishKernelCell(*state, 6, kAivBuildOwner, TaskKind::Qk), test, "publish QK reuse");
    Check(BindSharedExecTicketToToken(token, 6, 0, ExecEngineClass::Aic), test, "reuse token");
    Check(ProgressCrossCoreActiveToken<ExecScanTestOps>(
              state, worker, 0, DrainPlace::FinalDrain, stats, completed) && completed &&
              stats.exec_validated_routes == 0U && NoFatal(*state),
          test, "another task kind revalidates and finishes on the same slot");

    // A stale bit must never let a new ticket bypass its initial route checks.
    Check(PublishKernelCell(*state, 8, kAivBuildOwner, TaskKind::Pv, {6}),
          test, "publish invalid-reuse probe");
    Check(BindSharedExecTicketToToken(token, 8, 0, ExecEngineClass::Aic), test, "bind invalid-reuse probe");
    stats.exec_validated_routes = 1U;
    token.control.execute_owner = 1;
    Check(!ProgressCrossCoreActiveToken<ExecScanTestOps>(
              state, worker, 0, DrainPlace::FinalDrain, stats, completed) &&
              !completed && stats.exec_validated_routes == 0U &&
              token.control.phase == ExecTokenPhase::Faulted,
          test, "new ticket clears stale proof and rejects a mismatched owner");
    std::printf("[PASS] %s\n", test);
}
}  // namespace
