# CntNotify ADD completion chains deferred until driver support

Date: 2026-11-17. Status: withdrawn from the active implementation at the
user's request; retain the design and code for a supported ADD software stack.

## Decision and retained implementation

Commit `8a2c5e42bb5d918110af0ca3d9a3e45f5b72f922` prepared a real, non-returning
CntNotify ADD path. The installed driver still rejects the AICore ADD mapping
with 507899. CPU validation and a successful hardware GM fallback cannot qualify
that path for production or establish an ADD performance result.

Revert the complete preparation commit, including its runtime, platform, DFX,
tests and active design changes. Production behavior returns to
`f8a6b675617ff4d36655b380ce75a97ea11ca446`: SET/read logical DONE notifications,
remote GM MIX SET handshakes, the retained Host resource pool and the resource
exhaustion handling fix. The original preparation commit stays in Git history.
Pre-commit also normalizes whitespace in two restored protocol headers; it changes no runtime behavior.
This withdrawal is a capability/readiness decision, not a measured ADD regression.

## Preserved proposal

- A physical counter is shared only along a strict, real DAG ancestor chain.
  Each task has a cumulative threshold; independent tasks gain no dependency.
- Every executed lane contributes physical completion P and condition completion
  Q. Synchronous ordinary returns combine them into ADD(2); deferred returns and
  successful condition checks contribute ADD(1) separately.
- MIX retains the full resource-retirement boundary before publishing physical
  credits. Exactly-once condition ownership and dynamic GM task claims remain.
- Counters only increase during a run. Host planning splits before 32-bit
  overflow. READY preparation resets each chain once; there is no within-run
  reset, bitmap, bank ownership transfer or retired-state snapshot.
- Successors observe the threshold directly. Canonical completion accounting is
  separate and must not gate readiness, task ownership or resource retirement.
- The proposed DFX records actual P/Q contributions and counter observations,
  without inventing a returned FAA ticket or using accounting time as readiness.

The complete proposal is available in the original commit's
`docs/zh-cn/design/a5-hbg-graph-task/build-execute-separation.md`, under
the non-returning ADD completion-chain section. It is not current behavior.

## Evidence and re-entry requirements

The preparation passed focused runtime, Host and DFX CPU checks, A5 compilation
and 16/16 device translation-unit IR checks. A single-device SWA functional run
on automatically allocated device 1 passed numerical validation with the GM
fallback. The ADD mapping failed with 507899 and was attempted once in that
context. There is no real AICore ADD twelve-case performance result.

Local preservation paths:

- `artifacts/cnt-add-rollback-20261117/add-preparation.patch` and
  `design-before-revert.md`: complete patch and design snapshot.
- `artifacts/cnt-add-prep-20261117/`: validation summary, fallback hardware logs,
  installation provenance and the future ADD-coverage check.
- `artifacts/c2a-handoff-20261115/cnt-add-prepared-final/`: frozen source, runtime
  binaries and IR evidence for the prepared candidate.
- `artifacts/c2a-handoff-20261115/cnt-retained-final/`: restored supported runtime.

Reconsider after the installed SDK/driver provides the actual ADD alias. Start
with a new Worker/context, require observed ADD coverage, validate accumulation,
late observation, branch independence, MIX, deferred conditions and replay reset,
then compare the established twelve-case AICore L0 sum. Retain the current 0.5%
aggregate regression limit. Do not substitute SET-overwrite timing, stream-level
ADD capability, CPU emulation or a successful GM fallback for these checks.
