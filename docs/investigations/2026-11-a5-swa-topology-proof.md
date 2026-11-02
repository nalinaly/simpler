# A5 SWA: reuse the discovered compact topology

Date: 2026-11-02. The candidate is provisionally retained: all four frozen
alternating pairs improved, with a median paired improvement of 5.72%.
Selected HBG regression coverage is complete, with the shutdown warning below.
HBG has not overtaken TMR.

## Mechanism and publication contract

AICPU discovery already checks every worker's core type, compact launch
coordinate and complete cluster before publishing READY. Each AICore then
repeated the full peer-context scan when first constructing admission
membership. The candidate adds explicit worker-index validation to discovery
and publishes a compact-layout marker in the graph summary before READY.
Admission initialization uses this proof to populate its private membership
cache. The admission hot loop and task-specific membership checks are unchanged.

The marker occupies eight former padding bytes at offset 104. Summary size and
alignment remain 128 bytes. Zero keeps the general peer-scan implementation;
an unknown marker rejects initialization. Compact launch IDs are independent of
sparse physical core IDs. Local initialization still validates its own role,
cluster, peers and worker count. No legal worker is removed from competition.

The summary is initialized before READY and immutable throughout its concurrent
AICore readers. The existing summary cache observation remains. There is no new
runtime atomic, bypass-store or cache-handoff protocol. Error/epoch gates,
bootstrap barriers, Mix and sync-start behavior remain intact.

This follows the immutable-data and cacheline ownership rules in the sibling
`simpler-fdwic-swimlane-deps/tests/atomic_probe/ATOMIC_USAGE_GUIDE.md`, and the
entry-proof reuse and controlled-comparison lessons in its `perf_opt_record.md`
sections 15.62 and 15.65. The reference uses a different CANN/chip configuration;
its numeric results are not transferred to this CANN 9.3 / Ascend950DT run.
Existing shared-control publication methods require their own audit; this
candidate is not evidence that every pre-existing memory protocol matches that
guide.

## Frozen comparison

The unchanged DeepSeek V4.1 Flash SWA case uses 72 tokens, 12 requests, seed 17
and the same frozen inputs and five golden comparisons. A is `protocol-const`;
B is `topology-proof`. Each complete image includes the runtime libraries and
business chip-callable blob. The Python interface, generated sources and exact
business blob match between A and B. Artifact hashes and actual worker library
paths are checked on replay, retaining PTO-ISA validation.

Device 2 was held through one queue allocation for order `ABBABAAB`. Each batch
used five warmups and 100 uninstrumented device-wall samples. No compilation
ran concurrently. All eight numerical and resident-mode checks passed.

| Pair | A median (us) | B median (us) | B improvement |
| ---- | ------------: | ------------: | ------------: |
| A1/B2 | 933.9860 | 884.5895 | 5.289% |
| B3/A4 | 924.8365 | 891.7550 | 3.577% |
| B5/A6 | 974.5450 | 911.7215 | 6.446% |
| A7/B8 | 943.5605 | 885.5090 | 6.152% |

The comparison units are batch medians. All 800 samples are retained; they are
not pooled into an independence or confidence claim. Queue isolation does not
prove whole-system exclusivity. The earlier TMR repeat was about 642 us, so
these results do not satisfy the performance goal.

## Validation and diagnostic limits

All 99 A5 HBG C++ targets passed after the six focused groups. They compare exact compact/general membership
for all lane masks and valid sync-start shapes at 1, 2, 22, 32 and 36 clusters,
reject malformed local bindings and unknown markers, and exercise real AICPU
initialization to prove the marker precedes every READY reply. Invalid worker
identity prevents proof publication. The positive matrix uses supported
sync-start N=2 shapes; an initial unsupported N=1 fixture was corrected without
relaxing production validation.

Both frozen L4 captures pass the existing native audit: 96 workers, 549
executions, 24 logical executable tasks, 1344 member obligations and 8674 events.
The earliest context-publication end to first kernel interval is 116.766 us
for A and 83.562 us for B. This is one instrumented capture per image, consistent
with less startup work but not an exact decomposition of the L0 improvement.

Artifacts live under the development workspace's `logs/swa-hbg-opt-20261101/`:
`frozen-topology-proof-abba/summary.json`, the eight batch records,
`topology-proof-bundle-comparison.json`, `topology-proof-reference-review.json`,
`topology-proof-profile/audit.json`, `protocol-const-frozen-profile/audit.json`,
and `compare_topology_profiles.py` with its CSV/JSON outputs. Build and focused
test logs retain initial fixture/linker failures separately from passing runs.

The selected scene-test coverage below level 4 completed with 99 simulation
passes and two skips, and 212 hardware passes and eight skips on devices 2/3.
The original simulation and hardware sweeps hit their aggregate 600/1800-second
limits, so only unfinished cases were resumed; this is combined coverage, not
an uninterrupted sweep. No case assertion failed. The hardware first attempt
was also refused before execution because a selected case needed two devices.

Simulation and hardware L2 shutdown emitted nanobind reference warnings. A
single-case diagnostic reproduced the warning after successful `Worker.close`,
an empty compile cache and cleanup journal, and a CLOSED worker with no close
error. The precise reference holder is unresolved; this is not claimed as a
pre-existing issue or clean interpreter shutdown. No runtime check or warning
was suppressed. See `topology-proof-regression-summary.json` and
`nanobind-teardown-diagnostic.json` in the artifact directory.
