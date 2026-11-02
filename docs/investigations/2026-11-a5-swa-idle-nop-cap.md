# A5 SWA: a larger register-only idle spin cap

Date: 2026-11-02. Retained as a workload-local checkpoint after four frozen
alternating pairs improved. Median paired improvement is 2.80%; HBG has not
overtaken TMR.

## Change and reference boundary

Only the resident main loop's idle spin cap changes from 256 to 1024 iterations.
Initial delay remains eight iterations, useful progress resets it, and bootstrap
and drain barriers retain their 256-iteration cap. The loop uses an ordinary
counter and volatile NOP assembly; it adds no shared state, sleep or timer.
Ready, error, EXIT, ownership and execution checks remain in place.

The reference is the sibling `simpler-fdwic-swimlane-deps` checkout's
`tests/atomic_probe/ATOMIC_USAGE_GUIDE.md`, `pa_scheduler/loser_overhead.md` and
`perf_opt_record.md` sections 15.69–15.73. Reduced polling can trade discovery
latency against contention, so source operation counts do not establish a gain.
This changes no memory primitive or publication protocol, and does not treat
NOPs as a replacement for atomic synchronization or cache maintenance.

The earlier unmeasured `backoff1024` used a volatile counter and is a different
experiment. Here the production AIC and AIV disassemblies both show no
load/store in the idle spin loop. Each unprofiled dispatch function has exactly
the same instruction count as the preceding `topology-proof` image: 50426 for
AIC and 51727 for AIV. Positional comparison finds one changed instruction in
each, `MOVI X1, #256` to `MOVI X1, #1024`. These are static facts, not a dynamic
traffic or cycle measurement.

## Frozen comparison

A is `topology-proof`; B is `idle-nop1024`. Both use the same DeepSeek V4.1 Flash
SWA program, 72 tokens, 12 requests, seed 17, frozen inputs and five original
golden comparisons. Generated sources, the native Python interface and complete
business callable bytes are identical. The dispatcher file hash also changed,
but every differing byte is confined to its ELF build-id note; all its other
file bytes, including code and data, match. Complete native images are retained
without splicing libraries between them.

One device-2 allocation ran `ABBABAAB`, five warmups and 100 uninstrumented
device-wall samples per batch. All numerical and resident-mode checks passed.
No compilation ran during measurement, and every sample is retained.

| Pair | A median (us) | B median (us) | B improvement |
| ---- | ------------: | ------------: | ------------: |
| A1/B2 | 925.6020 | 891.2720 | 3.709% |
| B3/A4 | 951.2395 | 853.5935 | 10.265% |
| B5/A6 | 917.4865 | 900.3440 | 1.868% |
| A7/B8 | 915.4040 | 898.1545 | 1.884% |

All four pairs favor B; median paired improvement is 2.7966%. Batch medians are
the comparison units. The substantial between-batch variation precludes using
the lowest historical sample as the new baseline or treating individual rounds
as independent replications. This does not establish a gain for other workloads.

## Validation and artifacts

The existing nonce/exit C++ target passed. After rebuilding A5sim, two selected
bounded-timeout cases passed. Four native cases passed: pure C, pure V and Mix
bounded errors, plus a nonreturning Mix lane. Their original error, retirement
and containment assertions remain unchanged. These are focused checks for the
idle-cap change; the broader preceding checkpoint's coverage is documented in
[the topology investigation](2026-11-a5-swa-topology-proof.md).

The separate L4 native audit passed with 96 workers, 549 executions, 24 logical
tasks, 1344 member obligations and 8674 events. Source and the complete installed
native runtime were checked against the frozen B image after the tests.

Artifacts are under the development workspace's `logs/swa-hbg-opt-20261101/`:
`frozen-idle-nop1024-abba/summary.json`, all eight batch records,
`idle-nop1024-bundle-comparison.json`, `idle-nop1024-loop-codegen-comparison.json`,
`idle-nop1024-assembly-verification.json`, `idle-nop1024-profile/audit.json`,
`idle-nop1024-{a5sim,a5}-exit-regression.log` and
`idle-nop1024-retained-provenance.json`. The inherited nanobind shutdown
reference warning and its diagnostic limits remain recorded separately; no
binding or cleanup code was changed to suppress it.
