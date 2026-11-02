# A5 SWA initial Ready and wake-list preparation on the host

**Date**: 2026-11-02
**Verdict**: reverted; four frozen alternating pairs did not establish a stable benefit

## Candidate and boundary

For frozen task graphs without dynamic Graph instances, the candidate built
initial dependency wake lists and Ready roots in the unpublished host staging
image. It validated the entire plan before writing shared state. Ready roots
were distributed by task reference over dense ranks of the appropriate core
type. Block execution ownership remained dynamic.

An immutable summary flag, using existing padding, selected this prepared
state. Both all-worker bootstrap barriers remained. Admission, block claims,
completion, Mix and sync-start protocols were unchanged. Dynamic Graph
instances retained the existing AICore registration pass. All newly introduced
ordinary writes happened before metadata upload and READY; running workers
continued to use the existing atomic queue/wake-list operations.

The reference was the immutable host execution-image approach in the sibling
`simpler-fdwic-swimlane-deps/tests/atomic_probe/pa_scheduler/cross_core_only_scheduler/README.md`,
reviewed against its `tests/atomic_probe/ATOMIC_USAGE_GUIDE.md`. This experiment
prepared dependency state, rather than repeating the older rejected
per-record membership-bitmap precomputation.

## Evidence and decision

The workload was the unchanged DeepSeek V4.1 Flash SWA program: 72 tokens,
12 requests, seed 17, frozen inputs/goldens. Both runs used device 2, five
warmups, 100 measured rounds and no swimlane capture for timing.

| Version | Device median (us) | Device mean (us) |
| ------- | -----------------: | ---------------: |
| Latest preceding `protocol-const` repeat | 944.599 | 952.84995 |
| Host initial Ready/wake preparation | 964.4135 | 965.41794 |
| Restored preceding candidate, exact source/runtime hashes | 1011.1055 | 1014.41135 |

The candidate was 2.10% slower than the earlier baseline, but 4.62% faster than
the subsequent restored baseline. Source and all three native runtime artifact
hashes for the two baseline runs matched exactly. The original screening
rejection is therefore not a reliable estimate of the candidate's effect.
The cause of the between-batch drift has not been established; complete linked
callables and allocation placement were not frozen as a pair. Source remains
restored while a controlled alternating comparison is prepared. Initial Ready
placement and generated code were also not separately controlled.

The candidate's host-wall median/mean were 33071.770/33057.7227 us; the restored
baseline measured 32473.915/32610.5201 us. These include
the surrounding per-launch harness work, not just this helper. Earlier records
have only a rounded host mean in their logs, so these figures do not establish
the host-preparation cost or a host-time benefit.

Six C++ groups passed. Dedicated tests covered completed producers, multi-fanin
migration, rejection without partial image writes, repeated preparation,
dynamic-Graph rejection, unclaimed Mix/sync-start roots and the real concurrent
bootstrap barriers. The first concurrent test omitted the fixture's
`scheduler_count`; the fixture was corrected without changing runtime gates.

All five original SWA numerical comparisons passed. A separate L4 capture
passed the existing native audit: 96 workers, 549 executions, 24 logical tasks,
1344 member obligations and 8674 events, with no dropped records. All 96
bootstrap records reported zero device-scanned tasks, versus 24 tasks in the
original capture. The intended work was removed; that alone did not improve
full device wall time.

## Reconsideration and local artifacts

The follow-up froze complete chip-callable blobs, host/AICPU/AICore libraries,
shared dispatcher and PTO-ISA metadata from this checkout. Both variants had
identical generated program sources, native Python interface and exact business
callable bytes. Replay retained PTO-ISA validation, checked all artifact hashes
and asserted that workers loaded the archived runtime paths. No compilation
ran during the comparison.

One device-2 queue allocation ran `A B B A B A A B`, with five warmups and 100
samples per batch. A is the preceding candidate; B is host preparation.
Every batch passed all five numerical comparisons and resident-mode checks.

| Pair | A median (us) | B median (us) | B improvement |
| ---- | ------------: | ------------: | ------------: |
| A1/B2 | 1005.214 | 1021.508 | -1.621% |
| B3/A4 | 958.2605 | 962.573 | -0.450% |
| B5/A6 | 994.8395 | 982.4555 | +1.245% |
| A7/B8 | 969.2315 | 941.6865 | +2.842% |

B won two of four pairs; median paired improvement was 0.397%. This does not
justify retaining the additional preparation path. The candidate was reverted,
and exact preceding source and its authenticated same-checkout runtime archive
were restored. The cause of remaining batch variation is unproven. Individual
samples were not pooled into an independence or confidence claim, and queue
allocation is not a claim of whole-system exclusivity.

Reopen only with a different mechanism or stronger controlled evidence. Fewer
bootstrap operations did not establish a stable full-run latency benefit here.

Local artifacts are under `../logs/swa-hbg-opt-20261101/`: `host-bootstrap.json`,
`host-bootstrap-only.patch`, `host-bootstrap-reference-review.json`,
`host-bootstrap-tests.log`, `build-host-bootstrap.log`,
`host-bootstrap-device.log`, and `host-bootstrap-profile/audit.json`. The
measurement includes all device/host samples and source/native binary hashes.
The frozen comparison is in `frozen-host-bootstrap-abba/summary.json`; its eight
measurement records retain every sample and both bundle/source identities.
