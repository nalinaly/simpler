# A5 SWA: publisher-local admission adoption

Date: 2026-11-02. Reverted: only two of four frozen pairs improved, with a
median paired improvement of 0.162%. The retained checkpoint is `idle-nop1024`.
The goal of overtaking TMR remains open.

## Candidate and memory boundary

Ordered publication could immediately adopt the record it had just constructed
or fully revalidated, provided its publisher was a member, had no current
obligation, and its scan cursor exactly matched the new admission order.
An additional fresh error/epoch gate preceded local adoption. The record's
existing publication, ADMITTED publication, commit-tail barrier and lock release
were unchanged. Other workers and nonmatching frontiers used the original scan.

The record and bitmap are publisher-owned before admission and immutable after
publication until quiescent retirement. The optimization reused that same-call
proof for the publisher's own observation. It did not replace a cross-core
handoff, reserve a block, suppress competitors or cache an execution permission.
Low-level commit interfaces retained their separate-observation default; only
ordered publication opted in through an internal template parameter.

This was reviewed against the sibling `simpler-fdwic-swimlane-deps` checkout's
`tests/atomic_probe/ATOMIC_USAGE_GUIDE.md` and `perf_opt_record.md` sections 15.62
and 15.65. Those experiments also show that reduced local work need not improve
device wall time. Unlike the earlier cross-type direct-admission trial, this
candidate changed post-publication self-observation rather than Ready routing.

## Frozen device-wall comparison

A is `idle-nop1024`; B is publisher adoption. The unchanged DeepSeek V4.1 Flash
SWA workload used 72 tokens, 12 requests, seed 17 and frozen inputs/goldens.
Native Python interface, generated sources and complete business callable bytes
matched. Among native runtime files, only the AICore image differed. No
compilation ran during measurement.

Device 2 ran `ABBABAAB` in one queue allocation, with five warmups and 100
uninstrumented rounds per batch. All eight numerical and resident-mode checks
passed, and every sample is retained.

| Pair | A median (us) | B median (us) | B improvement |
| ---- | ------------: | ------------: | ------------: |
| A1/B2 | 888.7640 | 878.4390 | 1.162% |
| B3/A4 | 860.1730 | 844.0770 | 1.871% |
| B5/A6 | 890.3355 | 897.7905 | -0.837% |
| A7/B8 | 854.4905 | 888.5635 | -3.988% |

Two pairs favored each arm. Median paired improvement was 0.162%, so the
additional path was not retained. The lowest B batch is not evidence of a
stable gain. Batch medians are the comparison units; individual rounds are not
pooled into an independence or confidence claim.

## Correctness and diagnostic limits

Eight focused C++ targets passed, including admission, ordered execution,
dispatch, Mix, bootstrap and nonce/exit behavior. New checks covered every lane
mask, sync-start, fresh/retried commits, 6/69 workers, reversed permutations,
exactly-once member observation without block preclaim, earlier frontiers,
nonmember publishers and post-publication error/epoch rejection.

The existing L4 audit passed: 96 workers, 549 executions, 24 logical tasks,
1344 member obligations and 8674 events. Both AIC and AIV idle loops remained
free of load/store instructions. Independent L4 captures gave median
publisher admission-to-observation gaps of 5.0385 us for A and 4.4605 us for B.
Publisher placement can differ between captures; this is diagnostic context,
not an isolated helper cost or an attribution of L0 latency.

The candidate and its strategy-specific tests were withdrawn. Exact preceding
source and the authenticated complete native image from this same checkout
were restored, retaining native-interface and normal PTO-ISA validation.
No full scene sweep was run for the rejected candidate.

Local artifacts are in the development workspace's `logs/swa-hbg-opt-20261101/`:
`publisher-adoption-only.patch`, `publisher-adoption-reference-review.json`,
`publisher-adoption-tests.log`, `publisher-adoption-profile/audit.json`,
`publisher-adoption-observation-diagnostic.json`,
`frozen-publisher-adoption-abba/summary.json`, its eight complete batch records,
and `restored-after-publisher-adoption.json`. Reopen only with evidence that
publisher-only observation lies on the remaining critical delay, or with a
different mechanism that removes work for the wider member population.
