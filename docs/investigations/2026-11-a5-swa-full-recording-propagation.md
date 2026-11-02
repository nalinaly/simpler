# A5 SWA full propagation of disabled protocol recording

**Date**: 2026-11-02
**Verdict**: reject the full propagation extension after screening

## Question and scope

The resident no-profiling entry already selects `WithProfiling=false`, but
some nested helpers retain runtime checks and cold diagnostic bodies. The
first candidate propagated the existing recording template through admission
publication, member observation, execution permission, bootstrap and drain.
It measured 935.146 us, versus the preceding candidate's 948.032 us. That
small, sequential screening difference remains provisional.

Native inline stacks identified remaining diagnostic bodies in inline-task
completion and Graph child-completion contribution. A second candidate also
propagated the template through these paths, completion resolution and payload
publication. Default helper calls retained recording, including L0 sampled
execution. Business atomics, DCCI, ownership, error gates, dependency release,
kernel sources and ABI layouts were unchanged.

## Result

Device 2, frozen DeepSeek V4.1 Flash SWA inputs, 72 tokens, 12 requests, seed
17, five warmups and 100 measured rounds; no swimlane capture in the benchmark.
Both candidates passed all five original numerical comparisons and used
resident AICore scheduler mode 1.

| Candidate | Device median (us) | Mean (us) |
| --------- | -----------------: | --------: |
| First-stage propagation | 935.146 | 938.34826 |
| Full propagation extension | 1019.160 | 1021.15899 |
| Restored first-stage propagation | 944.599 | 952.84995 |

The extension was 8.98% slower in this screening comparison. These sequential
measurements do not prove a universal or repeatable regression, but they
provide no performance basis for retaining the extension.
After reverting the extension, both source and native binary hashes exactly
matched the first-stage measurement. Its repeated run passed all five
numerical comparisons and measured 944.599 us. This supports the screening
decision, while the first-stage gain over its older 948.032 us predecessor
still needs an alternating comparison.

Actual no-profiling loop instruction counts changed as follows. These are
static counts, not dynamic events, I-cache misses or recoverable time.

| Core | First stage | Full extension |
| ---- | ----------: | -------------: |
| AIC instructions | 50426 | 40888 |
| AIV instructions | 51726 | 41989 |
| AIC NOP instructions | 4803 | 3 |
| AIV NOP instructions | 4802 | 3 |

The remaining bounded idle delay still has no load/store in either core's
native loop. The removed large NOP groups belonged to disabled diagnostic
bodies; their static removal does not imply that the previous timed run
executed them.

Seven targeted C++ test groups passed, including ordinary/ordered inline
completion that still wakes and dispatches dependent work with recording
explicitly disabled, and admission/entry error-gate checks. This is not a full
runtime regression claim. Initial test-only compilation errors were corrected
and their logs retained; they are not evidence of pre-existing runtime defects.

A separate L4 capture also passed all five numerical checks and the existing
native protocol auditor: 96 workers, 549 executions, 24 logical tasks, 1344
member obligations and 8674 events. This validates the captured recording path,
not the benchmark's timing. The extension and its dedicated inline-service
test were then reverted; the first-stage candidate remains provisional.

## Evidence and reconsideration

The local directory `../logs/swa-hbg-opt-20261101/` contains
`protocol-const.json`, `protocol-const-full.json`, the associated source patches
and binary hashes, `protocol-const-full-codegen-comparison.json`,
`protocol-const-remaining-nop-sites.txt`, and build/test/device logs.

The reference checkout's `tests/atomic_probe/perf_opt_record.md` section 9.3
distinguishes runtime-disabled recording from compile-time removal, while
sections 15.64–15.67 require device evidence beyond code size. The main
checkout's `2026-10-csa-idle-loop-codegen.md` likewise records only small gains
from earlier propagation work. Do not reopen this extension merely because
it removes cold instructions; require a new mechanism or controlled evidence
that addresses the observed device-wall regression.
