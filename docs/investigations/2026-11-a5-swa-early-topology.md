# A5 SWA early admission-topology validation

**Date**: 2026-11-02
**Verdict**: dropped after screening; no demonstrated end-to-end improvement

## Question

The admission path validates the immutable worker topology once per worker, on
first use. Could doing the same work immediately after the all-scheduler scan
barrier reduce the first executable wave's startup skew?

## Candidate and correctness scope

The candidate extracted the existing validator and invoked it after
`bootstrap_scan_complete`, before initial slot fill. It populated only the
per-run private membership cache; admission, claim, payload and cursor updates
were unchanged. Validation on later use remained in place.

The review followed `tests/atomic_probe/ATOMIC_USAGE_GUIDE.md` in the sibling
`simpler-fdwic-swimlane-deps` checkout at
`3699b7ae4ff47abcfa1e2916881d4309cf39c00f`. Compiler-derived field offsets show
that topology reads touch worker-context cache lines at offsets 0, 64, 256 and
320. Those lines are read-only after READY publication. The separate line at
192 contains both immutable configuration and final statistics; topology
validation does not touch it. The new helper added no shared writes.

The admission, ordered-execution and scheduler-bootstrap C++ test targets
passed, including new checks for no publication before admission and rejection
of malformed topology without a populated cache. Native compilation and all
five SWA numerical comparisons passed. These checks do not establish the
correctness of every pre-existing memory-protocol path.

## Result

Both screening runs used device 2, the same frozen DeepSeek V4.1 Flash SWA
inputs and kernels, 72 tokens, 12 requests, seed 17, five warmups and 100
measured rounds. Swimlane collection was disabled; the metric is device wall
time. HBG remained in resident AICore scheduler mode 1.

| Version | Median (us) | Mean (us) |
| ------- | ----------: | --------: |
| Prior candidate, `local-noalias` | 948.0320 | 952.00168 |
| Added early topology preparation | 963.2445 | 966.62902 |

The candidate was 1.60% slower in this screening comparison. These are
sequential runs, not alternating pairs, so the result does not establish a
repeatable regression. It provides no evidence for keeping the extra startup
work. The candidate and its helper-specific tests were reverted.

## Reference lesson and reconsideration

The reference checkout's `tests/atomic_probe/perf_opt_record.md` sections 15.62
and 15.65 likewise distinguish reduced immutable-metadata work from unproven
end-to-end gains. Earlier validation alone does not prove a shorter critical
path. Reconsider with evidence that first-use validation is on the
uninstrumented critical path and an alternating comparison showing a gain.

Local evidence is under `../logs/swa-hbg-opt-20261101/` relative to this
checkout: `local-noalias.json`, `early-topology.json`,
`early-topology-only.patch`, `early-topology-tests.log`,
`build-early-topology.log`, `worker-context-layout.json` and
`reference-review-20261102.json`. Measurement records include raw samples,
source diff hashes and native binary hashes.
