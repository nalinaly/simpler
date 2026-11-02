# A5 SWA admission-record bypass reads

Date: 2026-11-01. The candidate was reverted after a slower screening run.

## Question and candidate

The DeepSeek V4.1 Flash SWA decode baseline spends substantial time between a
dependency completing and the successor's first block entering its kernel.
Published admission records are immutable until run retirement. The candidate
read their 64-byte headers into local snapshots with bypass loads and read the
membership bitmap through the same bypass primitive, instead of invalidating
the header and bitmap cache lines before checking them.

Epoch, order, task metadata, member-set, admission-state and error checks stayed
enabled. The candidate did not change kernels, tensor shapes, block counts,
worker eligibility, completion semantics or the admission publication sequence.

## Measurement

Both runs used A5 device 2, the same precompiled SWA program, and the same saved
inputs and golden outputs: 72 tokens across 12 requests, seed 17, mixed case.
Each run passed all five numerical checks before five warmups and 100 measured
dispatches. Chip swimlane collection was disabled. The metric is full device
wall time, excluding host preparation and validation.

| Version | Device median (us) | Min (us) | Max (us) |
| ------- | -----------------: | -------: | -------: |
| Register-counter NOP delay, maximum 256 iterations | 1037.2645 | 996.506 | 1090.258 |
| Same delay plus admission bypass snapshot | 1096.7535 | 1055.126 | 1143.343 |

The candidate was about 5.7% slower in this screening pair. This is sufficient
to reject it for the current optimization effort, but it is not an alternating
A/B/A/B experiment or a proof of a universal regression. The extra bypass reads
and snapshot construction were not separately timed, so their individual costs
are unknown.

The two C++ targets `test_a5_hbg_admission` and
`test_a5_hbg_ordered_execution`, native build and SWA numerical checks passed.
No full runtime regression claim is made for this rejected candidate.

## Evidence and reconsideration

The local experiment directory is `../logs/swa-hbg-opt-20261101/` relative to
the checkout. `nop256.json` and `admission-bypass.json` contain all samples,
source-diff hashes and runtime binary hashes. `admission-bypass-only.patch`
preserves the isolated candidate; build and test logs use the same prefix.

Reconsider only with a narrower mechanism that avoids the snapshot cost, or
new controlled measurements showing a repeatable improvement. Replacing cache
maintenance with bypass reads alone did not establish a speedup here.
