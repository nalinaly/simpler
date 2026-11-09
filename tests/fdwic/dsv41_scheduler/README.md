# DSV4.1 standalone Scalar scheduler experiments

A C++ Host runner and CCEC AIC/AIV executable for fast A5 scheduling experiments.
Build and execution require **neither PyPTO-lib, PyPTO JIT, Python bindings, nor a
Simpler runtime build**. Python is optional, standard-library-only tooling for
capture extraction and report conversion. CPU execution checks correctness;
its timings are never A5 performance evidence.

The included scheduler is a small **experimental reference scanner**, not the
production `defer_wait` resident scheduler. Use it to test mechanisms quickly,
then port successful changes to production and validate the real network.
The captured task shapes do not make this runner a production-performance proxy.

## Captured cases

The two workload directories contain slices of actual 40-layer HBG dependency
captures. Neither source is synthesized by flattening or grouping the other.
Logical task boundaries, block counts, all three AIC/AIV subtask slots, and
unique internal dependency edges are retained. Kernel IDs with no executable
body remain control nodes and release their dependencies without an engine call.

The source configuration is EP2, 72 valid tokens/rank (80 padded rows), 64 experts
per rank, and zero-weight inputs. The captures are rank 0 from the non-DSL run
on 2026-11-09 and the DSL run on 2026-11-07. Source hashes, extraction boundaries,
and table hashes are in each flavor's `source.json`. Raw captures are not copied
into the repository. Layer numbers below are zero-based.

| Case | Source slice | lib tasks / executions | DSL tasks / executions |
| ---- | ------------ | ---------------------- | ---------------------- |
| `swa` | Layer 0 attention | 25 / 631 | 19 / 339 |
| `c2a_full` | Layer 2 attention | 43 / 908 | 30 / 666 |
| `c2a_reuse` | Layer 3 attention | 24 / 583 | 17 / 391 |
| `c1a_full` | Layer 20 attention | 38 / 818 | 25 / 550 |
| `c1a_reindex` | Layer 24 attention | 34 / 801 | 23 / 625 |
| `c1a_reuse` | Layer 21 attention | 24 / 583 | 17 / 391 |
| `moe` | Layer 0 MoE | 479 / 3,069 | 32 / 913 |
| `engram` | Layer 1 Engram | 12 / 960 | 12 / 960 |
| `layers_19_21` | Three consecutive layers, attention and MoE | 1,527 / 11,218 | 159 / 4,168 |

An execution is one `(task, block, subtask)` invocation. A SPMD task completes
only after every required block and subtask returns and its deferred notification
is satisfied. Dependencies outside a slice are treated as prepared inputs;
their count is explicitly recorded. Duplicate tensor edges between the same two
tasks become one logical ordering edge. This changes edge-processing workload
relative to a runtime that stores duplicates, without changing reachability.

## Build and run

From this directory:

```bash
# C++17 compiler, pthread, and standard library only.
bash build.sh cpu
bash run.sh cpu --case all --flavor both --clusters 4 --rounds 2 --output outputs/cpu
bash run.sh cpu --self-test

# Source this machine's CANN environment first.
bash build.sh a5
bash run.sh a5 --case swa --flavor both --clusters 8 --rounds 3 --output outputs/a5-swa
bash run.sh a5 --case all --flavor both --clusters 32 --output outputs/a5-all

# Optional report and Perfetto conversion; no third-party Python packages.
python3 analyze.py outputs/a5-all --trace outputs/a5-all/perfetto.json \
  --split-traces outputs/a5-all/traces --workloads workloads
```

When `task-submit` is available, `run.sh a5` performs the repository's A5
architecture check and obtains one device through the queue. A job already
holding `TASK_DEVICE` uses that reservation. Keep CANN environment variables
available in the queued job. A standalone binary can also be invoked inside an
existing reservation with `--binary build/a5/kernel.o --device "$TASK_DEVICE"`.

`--profiling 0` uses a separate compile-time specialization with no per-execution
clock reads or sample writes. It retains the worker window, correctness counters,
and clocks required by the watchdog/deferred-notification protocol. `--profiling 1`
records kernel intervals and identities. `--profiling 3` additionally records
individual CHECK/control-service and ready-scan intervals, prepare, and finish.
The default is 3 for compatibility with initial experiments. `--warmup 3` validates
three discarded runs before the requested measured rounds. L0/L1 Scalar percentiles
are null rather than invented zero-cost measurements.

The L3 per-worker event buffer is bounded; overflow fails validation. A reference
scheduling window runs from the earliest worker entry to the latest loop exit;
it excludes Host launch/copies and the final trace flush. It includes replacement
kernels and modeled waits, so it is not pure Scalar overhead or production
whole-Device latency.

`--clusters` controls 1..32 AIC + 2 AIV triplets. `--repeats` changes the engine
work per invocation without recompilation. `--empty` omits arithmetic for a
dispatch lower-bound experiment. `--defer-us` controls the modeled notification
delay (default 20 us). All nine cases use the same device ELF. Changing cases,
rounds, core count, compute repetitions, or notification delay needs no rebuild.

An explicit build recompiles only this small runner and its two CCEC objects,
in parallel, with normal O3 for device code. It does not modify production
compiler flags or install over the shared environment. `build-time.txt` records
elapsed wall seconds and `artifacts.sha256` identifies the resulting binaries.

## Measurement and correctness

The runner records:

- Ready search, including failed claims and every complete unsuccessful scan.
- Parameter/dependency preparation and post-kernel completion processing.
- Deferred/control service, accumulated between useful executions, plus the
  full per-worker service totals and largest single service interval.
- Kernel start/end, physical core, logical worker, task, block, and subtask.
- Separate diagnostic-write time. Timer overhead is not subtracted.

Per-execution Scalar cost is `search + prepare + finish + service`; the kernel
body is excluded. Per-worker counters also retain empty scans and final drain
work that cannot be attributed to a subsequent execution. Core-time totals are
not Device wall time and cannot be subtracted from it. Samples are instrumented;
they do not establish an uninstrumented production 5 us bound.

A5 uses SYS_CNT at 1 GHz. Atomic-dependent boundaries consume the result before
reading the timestamp. The report contains median, P99, maximum, and all >5 us
counts. No outliers are discarded. Full search passes are not split into smaller
samples to pass the budget. The trace draws only measured contiguous intervals;
accumulated search/service costs stay in arguments rather than fabricated spans.

Validation checks every execution identity, role, block count, dependency start
ordering, group completion, notification deadline, and dependency checksum.
The small real engine bodies perform 16x16 FP32 matmul or vector addition and
their final workspace outputs are checked. `--self-test` verifies that corrupted
payload and premature asynchronous completion are rejected. A watchdog converts
a stalled experiment into failure; a failed device synchronize never grants the
runner permission to reclaim potentially in-use GM.

## Deliberate modeling boundaries

- Arithmetic is replaced with the small Cube/Vector bodies. Tensor sizes,
  memory traffic, real weights, and full-network numerical accuracy are not
  represented by these results.
- MoE has its captured task graph and asynchronous completion joins, but peer
  notification is modeled by a device deadline. There is no EP2 payload transfer,
  SDMA submission, or actual production `defer_wait` call in this runner.
- MIX subtask counts and logical joins are preserved. Physical triplet affinity,
  cooperative start barriers, and production admission policy are not modeled.
  These require the later real-kernel/production integration tier.
- The reference scanner intentionally exposes repeated scans and contention; it
  does not promise a 5 us result. Performance conclusions apply to this experiment.
- Captures are from two source revisions. Comparing the flavors describes their
  frozen workloads, not a controlled causal comparison of the two compilers.

## Refresh the tables

```bash
python3 export_workloads.py /path/to/rank0/d0 --flavor pypto_lib
python3 export_workloads.py /path/to/dsl/rank0/d0 --flavor pypto_lib_dsl
```

The exporter accepts `deps.json` or `deps.json.gz` plus `name_map.json`, checks
all 40 layer boundaries, and emits plain task/edge tables. It never imports a
model or compiler. Keep external boundary assumptions visible when changing a
slice. `common/workloads.h` only loads these tables; it does not infer a DSL
workload from the non-DSL version.

## Files

- `host.cpp`: direct C++ runner, CPU backend, A5 launch, and validation entry.
- `common/`: captured table loader, reference scheduling loop, wire records,
  correctness checks, and result export.
- `kernels/`: the AIC/AIV implementation and mixed ELF exports.
- `workloads/`: the 18 frozen graph slices and their source manifests.
- `build.sh`, `run.sh`: independent build and execution.
- `export_workloads.py`, `analyze.py`: optional offline tools.

Generated binaries, raw samples, traces, and logs stay in ignored `build/` and
`outputs/` directories.

## Initial validation (2026-11-09)

- CPU: all 18 configurations, two rounds at four clusters (36 runs); MoE and
  three-layer slices at 32 clusters (four runs); empty SWA at one cluster
  (two runs). All 42 runs and both corruption negative controls passed.
- A5: all 18 configurations, two rounds at eight clusters (36 runs); MoE and
  three-layer slices at 32 clusters (four runs); empty SWA at one cluster
  (two runs). All 42 runs passed on device 1 through reserved queue jobs.
- Final standalone build: CPU approximately 3 s; A5 approximately 4 s, including
  both O3 device objects, link, and Host runner. This is not the build time of
  the production runtime. Core count and case changes reused the same binary.
- Applicable formatting/lint hooks passed. No full repository test suite was run.
- The reference scanner has samples exceeding 5 us; this delivery establishes
  reproducible experiments, not achievement of the scheduling target.

Local evidence: `build/validation-summary.json`, `build/pre-commit.log`,
`build/a5/queue.log`, `build/a5/boundary-queue.log`, and `outputs/{cpu-final,
cpu-full-cores,cpu-empty,a5-captured,a5-boundaries}/`. The `a5-captured` directory
also contains an analyzed summary and a Perfetto trace.

## L0/L1/L3 baseline (2026-11-09)

The profiling specializations were subsequently validated on A5 device 3 with
32 clusters (32 AIC + 64 AIV), O3, one compute repeat, and a 20 us modeled
notification delay. All 306 runs passed: for each of the 18 configurations,
L0 used three warmups and ten measured rounds; L1 and L3 each used one warmup
and one capture. All 36 trace exports passed execution-count and dependency-flow
checks, with no event-buffer overflow. The full repository suite was not run.
The standalone A5 build took approximately 5 s with all three specializations.

L0 scheduling-window medians include replacement arithmetic and modeled waits;
they exclude Host launch/copies and final trace flush. The two flavors have
different task/execution counts, so these are not compiler speedup ratios.

| Case | PyPTO-lib (ms) | DSL (ms) |
| ---- | -------------- | -------- |
| SWA | 1.983221 | 0.713865 |
| C2A Full | 2.504815 | 1.825316 |
| C2A Reuse | 1.844366 | 0.984820 |
| C1A Full | 2.461688 | 1.940962 |
| C1A Reindex | 2.670212 | 1.885854 |
| C1A Reuse | 1.825725 | 0.989061 |
| MoE | 18.447890 | 3.005705 |
| Engram | 4.541466 | 4.499232 |
| Layers 19-21 | 234.262516 | 8.970382 |

Raw evidence is retained locally in `outputs/aicore-baseline-20261109-190613/`:
per-round summaries and CSVs in `L0/`, `L1/`, `L3/`; frozen binaries, workload
tables, source hashes, and the queue command beside them. `delivery/` contains
the baseline table, trace audit, and 36 Perfetto JSON files. The generated ZIP
and raw outputs are intentionally excluded from source control.
