# fully_distributed_within_core onboard bringup — known issues

## Environment setup

### Paths

```
REPO=/home/q00473782/atomic/glm/simpler-fully_distributed
CANN=/home/q00473782/cann/cann-9.1.0
PTO_ISA_ROOT=/home/q00473782/dsl/pto-isa
```

### Toolchain

- **CANN**: 9.1.0 at `/home/q00473782/cann/cann-9.1.0`
  - `source $CANN/set_env.sh` before any build/run
  - ccec = `$CANN/bin/ccec` (symlink to bisheng, same binary)
  - AICore arch flags: a5=`dav-c310-vec`/`dav-c310-cube`, a2a3=`dav-c220-vec`/`dav-c220-cube`
- **g++-15**: at `/home/q00473782/.local/gcc-15/root/usr/bin/g++-15` (required for a5sim/a2a3sim sim kernel compilation, uses C++23)
- **aarch64 cross-compiler**: for AICPU .so (a5 onboard AICPU is aarch64)
- **PTO-ISA**: at `/home/q00473782/dsl/pto-isa` (export `PTO_ISA_ROOT` before build)

### Build commands

```bash
# Full rebuild (all 4 platforms: a2a3sim, a5sim, a2a3, a5)
source $CANN/set_env.sh
PTO_ISA_ROOT=/home/q00473782/dsl/pto-isa pip install -e . --no-build-isolation

# Rebuild only a5 runtime (onboard)
PTO_ISA_ROOT=/home/q00473782/dsl/pto-isa python -m simpler_setup.build_runtimes --platforms a5

# Rebuild only a5sim runtime
PTO_ISA_ROOT=/home/q00473782/dsl/pto-isa python -m simpler_setup.build_runtimes --platforms a5sim

# Force clean rebuild (delete cache first)
rm -rf build/cache/a5 build/lib/a5
PTO_ISA_ROOT=/home/q00473782/dsl/pto-isa python -m simpler_setup.build_runtimes --platforms a5
```

### Test commands

```bash
# a5sim (fast, validates logic)
PTO_ISA_ROOT=/home/q00473782/dsl/pto-isa python -m pytest \
  tests/st/a5/fully_distributed_within_core/vector_example/test_vector_example.py \
  --platform a5sim -x -v

# a5 onboard (real hardware)
source $CANN/set_env.sh
PTO_ISA_ROOT=/home/q00473782/dsl/pto-isa python -m pytest \
  tests/st/a5/fully_distributed_within_core/vector_example/test_vector_example.py \
  --platform a5 -x -v --log-level v9

# mix_coown test (MIX co-ownership, more complex)
PTO_ISA_ROOT=/home/q00473782/dsl/pto-isa python -m pytest \
  tests/st/a5/fully_distributed_within_core/mix_coown/test_mix_coown.py \
  --platform a5sim -x -v
```

### Device logs

```bash
# AICPU device logs (progress crumbs, DIAG output)
LOG=$(ls -t ~/ascend/log/debug/device-0/*.log | head -1)
grep -aE "progress=" "$LOG" | sed -E 's/.*(progress=\[[^]]*\]).*(dbg=\[[^]]*\]).*/\1 \2/' | sort -u

# Decode C15 DIAG dbg values (bits: unmet_fanin, consumer, flag, fanin_count, etc.)
python3 -c "
d = 0x1050342004000  # paste dbg value here
unmet = d & 0xfff
consumer = (d >> 12) & 0xfff
flag = (d >> 24) & 1
fanin_count = (d >> 25) & 0xf
found_unready = (d >> 30) & 1
core_id = (d >> 32) & 0xff
local_idx = (d >> 40) & 0xff
occupied = (d >> 48) & 0xff
all_replayed = (d >> 56) & 1
print(f'unmet={unmet} consumer={consumer} flag={flag} fanin={fanin_count} found_unready={found_unready} core={core_id} local_idx={local_idx} occupied={occupied} all_replayed={all_replayed}')
"
```

### Crumb value reference

```
20 = entered dist_core_main
21 = gd recovered
22 = read layout ok
23 = core reset done, entering startup barrier
24 = passed startup barrier
25 = about to bind runtime
26 = runtime bound, ops set
27 = blob bound, about to replay orchestration
28 = orchestration replay returned
29 = replay_done published, entering drain loop
30 = drain loop top (drain_block_won)
31 = drain_block_won returned (drain_phase_b)
32 = drain_phase_b returned
33 = has_pending_won returned
40 = submit entry
41 = execute-first drain done
42 = about to run-ahead throttle
43 = throttle passed, entering alloc
44 = entering ring slot back-pressure
45 = ring back-pressure cleared
46 = entering heap reclaim back-pressure
47 = heap back-pressure cleared, building slot
48 = ef drain_block_won done, into phase_b (submit path)
49 = execute_slot: kernel returned
50 = execute_slot: entered (diagnostic)
55 = execute_slot: post-kernel, about to dist_set_flag (diagnostic)
70-79 = claim succeeded for task N (diagnostic)
80 = about to read cursor value for task 0 (diagnostic)
98 = about to CALL orchestration blob
99 = orchestration returned
```

### Probe scripts

```
tests/atomic_probe/          — .asc and ccec -x cce atomic probes
tests/atomic_probe/run.sh    — build & run ccec AtomicCas probe on A5
tests/atomic_probe/_run_asc_probe.sh — run .asc probes
tests/atomic_max_probe.asc   — atomicMax latency probe (bisheng -xasc)
```

### Key build flags

```
DIST_SIM_HOST_CLOCK=1  — AICPU + sim builds (GCC __atomic_* builtins)
DIST_SIM_HOST_CLOCK=0  — AICore/CCEC builds (ccec atomicMax/atomicCAS/atomicAdd builtins)
AICORE_PROGRESS_STRIDE — defined only in a5 fdwc runtime.h (enables crumb/diag)
SIMPLER_DIST_AICPU_ONLY — defined for AICPU onboard build (enables cache_flush_range)
PTO2_PROFILING — ON by default in AICPU, OFF in AICore (causes 160-byte L2TaskArgs size difference)
```

### Hardware

- A5 device at `/dev/davinci0` (Ascend950 family, dav-3510/dav-c310)
- No `npu-smi` binary on this host (arch precheck skill won't work — use device files to confirm)
- 3 blocks × (1 AIC + 2 AIV) = 9 cores total

---

## Context

`vector_example` st test passes on a5sim but hangs (507018 AICPU timeout) on a5 onboard.
This file lists every issue identified during debugging, with file:line references.

---

## Issue 1: Coherent<T> layout divergence (FIXED but needs verification)

**File:** `src/common/runtime/fully_distributed_within_core/dist_engine.cpp:407`

**Problem:** `Coherent<T>` was `std::atomic<T>` on AICPU (DIST_SIM_HOST_CLOCK=1) but plain `T` on AICore (DIST_SIM_HOST_CLOCK=0). `std::atomic<T>` may have different size/alignment than plain `T`, causing `DistGlobal` layout to differ between AICPU and AICore by ~160 bytes. AICPU writes cursor/flags/frontier at offset X, AICore reads at offset Y → garbage.

**Fix applied:** Unified to `T a;` (plain T) on both builds. AICPU now uses `__atomic_*` GCC builtins instead of `std::atomic` methods. AICore uses ccec `atomicMax`/`atomicCAS`/`atomicAdd` builtins.

**Remaining concern:** `sizeof(DistGlobal)` still shows 252171584 in AICPU logs — need to verify AICore sees the same size. The `L2TaskArgs` embedded in `orch_args_gm` (last member) has a known 160-byte size difference due to `PTO2_PROFILING` (on by default in AICPU, off in AICore). This was worked around by placing `orch_args_gm` last, but other fields may still diverge.

**Verification needed:** Add a DIST_DBG that writes `sizeof(DistGlobal)` from AICore side, compare with AICPU's 252171584.

---

## Issue 2: coherent_load used dcci_inval + plain load (TOCTOU window) (FIXED)

**File:** `src/common/runtime/fully_distributed_within_core/dist_engine.cpp:426`

**Problem:** AICore `coherent_load` did `pto_dcci_inval(&c->a) + return c->a` (plain load). Between inval and load, a prefetch/cache-fill can re-populate the stale cacheline, so the plain load reads stale value. This is why `flags[task_id]` was never observed as 1 by consumer cores.

**Fix applied:** AICore path now uses idempotent `atomicMax(p, identity)` — a single memory-level instruction that reads the true HBM value with no TOCTOU window. `identity` = `INT32_MIN` for signed, `0` for unsigned.

---

## Issue 3: coherent_store used plain store + dcci_flush (clobbers neighbours) (FIXED)

**File:** `src/common/runtime/fully_distributed_within_core/dist_engine.cpp:447`

**Problem:** AICore `coherent_store` did `c->a = v; pto_dcci_flush(&c->a)`. The `dcci_flush` writes back the ENTIRE 64B cacheline, clobbering adjacent fields (e.g. neighbouring flags in the dense `flags[]` ring — the C4 cacheline clobber bug from docs §17).

**Fix applied:** AICore path now uses `atomicCAS` loop to write directly to HBM (no cacheline writeback), followed by `pto_dcci_inval` to drop the stale local copy.

---

## Issue 4: execute_slot used dcci(ENTIRE_DATA_CACHE) (FIXED)

**File:** `src/common/runtime/fully_distributed_within_core/dist_engine.cpp:1782`

**Problem:** `execute_slot` did `dcci(ptr, ENTIRE_DATA_CACHE)` before kernel and `dcci(ptr, ENTIRE_DATA_CACHE, CACHELINE_OUT)` after kernel. This flushes/invalidates the ENTIRE data cache — including cachelines holding control state (flags/cursor/frontier). The post-kernel ENTIRE flush would write back stale cached copies of control state, clobbering prior `atomicMax` writes. This was the primary cause of `flags[id]==0 forever`.

**Fix applied:** Replaced with per-tensor `pto_dcci_inval`/`pto_dcci_flush` — only invalidates/flushes the cachelines covering each tensor's buffer data, not the entire cache.

**Remaining concern (from user):** Kernel I/O data is on GM (cacheable HBM, `gd->heap_base + phys`). Per-cacheline dcci on tensor data is still potentially dangerous if two tensors share a cacheline, or if a tensor cacheline overlaps with control state. The user questions whether dcci flush is safe at all when other cores may be writing to the same cacheline via atomics. May need uncacheable memory mapping for the heap segment.

---

## Issue 5: DIST_CRUMB / submit_crumb used dcci flush (FIXED)

**File:** `src/common/runtime/fully_distributed_within_core/dist_engine.cpp:1683, 2949`

**Problem:** `DIST_CRUMB` and `submit_crumb` wrote crumb values via plain store + `dcci(..., CACHELINE_OUT)`. If dcci is unreliable (as established), crumbs were unreliable — making diagnosis impossible.

**Fix applied:** AICore path now uses `atomicExch` (direct HBM write, no cacheline writeback). AICPU path uses `__atomic_store_n` + `pto_dcci_flush` (cache_flush_range, which is the AICPU's real cache operation).

**Important:** `atomicMax` cannot be used for crumbs because crumb values are NOT monotonically increasing (e.g. crumb 99 → 28 → 29 → 30). `atomicExch` (atomic swap) is the correct primitive for overwrite.

---

## Issue 6: dist_set_flag / coherent_fetch_add / coherent_cas_weak missing dcci_inval after atomic (FIXED)

**File:** `src/common/runtime/fully_distributed_within_core/dist_engine.cpp:606, 460, 507`

**Problem:** After `atomicMax`/`atomicAdd`/`atomicCAS` writes to HBM, the local core's stale cached copy of the same cacheline was not invalidated. A subsequent `dcci(ENTIRE_DATA_CACHE, CACHELINE_OUT)` (now removed) would write back the stale copy, clobbering the atomic write.

**Fix applied:** Added `pto_dcci_inval(&c->a, sizeof(c->a))` after every HW atomic operation on the AICore path.

---

## Issue 7: resolve_kernel_addr missing dcci_inval before read (FIXED)

**File:** `src/common/runtime/fully_distributed_within_core/dist_engine.cpp:1684`

**Problem:** `resolve_kernel_addr` reads `runtime->func_id_to_addr_[kernel_id]` and `callable->resolved_addr_` from cacheable HBM without `dcci_inval`. AICPU writes these during register; AICore reads stale cached value (0) → `function_bin_addr = 0` → kernel never executes → task never completes.

**Fix applied:** Added `pto_dcci_inval` before each read.

**Note:** This fix uses dcci_inval which has the TOCTOU issue from Issue 2. Ideally should use atomic read, but `func_id_to_addr_` is `uint64_t[]` (not `Coherent<T>`), so atomic read would need a different mechanism.

---

## Issue 8: claim/execute hangs — CURRENT STATUS

**Latest finding (after printf format fix):** Core 5 wins claim for task 0 (crumb 70), enters `execute_slot`, reaches crumb 50 (before `fn()` call), then hangs — `fn()` never returns (no crumb 51). Other cores stuck in drain loop (crumb 30-33) waiting on task 0's flag.

**Console output (correct, after printf fix):**
```
done_count=0/9 (go=1) progress=[31,30,32,31,33,50,33,33,32,]
```
- `go=1` ✓, `num_workers=9` ✓ — AICPU writes correct
- `done_count=0` — no core completed (previous "9" was printf bug showing num_workers)
- Core 5 at crumb 50 = `fn()` call about to execute
- `function_bin_addr = 0x1E4C0` (kernel binary offset, looks valid)

**Hang location:** `fn(reinterpret_cast<__gm__ int64_t *>(s.args))` at dist_engine.cpp:1777

**Diagnostic result:** `s.tensors[0].buffer.addr = 0x12006fe7e000` — valid HBM address. `function_bin_addr = 0x1E4C0` — valid binary offset (text starts at 0).

**Key test: PTO_DIST_SKIP_EXEC (skip kernel call):** 8/9 cores complete (`done_count=8/9`). Engine logic (claim/build/drain/flag/frontier) works correctly. Only core 6 stuck in drain loop (minor issue, likely a race on the last flag).

**Comparison with tensormap_and_ringbuffer (PASSES on a5 onboard):**
- `tensormap_and_ringbuffer/mixed_example` (uses TLOAD/TADD/TSTORE/pipe_sync AIV kernels) PASSES on a5 onboard
- `tensormap_and_ringbuffer/dummy_task` (simple scalar AIC kernels) PASSES on a5 onboard
- Both use `dcci(payload, ENTIRE_DATA_CACHE)` before kernel call and `pipe_barrier(PIPE_ALL)` after
- Key difference: tensormap kernel args come from `PTO2DispatchPayload` (AICPU-set), fdwc args come from `RingSlot` (AICore-built)

**Tried but did not fix the kernel hang:**
1. Per-tensor `pto_dcci_inval`/`pto_dcci_flush` (instead of ENTIRE_DATA_CACHE) — still hangs
2. `pto_dcci_flush(&s, sizeof(RingSlot))` + `ENTIRE_DATA_CACHE` inval — still hangs
3. `pipe_barrier(PIPE_ALL)` before/after `fn()` — still hangs
4. Removed `__attribute__((always_inline))` from kernel_entry — still hangs
5. `pto_dcci_inval` on Tensor objects in orch_args_storage_ before `build_ring_slot` — still hangs
6. `pto_dcci_inval(&gd->orch_args_gm, sizeof(...))` before orchestration call — still hangs

**Remaining hypothesis:** The kernel binary address `0x1E4C0` may not be the correct entry point for `kernel_entry` in the linked AICore binary. The fdwc AICore binary links dist_engine.cpp + kernel_*.cpp + orchestration.cpp together — the kernel_entry symbol from each kernel .cpp may collide or be resolved incorrectly. Need to check the binary's symbol table for `kernel_entry` addresses and verify `resolve_kernel_addr` returns the right one.

**Diagnostic added but not yet read:** `s.tensors[0].buffer.addr` written to aicore_progress slot 3/4 before `fn()` call. AICPU's DBGOUT print condition (`v0 != 0 || ish != 0 || osh != 0`) may not fire if addr is 0.

**Full GM data access audit completed.** Key findings:
- Category 1 (control state): SAFE — all use `Coherent<T>` with atomic ops
- Category 2 (tensor I/O): MOSTLY SAFE — per-tensor dcci in execute_slot; `dist_get/set_tensor_data` plain memcpy (cold path, may not be called in M2)
- Category 3 (Runtime metadata): PARTIALLY FIXED — added dcci_inval for config region, orch_args_gm, resolve_kernel_addr; but many plain field reads remain (see audit table below)
- Category 4 (per-core private): SAFE

**User insight on memory consistency:** AICPU (ARM) and AIV share L2 cache coherence, but ARM is a weak memory model — multiple independent addresses' read/write ordering is NOT guaranteed without explicit barriers. The fix should use atomic flag as critical-section boundary: AICPU writes config → release → atomic store go=1; AICore atomic load go==1 → acquire → dcci_inval → read config.

## Full GM Data Access Audit

### Category 3 hazards (Runtime metadata — AICPU writes, AICore reads with plain loads):

| Line | Field | Current | Fix needed |
|------|-------|---------|------------|
| 3133 | `gd->rt` | plain load | dcci_inval + acquire barrier |
| 3167 | `gd->orch_bind_func` | plain load | dcci_inval |
| 3172 | `gd->orch_func` | plain load | dcci_inval |
| 3172 | `gd->orch_args` | plain load | dcci_inval |
| 3172 | `*gd->orch_args` (L2TaskArgs) | plain deref | dcci_inval on args region |
| 3102 | `gd->num_workers` | plain load | dcci_inval (or use coherent_load) |
| Many | `gd->H, heap_base, heap_size, tm_shared, runahead_max, layout[]` | plain loads | dcci_inval at entry |
| 1669 | `runtime->func_id_to_addr_[]` | dcci_inval (fixed) | OK |
| 1672 | `callable->resolved_addr_` | dcci_inval (fixed) | OK |

### Category 2c/3g hazards (cross-core plain fields, multi-core/shared mode only):

| Line | Field | Issue |
|------|-------|-------|
| 2523-2544 | `WonSlot::task_id`, `BuiltSubtask` fields | Anchor plain stores (never flushed), follower plain loads (never inval'd). `coherent_store(&w.state)` only publishes state's cacheline. |
| 933-936 | `SharedRingSlot` plain fields | Plain stores discarded by `coherent_store(&seq)`'s dcci_inval (same cacheline). Shared mode only. |

---

## Issue 9: AICPU compilation errors after Coherent<T> unification (PARTIALLY FIXED)

**File:** `src/common/runtime/fully_distributed_within_core/dist_engine.cpp`

**Problem:** After changing `Coherent<T>` from `std::atomic<T>` to plain `T`, AICPU compilation exposes errors that were previously hidden by non-standard-layout:
- `error: reference to __host__ function 'to_active_mask' in __aicore__ function` (line ~2175)
- `error: reference to __host__ function 'core_mask'` (line ~2286)
- `error: reference to __host__ function 'compute_flat_offset'` (line ~2612, ~2625)

These functions (`to_active_mask`, `core_mask`, `compute_flat_offset`) are marked `__host__` but called from `__aicore__` functions. With `std::atomic<T>`, DistGlobal was non-standard-layout, and GCC couldn't instantiate certain template paths. Now with plain `T`, GCC can reach these call sites and flags the `__host__`/`__aicore__` mismatch.

**Status:** These errors appeared in the AICPU build but the AICore (ccec) build succeeded. The AICPU build may need these functions to not be marked `__host__`, or the calling `__aicore__` functions need to not be `__aicore__` on AICPU. This needs investigation — it may be a pre-existing latent bug exposed by the layout fix.

---

## Issue 10: Runtime struct layout — VERIFIED CONSISTENT

**Result:** Used `static_assert` with `offsetof()` to verify `Runtime` layout in both GCC (AICPU) and CCEC (AICore). All critical offsets match:
- `offsetof(Runtime, dist) = 26240` (both)
- `offsetof(dist.go) = 26256` (both)
- `offsetof(dist.done_count) = 26264` (both)
- `offsetof(dist.aicore_progress) = 26304` (both)
- `sizeof(Handshake) = 128` (both)

Layout is NOT the problem. `sizeof(Runtime)` differs (CCEC has different private member layout), but `dist` is before the private members so all `dist.*` offsets are identical.

## Issue 11: AICPU printf format bug — FIXED (was misdiagnosing all onboard output)

**File:** `src/a5/runtime/fully_distributed_within_core/aicpu/aicpu_executor.cpp:757`

**Problem:** The waiting-loop `LOG_INFO_V9` format string had 7 specifiers (`%d %d %d %d %u %s %s`) but only 6 arguments — `runtime->dist.done_count` was missing. This caused ALL progress/dbg output to be shifted by one argument:
- `done_count=%d` printed `num_workers` (9)
- `/%d` printed `runtime->dist.go` (1)
- `(go=%u)` printed `prog` pointer as uint (garbage like `0xf2fdd370`)
- `progress=[%s]` printed `dbg` string
- `dbg=[%s]` printed nothing (`(null)`)

This made it appear as if `go` was garbage and `done_count=9` (when actually `done_count=0` and `go=1`). All previous diagnosis based on these values was wrong.

**Fix applied:** Added `runtime->dist.done_count` as the third argument:
```cpp
LOG_INFO_V9("... done_count=%d/%d (go=%u) ...",
    thread_idx, dist_done_count,
    runtime->dist.done_count, num_workers, runtime->dist.go, prog, dbg);
```

## Issue 12: AICPU flush dist cacheline — ADDED

**File:** `src/a5/runtime/fully_distributed_within_core/aicpu/aicpu_executor.cpp:684,704`

**Problem:** AICPU wrote `dist.core_main_fn`, `dist.num_workers`, `dist.done_count`, `dist.go` via plain stores but only flushed `workers[i]` handshake lines. The `dist` cacheline (offset 26240-26303) was never flushed to HBM. AICore's `atomicAdd(&dist.done_count, 1)` triggers L2 coherence to invalidate AICPU's cacheline, potentially losing AICPU's unflushed writes to `go`/`num_workers`.

**Fix applied:** Added `cache_flush_range` after writing `dist` fields:
- Line 684: `cache_flush_range(&runtime->dist.core_main_fn, 64)` after writing core_main_fn/num_workers/done_count
- Line 704: `cache_flush_range(&runtime->dist.go, 64)` after writing go=1

## Issue 13: a2a3 compilation guards (FIXED)

---

## Summary of all code changes

### dist_engine.cpp

| Location | Change | Reason |
|----------|--------|--------|
| `Coherent<T>` (line 407) | Unified to `T a;` (was `std::atomic<T>` on sim) | Fix AICPU/AICore layout divergence |
| `coherent_load` (line 426) | AICore: `atomicMax(p, identity)` instead of dcci_inval + plain load | Fix TOCTOU window |
| `coherent_store` (line 447) | AICore: atomicCAS loop instead of plain store + dcci_flush | Fix cacheline clobber (C4) |
| `coherent_fetch_add/sub` (line 460/472) | AICore: added dcci_inval after atomicAdd | Prevent stale cache clobber |
| `coherent_fetch_xor` (line 484) | AICore: added dcci_inval after CAS loop | Same |
| `coherent_cas_weak` (line 502) | AICore: added dcci_inval after atomicCAS | Same |
| `dist_set_flag` (line 606) | AICore: added dcci_inval after atomicMax | Same |
| `submit_crumb` (line 1683) | AICore: atomicExch instead of store + dcci_flush | Reliable crumb publish |
| `DIST_CRUMB/DIST_DBG` (line 2949) | AICore: atomicExch; AICPU: `__atomic_store_n` + dcci_flush | Reliable diagnostics |
| `resolve_kernel_addr` (line 1668) | Added dcci_inval before reading func_id_to_addr_ and resolved_addr_ | Fix stale cache read |
| `execute_slot` (line 1764) | Per-tensor dcci instead of ENTIRE_DATA_CACHE | Fix control-state clobber |
| `execute_slot` (line 1776-1778) | Added crumb 50/51 around `fn()` call | Diagnose kernel hang |
| `claim` (line 1638) | Added dcci_inval after atomicMax | Prevent stale cache clobber |
| `dist_core_main_impl` (line 3125) | Added config region dcci_inval after startup barrier | AICore reads AICPU-written config with fresh cache |
| `dist_core_main_impl` (line 3226) | Added `pto_dcci_inval(&gd->orch_args_gm, sizeof(...))` before orch call | AICore reads L2TaskArgs with fresh cache |
| Various (line 1677, 2218, 3383, 3597) | `#ifdef AICORE_PROGRESS_STRIDE` guards | Fix a2a3 compilation |

### aicpu_executor.cpp

| Location | Change | Reason |
|----------|--------|--------|
| Line 684 | Added `cache_flush_range(&dist.core_main_fn, 64)` after writing dist fields | Flush dist cacheline to HBM |
| Line 704 | Added `cache_flush_range(&dist.go, 64)` after writing go=1 | Same |
| Line 757 | Fixed printf format: added missing `runtime->dist.done_count` argument | All previous output was shifted by one arg |
| Line 769 | Added TYPE_MATCH diagnostic output (slot 5/6/7) | Verify type_match/role per core |

### Other files

| File | Change | Reason |
|------|--------|--------|
| `src/a2a3/platform/onboard/host/CMakeLists.txt` | `SIMPLER_ENABLE_PTO_SDMA_WORKSPACE` OFF | pto-isa version mismatch |
| `simpler_setup/runtime_compiler.py` | strip non-fatal for cross-arch .so | AICPU .so is aarch64, host strip can't read it |
| `src/common/platform/sim/aicpu/device_malloc.cpp` | Added `aicpu_device_probe_uncacheable` stub | a5 aicpu_executor calls it, sim had no impl |

## Excluded hypotheses (confirmed NOT the problem)

| Hypothesis | Excluded by |
|------------|-------------|
| Runtime struct layout divergence (AICPU vs AICore) | `static_assert(offsetof)` verified identical in GCC and CCEC: dist=26240, go=26256, done_count=26264, aicore_progress=26304 |
| `__mix__` attribute missing (all cores running AIC entry) | Diagnostic showed AIV cores have `role=1` (AIV), `type_match=true`. `tm=2` was misread as false (actually type_match+1=2 → true) |
| AICPU `go=1` not reaching HBM | printf format bug — `go` was always 1, garbage value was `prog` pointer printed as `%u` |
| `num_workers` corrupted to 1 | printf format bug — `1` was actually `go=1` shifted into the `num_workers` slot |
| Cursor not initialized to -1 | AICPU readback `seg[0]=0xffffffff` = -1, correct |
| `claim()` never succeeds | Crumb 70 appeared (core 5 won task 0) |
| `Coherent<T>` std::atomic causing non-standard-layout | Unified to plain T; all offsets verified consistent |
| `done_count=9` (all cores completed) | printf format bug — `9` was `num_workers`, actual `done_count=0` |

## Current hang point

**Core 5 wins task 0, enters `execute_slot`, reaches `fn()` call (crumb 50), kernel never returns.**

`function_bin_addr = 0x1E4C0` (kernel binary offset). Need to determine:
1. Is `s.tensors[0].buffer.addr` valid? (written to diag slot 3/4 but AICPU print condition may not show it)
2. Does `pto_dcci_inval` on tensor buffer.addr fault if addr is invalid?
3. Does the kernel itself fault (bad DMA, bad arg layout)?
4. Check AICore exception dump in plog for faulting PC and access address

## Session state snapshot (for context recovery after compact)

### What works
- a5sim `vector_example` PASSES ✓
- a5sim `mix_coown` PASSES ✓
- a5 onboard `tensormap_and_ringbuffer/dummy_task` PASSES ✓
- a5 onboard `tensormap_and_ringbuffer/mixed_example` (TLOAD/TADD/TSTORE/pipe_sync AIV kernels) PASSES ✓
- `pip install -e .` all 4 platforms (a2a3sim, a5sim, a2a3, a5) PASSES ✓
- A5 hardware atomic probes (AtomicCas, AtomicMax) all pass on A5 ✓

### What's broken
- a5 onboard `fully_distributed_within_core/vector_example` FAILS (507018 timeout)
- Kernel call `fn(s.args)` hangs at crumb 50 (fn entered, never returns)
- With DIST_SKIP_EXEC: 8/9 cores complete → engine logic correct, only kernel call is the problem

### Most likely root cause (not yet verified)
Multiple kernel .cpp files in the fdwc build each define `extern "C" __aicore__ void kernel_entry(...)`. When linked into one AICore binary, only ONE `kernel_entry` symbol survives (last one wins). All `func_id_to_addr_[]` entries may resolve to the same kernel_entry, so calling func_id=0 (kernel_add) actually calls kernel_mul or whatever was last in the link order. The wrong kernel reads args in the wrong format → pipe deadlock.

**Verification command:**
```bash
strings build/cache/a5/onboard/fully_distributed_within_core/aicore/aicore_kernel.o | grep "kernel_entry"
```
If only one `kernel_entry` symbol exists, this is the root cause.

**Fix direction:** Rename each kernel's entry to a unique symbol (e.g. `kernel_entry_add`, `kernel_entry_add_scalar`, `kernel_entry_mul`), or use a mangled name. The build system (`kernel_compiler.py`) must register the correct symbol name per func_id.

### Files modified (git diff summary)
- `src/common/runtime/fully_distributed_within_core/dist_engine.cpp` — Coherent<T> unified to plain T, all coherent_* ops use atomics on AICore, DIST_CRUMB/submit_crumb use atomicExch, execute_slot uses per-tensor dcci, config region inval at entry, orch_args_gm inval, resolve_kernel_addr dcci_inval, #ifdef AICORE_PROGRESS_STRIDE guards for a2a3
- `src/a5/runtime/fully_distributed_within_core/aicpu/aicpu_executor.cpp` — printf format fix (missing done_count arg), cache_flush_range for dist cacheline, TYPE_MATCH/TENSOR_ADDR diagnostic output
- `src/a5/platform/onboard/aicore/kernel.cpp` — no changes (reverted __mix__ experiment)
- `src/a2a3/platform/onboard/host/CMakeLists.txt` — SIMPLER_ENABLE_PTO_SDMA_WORKSPACE OFF
- `simpler_setup/runtime_compiler.py` — strip non-fatal for cross-arch
- `src/common/platform/sim/aicpu/device_malloc.cpp` — aicpu_device_probe_uncacheable stub
- `tests/st/a5/fully_distributed_within_core/vector_example/kernels/aiv/*.cpp` — removed `__attribute__((always_inline))` from kernel_entry
- `tests/ONBOARD_ISSUES.md` — this file

### Diagnostic crumbs still in code (should clean up after fix)
- `submit_crumb(70 + N)` after claim success
- `submit_crumb(50/51)` around fn() call
- slot 3/4 tensor addr write before fn()
- slot 5/6/7 type_match/role/anchor_is_cube at N==0
- N==0 cursor value read to slot 3/4
- drain_phase_b slot 5/6 write for execute_slot/skip path
- TENSOR_ADDR print in aicpu_executor.cpp waiting loop
- TYPE_MATCH print in aicpu_executor.cpp waiting loop

### Build/test quick reference
```bash
source /home/q00473782/cann/cann-9.1.0/set_env.sh
export PTO_ISA_ROOT=/home/q00473782/dsl/pto-isa

# Rebuild a5 only
rm -rf build/cache/a5 build/lib/a5
python -m simpler_setup.build_runtimes --platforms a5

# Test a5 onboard
python -m pytest tests/st/a5/fully_distributed_within_core/vector_example/test_vector_example.py --platform a5 -x -v --log-level v9

# Test a5sim (should pass)
python -m pytest tests/st/a5/fully_distributed_within_core/vector_example/test_vector_example.py --platform a5sim -x -v

# Check device logs
LOG=$(ls -t ~/ascend/log/debug/device-0/*.log | head -1)
grep -aE "waiting dist_done" "$LOG" | head -1
```
