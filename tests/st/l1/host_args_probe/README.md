# L1 `WithHostArgs`/placeholder device probe

This standalone probe measures the CANN ownership boundary that cannot be
proved by PyPTO host unit tests.  It deliberately does not call PyPTO's HBG
serializer or parser: the device AICPU kernel independently checks the actual
runtime task-args base, three patched pointers, full payload checksums, and
head/middle/tail bytes.

The probe covers:

- eager snapshots after the writable host image is poisoned and released;
- three independent placeholders in one variable-size argument image;
- 64 KiB, 1 MiB, 16 MiB, and 64 MiB argument-size scans by default;
- two independently captured graphs with different payloads, alternating
  replay, and configurable replay count (100 by default);
- intervening `WithHostArgs` launches to pressure the runtime args allocator;
- HBM samples before capture, after capture, after pressure, after replay, and
  after graph destruction.

It is a measurement tool, not a product limit declaration.  A failing size is
reported with its real ACL error; no internal runtime constant is copied into
PyPTO policy.

## Safety boundary

The executable requires `--device=<id>` and has no default device.  It never
calls `aclrtResetDevice`.  Run it only through the repository's normal device
scheduler/precheck and only on an idle device.  For this development branch,
use device 1; device 0 is reserved for the parallel worktree.

## Build

```bash
source ../.claude/skills/testing/load-env.sh
tests/st/l1/host_args_probe/build.sh /tmp/gpt-host-args-probe
```

The build also runs a no-device self-test.  It invokes the same AICPU parser on
a deliberately one-byte-offset argument image and verifies both the valid
three-pointer case and the bad-placeholder diagnostic.

Set the dispatcher and probe AICPU SO paths, then run on the explicitly
scheduled device:

```bash
SIMPLER_DISPATCHER_SO=/path/to/libaicpu_dispatcher.so \
HOST_ARGS_PROBE_AICPU_SO=/tmp/gpt-host-args-probe/libhost_args_probe_aicpu.so \
/tmp/gpt-host-args-probe/host_args_probe \
  --device=1 \
  --eager-sizes=65536,1048576,16777216,67108864 \
  --graph-sizes=1048576,16777216 \
  --replays=100 \
  --pressure-count=512 \
  --pressure-size=65536
```

Use a separate run with a 64 MiB graph image after the regular matrix is
green; keeping the replay count explicit makes the cost visible:

```bash
SIMPLER_DISPATCHER_SO=/path/to/libaicpu_dispatcher.so \
HOST_ARGS_PROBE_AICPU_SO=/tmp/gpt-host-args-probe/libhost_args_probe_aicpu.so \
/tmp/gpt-host-args-probe/host_args_probe \
  --device=1 --graph-sizes=67108864,1048576 --replays=100 \
  --pressure-count=2048 --pressure-size=65536
```

Do not interpret `2048` as a runtime specification.  It is merely one pressure
point; repeat with other counts and retain the actual successful count/error
code in the implementation record.
