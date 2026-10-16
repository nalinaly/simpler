#!/usr/bin/env python3
# Copyright (c) PyPTO Contributors.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------
"""沿用 Worker 的 kernel 模式，以调用方持有的设备 Tensor 验证大参数 MIX。"""

import argparse
import json
from pathlib import Path

from simpler.task_interface import CallConfig, ChipStorageTaskArgs, ChipTensor, DataType

from tests.st.a2a3.core_argument_capacity.test_core_argument_capacity import (
    TestHbg133,
    TestHbg256,
    TestTmr133,
    TestTmr256,
)
from tests.ut.py.test_worker.test_worker_kernel_mode_hw import (
    _COUNT,
    _assert_closed_cleanly,
    _caller_device,
    _kernel_worker,
)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--device", type=int, required=True)
    parser.add_argument("--runtime", choices=["host_build_graph", "tensormap_and_ringbuffer"], required=True)
    parser.add_argument("--count", choices=[133, 256], type=int, required=True)
    parser.add_argument("--output", type=Path, required=True)
    options = parser.parse_args()
    cases = {
        ("host_build_graph", 133): TestHbg133,
        ("host_build_graph", 256): TestHbg256,
        ("tensormap_and_ringbuffer", 133): TestTmr133,
        ("tensormap_and_ringbuffer", 256): TestTmr256,
    }
    chip = cases[options.runtime, options.count]().build_callable("a2a3")
    result = {
        "runtime": options.runtime,
        "tensor_count": options.count,
        "core_scalar_count": 16,
        "mode": "kernel",
        "passed": False,
    }
    options.output.parent.mkdir(parents=True, exist_ok=True)
    try:
        with _caller_device(options.device) as caller, _kernel_worker(caller, "a2a3", options.runtime) as worker:
            config = CallConfig()
            config.runtime_env.ring_task_window = 64
            worker.init(config=config)
            pin = worker._kernel_pin_finalizer
            print("kernel init 完成", flush=True)
            callable_id = worker.kernel_prepare_callable(chip)
            args = ChipStorageTaskArgs()
            for i in range(options.count - 1):
                value = options.count if i == 0 else i + 1
                address = caller.device_buffer([float(value)])
                args.add_tensor(ChipTensor.make(address, (1,), DataType.FLOAT32, child_memory=True))
            output = caller.device_buffer([-1.0] * _COUNT)
            args.add_tensor(ChipTensor.make(output, (_COUNT,), DataType.FLOAT32, child_memory=True))
            expected = [-1.0] * _COUNT
            value = options.count + sum((i + 1) ** 2 for i in range(1, options.count - 1))
            value += sum((101 + i) * (i + 1) for i in range(16))
            for block in range(24):
                for lane in range(3):
                    expected[(block * 3 + lane) * 16] = float(value + block * 1000 + lane * 100 + 24)
            # 重复 launch 同一 callable/地址，检查三次运行及所有未写入保护区。
            for repeat in range(3):
                worker.kernel_launch(callable_id, args, caller_stream=caller.stream)
                assert caller.synchronize() == 0
                actual = caller.read(output)
                assert actual == expected, f"第 {repeat} 次输出或保护区不符"
                print(f"kernel launch {repeat + 1}/3 完整输出通过", flush=True)
            worker.close()
            _assert_closed_cleanly(worker, pin)
            result["passed"] = True
            result["launches"] = 3
    except Exception as exc:
        result["error"] = str(exc)
        raise
    finally:
        options.output.write_text(json.dumps(result, ensure_ascii=False, indent=2) + "\n")
        print(json.dumps(result, ensure_ascii=False), flush=True)


if __name__ == "__main__":
    main()
