#!/usr/bin/env python3
# Copyright (c) PyPTO Contributors.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------
"""只读检查设备生成的选择性 dump，包括超过 128 的 Tensor 和最后一个标量。"""

import argparse
import json
import struct
from pathlib import Path


def verify(path: Path, count: int):
    data = json.loads(path.read_text())
    assert data["dump_args_level"] == 1
    assert data["truncated_args"] == data["dropped_records"] == 0
    records = data["args"]
    selected = {63, 64, 128, count - 1, count + 15}
    assert {record["arg_index"] for record in records} == selected
    blob = (path.parent / data["bin_file"]).read_bytes()
    tasks = {record["task_id"] for record in records}
    assert len(tasks) == 2
    for task in tasks:
        current = [record for record in records if record["task_id"] == task]
        assert {record["arg_index"] for record in current} == selected
        for record in current:
            if record["kind"] == "scalar":
                assert record["arg_index"] == count + 15
                assert record["dtype"] == "INT64" and record["value"] == 116
            elif record["arg_index"] in (63, 64, 128):
                value = struct.unpack_from("<f", blob, record["bin_offset"])[0]
                assert value == record["arg_index"] + 1
        outputs = [record for record in current if record["stage"] == "after_completion"]
        assert len(outputs) == 1 and outputs[0]["arg_index"] == count - 1
    return {
        "path": str(path),
        "tensor_count": count,
        "selected_indices": sorted(selected),
        "tasks": len(tasks),
        "records": len(records),
        "passed": True,
    }


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("path", type=Path)
    parser.add_argument("--count", type=int, required=True)
    options = parser.parse_args()
    print(json.dumps(verify(options.path, options.count), ensure_ascii=False, indent=2))
