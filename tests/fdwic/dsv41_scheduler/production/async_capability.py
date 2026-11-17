# Copyright (c) PyPTO Contributors.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------

"""Explicit completion declarations, maintained with each kernel's signature."""

MODES = {"SYNCHRONOUS": 1, "DEFERRED": 2}


def capability_words(kernel):
    """Validate a declaration and return its four wire fields; never infer one."""
    value = kernel.get("async_capability")
    if not isinstance(value, dict) or set(value) != {"version", "mode", "max_conditions", "condition_types"}:
        raise ValueError("Invalid async capability fields")
    version, mode = value["version"], value["mode"]
    count, types = value["max_conditions"], value["condition_types"]
    if type(version) is not int or version != 1 or type(mode) is not str or mode not in MODES:
        raise ValueError("Unsupported async capability version or mode")
    if type(count) is not int or type(types) is not int:
        raise ValueError("Invalid async capacity/types")
    valid = (1 <= count <= 64 and 1 <= types <= 3) if mode == "DEFERRED" else count == types == 0
    if not valid:
        raise ValueError("Invalid async capacity/types")
    return version, MODES[mode], count, types
