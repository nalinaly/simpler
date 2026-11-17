# Copyright (c) PyPTO Contributors.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------

"""Completion declarations normalized through the public registration boundary."""

from simpler.task_interface import CoreCallable

MODES = {"SYNCHRONOUS": 1, "DEFERRED": 2}


def capability_declaration(kernel):
    """Use the same declaration policy as every registered CoreCallable."""
    return CoreCallable.build(
        signature=[], binary=b"", async_capability=kernel.get("async_capability")
    ).async_capability


def capability_words(kernel):
    """Return canonical wire fields without a separate benchmark capability policy."""
    value = capability_declaration(kernel)
    return value["version"], MODES[value["mode"]], value["max_conditions"], value["condition_types"]
