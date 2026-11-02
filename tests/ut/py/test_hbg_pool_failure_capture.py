# Copyright (c) PyPTO Contributors.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------

"""Incomplete pool captures preserve ownership and retirement evidence."""

import importlib.util
from pathlib import Path

import pytest
from test_hbg_ordinary_pool_protocol import word16_capture


@pytest.fixture(scope="module")
def checker(pytestconfig):
    pytestconfig.addinivalue_line("markers", "scheduler_mode(mode): expected runtime scheduler path")
    source = Path(__file__).resolve().parents[2] / "st/a5/host_build_graph/resident_failure/test_resident_failure.py"
    spec = importlib.util.spec_from_file_location("_pool_failure_fixture", source)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module._check_failed_pool_capture


def partial_capture(mask, claimed):
    raw, _ = word16_capture(((mask, 3, ()), (1, 1, (0,))))
    root = raw["tasks"][0]
    controls = {row["task_ref"]: row for row in raw["task_controls"]}
    controls[0].update(next_block=claimed, claim_tickets=claimed, completed_blocks=0, state=0)
    controls[1]["state"] = 0
    for row in raw["task_events"]:
        row["completion"] = None
    raw["blocks"] = [row for row in raw["blocks"] if row["owner"]["block_idx"] < claimed]
    raw["executions"] = [row for row in raw["executions"] if row["task_ref"] == 0 and row["block_idx"] < claimed]
    if claimed < 3:
        raw["ordinary_pool"]["words"][int(mask != 1)] = 1
    return raw, root, controls


@pytest.mark.parametrize("mask", (1, 2, 4))
@pytest.mark.parametrize("claimed", (1, 2, 3))
def test_partial_pool_capture_distinguishes_unclaimed_work_from_retired_bit(checker, mask, claimed):
    checker(*partial_capture(mask, claimed))


@pytest.mark.parametrize(
    "mutation",
    (
        "lost_live_bit",
        "extra_live_bit",
        "wrong_engine",
        "lost_owner",
        "fake_order",
        "duplicate_confirmation",
        "missing_claim",
        "stale_epoch",
        "fake_seal",
        "fake_completion",
        "early_retirement",
        "wrong_capacity",
    ),
)
def test_partial_pool_capture_rejects_fabricated_ownership(checker, mutation):
    raw, root, controls = partial_capture(2, 2)
    first, second = raw["blocks"]
    if mutation == "lost_live_bit":
        raw["ordinary_pool"]["words"][1] = 0
    elif mutation == "extra_live_bit":
        raw["ordinary_pool"]["words"][1] |= 2
    elif mutation == "wrong_engine":
        first["owner"]["worker_index"] = first["pool_claim"]["worker_index"] = 0
    elif mutation == "lost_owner":
        controls[0]["admission_claim_lock"] = 0
    elif mutation == "fake_order":
        first["owner"]["order_key"] = 0
    elif mutation == "duplicate_confirmation":
        second["pool_claim"]["confirmed_count"] = first["pool_claim"]["confirmed_count"]
    elif mutation == "missing_claim":
        raw["blocks"].pop()
    elif mutation == "stale_epoch":
        first["pool_claim"]["run_epoch"] -= 1
    elif mutation == "fake_seal":
        controls[0]["claim_seal"] = 3
    elif mutation == "fake_completion":
        raw["task_events"][0]["completion"] = raw["ordinary_pool"]["publications"][0]["publication"]
    elif mutation == "early_retirement":
        first["pool_claim"]["retired_word_before"] = 1
    else:
        raw["ordinary_pool"]["word_task_capacity"] = 32
    with pytest.raises((AssertionError, ValueError)):
        checker(raw, root, controls)
