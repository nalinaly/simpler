# Copyright (c) PyPTO Contributors.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------

"""Audit once-published ordinary block pools without inventing admission orders."""

from .native_causality import MAX, point, require
from .native_protocol import has_pool_protocol, pool_word_task_capacity


def audit_pool_publications(audit):
    if not has_pool_protocol(audit.raw):
        return
    pool = audit.raw["ordinary_pool"]
    require(pool["task_count"] == len(audit.pool_tasks) > 0, "ordinary pool task count disagrees with frozen shapes")
    require(all(word == 0 for word in pool["words"]), "ordinary pool retains unclaimed work at terminal state")
    publications = {row["index"]: row["publication"] for row in pool["publications"]}
    require(publications.keys() == audit.pool_tasks, "ordinary pool publication coverage mismatch")
    for ref, actor in publications.items():
        audit.actor(actor, ref)
        require(actor["order_key"] == MAX and ref not in audit.orders, "pool publication fabricated an admission order")
        require(
            audit.workers[actor["worker_index"]]["core_type"] == int(audit.tasks[ref]["active_mask"] != 1),
            "pool publication came from the wrong engine",
        )
        require(audit.actors[ref]["admission"] is None, "pool task has both publication and admission ownership")
        audit.pool_publication_before[ref] = point(actor, "before")
        audit.pool_publication_points[ref] = audit.causal.bounded(
            ("pool_publication", ref), point(actor, "before"), point(actor, "after")
        )


def audit_pool_claims(audit):
    if not has_pool_protocol(audit.raw):
        return
    capacity = pool_word_task_capacity(audit.raw)
    retired_words = {}
    for index, block in audit.blocks.items():
        ref, _ = audit.expected_blocks[index]
        require(("pool_claim" in block) == (ref in audit.pool_tasks), "block uses the wrong claim certificate kind")
    for ref in audit.pool_tasks:
        task = audit.tasks[ref]
        blocks = audit.task_blocks[ref]
        by_confirmation = sorted(blocks, key=lambda block: block["pool_claim"]["confirmed_count"])
        require(
            [block["pool_claim"]["confirmed_count"] for block in by_confirmation]
            == list(range(1, task["block_num"] + 1)),
            "duplicate or missing pool claim confirmation",
        )
        previous_confirmation = None
        for block in by_confirmation:
            index, owner, certificate = block["index"], block["owner"], block["pool_claim"]
            require(
                certificate["run_epoch"] == audit.raw["run_epoch"]
                and certificate["worker_index"] == owner["worker_index"]
                and certificate["block_idx"] == owner["block_idx"],
                "pool claim certificate changed its owner, block or epoch",
            )
            audit.causal.add(audit.pool_publication_points[ref], audit.claim_points[(ref, owner["block_idx"])])
            final = certificate["confirmed_count"] == task["block_num"]
            upper = point(certificate, "retire_before") if final else audit.payload_points[index][0]
            confirmation = audit.causal.bounded(
                ("pool_confirmation", ref, certificate["confirmed_count"]), point(owner, "claim_after"), upper
            )
            if previous_confirmation is not None:
                audit.causal.add(previous_confirmation, confirmation)
            previous_confirmation = confirmation
            if not final:
                require(
                    certificate["retire_before"] == certificate["retire_after"] == [0, 0]
                    and certificate["retired_word_before"] == 0,
                    "nonfinal pool confirmer fabricated a retirement",
                )
                continue
            bit = 1 << (ref % capacity)
            require(certificate["retired_word_before"] & bit, "pool retirement did not own a published bit")
            audit.pool_retirement_points[ref] = audit.causal.bounded(
                ("pool_retirement", ref), point(certificate, "retire_before"), point(certificate, "retire_after")
            )
            audit.causal.add(point(certificate, "retire_after"), audit.payload_points[index][0])
            audit.before(
                point(certificate, "retire_after"),
                point(audit.executions[index], "permit"),
                "final pool confirmer entered before retiring its bit",
            )
            retired_words[ref] = certificate["retired_word_before"]

    # A returned atomic word can also witness a different task's live bit.
    # Bind those observations to publication/retirement intervals, without
    # imposing a shared clock on the workers or guessing what an absent bit means.
    for ref, observed in retired_words.items():
        role = int(audit.tasks[ref]["active_mask"] != 1)
        neighbors = {
            other
            for other in audit.pool_tasks
            if other // capacity == ref // capacity and int(audit.tasks[other]["active_mask"] != 1) == role
        }
        valid_bits = sum(1 << (other % capacity) for other in neighbors)
        require(observed & ~valid_bits == 0, "pool retirement observed an ineligible bitmap bit")
        for other in neighbors - {ref}:
            if observed & (1 << (other % capacity)):
                audit.causal.add(audit.pool_publication_points[other], audit.pool_retirement_points[ref])
                audit.causal.add(audit.pool_retirement_points[ref], audit.pool_retirement_points[other])
