#!/usr/bin/env python3
"""Exclusive coverage of Scalar lanes, in core-us (not elapsed wall time)."""
import argparse
import json
from collections import Counter
from pathlib import Path

PRIORITY = ["Kernel + synchronous wait", "Atomic", "Dcci", "ExecBind",
            "ExecFanin", "ExecComplete", "ExecTicketBind", "ExecTokenScan",
            "ExecDispatch", "ExecDrainCheck", "ExecAdmission", "ExecStartup", "Unattributed"]


def category(event):
    cat = event.get("cat")
    if cat == "scalar.blocked":
        return PRIORITY[0]
    if cat == "scalar.atomic":
        return "Atomic"
    if cat == "scalar.dcci":
        return "Dcci"
    if cat == "scalar.stage":
        return event["args"]["stage"]
    return None


def partition(lo, hi, intervals):
    """Sweep nested intervals once; never add parent and child durations."""
    points = [(lo, 0, ""), (hi, 0, "")]
    for start, end, label in intervals:
        start, end = max(lo, start), min(hi, end)
        if end > start:
            points.extend(((start, 1, label), (end, -1, label)))
    active, totals = Counter(), Counter()
    last = lo
    for tick, delta, label in sorted(points):
        key = next((p for p in PRIORITY if active[p]), "Unattributed")
        totals[key] += tick - last
        active[label] += delta
        last = tick
    assert abs(sum(totals.values()) - (hi - lo)) < 1e-6
    return totals


def analyze(trace):
    events = trace["traceEvents"]
    groups = {"AIC": Counter(), "AIV": Counter()}
    gaps = []
    unlabeled = 0
    for w in range(16):
        lane = [e for e in events if e.get("pid") == w + 1 and e.get("tid") == 1]
        scope = next(e for e in lane if e.get("name") == "Scheduler + final drain")
        lo, hi = scope["ts"], scope["ts"] + scope["dur"]
        measured = [e for e in lane if e.get("ph") == "X" and category(e)]
        intervals = [(e["ts"], e["ts"] + e["dur"], category(e)) for e in measured]
        groups["AIC" if w < 8 else "AIV"].update(partition(lo, hi, intervals))
        labeled = intervals + [(e["ts"], e["ts"] + e["dur"], "ExecDispatch")
                               for e in lane if e.get("cat") == "scalar.residual"]
        unlabeled += partition(lo, hi, labeled)["Unattributed"]
        last, previous = lo, "Scheduler begin"
        for e in sorted(measured, key=lambda e: (e["ts"], -e["dur"])):
            start, end = max(lo, e["ts"]), min(hi, e["ts"] + e["dur"])
            if end <= start:
                continue
            if start > last:
                gaps.append(dict(worker=w, start_us=last, duration_us=start-last,
                                 previous=previous, following=e["name"]))
            if end > last:
                last, previous = end, e["name"]
        if last < hi:
            gaps.append(dict(worker=w, start_us=last, duration_us=hi-last,
                             previous=previous, following="Scheduler end"))
    return {
        "counts": trace["metadata"]["counts"],
        "elapsed_us": trace["metadata"]["startup_to_final_drain_us"],
        "units": "sum across 8 Scalars per role, core-us; mutually exclusive",
        "business_unlabeled_core_us": round(unlabeled, 6),
        "business_coverage_caveat": "Labels include derived source-path complements, not additional measurements.",
        "caveat": "Unattributed is NOT idle/pure software: ordinary memory, control, "
                  "observer writes and issue-only operation tails are not individually measured.",
        "exclusive_core_us": {role: {k: round(v[k], 3) for k in PRIORITY}
                              for role, v in groups.items()},
        "poll_batch_markers": sum(e.get("ph") == "i" and e.get("cat") == "scalar.atomic"
                                  for e in events),
        "largest_unattributed": sorted(gaps, key=lambda g: g["duration_us"], reverse=True)[:8],
    }


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("traces", nargs="+", type=Path)
    args = parser.parse_args()
    for path in args.traces:
        print(json.dumps({"file": str(path), **analyze(json.loads(path.read_text()))},
                         ensure_ascii=False, indent=2))
