#!/usr/bin/env python3
"""Attribute Kineto CUDA kernels to explicit PP lane ranges and measure overlap.

No lane is inferred from a stream number or kernel name. Chrome trace timestamps
and durations are microseconds; displayTimeUnit changes presentation, not units.
"""

import argparse
from bisect import bisect_right
from collections import Counter, defaultdict
import gzip
import json
import math
from pathlib import Path
import re


LABEL = re.compile(r"^pdmux_pp\|pp=(\d+)\|tp=(\d+)\|lane=(prefill|decode)\|partition=(\d+)\|phase=([^|]+)$")
KERNEL_CATEGORIES = {"kernel", "gpu_kernel"}
CPU_CATEGORIES = {"cpu_op", "user_annotation", "python_function", "cuda_runtime", "cuda_driver"}


def union(intervals):
    merged = []
    for start, end in sorted(intervals):
        if merged and start <= merged[-1][1]:
            merged[-1][1] = max(end, merged[-1][1])
        else:
            merged.append([start, end])
    return merged


def duration(intervals):
    return sum(end - start for start, end in intervals)


def intersection(left, right):
    left, right = union(left), union(right)
    i = j = 0
    result = []
    while i < len(left) and j < len(right):
        start, end = max(left[i][0], right[j][0]), min(left[i][1], right[j][1])
        if start < end:
            result.append([start, end])
        if left[i][1] <= right[j][1]:
            i += 1
        else:
            j += 1
    return result


def thread_key(event):
    return str(event.get("pid")), str(event.get("tid"))


class RangeIndex:
    """Thread-local interval lookup; prefix maxima also handle nested ranges."""

    def __init__(self, rows):
        groups = defaultdict(list)
        for row in rows:
            groups[thread_key(row["event"])].append(row)
        self.groups = {}
        for key, entries in groups.items():
            entries.sort(key=lambda row: row["start"])
            ends, latest = [], -math.inf
            for row in entries:
                latest = max(latest, row["end"])
                ends.append(latest)
            self.groups[key] = ([row["start"] for row in entries], ends, entries)

    def containing(self, event, point=False):
        start = event["ts"]
        end = start if point else start + event["dur"]
        starts, ends, rows = self.groups.get(thread_key(event), ([], [], []))
        index = bisect_right(starts, start) - 1
        matches = []
        while index >= 0 and ends[index] >= end:
            row = rows[index]
            if row["end"] >= end and row["end"] > start:
                matches.append(row)
            index -= 1
        return matches


def identity(label):
    return tuple(label[key] for key in ("pp_rank", "tp_rank", "lane", "partition"))


def argument_id(event, names):
    for name, value in event.get("args", {}).items():
        normalized = re.sub(r"[ _]", "", name.lower())
        if normalized in names and type(value) in (str, int) and str(value) not in ("", "0", "-1"):
            return str(value)
    return None


def finite(value):
    return type(value) in (int, float) and math.isfinite(value)


def analyze_trace(data, source, evidence_limit=100):
    events = data.get("traceEvents") if isinstance(data, dict) else data
    if not isinstance(events, list):
        raise ValueError("Expected Chrome traceEvents list")
    complete, flows, invalid = [], [], 0
    for index, event in enumerate(events):
        if not isinstance(event, dict):
            invalid += 1
            continue
        if event.get("ph") == "X":
            if not finite(event.get("ts")) or not finite(event.get("dur")) or event["dur"] < 0:
                invalid += 1
                continue
            if event["dur"] > 0:
                complete.append({"event": event, "index": index,
                                 "start": event["ts"], "end": event["ts"] + event["dur"]})
        elif event.get("ph") in ("s", "f") and finite(event.get("ts")):
            flows.append(event)
    kernels, labels = [], []
    for row in complete:
        event = row["event"]
        categories = set(str(event.get("cat", "")).split(","))
        if categories & KERNEL_CATEGORIES:
            kernels.append(row)
        elif categories & CPU_CATEGORIES and (match := LABEL.fullmatch(event.get("name", ""))):
            pp, tp, lane, partition, phase = match.groups()
            labels.append({**row, "label": {"pp_rank": int(pp), "tp_rank": int(tp),
                            "lane": lane, "partition": int(partition), "phase": phase}})
    label_index = RangeIndex(labels)

    def classify_cpu(event, point=False):
        containing = label_index.containing(event, point)
        if not containing or len({identity(row["label"]) for row in containing}) != 1:
            return None
        return min(containing, key=lambda row: row["end"] - row["start"])

    external, correlations = defaultdict(list), defaultdict(list)
    kernel_ids = {row["index"] for row in kernels}
    for row in complete:
        if row["index"] in kernel_ids:
            continue
        event = row["event"]
        if not set(str(event.get("cat", "")).split(",")) & CPU_CATEGORIES:
            continue
        classified = classify_cpu(event)
        if external_id := argument_id(event, {"externalid"}):
            external[external_id].append(classified)
        if any(category in str(event.get("cat", "")) for category in ("cuda_runtime", "cuda_driver")):
            if correlation := argument_id(event, {"correlation", "correlationid"}):
                correlations[correlation].append(classified)

    # Accept a flow only when both endpoints identify exactly one source/target.
    flow_groups = defaultdict(list)
    ignored_non_cuda_flows = 0
    for event in flows:
        if event.get("cat") != "ac2g" or event.get("name") != "ac2g":
            ignored_non_cuda_flows += 1
            continue
        flow_id = event.get("id", event.get("id2"))
        if flow_id is not None:
            scope = event.get("pid") if isinstance(flow_id, dict) and "local" in flow_id else None
            key = (event.get("cat"), event.get("name"), json.dumps(flow_id, sort_keys=True), scope)
            flow_groups[key].append(event)
    kernel_index = RangeIndex(kernels)
    flow_sources = defaultdict(list)
    ignored_flows = 0
    for group in flow_groups.values():
        starts = [event for event in group if event["ph"] == "s"]
        ends = [event for event in group if event["ph"] == "f"]
        if len(starts) != 1 or len(ends) != 1:
            ignored_flows += 1
            continue
        targets = kernel_index.containing(ends[0], point=True)
        if len(targets) != 1:
            ignored_flows += 1
            continue
        flow_sources[targets[0]["index"]].append(classify_cpu(starts[0], point=True))

    mapping = Counter(mapped_kernels=0, unmapped_kernels=0, ambiguous_kernels=0)
    methods = Counter()
    evidence, by_device = [], defaultdict(list)
    for row in kernels:
        event = row["event"]
        links = {}
        ext = argument_id(event, {"externalid"})
        corr = argument_id(event, {"correlation", "correlationid"})
        if ext in external:
            links["external_id"] = external[ext]
        if corr in correlations:
            links["correlation"] = correlations[corr]
        if row["index"] in flow_sources:
            links["flow"] = flow_sources[row["index"]]
        candidates = [source_row for group in links.values() for source_row in group]
        known = [source_row for source_row in candidates if source_row is not None]
        classification = None
        if known and len(known) == len(candidates) and len({identity(item["label"]) for item in known}) == 1:
            source_row = min(known, key=lambda item: item["end"] - item["start"])
            classification = source_row["label"]
            status = "mapped"
            methods.update(links.keys())
        elif known:
            status = "ambiguous"
        else:
            status = "unmapped"
        mapping[f"{status}_kernels"] += 1
        args = event.get("args", {})
        device = str(args.get("device", f"pid:{event.get('pid')}"))
        stream = str(args.get("stream", f"tid:{event.get('tid')}"))
        if "context" in args:
            stream = f"context:{args['context']}/stream:{stream}"
        item = {"event_index": row["index"], "name": event.get("name"),
                "start_us": row["start"], "duration_us": event["dur"],
                "device": device, "stream": stream, "status": status,
                "device_stream_known": args.get("device") is not None and args.get("stream") is not None,
                "classification": classification, "methods": sorted(links)}
        by_device[device].append(item)
        if len(evidence) < evidence_limit:
            evidence.append(item)

    devices = {}
    for device, rows in by_device.items():
        streams = defaultdict(list)
        lane_streams = {lane: defaultdict(list) for lane in ("prefill", "decode")}
        lanes = {}
        for row in rows:
            interval = (row["start_us"], row["start_us"] + row["duration_us"])
            if row["device_stream_known"]:
                streams[row["stream"]].append(interval)
            if row["classification"] and row["device_stream_known"]:
                lane_streams[row["classification"]["lane"]][row["stream"]].append(interval)
        stream_keys = list(streams)
        generic = union([piece for i, a in enumerate(stream_keys) for b in stream_keys[i + 1:]
                         for piece in intersection(streams[a], streams[b])])
        for lane in ("prefill", "decode"):
            selected = [row for row in rows if row["classification"] and row["classification"]["lane"] == lane]
            intervals = [(row["start_us"], row["start_us"] + row["duration_us"]) for row in selected]
            phase_intervals = defaultdict(list)
            for row, interval in zip(selected, intervals):
                phase_intervals[row["classification"]["phase"]].append(interval)
            lanes[lane] = {"kernel_count": len(selected), "kernel_sum_us": duration(intervals),
                           "kernel_union_us": duration(union(intervals)),
                           "streams": sorted(lane_streams[lane]),
                           "identity_proven_kernel_count": sum(row["device_stream_known"] for row in selected),
                           "phase_stats": {phase: {"kernel_count": len(values), "kernel_sum_us": duration(values),
                                                   "kernel_union_us": duration(union(values))}
                                           for phase, values in phase_intervals.items()},
                           "phases": dict(Counter(row["classification"]["phase"] for row in selected))}
        available = all(lane_streams[lane] for lane in ("prefill", "decode"))
        overlap = union([piece for p, p_intervals in lane_streams["prefill"].items()
                         for d, d_intervals in lane_streams["decode"].items() if p != d
                         for piece in intersection(p_intervals, d_intervals)])
        devices[device] = {
            "kernel_count": len(rows),
            "kernel_union_us": duration(union([(row["start_us"], row["start_us"] + row["duration_us"]) for row in rows])),
            "stream_count": len(streams), "lanes": lanes,
            "kernels_missing_device_or_stream": sum(not row["device_stream_known"] for row in rows),
            "generic_different_stream_overlap_us": duration(generic) if streams else None,
            "semantic_overlap_available": bool(available),
            "prefill_decode_overlap_us": duration(overlap) if available else None,
            "prefill_decode_overlap_intervals_us": overlap if available else [],
            "overlap_fraction_of_prefill_union": duration(overlap) / lanes["prefill"]["kernel_union_us"] if available else None,
            "overlap_fraction_of_decode_union": duration(overlap) / lanes["decode"]["kernel_union_us"] if available else None,
        }
    ranks = sorted({row["label"]["pp_rank"] for row in labels})
    filename_rank = re.search(r"(?:^|-)PP-(\d+)(?:-|\.)", Path(source).name)
    rank = int(filename_rank[1]) if filename_rank else (ranks[0] if len(ranks) == 1 else None)
    limitations = [
        "Kernel intervals measure recorded GPU execution overlap, not SM occupancy or achieved throughput.",
        "Only explicit pdmux_pp ranges with external-id, runtime-correlation, or unique flow links assign lanes; stream numbers and kernel names never assign lanes.",
        "Mapped overlap is a lower bound when any kernels are unmapped or ambiguous; zero mapped overlap does not rule out unobserved overlap.",
        "Each trace/device is measured separately; timestamps are not aligned or combined across PP ranks.",
        "Memcpy, memset, communication outside lane ranges, and host launch counters are excluded from semantic kernel overlap.",
        "Device/stream identity must be explicit for overlap evidence; pid/tid fallbacks label unknown groups only. Complete X events and ac2g s/f flows are supported, not reconstructed B/E ranges.",
    ]
    if not labels:
        limitations.append("No explicit lane ranges found; enable SGLANG_PDMUX_PP_PROFILE_RANGES=1 during the diagnostic capture.")
    if not kernels:
        limitations.append("No GPU kernel events found; CPU-only captures cannot establish GPU overlap.")
    if ranks and ranks != [rank]:
        limitations.append(f"PP rank labels {ranks} do not match filename rank {rank}; inspect trace provenance.")
    if invalid:
        limitations.append(f"Ignored {invalid} malformed events; coverage is incomplete.")
    return {"source": str(source), "pp_rank": rank, "label_pp_ranks": ranks, "time_unit": "us",
            "event_count": len(events), "kernel_count": len(kernels), "lane_range_count": len(labels),
            "lane_range_names": dict(Counter(row["event"]["name"] for row in labels)),
            "mapping": {**mapping, "methods": dict(methods), "ignored_flow_groups": ignored_flows,
                        "ignored_non_cuda_flow_events": ignored_non_cuda_flows},
            "devices": devices, "kernel_evidence": evidence, "evidence_limit": evidence_limit,
            "evidence_truncated": len(kernels) > evidence_limit, "limitations": limitations}


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--trace", type=Path, nargs="+", required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--evidence-limit", type=int, default=100)
    args = parser.parse_args(argv)
    if args.evidence_limit < 0:
        parser.error("--evidence-limit must be nonnegative")
    report = {"schema_version": 1, "traces": [], "errors": [], "completed": False}
    for path in args.trace:
        try:
            with (gzip.open(path, "rt") if path.suffix == ".gz" else path.open()) as source:
                report["traces"].append(analyze_trace(json.load(source), str(path.resolve()), args.evidence_limit))
        except Exception as error:
            report["errors"].append({"source": str(path), "error": str(error)})
    report["completed"] = not report["errors"]
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2, allow_nan=False) + "\n")
    print(f"Parsed {len(report['traces'])} traces; errors={len(report['errors'])}")
    return 0 if report["completed"] else 2


if __name__ == "__main__":
    raise SystemExit(main())
