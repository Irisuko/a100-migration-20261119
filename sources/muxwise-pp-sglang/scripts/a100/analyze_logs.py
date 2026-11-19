#!/usr/bin/env python3
"""Extract MuxWise-PP evidence without importing the runtime or using a GPU.

An exit status of zero means the report was written, not that an experiment
passed. Inspect metadata.parse_errors, metadata.issues, and calibration coverage.
Nonfinite Python floats become JSON null with their locations recorded. Counters
are cumulative: snapshots are never summed, and observed deltas exclude resets
and the unknown interval before the first snapshot. Times remain in seconds.
"""

import argparse
import ast
from collections import Counter
import csv
import json
import math
from pathlib import Path
import re


COUNTERS = ("prefill_slices", "decode_steps", "cached_tokens", "overlap_launches")
CSV_FIELDS = ("stage", "tp_rank", "partition", "prefill_sms", "decode_sms",
              "batch_size", "context_tokens", "seconds", "valid", "line_number",
              "timestamp")
ANSI = re.compile(r"\x1b\[[0-?]*[ -/]*[@-~]")


class NonfiniteNames(ast.NodeTransformer):
    def visit_Name(self, node):
        values = {"inf": math.inf, "nan": math.nan,
                  "Infinity": math.inf, "NaN": math.nan}
        if node.id not in values:
            raise ValueError(f"unsupported literal name: {node.id}")
        return ast.copy_location(ast.Constant(values[node.id]), node)


def literal(text):
    """Allow Python literals and explicit inf/nan names; never eval expressions."""
    tree = NonfiniteNames().visit(ast.parse(text.strip(), mode="eval"))
    return ast.literal_eval(tree)


def literal_prefix(text):
    """Split a nested repr from a suffix, respecting quoted/escaped delimiters."""
    text = text.lstrip()
    if not text or text[0] not in "[{(":
        raise ValueError("expected a bracketed literal")
    stack, quote, escaped = [], None, False
    pairs = {"[": "]", "{": "}", "(": ")"}
    for index, char in enumerate(text):
        if quote:
            if escaped:
                escaped = False
            elif char == "\\":
                escaped = True
            elif char == quote:
                quote = None
        elif char in "\"'":
            quote = char
        elif char in pairs:
            stack.append(pairs[char])
        elif char in "]})":
            if not stack or stack.pop() != char:
                raise ValueError("unbalanced literal delimiters")
            if not stack:
                return text[:index + 1], text[index + 1:].strip()
    raise ValueError("unterminated literal")


def key_values(text):
    result = {}
    while text.strip():
        match = re.match(r"\s*(\w+)=", text)
        if not match:
            raise ValueError(f"expected key=value near {text[:80]!r}")
        key, rest = match.group(1), text[match.end():]
        if rest.startswith(("[", "{", "(")):
            value, text = literal_prefix(rest)
            result[key] = literal(value)
        else:
            value, _, text = rest.partition(" ")
            if key == "waiting":
                requests, tokens = value.split("/", 1)
                result["waiting_requests"], result["waiting_tokens"] = int(requests), int(tokens)
            else:
                try:
                    result[key] = literal(value)
                except (ValueError, SyntaxError):
                    # Stage startup fields include names such as radix=lru.
                    if not re.fullmatch(r"[\w.-]+", value):
                        raise
                    result[key] = value
    return result


def positive_int(value):
    return type(value) is int and value > 0


def finite_number(value):
    return type(value) in (int, float) and math.isfinite(value)


class Analyzer:
    def __init__(self, source):
        self.report = {
            "schema_version": 1, "source_log": source,
            "metadata": {"line_count": 0, "recognized_event_count": 0,
                         "parse_errors": [], "nonfinite_values": [],
                         "issues": [], "inactivity": []},
            "calibration_rows": [], "calibration_coverage": {},
            "calibration_elapsed": [], "graph_memory": [], "idle_checks": [],
            "summaries": [], "stages": {},
            "migrations": {"max": None, "observed_increments": 0, "resets": []},
        }
        self.previous_idle = {}
        self.previous_migrations = None

    def stage(self, rank):
        if rank is None or not str(rank).isdigit():
            raise ValueError("PP stage is missing or invalid")
        return self.report["stages"].setdefault(str(rank), {
            "sm_counts": None, "startup": None, "startup_count": 0,
            "latest_telemetry": None, "latest_summary_line": None,
            "summary_count": 0, "partition_observations": {},
            "partition_transitions_observed": 0, "partition_debug_counts": {},
            "partition_seconds": {}, "idle_check_count": 0,
            "idle_counter_max": {}, "idle_observed_increments": {},
            "idle_counter_resets": [],
        })

    def clean(self, value, line, path):
        if isinstance(value, float) and not math.isfinite(value):
            self.report["metadata"]["nonfinite_values"].append(
                {"line_number": line, "path": path, "value": str(value)})
            return None
        if isinstance(value, dict):
            if any(type(key) not in (str, int) for key in value):
                raise ValueError("dictionary keys must be strings or integers")
            return {str(key): self.clean(item, line, f"{path}.{key}")
                    for key, item in value.items()}
        if isinstance(value, (tuple, list)):
            return [self.clean(item, line, f"{path}[{index}]")
                    for index, item in enumerate(value)]
        if value is None or type(value) in (str, int, float, bool):
            return value
        raise ValueError(f"unsupported literal type: {type(value).__name__}")

    def issue(self, code, **fields):
        self.report["metadata"]["issues"].append({"code": code, **fields})

    def parse(self, raw, number):
        self.report["metadata"]["line_count"] = number
        line = ANSI.sub("", raw).strip()
        markers = ("MuxWise-PP decode calibration ", "MuxWise-PP graph memory ",
                   "MuxWise-PP idle self-check ", "MuxWise-PP summary stages=",
                   "MuxWise-PP stage=", "PD-Multiplexing enabled with ",
                   "MuxWise-PP partition=")
        marker = next((item for item in markers if item in line), None)
        if marker is None:
            return
        self.report["metadata"]["recognized_event_count"] += 1
        prefix, payload = line.split(marker, 1)
        pp = re.search(r"\bPP(\d+)\b", prefix)
        tp = re.search(r"\bTP(\d+)\b", prefix)
        stamp = re.search(r"\[(\d{4}-\d\d-\d\d \d\d:\d\d:\d\d(?:[.,]\d+)?)", prefix)
        context = {"line_number": number, "timestamp": stamp.group(1) if stamp else None,
                   "stage": int(pp.group(1)) if pp else None,
                   "tp_rank": int(tp.group(1)) if tp else None}
        try:
            if marker.endswith("summary stages="):
                self.summary(payload, context)
                return
            if marker.endswith("idle self-check "):
                self.idle(payload, context)
                return
            if marker.endswith("stage="):
                match = re.fullmatch(r"(\d+) layers=\[(\d+),(\d+)\) (.*)", payload)
                if not match:
                    raise ValueError("malformed stage startup")
                rank, start, end, rest = match.groups()
                stage = self.stage(rank)
                stage["startup_count"] += 1
                stage["startup"] = dict(key_values(rest), start_layer=int(start), end_layer=int(end))
                return
            stage = self.stage(context["stage"])
            if marker == "PD-Multiplexing enabled with ":
                match = re.fullmatch(r"(\d+) stream groups, sm_counts \(prefill_sm, decode_sm\): (.*)", payload)
                if not match:
                    raise ValueError("malformed stream-group configuration")
                counts = self.clean(literal(match.group(2)), number, "sm_counts")
                if not isinstance(counts, list) or len(counts) != int(match.group(1)):
                    raise ValueError("stream group count does not match SM tuples")
                if any(not isinstance(pair, list) or len(pair) != 2 or
                       any(type(n) is not int or n < 0 for n in pair) for pair in counts):
                    raise ValueError("invalid SM tuple")
                stage["sm_counts"] = counts
            elif marker.endswith("decode calibration "):
                if payload.startswith("elapsed="):
                    seconds = float(payload.removeprefix("elapsed=").removesuffix("s"))
                    if not finite_number(seconds) or seconds < 0:
                        raise ValueError("invalid calibration elapsed time")
                    self.report["calibration_elapsed"].append(dict(context, seconds=seconds))
                else:
                    self.calibration(payload, context)
            elif marker.endswith("graph memory "):
                values = self.clean(key_values(payload), number, "graph_memory")
                for key in ("partition", "graphs", "added_reserved_MiB", "total_reserved_MiB"):
                    if not finite_number(values.get(key)):
                        raise ValueError(f"invalid graph-memory field: {key}")
                self.report["graph_memory"].append(dict(context, **values))
            else:
                values = self.clean(key_values("partition=" + payload), number, "partition")
                index = str(values["partition"])
                counts = stage["partition_debug_counts"]
                counts[index] = counts.get(index, 0) + 1
        except (ValueError, TypeError, SyntaxError, KeyError, OverflowError, RecursionError) as error:
            self.report["metadata"]["parse_errors"].append(
                dict(context, event=marker.strip(), error=str(error), excerpt=payload[:240]))

    def calibration(self, payload, context):
        match = re.fullmatch(r"partition=(\d+) SMs=(\([^)]*\)) samples=(.*)", payload)
        if not match:
            raise ValueError("malformed calibration table")
        partition, sms_text, samples_text = match.groups()
        sms = literal(sms_text)
        if not isinstance(sms, tuple) or len(sms) != 2:
            raise ValueError("calibration SMs must be a pair")
        samples = self.clean(literal(samples_text), context["line_number"], "calibration.samples")
        if not isinstance(samples, list):
            raise ValueError("calibration samples must be a list")
        for sample in samples:
            if not isinstance(sample, list) or len(sample) != 3:
                raise ValueError("calibration sample must contain batch, context, seconds")
            batch, length, seconds = sample
            errors = []
            if not positive_int(batch) or not positive_int(length):
                errors.append("batch/context must be positive integers")
            if not finite_number(seconds) or seconds <= 0:
                errors.append("seconds must be positive and finite")
            if type(sms[0]) is not int or sms[0] < 0 or not positive_int(sms[1]):
                errors.append("invalid prefill/decode SM counts")
            row = dict(context, partition=int(partition), prefill_sms=sms[0], decode_sms=sms[1],
                       batch_size=batch, context_tokens=length, seconds=seconds,
                       valid=not errors, validation_errors=errors)
            self.report["calibration_rows"].append(self.clean(row, context["line_number"], "calibration.row"))

    def idle(self, payload, context):
        status, _, rest = payload.partition(" ")
        values = key_values(rest)
        rank = values.pop("stage", context["stage"])
        stage = self.stage(rank)
        if any(type(value) is not int or value < 0 for value in values.values()):
            raise ValueError("idle-check fields must be nonnegative integers")
        previous = self.previous_idle.get(str(rank), {})
        increments = {}
        for key in COUNTERS:
            if key not in values:
                continue
            current, prior = values[key], previous.get(key)
            increments[key] = current - prior if prior is not None and current >= prior else None
            stage["idle_counter_max"][key] = max(current, stage["idle_counter_max"].get(key, 0))
            stage["idle_observed_increments"][key] = stage["idle_observed_increments"].get(key, 0) + (increments[key] or 0)
            if prior is not None and current < prior:
                stage["idle_counter_resets"].append(
                    dict(line_number=context["line_number"], counter=key, previous=prior, current=current))
        self.previous_idle[str(rank)] = values
        stage["idle_check_count"] += 1
        self.report["idle_checks"].append(dict(context, stage=int(rank), status=status,
                                                counters=values, increments=increments))
        if status != "PASS":
            self.issue("idle_check_not_pass", stage=int(rank), line_number=context["line_number"], status=status)

    def summary(self, payload, context):
        text, tail = literal_prefix(payload)
        stages = self.clean(literal(text), context["line_number"], "summary.stages")
        if not isinstance(stages, dict):
            raise ValueError("summary stages must be a dictionary")
        global_values = self.clean(key_values(tail), context["line_number"], "summary.global")
        # Validate before changing cumulative state: a rejected snapshot must
        # not count as an observation or replace the last valid telemetry.
        for rank, telemetry in stages.items():
            if not str(rank).isdigit() or not isinstance(telemetry, dict):
                raise ValueError("stage telemetry must map integer ranks to dictionaries")
            if not isinstance(telemetry.get("timing", {}), dict):
                raise ValueError("stage timing must be a dictionary")
            if telemetry.get("partition") is not None and type(telemetry["partition"]) is not int:
                raise ValueError("partition must be an integer")
        migrations = global_values.get("migrations")
        if migrations is not None and (type(migrations) is not int or migrations < 0):
            raise ValueError("migrations must be a nonnegative integer")
        for rank, telemetry in stages.items():
            stage = self.stage(rank)
            previous = stage["latest_telemetry"]
            index = telemetry.get("partition")
            if index is not None:
                if previous is not None and previous.get("partition") != index:
                    stage["partition_transitions_observed"] += 1
                counts = stage["partition_observations"]
                counts[str(index)] = counts.get(str(index), 0) + 1
            stage["latest_telemetry"] = telemetry
            stage["latest_summary_line"] = context["line_number"]
            stage["summary_count"] += 1
            stage["partition_seconds"] = telemetry.get("timing", {}).get("partitions", {})
        if migrations is not None:
            stats = self.report["migrations"]
            stats["max"] = max(migrations, stats["max"] or 0)
            if self.previous_migrations is not None:
                if migrations >= self.previous_migrations:
                    stats["observed_increments"] += migrations - self.previous_migrations
                else:
                    stats["resets"].append(dict(line_number=context["line_number"],
                                                previous=self.previous_migrations, current=migrations))
            self.previous_migrations = migrations
        self.report["summaries"].append(dict(context, stages=stages, global_values=global_values))

    def finish(self):
        rows = self.report["calibration_rows"]
        batches = sorted({r["batch_size"] for r in rows if positive_int(r["batch_size"])})
        contexts = sorted({r["context_tokens"] for r in rows if positive_int(r["context_tokens"])})
        for rank, stage in self.report["stages"].items():
            selected = [r for r in rows if str(r["stage"]) == rank]
            observed = sorted({r["partition"] for r in selected})
            expected = None
            if stage["sm_counts"] is not None:
                expected = [i for i, pair in enumerate(stage["sm_counts"]) if pair[1] > 0]
            elif stage["startup"] and positive_int(stage["startup"].get("partitions")):
                expected = list(range(1, stage["startup"]["partitions"]))
            cells = Counter((r["partition"], r["batch_size"], r["context_tokens"])
                            for r in selected if r["valid"])
            missing = [{"partition": p, "batch_size": b, "context_tokens": c}
                       for p in (expected if expected is not None else observed)
                       for b in batches for c in contexts if not cells[p, b, c]]
            invalid = sum(not r["valid"] for r in selected)
            complete = None if expected is None or not batches or not contexts else not missing and not invalid
            self.report["calibration_coverage"][rank] = {
                "expected_partitions": expected, "observed_partitions": observed,
                "batch_sizes": batches, "contexts": contexts,
                "grid_basis": "union of observed calibration batch/context coordinates",
                "row_count": len(selected), "invalid_row_count": invalid,
                "all_values_valid": not invalid if selected else None,
                "missing_partitions": sorted(set(expected or []) - set(observed)),
                "missing_cells": missing, "complete": complete,
                "duplicate_cells": [dict(partition=p, batch_size=b, context_tokens=c, count=n)
                                    for (p, b, c), n in sorted(cells.items()) if n > 1],
            }
            if not selected:
                self.issue("no_decode_calibration", stage=int(rank))
            elif missing or invalid:
                self.issue("incomplete_or_invalid_calibration", stage=int(rank))
            if stage["startup_count"] > 1 or stage["idle_counter_resets"]:
                self.issue("possible_multiple_process_lifetimes", stage=int(rank))
            if not stage["idle_check_count"]:
                self.issue("no_idle_self_check_recorded", stage=int(rank))
            latest = stage["latest_telemetry"]
            if latest is None:
                self.issue("no_scheduling_summary", stage=int(rank))
            inactive = self.report["metadata"]["inactivity"]
            if stage["idle_counter_max"].get("overlap_launches") == 0:
                inactive.append(dict(stage=int(rank), mechanism="overlap_launches", evidence="maximum recorded counter is zero"))
            if latest:
                choices = latest.get("slice_choices")
                if isinstance(choices, list) and len(choices) > 1 and choices[1] == 0:
                    inactive.append(dict(stage=int(rank), mechanism="decode_aware_slice_shortening", evidence="shortened-slice counter is zero"))
                samples = latest.get("samples")
                if isinstance(samples, list) and len(samples) > 1 and finite_number(samples[1]) and samples[1] < 16:
                    inactive.append(dict(stage=int(rank), mechanism="prefill_rls_warmup", evidence=f"{samples[1]} complete observations; warmup requires 16"))
        if self.report["migrations"]["max"] == 0:
            self.report["metadata"]["inactivity"].append(
                dict(stage=0, mechanism="migration", evidence="maximum recorded migrations counter is zero"))
        if not self.report["metadata"]["recognized_event_count"]:
            self.issue("no_muxwise_events", message="This may be an ordinary baseline log.")
        self.report["metadata"]["stage_ids"] = sorted(map(int, self.report["stages"]))
        self.report["metadata"]["notes"] = [
            "Summary times/counters include warmup and idle periods; use aligned snapshots for trial deltas.",
            "Snapshot partition transitions are a lower bound; short visits can fall between summaries.",
            "overlap_launches records scheduling activity, not proven simultaneous GPU execution.",
            "Coverage detects holes in the observed grid, not wholly absent batch/context coordinates.",
            "No recorded activity can mean disabled, unexercised, or unobserved; it is not a correctness verdict.",
        ]
        return self.report


def analyze_lines(lines, source=None):
    analyzer = Analyzer(source)
    for number, line in enumerate(lines, 1):
        analyzer.parse(line, number)
    return analyzer.finish()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--log", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--csv", type=Path, help="Optional flattened calibration table")
    args = parser.parse_args()
    if args.log.resolve() in {args.output.resolve(), args.csv.resolve() if args.csv else None}:
        parser.error("Output paths must not overwrite the raw input log.")
    if args.csv and args.csv.resolve() == args.output.resolve():
        parser.error("JSON and CSV outputs must have different paths.")
    try:
        with args.log.open(encoding="utf-8", errors="replace") as stream:
            report = analyze_lines(stream, source=str(args.log.resolve()))
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(json.dumps(report, indent=2, allow_nan=False) + "\n")
        if args.csv:
            args.csv.parent.mkdir(parents=True, exist_ok=True)
            with args.csv.open("w", newline="") as stream:
                writer = csv.DictWriter(stream, fieldnames=CSV_FIELDS, extrasaction="ignore")
                writer.writeheader()
                writer.writerows(report["calibration_rows"])
    except (OSError, ValueError) as error:
        parser.error(str(error))
    print(f"Wrote {args.output}: {len(report['calibration_rows'])} calibration rows, "
          f"{len(report['summaries'])} summaries, "
          f"{len(report['metadata']['parse_errors'])} parse errors.")


if __name__ == "__main__":
    main()
