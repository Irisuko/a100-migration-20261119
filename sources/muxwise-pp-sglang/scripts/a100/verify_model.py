#!/usr/bin/env python3
"""Wait for an existing download, then verify SHA256SUMS and safetensors.

Reads the model directory without modifying its files or download processes.
Only the requested JSON report is written. No GPU or ML packages are imported.
"""

import argparse
import datetime
import hashlib
import json
from pathlib import Path
import struct
import subprocess
import time


DTYPE_BYTES = {
    "BOOL": 1, "U8": 1, "I8": 1, "F8_E4M3": 1, "F8_E5M2": 1,
    "I16": 2, "U16": 2, "F16": 2, "BF16": 2,
    "I32": 4, "U32": 4, "F32": 4, "I64": 8, "U64": 8, "F64": 8,
}


def inside(model, name):
    path = (model / name).resolve()
    if not path.is_relative_to(model):
        raise ValueError("Path escapes model directory: " + name)
    return path


def inspect_shard(path):
    size = path.stat().st_size
    with path.open("rb") as source:
        raw_length = source.read(8)
        if len(raw_length) != 8:
            raise ValueError("Truncated safetensors prefix: " + path.name)
        header_length = struct.unpack("<Q", raw_length)[0]
        if header_length > min(size - 8, 100_000_000):
            raise ValueError("Invalid safetensors header length: " + path.name)
        header = json.loads(source.read(header_length))
    tensors = {k: v for k, v in header.items() if k != "__metadata__"}
    if not tensors:
        raise ValueError("No tensors in " + path.name)
    spans = []
    for name, tensor in tensors.items():
        start, end = tensor["data_offsets"]
        if not isinstance(start, int) or not isinstance(end, int) or start < 0 or end < start:
            raise ValueError("Invalid tensor offsets: " + name)
        elements = 1
        for dimension in tensor["shape"]:
            if not isinstance(dimension, int) or dimension < 0:
                raise ValueError("Invalid tensor shape: " + name)
            elements *= dimension
        dtype_bytes = DTYPE_BYTES[tensor["dtype"]]
        if end - start != elements * dtype_bytes:
            raise ValueError("Tensor shape/byte count mismatch: " + name)
        spans.append((start, end, name))
    offset = 0
    for start, end, name in sorted(spans):
        if start != offset:
            raise ValueError("Noncontiguous or overlapping tensor: " + name)
        offset = end
    if offset + 8 + header_length != size:
        raise ValueError("Safetensors file length mismatch: " + path.name)
    return {
        "file": path.name,
        "bytes": size,
        "header_bytes": header_length,
        "tensor_data_bytes": offset,
        "tensor_count": len(tensors),
        "dtypes": sorted({t["dtype"] for t in tensors.values()}),
        "header_bounds_valid": True,
        "tensor_shapes_match_bytes": True,
    }, set(tensors)


def verify(model, report):
    checksum_file = model / "SHA256SUMS"
    checksums = {}
    for line in checksum_file.read_text().splitlines():
        if not line.strip():
            continue
        expected, name = line.split(maxsplit=1)
        name = name.lstrip("*")
        if len(expected) != 64 or any(c not in "0123456789abcdefABCDEF" for c in expected):
            raise ValueError("Invalid SHA-256 entry")
        if name in checksums:
            raise ValueError("Duplicate checksum filename: " + name)
        inside(model, name)
        checksums[name] = expected.lower()
    before = {name: (inside(model, name).stat().st_size,
                     inside(model, name).stat().st_mtime_ns) for name in checksums}
    print("Download sidecars are gone; checking all SHA256SUMS entries", flush=True)
    checked = subprocess.run(
        ["sha256sum", "--check", "SHA256SUMS"], cwd=model,
        capture_output=True, text=True,
    )
    report["sha256sum"] = {
        "command": ["sha256sum", "--check", "SHA256SUMS"],
        "cwd": str(model), "returncode": checked.returncode,
        "stdout": checked.stdout, "stderr": checked.stderr,
        "expected_hashes": checksums,
        "checksum_file_sha256": hashlib.sha256(checksum_file.read_bytes()).hexdigest(),
    }
    if checked.returncode:
        raise ValueError("SHA256SUMS validation failed")
    index = json.loads((model / "model.safetensors.index.json").read_text())
    weight_map = index["weight_map"]
    shard_names = sorted(set(weight_map.values()))
    shards, all_tensors = [], set()
    for name in shard_names:
        if name not in checksums:
            raise ValueError("Shard is not covered by SHA256SUMS: " + name)
        record, tensor_names = inspect_shard(inside(model, name))
        expected_names = {key for key, value in weight_map.items() if value == name}
        if tensor_names != expected_names:
            raise ValueError("Safetensors index/header mismatch: " + name)
        if all_tensors.intersection(tensor_names):
            raise ValueError("Duplicate tensor names across shards")
        all_tensors.update(tensor_names)
        shards.append(record)
    total_data_bytes = sum(s["tensor_data_bytes"] for s in shards)
    if total_data_bytes != index["metadata"]["total_size"]:
        raise ValueError("Index total_size disagrees with tensor data bytes")
    after = {name: (inside(model, name).stat().st_size,
                    inside(model, name).stat().st_mtime_ns) for name in checksums}
    if before != after or list(model.glob("*.aria2")):
        raise ValueError("Model files changed during verification")
    config = json.loads((model / "config.json").read_text())
    report.update({
        "shards": shards, "all_indexed_tensors_found": True,
        "tensor_count": len(all_tensors), "total_tensor_data_bytes": total_data_bytes,
        "checksum_file_count": len(checksums), "files_unchanged_during_check": True,
        "config": {key: config.get(key) for key in [
            "architectures", "model_type", "torch_dtype", "num_hidden_layers",
            "hidden_size", "num_attention_heads", "num_key_value_heads", "vocab_size",
        ]},
    })
    metadata = model.parent / ".llama-download" / (model.name + ".hf.json")
    if metadata.is_file():
        source = json.loads(metadata.read_text())
        report["source"] = {"repo_id": source.get("id"), "revision": source.get("sha")}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--wait-seconds", type=float, default=900)
    parser.add_argument("--poll-seconds", type=float, default=15)
    args = parser.parse_args()
    if args.wait_seconds < 0 or not 0 < args.poll_seconds <= 60:
        parser.error("wait must be nonnegative; polling interval must be in (0, 60]")
    model = args.model.resolve()
    report = {"model_directory": str(model), "started_at_utc":
              datetime.datetime.now(datetime.timezone.utc).isoformat(), "result": "FAIL"}
    started = time.monotonic()
    deadline = started + args.wait_seconds
    previous = None
    try:
        while True:
            sidecars = sorted(p.name for p in model.glob("*.aria2"))
            if not sidecars:
                break
            status_path = model.parent / ".llama-download" / "status.json"
            status = {}
            try:
                status = json.loads(status_path.read_text()).get("models", {}).get(model.name, {})
            except (OSError, ValueError):
                pass
            current = (len(sidecars), int(status.get("percent", 0)) // 5)
            if current != previous:
                print(json.dumps({"waiting_for_download": True,
                                  "remaining_sidecars": len(sidecars),
                                  "status": status}), flush=True)
                previous = current
            if time.monotonic() >= deadline:
                raise TimeoutError("Download still has aria2 sidecars after bounded wait")
            time.sleep(min(args.poll_seconds, deadline - time.monotonic()))
        verify(model, report)
        report["result"] = "PASS"
    except Exception as error:
        report["error"] = type(error).__name__ + ": " + str(error)
    report["elapsed_seconds"] = time.monotonic() - started
    report["finished_at_utc"] = datetime.datetime.now(datetime.timezone.utc).isoformat()
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps({"result": report["result"], "report": str(args.output),
                      "error": report.get("error"),
                      "tensor_count": report.get("tensor_count")}), flush=True)
    raise SystemExit(0 if report["result"] == "PASS" else 1)


if __name__ == "__main__":
    main()
