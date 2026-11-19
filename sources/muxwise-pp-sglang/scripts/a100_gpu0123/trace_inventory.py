#!/usr/bin/env python3
"""CPU-only audit and immutable-content references for the GPU 0--3 campaign.

Reads existing traces and model headers; never loads weights or uses CUDA.
The new namespace contains symlinks plus an exclusive-create audited manifest.
Use --verify to recheck an existing manifest without changing any files.
"""
import argparse
from collections import Counter
from datetime import datetime, timezone
from functools import lru_cache
import hashlib
import json
from pathlib import Path
import struct
import sys

import numpy as np


REPO = Path(__file__).resolve().parents[2]
WORKSPACE = REPO.parent
OLD_ROOT = REPO / "run/a100-4gpu-20261114/traces"
NEW_ROOT = REPO / "run/a100-gpu0123-20261114/traces"
MODELS = WORKSPACE / "models"
BUILDER = WORKSPACE / "muxwise-pp-experiments/bench/build_trace.py"


def sha256(path):
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def require(condition, message):
    if not condition:
        raise ValueError(message)


def model_audit(path):
    provenance = json.loads((path / "download-provenance.json").read_text())
    sizes, small_hashes = {}, {}
    for entry in provenance["files"]:
        target = path / entry["file"]
        require(target.is_file(), f"Missing model file: {target}")
        sizes[entry["file"]] = target.stat().st_size
        require(sizes[entry["file"]] == entry["bytes"], f"Model size mismatch: {target}")
        if target.suffix != ".safetensors":
            small_hashes[entry["file"]] = sha256(target)
            require(small_hashes[entry["file"]] == entry["sha256"], f"Model hash mismatch: {target}")
    index = json.loads((path / "model.safetensors.index.json").read_text())
    tensors, dtypes, tensor_bytes = {}, Counter(), 0
    for filename in sorted(set(index["weight_map"].values())):
        with (path / filename).open("rb") as stream:
            header_length = struct.unpack("<Q", stream.read(8))[0]
            header = json.loads(stream.read(header_length))
        ends = []
        for name, tensor in header.items():
            if name == "__metadata__":
                continue
            require(name not in tensors, f"Duplicate tensor: {name}")
            tensors[name] = filename
            dtypes[tensor["dtype"]] += 1
            start, end = tensor["data_offsets"]
            tensor_bytes += end - start
            ends.append(end)
        require(max(ends) + 8 + header_length == (path / filename).stat().st_size,
                f"Incomplete tensor payload: {filename}")
    require(tensors == index["weight_map"], f"Tensor index mismatch: {path}")
    require(set(dtypes) == {"BF16"}, f"Unexpected model precision: {path}")
    require(tensor_bytes == index["metadata"]["total_size"], f"Tensor size mismatch: {path}")
    sys.path.insert(0, str(BUILDER.parent))
    from build_trace import load_vocabulary
    ordinary, bos, excluded = load_vocabulary(path)
    require(np.array_equal(ordinary, np.arange(128000, dtype=np.int32)),
            f"Unexpected ordinary vocabulary: {path}")
    require(bos == 128000 and excluded == 256, f"Unexpected special tokens: {path}")
    return dict(path=str(path), official_revision=provenance["official_revision"],
                provenance_sha256=sha256(path / "download-provenance.json"),
                shard_count=len(set(index["weight_map"].values())), tensor_count=len(tensors),
                tensor_bytes=tensor_bytes, dtypes=dict(dtypes), all_file_sizes=sizes,
                nonweight_file_sha256=small_hashes, fresh_header_index_checks="PASS",
                current_full_weight_sha256_recomputed=False,
                prior_full_weight_sha256_verified=provenance["verification"]["sha256_all_files"],
                prior_official_weight_hashes_match=provenance["weight_hashes_match_official"],
                ordinary_vocab_size=len(ordinary), bos_token_id=bos, excluded_tokens=excluded,
                ordinary_vocab_int32_sha256=hashlib.sha256(ordinary.tobytes()).hexdigest())


@lru_cache(maxsize=8192)
def token_block(hash_id):
    # Independently reconstruct the builder's ordinary-ID sampling, without
    # calling its block-generation code or decoding/retokenizing synthetic text.
    require(type(hash_id) is int and hash_id >= 0, "Expected nonnegative integer source hashes")
    return np.random.Generator(np.random.PCG64(hash_id)).integers(0, 128000, size=512)


def validate_generated(path, selected, args):
    indices, inputs, outputs, arrivals = [], [], [], []
    with path.open() as stream:
        for position, line in enumerate(stream):
            require(position < len(selected), f"Unexpected extra generated row: {path}")
            generated = json.loads(line)
            original_index, original = selected[position]
            require(generated["idx"] == original_index, f"Source ordering mismatch: {path}:{position}")
            require(generated["input_len"] == original["input_length"] and
                    generated["output_len"] == original["output_length"], f"Length changed: {path}:{position}")
            require(generated["hash_ids"] == original["hash_ids"], f"Prefix order changed: {path}:{position}")
            expected_arrival = (original["timestamp"] - args["start_sec"] * 1000) / 1000 * args["stretch"]
            require(generated["arrival"] == expected_arrival, f"Arrival mismatch: {path}:{position}")
            ids = generated["input_ids"]
            require(len(ids) == original["input_length"] and ids[0] == 128000,
                    f"Length or BOS mismatch: {path}:{position}")
            require(all(type(x) is int for x in ids), f"Noninteger token: {path}:{position}")
            actual = np.asarray(ids, dtype=np.int64)
            offset = 0
            for hash_id in original["hash_ids"]:
                count = min(512, len(ids) - offset)
                if count == 0:
                    break
                expected = token_block(hash_id)[:count]
                skip = 1 if offset == 0 else 0
                require(np.array_equal(actual[offset + skip:offset + count], expected[skip:]),
                        f"Hash token reconstruction mismatch: {path}:{position}:{offset}")
                offset += count
            require(offset == len(ids), f"Unexpected padding required: {path}:{position}")
            indices.append(original_index)
            inputs.append(len(ids))
            outputs.append(generated["output_len"])
            arrivals.append(generated["arrival"])
    require(len(indices) == len(selected), f"Missing generated rows: {path}")
    require(len(set(indices)) == len(indices), f"Duplicated source requests: {path}")
    return dict(requests=len(indices), first_source_idx=indices[0], last_source_idx=indices[-1],
                source_indices_sha256=hashlib.sha256(json.dumps(indices, separators=(",", ":")).encode()).hexdigest(),
                input_tokens=sum(inputs), output_tokens=sum(outputs),
                max_input_len=max(inputs), max_output_len=max(outputs),
                max_total_len=max(a + b for a, b in zip(inputs, outputs)),
                first_arrival_sec=arrivals[0], last_arrival_sec=arrivals[-1],
                base_arrival_window_sec=args["duration_sec"] * args["stretch"],
                duplicate_source_rows=0, length_clipping=False, padding_tokens=0,
                all_rows_and_hash_blocks_validated=True)


def audit(old_root):
    sources, raw = [], {}
    for entry in json.loads((old_root / "source/provenance.json").read_text()):
        path = old_root / "source" / Path(entry["path"]).name
        digest = sha256(path)
        require(digest == entry["sha256"] and path.stat().st_size == entry["bytes"],
                f"Source provenance mismatch: {path}")
        rows = [json.loads(line) for line in path.read_text().splitlines()]
        workload = path.name.removesuffix("_trace.jsonl")
        raw[workload] = rows
        valid = [row for row in rows if row["input_length"] + row["output_length"] <= 32768]
        require(all(row["input_length"] >= 1 and row["output_length"] >= 0 for row in rows), "Invalid source lengths")
        timestamps = [row["timestamp"] for row in rows]
        require(timestamps == sorted(timestamps), f"Unsorted raw timestamps: {path}")
        sources.append(dict(path=str(path), workload=workload, sha256=digest, bytes=path.stat().st_size,
                            source_url=entry["url"], source_revision=entry["revision"], requests=len(rows),
                            retained_le_32768=len(valid), excluded_gt_32768=len(rows) - len(valid),
                            source_timestamp_span_ms=[min(timestamps), max(timestamps)],
                            max_retained_total_len=max(row["input_length"] + row["output_length"] for row in valid)))
    old_records = []
    for name in ["generated-manifest.json", "heldout-manifest.json"]:
        old_records.extend(json.loads((old_root / name).read_text())["records"])
    validation_cache, index_sets, records = {}, {}, []
    for entry in old_records:
        args = entry["args"]
        path = old_root / "generated" / (entry["name"] + ".jsonl")
        digest = sha256(path)
        require(digest == entry["sha256"] and path.stat().st_size == entry["bytes"],
                f"Generated trace hash/size mismatch: {path}")
        require(args["max_total_len"] == 32768 and args["block_size"] == 512, "Wrong trace construction parameters")
        selected = [(i, row) for i, row in enumerate(raw[entry["workload"]])
                    if args["start_sec"] * 1000 <= row["timestamp"] < (args["start_sec"] + args["duration_sec"]) * 1000
                    and row["input_length"] + row["output_length"] <= 32768]
        selected.sort(key=lambda item: (item[1]["timestamp"], item[0]))
        key = (entry["workload"], entry["purpose"])
        ids = {i for i, row in selected}
        if key in index_sets:
            require(index_sets[key] == ids, "Models use different source rows")
        index_sets[key] = ids
        if digest not in validation_cache:
            validation_cache[digest] = validate_generated(path, selected, args)
        checked = validation_cache[digest]
        require(checked["requests"] == entry["requests"], "Manifest request count mismatch")
        stats_path = path.with_suffix(".stats.json")
        stats = json.loads(stats_path.read_text())
        require(stats["duration_sec"] == checked["base_arrival_window_sec"] and
                stats["requests"] == checked["requests"] and stats["padded_tokens"] == 0,
                f"Generated stats mismatch: {stats_path}")
        records.append(dict(name=entry["name"], model=entry["model"], workload=entry["workload"],
                            purpose=entry["purpose"], original_path=str(path), sha256=digest,
                            bytes=path.stat().st_size, original_stats_path=str(stats_path),
                            stats_sha256=sha256(stats_path), source_start_sec=args["start_sec"],
                            source_end_exclusive_sec=args["start_sec"] + args["duration_sec"],
                            stretch=args["stretch"], **checked))
    disjoint = []
    for workload in raw:
        for left, right in [("dev", "formal"), ("dev", "heldout"), ("formal", "heldout")]:
            overlap = len(index_sets[(workload, left)] & index_sets[(workload, right)])
            require(overlap == 0, f"Development/formal leakage: {workload}:{left}:{right}")
            disjoint.append(dict(workload=workload, left=left, right=right, overlapping_source_rows=overlap))
    for workload in raw:
        for purpose in ["dev", "formal", "heldout"]:
            pairs = [r for r in records if r["workload"] == workload and r["purpose"] == purpose]
            require(len(pairs) == 2 and pairs[0]["sha256"] == pairs[1]["sha256"], "Model token-ID trace mismatch")
    models = [model_audit(MODELS / name) for name in ["Llama-3.1-8B-Instruct", "Llama-3.3-70B-Instruct"]]
    return dict(schema_version=1, audited_at_utc=datetime.now(timezone.utc).isoformat(),
                status="PASS", builder=dict(path=str(BUILDER), sha256=sha256(BUILDER)),
                sources=sources, models=models, records=records, source_row_disjointness=disjoint,
                prefix_definition="512 ordinary-ID tokens per source hash; deterministic PCG64; BOS replaces token zero; exact input/output lengths; no clipping, padding, text retokenization, or duplicated source rows.",
                model_pairs_byte_identical=True,
                limitations=["Original private text/tokens are unavailable; reconstruction preserves anonymized lengths and prefix block structure.",
                             "Theoretical hash reuse is not measured contiguous radix-cache reuse; flush before each measurement.",
                             "Current model tensor payload hashes were not recomputed; prior full hashes are in download provenance.",
                             "Symlinks preserve storage, not write protection. Recheck content hashes before measurement.",
                             "High-rate saturation and SLO crossover remain unmeasured; proposed rates are starting brackets, not selected parameters."])


def rate_plan(records):
    proposals = {}
    for model, speedups in [("Llama-3.1-8B-Instruct", [1, 2, 4, 8, 16]),
                            ("Llama-3.3-70B-Instruct", [0.5, 1, 2, 4, 8])]:
        rows = []
        for workload in ["toolagent", "conversation"]:
            dev = next(r for r in records if r["model"] == model and r["workload"] == workload and r["purpose"] == "dev")
            held = next(r for r in records if r["model"] == model and r["workload"] == workload and r["purpose"] == "heldout")
            for factor in speedups:
                rows.append(dict(workload=workload, speedup=factor, time_scale=1 / factor,
                                 dev_window_sec=dev["base_arrival_window_sec"] / factor,
                                 dev_planned_requests_per_sec=dev["requests"] * factor / dev["base_arrival_window_sec"],
                                 dev_is_sustained_180sec=dev["base_arrival_window_sec"] / factor >= 180,
                                 heldout_window_sec=held["base_arrival_window_sec"] / factor,
                                 heldout_planned_requests_per_sec=held["requests"] * factor / held["base_arrival_window_sec"],
                                 heldout_is_sustained_180sec=held["base_arrival_window_sec"] / factor >= 180))
        proposals[model] = rows
    return dict(status="proposed_bounded_starting_points_not_selected", minimum_sustained_window_sec=180,
                maximum_sustained_heldout_speedup_at_180sec=3900 / 180,
                initial_common_upper_speedup=16,
                policy="Choose rates using development only. Short dev screens above 3.333x are not sustained evidence. Freeze parameters/rates before untouched formal and heldout runs. Use heldout for >=180s high-rate measurements; never loop, duplicate, or feedback-tune on heldout.",
                lower_70b_start_reason="Conservative exploratory bracket for greater per-token model compute; this is not a measured throughput estimate.",
                proposals=proposals)


def link_exact(source, destination):
    destination.parent.mkdir(parents=True, exist_ok=True)
    if destination.is_symlink():
        require(destination.resolve() == source.resolve(), f"Refusing to replace symlink: {destination}")
    else:
        require(not destination.exists(), f"Refusing to replace file: {destination}")
        destination.symlink_to(source)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--old-root", type=Path, default=OLD_ROOT)
    parser.add_argument("--out-root", type=Path, default=NEW_ROOT)
    parser.add_argument("--verify", action="store_true")
    args = parser.parse_args()
    manifest_path = args.out_root / "manifest.json"
    if args.verify:
        manifest = json.loads(manifest_path.read_text())
        for row in manifest["sources"]:
            require(sha256(Path(row["path"])) == row["sha256"], f"Changed source: {row['path']}")
        for row in manifest["records"]:
            require(sha256(Path(row["path"])) == row["sha256"], f"Changed trace: {row['path']}")
            require(sha256(Path(row["stats_path"])) == row["stats_sha256"], f"Changed stats: {row['stats_path']}")
        require(sha256(Path(manifest["builder"]["path"])) == manifest["builder"]["sha256"], "Builder changed")
        print(json.dumps(dict(status="PASS", checked_traces=len(manifest["records"]), manifest=str(manifest_path))))
        return
    require(not manifest_path.exists(), f"Manifest exists; use --verify: {manifest_path}")
    manifest = audit(args.old_root.resolve())
    manifest["rate_plan"] = rate_plan(manifest["records"])
    for row in manifest["records"]:
        path = args.out_root.resolve() / "generated" / Path(row["original_path"]).name
        link_exact(Path(row["original_path"]), path)
        link_exact(Path(row["original_stats_path"]), path.with_suffix(".stats.json"))
        row.update(path=str(path), stats_path=str(path.with_suffix(".stats.json")))
    for source in manifest["sources"]:
        link_exact(Path(source["path"]), args.out_root.resolve() / "source" / Path(source["path"]).name)
    with manifest_path.open("x") as stream:
        json.dump(manifest, stream, indent=2, allow_nan=False)
        stream.write("\n")
    print(json.dumps(dict(status="PASS", independently_validated_unique_traces=6,
                          model_specific_trace_references=len(manifest["records"]),
                          source_row_disjointness="PASS", manifest=str(manifest_path))))


if __name__ == "__main__":
    main()
