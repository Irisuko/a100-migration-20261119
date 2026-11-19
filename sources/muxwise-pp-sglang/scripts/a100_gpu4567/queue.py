#!/usr/bin/env python3
"""Resumable serial controller queue; formal tasks require an exact freeze.

tasks.json: {"tasks": [{"id": "unique-label", "phase": "development",
  "argv": ["--model", "8b", "--family", "plain", "--out", "..."],
  "timeout": 2400, "dependencies": []}], "source_sha256": {"path": "sha256"}}

freeze.json: {"status": "frozen", "source_sha256": {"path": "sha256"},
  "tasks": {"formal-label": {"phase": "formal_main", "argv": [...],
    "timeout": 2400, "trace_sha256": "sha256"}}}

Exact argv freezes server parameters and replay rates. Three independent formal
deployments require three distinct task IDs and output directories. This tool
does not select configurations, create freezes, retry failed labels, or delete
measurement artifacts. Unknown phases require a freeze by default.
"""
# This filename is on server_run.py's import path. Preserve Python's queue API
# for concurrent.futures and multiprocessing instead of shadowing it with an
# incompatible application module.
import importlib.util as _import_util
from pathlib import Path
import sysconfig as _sysconfig
_spec = _import_util.spec_from_file_location("_a100_stdlib_queue", Path(_sysconfig.get_path("stdlib")) / "queue.py")
_stdlib_queue = _import_util.module_from_spec(_spec)
_spec.loader.exec_module(_stdlib_queue)
for _name in _stdlib_queue.__all__:
    globals()[_name] = getattr(_stdlib_queue, _name)
__all__ = _stdlib_queue.__all__

import argparse
import fcntl
import hashlib
import json
import math
import os
import re
import signal
import subprocess
import sys
import tempfile
import time


REPO = Path(__file__).resolve().parents[2]
ROOT = REPO / "run/a100-gpu0123-20261114/pool-gpu4567"
PYTHON = REPO / ".venv/bin/python"
HERE = Path(__file__).resolve().parent
NONFORMAL = {"support", "feasibility", "smoke", "preflight", "correctness", "model_correctness",
             "topology_feasibility", "development", "development_selection",
             "rate_calibration", "pilot"}
TERMINAL_FAILURE = {"failed", "interrupted", "timed_out"}


def require(condition, message):
    if not condition:
        raise ValueError(message)


def canonical_hash(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(",", ":")).encode()).hexdigest()


def file_hash(path):
    value = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for chunk in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            value.update(chunk)
    return value.hexdigest()


def resolve(path):
    path = Path(path)
    return (path if path.is_absolute() else REPO / path).resolve()


def atomic_json(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(mode="w", dir=path.parent, delete=False) as stream:
        json.dump(value, stream, indent=2, allow_nan=False)
        stream.write("\n")
        stream.flush()
        os.fsync(stream.fileno())
    os.replace(stream.name, path)


def option(argv, name):
    values = []
    for i, value in enumerate(argv):
        if value == "--extra":
            break
        if value == name:
            require(i + 1 < len(argv) and not argv[i + 1].startswith("--"), f"Missing value for {name}")
            values.append(argv[i + 1])
        elif value.startswith(name + "="):
            values.append(value[len(name) + 1:])
    require(len(values) <= 1, f"Repeated option {name}")
    return values[0] if values else None


def validate_plan(plan, root):
    require(Path(root).resolve() == ROOT.resolve(), "Fixed pool4567 root required")
    require(isinstance(plan, dict) and isinstance(plan.get("tasks"), list) and plan["tasks"], "Nonempty tasks list required")
    ids, outputs = set(), set()
    for task in plan["tasks"]:
        label = task.get("id", "")
        require(isinstance(label, str) and re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9_.-]{0,95}", label), "Invalid task ID")
        require(label not in ids, f"Duplicate task ID: {label}")
        ids.add(label)
        require(isinstance(task.get("phase"), str) and task["phase"], f"Missing phase: {label}")
        argv = task.get("argv")
        require(isinstance(argv, list) and argv and all(isinstance(x, str) and x for x in argv), f"Invalid argv: {label}")
        require(isinstance(task.get("timeout"), (int, float)) and math.isfinite(task["timeout"]) and task["timeout"] > 0,
                f"Positive finite timeout required: {label}")
        out = option(argv, "--out")
        require(out is not None, f"Explicit --out required: {label}")
        out = resolve(out)
        require(root in out.parents and (root / "runs") not in out.parents and out != root / "runs",
                f"--out must be a measurement directory under campaign root, outside controller runs: {out}")
        require(out not in outputs, f"Duplicate output directory: {out}")
        outputs.add(out)
        dependencies = task.get("dependencies", [])
        require(isinstance(dependencies, list) and all(isinstance(x, str) for x in dependencies), f"Invalid dependencies: {label}")
    by_id = {task["id"]: task for task in plan["tasks"]}
    visited, active = set(), set()
    def visit(label):
        require(label in by_id, f"Unknown dependency: {label}")
        require(label not in active, f"Dependency cycle: {label}")
        if label in visited:
            return
        active.add(label)
        for dependency in by_id[label].get("dependencies", []):
            visit(dependency)
        active.remove(label)
        visited.add(label)
    for label in by_id:
        visit(label)
    return by_id


def verify_hashes(mapping):
    require(isinstance(mapping, dict), "source_sha256 must map file paths to hashes")
    for filename, expected in mapping.items():
        require(isinstance(expected, str) and re.fullmatch(r"[0-9a-f]{64}", expected), f"Invalid hash: {filename}")
        require(file_hash(resolve(filename)) == expected, f"Source hash mismatch: {filename}")


def check_task(task, plan, freeze_path):
    sources = dict(plan.get("source_sha256", {}))
    for key, value in task.get("source_sha256", {}).items():
        require(key not in sources or sources[key] == value, f"Conflicting source hash: {key}")
        sources[key] = value
    verify_hashes(sources)
    spec = _import_util.spec_from_file_location('_queue_pool_evidence', HERE / 'pool_evidence.py')
    pool = _import_util.module_from_spec(spec); spec.loader.exec_module(pool)
    pool.validate_source_pins(sources)
    trace = option(task["argv"], "--trace")
    trace_digest = file_hash(resolve(trace)) if trace else None
    if task.get("trace_sha256"):
        require(trace_digest == task["trace_sha256"], f"Trace hash mismatch: {trace}")
    evidence = dict(source_sha256=sources, trace_sha256=trace_digest)
    if task["phase"] not in NONFORMAL:
        require(freeze_path is not None, f"Explicit freeze required for phase {task['phase']}")
        freeze = json.loads(Path(freeze_path).read_text())
        require(freeze.get("status") == "frozen", "Freeze status must be frozen")
        frozen_sources = freeze.get("source_sha256")
        require(isinstance(frozen_sources, dict) and frozen_sources, "Freeze source hashes required")
        verify_hashes(frozen_sources)
        require(all(frozen_sources.get(key) == value for key, value in sources.items()), "Freeze source fingerprint differs")
        frozen_task = freeze.get("tasks", {}).get(task["id"])
        require(isinstance(frozen_task, dict), f"Task absent from freeze: {task['id']}")
        for field in ["phase", "argv", "timeout"]:
            require(frozen_task.get(field) == task[field], f"Frozen {field} differs for {task['id']}")
        require(trace is not None and frozen_task.get("trace_sha256") == trace_digest,
                f"Frozen trace hash missing or mismatched: {trace}")
        evidence.update(source_sha256=frozen_sources, freeze_sha256=file_hash(freeze_path), freeze_path=str(resolve(freeze_path)))
    evidence["source_fingerprint"] = canonical_hash(evidence["source_sha256"])
    return evidence


def commands(task, root):
    server = ["taskset", "-c", "32-63,96-127", str(PYTHON), str(HERE / "server_run.py"), *task["argv"]]
    controller = [str(PYTHON), str(HERE / "control.py"), "--root", str(root), "run",
                  "--label", task["id"], "--timeout", str(task["timeout"]), "--", *server[3:]]
    return controller, server


def collect_artifacts(task, root, record):
    outcome_path = root / "runs" / task["id"] / "outcome.json"
    record["outcome_path"] = str(outcome_path)
    summary = resolve(option(task["argv"], "--out")) / "trace_summary.json"
    record["request_summary_path"] = str(summary)
    if summary.is_file():
        record["request_summary_sha256"] = file_hash(summary)
    if outcome_path.is_file():
        outcome = json.loads(outcome_path.read_text())
        record["controller_outcome"] = outcome
        record["status"] = outcome.get("status", "failed")
        require(record["status"] in {"completed", *TERMINAL_FAILURE}, "Unknown controller outcome status")
        if record["status"] == "completed":
            command_path = outcome_path.with_name("command.json")
            require(command_path.is_file(), "Completed outcome lacks command provenance")
            command = json.loads(command_path.read_text())
            require(command["argv"] == commands(task, root)[1] and command["timeout"] == task["timeout"],
                    "Existing completed outcome command does not match task")
    return outcome_path.is_file()


def launch_controller(command, log_path):
    """Only live boundary. Controller handles timeout, ownership and cleanup."""
    interrupted, previous = [], {}
    with log_path.open("x") as log:
        child = subprocess.Popen(command, cwd=REPO, stdout=log, stderr=subprocess.STDOUT,
                                 start_new_session=True)
        def forward(signum, _frame):
            interrupted.append(signum)
            if child.poll() is None:
                child.send_signal(signum)
        for signum in [signal.SIGINT, signal.SIGTERM]:
            previous[signum] = signal.signal(signum, forward)
        try:
            code = child.wait()
        finally:
            for signum, handler in previous.items():
                signal.signal(signum, handler)
    return 128 + interrupted[0] if interrupted else code


def run_queue(tasks_path, root=ROOT, queue_id=None, freeze_path=None,
              continue_independent=False, launcher=None):
    root, tasks_path = resolve(root), resolve(tasks_path)
    plan = json.loads(tasks_path.read_text())
    by_id = validate_plan(plan, root)
    queue_id = queue_id or tasks_path.stem
    require(re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9_.-]{0,95}", queue_id), "Invalid queue ID")
    queue_root = root / "queues" / queue_id
    queue_root.mkdir(parents=True, exist_ok=True)
    state_path = queue_root / "status.json"
    digest = canonical_hash(plan)
    launcher = launcher or launch_controller
    with (queue_root / "queue.lock").open("a") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        if state_path.exists():
            state = json.loads(state_path.read_text())
            require(state["plan_sha256"] == digest, "Task plan changed; use a new queue ID and new task labels")
        else:
            state = dict(schema_version=1, queue_id=queue_id, tasks_path=str(tasks_path),
                         plan_sha256=digest, status="pending", tasks={label: {"status": "pending"} for label in by_id})
            atomic_json(queue_root / "plan.json", plan)
        freeze_digest = file_hash(freeze_path) if freeze_path else None
        previous_freeze = state.get("freeze_sha256")
        require(previous_freeze is None or previous_freeze == freeze_digest, "Queue freeze changed")
        if freeze_digest:
            state["freeze_sha256"] = freeze_digest
        state["updated_unix"] = time.time()
        # Reconcile durable outcomes before deciding what is runnable. Existing
        # output without a complete outcome is evidence to preserve, not retry.
        for label, task in by_id.items():
            record = state["tasks"][label]
            try:
                terminal_failure = record["status"] if record["status"] in TERMINAL_FAILURE else None
                found = collect_artifacts(task, root, record)
                if terminal_failure:
                    record["status"] = terminal_failure
                if not found and ((root / "runs" / label).exists() or record["status"] == "running"):
                    record.update(status="interrupted", error="Prior execution has no complete outcome; use a new task ID")
                elif not found and record["status"] == "completed":
                    record.update(status="failed", error="Completed controller evidence disappeared")
            except (ValueError, OSError, KeyError, TypeError) as error:
                record.update(status="failed", error=str(error))
        state["status"] = "running"
        atomic_json(state_path, state)
        while True:
            failed = [label for label, row in state["tasks"].items() if row["status"] in TERMINAL_FAILURE]
            if failed and not continue_independent:
                break
            pending = [task for task in plan["tasks"] if state["tasks"][task["id"]]["status"] in {"pending", "blocked"}]
            if not pending:
                break
            ready = [task for task in pending if all(state["tasks"][dep]["status"] == "completed" for dep in task.get("dependencies", []))]
            if not ready:
                for task in pending:
                    state["tasks"][task["id"]].update(status="blocked", error="Dependencies did not complete")
                break
            task = ready[0]
            record = state["tasks"][task["id"]]
            try:
                if freeze_path:
                    require(file_hash(freeze_path) == freeze_digest, "Freeze changed while queue was running")
                evidence = check_task(task, plan, freeze_path)
                out = resolve(option(task["argv"], "--out"))
                require(not out.exists(), f"Measurement directory already exists; use new task ID and output: {out}")
                record.update(evidence, status="running", phase=task["phase"], started_unix=time.time())
                record.pop("error", None)
                atomic_json(state_path, state)
                log = queue_root / (task["id"] + ".controller.log")
                require(not log.exists(), f"Prior launch log exists; use a new task ID: {log}")
                code = launcher(commands(task, root)[0], log)
                record.update(controller_returncode=code, ended_unix=time.time())
                found = collect_artifacts(task, root, record)
                if not found:
                    record.update(status="failed", error="Controller exited without an outcome; launch log preserved")
                elif code != 0 and record["status"] == "completed":
                    record.update(status="failed", error="Controller exit code contradicts completed outcome")
                if code in [130, 143]:
                    record["status"] = "interrupted"
                run_dir = root / "runs" / task["id"]
                if run_dir.is_dir():
                    provenance = run_dir / "queue-task.json"
                    if not provenance.exists():
                        atomic_json(provenance, dict(queue_id=queue_id, task=task, plan_sha256=digest, evidence=evidence))
                atomic_json(state_path, state)
                if record["status"] == "interrupted":
                    break
            except (ValueError, OSError, KeyError, TypeError) as error:
                # A task that never launched may be reconsidered after a missing
                # freeze is supplied. A launched label is always terminal.
                started = record.get("status") == "running"
                record.update(status="failed" if started else "blocked", error=str(error), ended_unix=time.time())
                atomic_json(state_path, state)
                break
        rows = list(state["tasks"].values())
        state["status"] = ("completed" if all(row["status"] == "completed" for row in rows)
                           else "failed" if any(row["status"] in TERMINAL_FAILURE for row in rows) else "blocked")
        state["updated_unix"] = time.time()
        atomic_json(state_path, state)
        return state


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("action", choices=["run", "validate", "status"])
    parser.add_argument("--tasks", type=Path)
    parser.add_argument("--root", type=Path, default=ROOT)
    parser.add_argument("--queue-id")
    parser.add_argument("--freeze", type=Path)
    parser.add_argument("--continue-independent", action="store_true")
    args = parser.parse_args()
    if args.action == "status":
        require(args.queue_id is not None, "status requires --queue-id")
        print((resolve(args.root) / "queues" / args.queue_id / "status.json").read_text())
        return 0
    require(args.tasks is not None, "--tasks is required")
    if args.action == "validate":
        plan = json.loads(args.tasks.read_text())
        validate_plan(plan, resolve(args.root))
        for task in plan["tasks"]:
            check_task(task, plan, args.freeze)
        print(json.dumps(dict(status="PASS", tasks=len(plan["tasks"]), gpu_processes_started=0)))
        return 0
    state = run_queue(args.tasks, args.root, args.queue_id, args.freeze, args.continue_independent)
    print(json.dumps(state, indent=2))
    return 0 if state["status"] == "completed" else 1


if __name__ == "__main__":
    raise SystemExit(main())
