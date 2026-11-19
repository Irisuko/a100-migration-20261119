#!/usr/bin/env python3
"""Serialize authorized GPU4-7 experiments and reserve their memory while idle.

Usage: control.py [--root PATH] status|reserve
       control.py [--root PATH] run --label NAME --timeout SECONDS -- COMMAND ...
Only the private holder imports torch. Commands must honor CUDA_VISIBLE_DEVICES.
"""
import argparse
from contextlib import contextmanager, ExitStack
import fcntl
import json
import os
from pathlib import Path
import re
import signal
import subprocess
import sys
import tempfile
import threading
import time
import uuid

REPO = Path(__file__).resolve().parents[2]
DEFAULT_ROOT = REPO / "run/a100-gpu0123-20261114/pool-gpu4567"
GLOBAL_SUSPENSION = DEFAULT_ROOT / "RESERVATION_DISABLED.json"
GPUS = {4: "GPU-725d9550-4cdc-f401-0cb0-3d31bccec1af",
        5: "GPU-69f3abce-1e88-4391-95dc-4ab4c2c77593",
        6: "GPU-fd1b979e-abe3-dc82-8a17-890c8cd3cf14",
        7: "GPU-0c16ece9-91b7-1520-f2ea-69811b52901a"}
TOKEN = "A100_GPU4567_CONTROL_TOKEN"
CPU_AFFINITY = "32-63,96-127"
POOL_CPUS = set(range(32,64)) | set(range(96,128))
GPU_OWNERS = {}


def validated_root(root):
    root = Path(root).resolve()
    if root != DEFAULT_ROOT.resolve():
        raise ValueError("This controller requires its isolated GPU4-7 pool root")
    return root


def require_cpu_affinity():
    actual = set(os.sched_getaffinity(0))
    if not actual or not actual.issubset(POOL_CPUS):
        raise RuntimeError("GPU4-7 serving/holder requires NUMA1 CPU affinity32-63,96-127")


def bound_command(command):
    return ["taskset", "-c", CPU_AFFINITY, *command]


@contextmanager
def pool_locks():
    # GPU4-7 overlaps the retired temporary45 and auxiliary67 pools. Hold
    # their existing locks too, without acquiring the independent main lock.
    with ExitStack() as stack:
        for pool in ("4567", "45", "67"):
            path = REPO / f"run/a100-gpu{pool}-controller.lock"
            path.parent.mkdir(parents=True, exist_ok=True)
            lock = stack.enter_context(path.open("a"))
            fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        yield


def atomic_json(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(mode="w", dir=path.parent, delete=False) as out:
        json.dump(value, out, indent=2)
        out.write("\n")
        out.flush()
        os.fsync(out.fileno())
    os.replace(out.name, path)


def identity(pid):
    try:
        proc = Path("/proc") / str(pid)
        fields = (proc / "stat").read_text().rsplit(")", 1)[1].split()
        if fields[0] == "Z":
            return None
        return {"pid": int(pid), "uid": proc.stat().st_uid, "starttime": int(fields[19]),
                "pgid": int(fields[2]), "session": int(fields[3]),
                "cmdline": (proc / "cmdline").read_bytes().decode().rstrip("\0").split("\0")}
    except (OSError, ValueError, UnicodeError, IndexError):
        return None


def matches(record):
    return bool(record and record.get("uid") == os.getuid() and identity(record["pid"]) == record)


def signal_process(record, sig):
    if not matches(record):
        return False
    try:
        descriptor = os.pidfd_open(record["pid"])
        try:
            if not matches(record):
                return False
            signal.pidfd_send_signal(descriptor, sig)
            return True
        finally:
            os.close(descriptor)
    except ProcessLookupError:
        return False


def has_token(pid, token):
    if not token:
        return False
    try:
        return f"{TOKEN}={token}".encode() in (Path("/proc") / str(pid) / "environ").read_bytes().split(b"\0")
    except OSError:
        return False


def confirm_holder_identity(child, launched, ready, token):
    """Use the ready child's identity after exec/import, retaining PID reuse checks."""
    announced = json.loads(Path(ready).read_text()).get('identity')
    live = identity(child.pid)
    code = child.poll()
    diagnostic = {'launched': launched, 'announced': announced, 'live': live,
                  'returncode': code, 'token_present': has_token(child.pid, token)}
    atomic_json(Path(ready).with_suffix('.verification.json'), diagnostic)
    stable = ('pid', 'uid', 'starttime')
    if (code is not None or not announced or announced.get('pid') != child.pid
            or not matches(announced) or not diagnostic['token_present']
            or (launched and any(launched.get(k) != announced.get(k) for k in stable))):
        raise RuntimeError(f'Holder identity verification failed: {diagnostic}')
    # The immediate Popen snapshot can precede a stable /proc cmdline. The child
    # is still our live unreaped child, has our token, and confirms its identity
    # after reserving memory. Do not weaken strict identity checks for signalling.
    return announced


def owned_processes(token):
    found = []
    for path in Path("/proc").iterdir():
        if path.name.isdigit() and has_token(path.name, token):
            record = identity(int(path.name))
            if record and record["uid"] == os.getuid():
                found.append(record)
    return found


def stop_owned(token):
    # The unguessable inherited token also follows reparented/session-owning children.
    for sig, seconds in ((signal.SIGTERM, 12), (signal.SIGKILL, 5)):
        deadline = time.monotonic() + seconds
        while True:
            members = owned_processes(token)
            if not members:
                return
            for record in members:
                signal_process(record, sig)
            if time.monotonic() >= deadline:
                break
            time.sleep(.1)
    raise RuntimeError("Owned child processes did not exit; reservation withheld")


def gpu_processes():
    def query(args):
        return subprocess.check_output(["nvidia-smi", *args, "--format=csv,noheader"], text=True, timeout=15)
    layout = query(["-i", ",".join(map(str, GPUS)), "--query-gpu=index,uuid"])
    actual = {int(row.split(",")[0]): row.split(",")[1].strip() for row in layout.splitlines()}
    if actual != GPUS:
        raise RuntimeError(f"GPU4-7 UUID mapping changed: {actual}")
    rows = query(["-i", ",".join(map(str, GPUS)), "--query-compute-apps=gpu_uuid,pid"])
    return [{"gpu_uuid": row.split(",")[0].strip(), "pid": int(row.split(",")[1])}
            for row in rows.splitlines() if row.strip()]


def assert_clear(allowed_token=None):
    known = GPU_OWNERS.setdefault(allowed_token, {}) if allowed_token else {}
    if allowed_token:
        for record in owned_processes(allowed_token):
            known[record['pid']]=record
    def retained(current):
        previous=known.get(current['pid']) if current else None
        return bool(previous and all(current.get(key)==previous.get(key)
                                     for key in ('pid','uid','starttime')))
    # nvidia-smi may retain a context row after /proc/PID/environ disappears.
    # Retain full identities, and requery disappeared/zombie PIDs for at most
    # two seconds. A live unknown identity always fails immediately; persistent
    # stale rows still block the next allocation rather than being ignored.
    for attempt in range(11):
        foreign, departed = [], []
        for row in gpu_processes():
            pid = row['pid']
            token_present=has_token(pid, allowed_token)
            current = identity(pid)
            if token_present:
                if current and current.get('uid') == os.getuid():
                    known[pid] = current
                continue
            if retained(current):
                continue
            # New ranks can tear down environ before their first GPU poll.
            # A live token-owned process-group leader inside the same session
            # establishes descent without accepting a reused PID or foreign UID.
            leader=None
            if (allowed_token and current and current.get('uid')==os.getuid()
                    and current.get('pgid') and current['pgid']!=pid):
                leader=identity(current['pgid'])
                if (leader and leader.get('uid')==os.getuid()
                        and leader.get('session')==current.get('session')
                        and (has_token(leader['pid'],allowed_token) or retained(leader))):
                    known[pid]=current
                    continue
            # Recheck after reading environ/group state: process exit can race
            # between those reads and an earlier live /proc snapshot.
            fresh=identity(pid) if current else None
            diagnostic={**row,'identity':fresh,'previous_identity':known.get(pid),
                        'token_present':token_present,'group_identity':leader}
            (departed if fresh is None else foreign).append(diagnostic)
        if foreign:
            raise RuntimeError(f"Foreign GPU process; only this controller's children may be stopped: {foreign}")
        if not departed:
            return
        if attempt == 10:
            raise RuntimeError(f"GPU context rows persist after process exit: {departed}")
        time.sleep(.2)


def environment(root, base=None):
    root = validated_root(root)
    env = dict(os.environ if base is None else base)
    for key in list(env):
        if (key.startswith(("NCCL_", "UCX_", "A100_GPU"))
                or key in ("MC_FORCE_TCP", "SGLANG_PP_LAYER_PARTITION")
                or (key.startswith("SGLANG_PDMUX_PP_") and ("TRACE" in key or "PROFILE" in key or "CHECK" in key))):
            env.pop(key)
    cuda = Path.home() / ".local/cuda-12.8"
    env.update(CUDA_VISIBLE_DEVICES=",".join(GPUS.values()), CUDA_HOME=str(cuda),
               A100_GPU4567_CAMPAIGN_ROOT=str(root),
               PATH=f"{REPO / '.venv/bin'}:{cuda / 'bin'}:{env.get('PATH', '')}",
               PYTHONPATH=str(REPO / "python"), TORCH_CUDA_ARCH_LIST="8.0",
               FLASHINFER_CUDA_ARCH_LIST="8.0", FLASHINFER_WORKSPACE_BASE=str(root / "cache/flashinfer"),
               TRITON_CACHE_DIR=str(root / "cache/triton"), TORCH_EXTENSIONS_DIR=str(root / "cache/torch"),
               OMP_NUM_THREADS="4", MAX_JOBS="4", TOKENIZERS_PARALLELISM="false",
               HF_HUB_OFFLINE="1", TRANSFORMERS_OFFLINE="1", NO_PROXY="localhost,127.0.0.1,::1",
               no_proxy="localhost,127.0.0.1,::1")
    return env


def validate_label(label):
    if not re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9_.-]{0,95}", label):
        raise ValueError("Label must be 1-96 plain letters, numbers, underscores, dots or hyphens")


class Controller:
    def __init__(self, root):
        self.root = validated_root(root)
        self.state_path = self.root / "controller.json"
        self.state = json.loads(self.state_path.read_text()) if self.state_path.exists() else {"holder": None, "active": None}

    def save(self):
        atomic_json(self.state_path, self.state)

    def assert_enabled(self):
        if (GLOBAL_SUSPENSION.exists() or (self.root / "RESERVATION_DISABLED.json").exists()
                or self.state.get("reservation_suspended")):
            raise RuntimeError("GPU work is suspended by the user; explicit renewed authorization is required")

    def recover(self):
        active = self.state.get("active")
        if not active:
            return
        if matches(active.get("controller")):
            raise RuntimeError("An experiment controller is still alive")
        stop_owned(active["token"])
        path = self.root / "runs" / active["label"] / "outcome.json"
        if not path.exists():
            atomic_json(path, {"status": "interrupted", "error": "Recovered abandoned controller", "ended": time.time()})
        self.state["active"] = None
        self.save()

    def release(self):
        record = self.state.get("holder")
        if record:
            if matches(record.get("identity")):
                stop_owned(record["token"])
            elif owned_processes(record["token"]):
                raise RuntimeError("Holder identity changed; refusing to signal it")
            self.state["holder"] = None
            self.save()

    def reserve(self):
        self.assert_enabled()
        current = self.state.get("holder")
        if current and matches(current.get("identity")) and Path(current["ready"]).exists():
            assert_clear(current["token"])
            return
        self.release()
        assert_clear()
        token = uuid.uuid4().hex
        ready = self.root / "holders" / (token + ".ready.json")
        ready.parent.mkdir(parents=True, exist_ok=True)
        env = environment(self.root)
        env[TOKEN] = token
        with ready.with_suffix(".log").open("x") as log:
            child = subprocess.Popen(bound_command([sys.executable, str(Path(__file__).resolve()), "--root", str(self.root),
                                      "_holder", "--ready", str(ready)]), env=env, stdout=log,
                                     stderr=subprocess.STDOUT, start_new_session=True, cwd=REPO)
        self.state["holder"] = {"identity": identity(child.pid), "token": token, "ready": str(ready)}
        self.save()
        try:
            deadline = time.monotonic() + 120
            while not ready.exists():
                if child.poll() is not None or time.monotonic() > deadline:
                    raise RuntimeError(f"Holder failed or timed out; see {ready.with_suffix('.log')}")
                time.sleep(.2)
            self.state['holder']['identity'] = confirm_holder_identity(
                child, self.state['holder']['identity'], ready, token)
            self.save()
        except BaseException:
            stop_owned(token)
            self.state["holder"] = None
            self.save()
            raise

    def run(self, label, timeout, command):
        self.assert_enabled()
        validate_label(label)
        if not command or timeout <= 0:
            raise ValueError("A command and positive timeout are required")
        command = bound_command(command)
        path = self.root / "runs" / label
        path.mkdir(parents=True, exist_ok=False)
        env = environment(self.root)
        token = uuid.uuid4().hex
        env[TOKEN] = token
        public_env = {k: v for k, v in env.items() if k in environment(self.root, {}) and k != "PATH"}
        atomic_json(path / "command.json", {"argv": command, "environment": public_env,
                    "gpu_mapping": GPUS, "timeout": timeout, "started": time.time()})
        outcome = {"status": "failed", "returncode": None, "started": time.time()}
        child = None
        try:
            self.release()
            assert_clear()
            self.state["active"] = {"label": label, "token": token, "command": command, "controller": identity(os.getpid())}
            self.save()
            with (path / "command.log").open("x") as log:
                child = subprocess.Popen(command, env=env, stdout=log, stderr=subprocess.STDOUT,
                                         start_new_session=True, cwd=REPO)
            self.state["active"]["identity"] = identity(child.pid)
            self.save()
            deadline, next_check = time.monotonic() + timeout, 0
            while child.poll() is None:
                self.assert_enabled()
                now = time.monotonic()
                if now >= deadline:
                    outcome["status"] = "timed_out"
                    break
                if now >= next_check:
                    assert_clear(token)
                    next_check = now + 2
                time.sleep(min(.2, max(0, deadline - now)))
            else:
                outcome.update(status="completed" if child.returncode == 0 else "failed", returncode=child.returncode)
        except (KeyboardInterrupt, SystemExit) as exc:
            outcome.update(status="interrupted", error=repr(exc))
        except Exception as exc:
            outcome["error"] = repr(exc)
        finally:
            try:
                stop_owned(token)
                if child:
                    outcome["returncode"] = child.wait(timeout=5)
                self.state["active"] = None
                self.save()
                self.reserve()
                outcome["reservation"] = "held"
            except BaseException as exc:
                outcome["cleanup_error"] = repr(exc)
                outcome["reservation"] = "not_held"
                if outcome["status"] == "completed":
                    outcome["status"] = "failed"
            outcome["ended"] = time.time()
            outcome["duration"] = outcome["ended"] - outcome["started"]
            atomic_json(path / "outcome.json", outcome)
        return outcome


def holder(root, ready):
    root = validated_root(root)
    require_cpu_affinity()
    if Path(ready).resolve().parent != root / "holders" or not os.environ.get(TOKEN):
        raise ValueError("Private holder requires owned token and isolated pool ready path")
    Controller(root).assert_enabled()
    if os.environ.get("CUDA_VISIBLE_DEVICES") != ",".join(GPUS.values()):
        raise RuntimeError("Holder requires the four assigned GPU UUIDs")
    assert_clear()
    stopped = threading.Event()
    for sig in (signal.SIGINT, signal.SIGTERM):
        signal.signal(sig, lambda *_: stopped.set())
    import torch  # The sole CUDA-using mode; deliberately absent from status/tests.
    blocks, allocations = [], []
    try:
        if torch.cuda.device_count() != 4:
            raise RuntimeError("Expected exactly the four assigned CUDA devices")
        for index, gpu_uuid in enumerate(GPUS.values()):
            Controller(root).assert_enabled()
            if stopped.is_set():
                return
            assert_clear(os.environ[TOKEN])
            with torch.cuda.device(index):
                free, _ = torch.cuda.mem_get_info()
                count = free - 1024**3
                if count <= 0:
                    raise RuntimeError("Insufficient memory for idle reservation")
                blocks.append(torch.empty(count, dtype=torch.uint8, device=f"cuda:{index}"))
                allocations.append({"uuid": gpu_uuid, "reserved_bytes": count})
        atomic_json(ready, {"identity": identity(os.getpid()), "allocations": allocations, "created": time.time()})
        while not stopped.wait(1):
            Controller(root).assert_enabled()
            assert_clear(os.environ[TOKEN])
    finally:
        ready.unlink(missing_ok=True)
        blocks.clear()
        torch.cuda.empty_cache()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=DEFAULT_ROOT)
    sub = parser.add_subparsers(dest="mode", required=True)
    sub.add_parser("status")
    sub.add_parser("reserve")
    run = sub.add_parser("run")
    run.add_argument("--label", required=True)
    run.add_argument("--timeout", type=float, required=True)
    run.add_argument("command", nargs=argparse.REMAINDER)
    private = sub.add_parser("_holder", help=argparse.SUPPRESS)
    private.add_argument("--ready", type=Path, required=True)
    args = parser.parse_args()
    if args.mode == "_holder":
        holder(args.root, args.ready)
        return 0
    controller = Controller(args.root)
    if args.mode == "status":
        print(json.dumps({"state": controller.state, "gpu_processes": gpu_processes(),
                          "holder_alive": matches((controller.state.get("holder") or {}).get("identity"))}, indent=2))
        return 0
    with pool_locks():
        controller = Controller(args.root)
        controller.recover()
        if args.mode == "reserve":
            controller.reserve()
            print(json.dumps(controller.state, indent=2))
            return 0
        signal.signal(signal.SIGTERM, lambda *_: (_ for _ in ()).throw(KeyboardInterrupt()))
        command = args.command[1:] if args.command[:1] == ["--"] else args.command
        result = controller.run(args.label, args.timeout, command)
        print(json.dumps(result, indent=2))
        return 0 if result["status"] == "completed" else 1


if __name__ == "__main__":
    raise SystemExit(main())
