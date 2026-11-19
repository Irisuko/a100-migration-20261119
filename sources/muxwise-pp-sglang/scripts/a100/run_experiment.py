#!/usr/bin/env python3
"""Run one isolated server experiment, preserve artifacts, and always stop it."""

import argparse
import datetime
import hashlib
import json
import os
from pathlib import Path
import signal
import socket
import subprocess
import sys
import time
from urllib.request import ProxyHandler, Request, build_opener

REPO = Path(__file__).resolve().parents[2]
PYTHON = REPO / ".venv/bin/python"
HTTP = build_opener(ProxyHandler({}))


def request(base_url, path, payload=None, timeout=30):
    body = None if payload is None else json.dumps(payload).encode()
    req = Request(base_url + path, data=body, headers={"Content-Type": "application/json"})
    with HTTP.open(req, timeout=timeout) as response:
        data = response.read()
        try:
            return json.loads(data)
        except (ValueError, UnicodeDecodeError):
            return data.decode(errors="replace")


def flush_cache(base_url):
    with HTTP.open(base_url + "/flush_cache", timeout=30) as response:
        body = response.read().decode()
        # This repository returns 200 only when FlushCacheReqOutput.success is true.
        if response.status != 200 or not body.startswith("Cache flushed."):
            raise RuntimeError(f"Cache reset was not acknowledged: {response.status} {body}")
        return {"http_status": response.status, "body": body, "acknowledged_at": time.time(),
                "initiator": "external experiment runner", "success": True}


def environment():
    env = os.environ.copy()
    cuda = Path.home() / ".local/cuda-12.8"
    env.update(
        CUDA_VISIBLE_DEVICES="6,7",
        CUDA_HOME=str(cuda),
        PATH=f"{REPO / '.venv/bin'}:{cuda / 'bin'}:{env.get('PATH', '')}",
        FLASHINFER_CUDA_ARCH_LIST="8.0",
        FLASHINFER_WORKSPACE_BASE=str(REPO / "run/a100/flashinfer-cache"),
        OMP_NUM_THREADS="4",
        MAX_JOBS="4",
        TOKENIZERS_PARALLELISM="false",
        HF_HUB_OFFLINE="1",
        TRANSFORMERS_OFFLINE="1",
        NO_PROXY="localhost,127.0.0.1,::1",
        no_proxy="localhost,127.0.0.1,::1",
    )
    # Never inherit debug/profiling switches into a performance experiment.
    for key in ("SGLANG_PDMUX_PP_NUMERIC_TRACE", "SGLANG_PDMUX_PP_TIMING_TRACE",
                "SGLANG_PDMUX_PP_ROUND_TRACE", "SGLANG_PDMUX_PP_PROFILE_RANGES",
                "SGLANG_PDMUX_PP_DECODE_PLAN_REUSE_CHECK"):
        env.pop(key, None)
    return env


def gpu_snapshot():
    return subprocess.check_output(
        ["nvidia-smi", "--query-gpu=index,uuid,name,memory.used,memory.free,utilization.gpu,clocks.sm,clocks.mem,power.draw", "--format=csv"],
        text=True,
    )


def active_group_members(pgid):
    members = []
    for proc in Path("/proc").iterdir():
        if not proc.name.isdigit():
            continue
        try:
            fields = (proc / "stat").read_text().rsplit(")", 1)[1].split()
            if int(fields[2]) == pgid and fields[0] != "Z":
                if proc.stat().st_uid != os.getuid():
                    raise RuntimeError("Unexpected foreign process in experiment process group")
                members.append(int(proc.name))
        except (OSError, ValueError, IndexError):
            pass
    return members


class Server:
    def __init__(self, args, run_dir):
        self.args, self.run_dir = args, run_dir
        self.base_url = f"http://127.0.0.1:{args.port}"
        self.proc = None
        self.log = None

    def start(self):
        args = self.args
        with socket.socket() as sock:
            if sock.connect_ex(("127.0.0.1", args.port)) == 0:
                raise RuntimeError(f"Port {args.port} is occupied")
        free = subprocess.check_output(
            ["nvidia-smi", "-i", "6,7", "--query-gpu=memory.free", "--format=csv,noheader,nounits"],
            text=True,
        )
        if min(int(x) for x in free.split()) < args.min_free_mib:
            raise RuntimeError("Selected GPUs no longer have sufficient free memory")
        active = subprocess.check_output(
            ["nvidia-smi", "-i", "6,7", "--query-compute-apps=gpu_uuid,pid", "--format=csv,noheader"],
            text=True,
        ).strip()
        if active:
            raise RuntimeError(f"Selected GPUs have active compute processes; refusing to share: {active}")
        if list(args.model.glob("*.aria2")):
            raise RuntimeError("Model has incomplete aria2 downloads")
        command = [
            str(PYTHON), "-m", "sglang.launch_server", "--model-path", str(args.model),
            "--served-model-name", "a100-llama", "--host", "127.0.0.1", "--port", str(args.port),
            "--tp-size", "1", "--pp-size", "2", "--dtype", "bfloat16",
            "--context-length", str(args.context), "--max-total-tokens", str(args.kv_tokens),
            "--max-running-requests", str(args.running_requests),
            "--max-micro-batch-size", str(args.microbatch),
            "--cuda-graph-max-bs", str(args.microbatch), "--mem-fraction-static", "0.65",
            "--max-prefill-tokens", str(args.prefill_tokens),
            "--attention-backend", "flashinfer", "--sampling-backend", "pytorch",
            "--disable-overlap-schedule", "--random-seed", "20261114", "--log-level", "info",
        ]
        env = environment()
        if args.profile:
            env["SGLANG_PDMUX_PP_PROFILE_RANGES"] = "1"
        if args.plan_reuse_check:
            env["SGLANG_PDMUX_PP_DECODE_PLAN_REUSE_CHECK"] = "1"
        if args.mode == "mux":
            import yaml
            config = yaml.safe_load(args.config.read_text())
            command += [
                "--enable-pdmux", "--chunked-prefill-size", "-1",
                "--pdmux-config-path", str(args.config), "--sm-group-num", str(config["sm_group_num"]),
                "--pdmux-pp-policy", args.policy,
                "--pdmux-pp-prefill-chunk-size", str(args.chunk),
                "--pdmux-pp-slo-ttft", "2", "--pdmux-pp-slo-tpot", str(args.tpot),
                "--pdmux-pp-summary-interval", "2",
            ]
            if args.trace:
                prefix = self.run_dir / "trace"
                command += ["--pdmux-pp-request-trace", str(prefix)]
                env["SGLANG_PDMUX_PP_TIMING_TRACE"] = str(self.run_dir / "timing")
                env["SGLANG_PDMUX_PP_ROUND_TRACE"] = str(self.run_dir / "rounds.jsonl")
        else:
            command += ["--chunked-prefill-size", str(args.baseline_chunk)]
        command += args.extra_args
        metadata = {
            "command": command, "model": str(args.model), "mode": args.mode,
            "profile": args.profile, "trace": args.trace, "plan_reuse_check": args.plan_reuse_check,
            "experiment_purpose": args.purpose,
            "measurement_settings": {"suites": args.suites, "repetitions": args.repetitions,
                                     "concurrency": args.concurrency, "request_rate": args.request_rate,
                                     "also_open_loop_rate": args.also_open_loop_rate,
                                     "order": args.order, "tpot_target_s": args.tpot},
            "started_at": datetime.datetime.now(datetime.timezone.utc).isoformat(),
            "environment": {k: env[k] for k in ("CUDA_VISIBLE_DEVICES", "CUDA_HOME", "FLASHINFER_CUDA_ARCH_LIST", "FLASHINFER_WORKSPACE_BASE", "OMP_NUM_THREADS")},
            "gpu_before": gpu_snapshot(),
            "repository_commit": subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=REPO, text=True).strip(),
            "runtime_source_sha256": {
                str(path.relative_to(REPO)): hashlib.sha256(path.read_bytes()).hexdigest()
                for path in [REPO / "python/sglang/srt/multiplex/multiplexing_pp.py",
                             REPO / "python/sglang/srt/multiplex/pp_policy.py",
                             REPO / "python/sglang/srt/layers/attention/flashinfer_backend.py",
                             REPO / "python/sglang/srt/managers/tokenizer_manager.py",
                             REPO / "python/sglang/srt/managers/scheduler_profiler_mixin.py"]},
        }
        if args.mode == "mux":
            metadata["config"] = config
        (self.run_dir / "launch.json").write_text(json.dumps(metadata, indent=2) + "\n")
        self.log = (self.run_dir / "server.log").open("w")
        self.proc = subprocess.Popen(command, cwd=REPO, env=env, stdin=subprocess.DEVNULL,
                                     stdout=self.log, stderr=subprocess.STDOUT, start_new_session=True)
        (self.run_dir / "server.pid").write_text(str(self.proc.pid) + "\n")
        deadline = time.monotonic() + args.startup_timeout
        while time.monotonic() < deadline:
            if self.proc.poll() is not None:
                raise RuntimeError(f"Server exited with {self.proc.returncode}; see {self.run_dir / 'server.log'}")
            try:
                request(self.base_url, "/health", timeout=2)
                print(f"READY {self.run_dir.name} pid={self.proc.pid}", flush=True)
                return
            except Exception:
                time.sleep(2)
        raise TimeoutError("Server readiness timeout")

    def stop(self):
        cleanup = {"remaining": [], "forced": False}
        if self.proc is not None:
            try:
                os.killpg(self.proc.pid, signal.SIGTERM)
            except ProcessLookupError:
                pass
            deadline = time.monotonic() + 30
            while active_group_members(self.proc.pid) and time.monotonic() < deadline:
                self.proc.poll()
                time.sleep(0.2)
            # Parent exit does not establish that scheduler/compiler children exited.
            if active_group_members(self.proc.pid):
                cleanup["forced"] = True
                try:
                    os.killpg(self.proc.pid, signal.SIGKILL)
                except ProcessLookupError:
                    pass
                deadline = time.monotonic() + 10
                while active_group_members(self.proc.pid) and time.monotonic() < deadline:
                    self.proc.poll()
                    time.sleep(0.2)
            self.proc.wait(timeout=2)
            cleanup["remaining"] = active_group_members(self.proc.pid)
        if self.log is not None:
            self.log.close()
        try:
            (self.run_dir / "gpu-after.txt").write_text(gpu_snapshot())
        except Exception as error:
            cleanup["gpu_snapshot_error"] = repr(error)
        (self.run_dir / "shutdown.json").write_text(json.dumps(cleanup, indent=2) + "\n")
        if cleanup["remaining"]:
            raise RuntimeError(f"Experiment processes did not exit: {cleanup['remaining']}")
        (self.run_dir / "server.pid").unlink(missing_ok=True)


def main():
    def terminate(signum, frame):
        raise KeyboardInterrupt(f"Experiment interrupted by signal {signum}")

    signal.signal(signal.SIGTERM, terminate)
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--name", required=True)
    parser.add_argument("--purpose", choices=("correctness", "pilot", "main", "mechanism", "diagnostic"),
                        default=None, help="Explicit reporting category; targeted mechanisms are not main performance trials")
    parser.add_argument("--output-root", type=Path, default=REPO / "run/a100")
    parser.add_argument("--model", type=Path, default=REPO.parent / "models/Llama-3.1-8B-Instruct")
    parser.add_argument("--config", type=Path, default=REPO / "configs/pdmux_pp_a100_calibration.yml")
    parser.add_argument("--fixtures", type=Path, default=REPO / "run/a100/fixtures.json")
    parser.add_argument("--mode", choices=("baseline", "mux"), default="mux")
    parser.add_argument("--policy", choices=("static", "alp"), default="alp")
    parser.add_argument("--chunk", type=int, default=0)
    parser.add_argument("--baseline-chunk", type=int, default=-1)
    parser.add_argument("--context", type=int, default=8192)
    parser.add_argument("--kv-tokens", type=int, default=135168)
    parser.add_argument("--running-requests", type=int, default=32)
    parser.add_argument("--microbatch", type=int, default=16)
    parser.add_argument("--prefill-tokens", type=int, default=8192)
    parser.add_argument("--tpot", type=float, default=0.05)
    parser.add_argument("--port", type=int, default=30100)
    parser.add_argument("--min-free-mib", type=int, default=40000)
    parser.add_argument("--startup-timeout", type=int, default=900)
    parser.add_argument("--suites", default="warmup,serial,cache,concurrent")
    parser.add_argument("--repetitions", type=int, default=1)
    parser.add_argument("--correctness-repetitions", type=int, default=None)
    parser.add_argument("--concurrency", type=int, default=16)
    parser.add_argument("--request-rate", type=float, default=0)
    parser.add_argument("--also-open-loop-rate", type=float, default=0,
                        help="After perf trials, repeat perf at this Poisson rate with concurrency 64")
    parser.add_argument("--order", choices=("fixture", "reverse", "shuffle"), default="fixture")
    parser.add_argument("--trace", action="store_true")
    parser.add_argument("--profile", action="store_true", help="Separate diagnostic GPU trace; do not use timings as benchmark results")
    parser.add_argument("--plan-reuse-check", action="store_true", help="Diagnostic pinned plan buffer reuse counters")
    parser.add_argument("--oracle-comparisons", type=Path,
                        help="JSON list of named reference/candidate pairs to teacher-force on a baseline server")
    parser.add_argument("--extra-args", nargs=argparse.REMAINDER, default=[])
    args = parser.parse_args()
    args.model, args.config = args.model.resolve(), args.config.resolve()
    run_dir = args.output_root.resolve() / args.name
    run_dir.mkdir(parents=True, exist_ok=False)
    server = Server(args, run_dir)
    outcome = {"passed": False, "runs": []}
    try:
        server.start()
        if args.oracle_comparisons:
            if args.mode != "baseline":
                raise ValueError("Logprob oracle requires an ordinary baseline server")
            outcome["oracle_diagnostics"] = []
            for item in json.loads(args.oracle_comparisons.read_text()):
                flush_cache(server.base_url)
                output = run_dir / f"oracle-{item['name']}.json"
                command = [str(PYTHON), str(REPO / "scripts/a100/diagnose_differences.py"),
                           "--fixtures", str(args.fixtures.resolve()), "--reference", item["reference"],
                           "--candidate", item["candidate"], "--base-url", server.base_url,
                           "--output", str(output), "--max-cases", "32"]
                result = subprocess.run(command, cwd=REPO)
                outcome["oracle_diagnostics"].append({"name": item["name"], "command": command,
                                                       "exit_code": result.returncode})
                if result.returncode:
                    raise RuntimeError(f"Oracle diagnostic failed: {output}")
        suites = [s for s in args.suites.split(",") if s]
        if args.also_open_loop_rate and "perf" in suites:
            suites.insert(suites.index("perf") + 1, "perf-openloop")
        for suite in suites:
            client_suite = "perf" if suite == "perf-openloop" else suite
            rate = args.also_open_loop_rate if suite == "perf-openloop" else args.request_rate
            concurrency = 64 if suite == "perf-openloop" else args.concurrency
            repeats = args.repetitions if client_suite in ("perf", "pilot", "serial", "concurrent") else 1
            if client_suite in ("serial", "concurrent") and args.correctness_repetitions is not None:
                repeats = args.correctness_repetitions
            for repetition in range(repeats):
                # Warmup learns RLS, and flush_cache preserves the learned model.
                reset = flush_cache(server.base_url)
                output = run_dir / f"{suite}-{repetition}.json"
                command = [str(PYTHON), str(REPO / "scripts/a100/experiment_client.py"), "run",
                           "--fixtures", str(args.fixtures.resolve()), "--suite", client_suite,
                           "--base-url", server.base_url, "--output", str(output),
                           "--concurrency", str(concurrency), "--request-rate", str(rate),
                           "--seed", str(20261114 + repetition),
                           "--order", args.order if client_suite in ("serial", "concurrent", "pilot", "perf") else "fixture"]
                if args.profile and suite != "warmup":
                    prior_profile_done = (run_dir / "server.log").read_text(errors="replace").count("Profiling done.")
                    request(server.base_url, "/start_profile", {
                        "output_dir": str(run_dir / f"profile-{suite}-{repetition}"),
                        "activities": ["CPU", "GPU"], "with_stack": False, "record_shapes": False,
                    }, timeout=90)
                before = time.time()
                with (run_dir / f"{suite}-{repetition}.log").open("w") as log:
                    result = subprocess.run(command, cwd=REPO, env=environment(), stdout=log, stderr=subprocess.STDOUT)
                if args.profile and suite != "warmup":
                    request(server.base_url, "/stop_profile", {}, timeout=180)
                    deadline = time.monotonic() + 180
                    while (run_dir / "server.log").read_text(errors="replace").count("Profiling done.") < prior_profile_done + 2:
                        if time.monotonic() > deadline or server.proc.poll() is not None:
                            raise RuntimeError("Both PP stages did not finish exporting their GPU traces")
                        time.sleep(0.5)
                outcome["runs"].append({"suite": suite, "repeat": repetition, "started_at": before,
                                        "ended_at": time.time(), "exit_code": result.returncode,
                                        "command": command, "output": str(output), "cache_reset": reset})
                print(f"RESULT {args.name} {suite}/{repetition} exit={result.returncode}", flush=True)
                if result.returncode:
                    raise RuntimeError(f"Client failed; see {output.with_suffix('.log')}")
                data = json.loads(output.read_text())
                data["cache_reset"] = True
                data["external_cache_reset"] = reset
                records = data["requests"]
                data["experiment_slo"] = {
                    "ttft_target_s": 2.0, "tpot_target_s": args.tpot,
                    "tpot_met_fraction": sum(not r.get("error") and r.get("tpot_s", float("inf")) <= args.tpot for r in records) / len(records),
                    "both_met_fraction": sum(not r.get("error") and r.get("ttft_s", float("inf")) <= 2.0 and r.get("tpot_s", float("inf")) <= args.tpot for r in records) / len(records),
                    "arrival_mode": "open_loop" if rate > 0 and concurrency >= len(records) else "concurrency_limited",
                }
                output.write_text(json.dumps(data, indent=2, allow_nan=False) + "\n")
                time.sleep(2.5)  # Give deferred timing samples and the periodic summary a final idle turn.
        outcome["passed"] = True
    except BaseException as error:
        outcome["error"] = repr(error)
        raise
    finally:
        (run_dir / "outcome.json").write_text(json.dumps(outcome, indent=2) + "\n")
        try:
            server.stop()
        except BaseException as error:
            outcome["passed"] = False
            outcome["cleanup_error"] = repr(error)
            (run_dir / "outcome.json").write_text(json.dumps(outcome, indent=2) + "\n")
            raise
        finally:
            analyzer = REPO / "scripts/a100/analyze_logs.py"
            if analyzer.exists() and (run_dir / "server.log").exists():
                subprocess.run([str(PYTHON), str(analyzer), "--log", str(run_dir / "server.log"),
                                "--output", str(run_dir / "scheduler.json"), "--csv", str(run_dir / "calibration.csv")],
                               cwd=REPO, check=False)


if __name__ == "__main__":
    main()
