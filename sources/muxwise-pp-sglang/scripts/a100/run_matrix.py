#!/usr/bin/env python3
"""Execute a recorded experiment matrix sequentially on GPUs 6/7."""
import argparse
import json
from pathlib import Path
import signal
import subprocess
import sys
import time


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--jobs", type=Path, required=True)
    parser.add_argument("--name-prefix", default="", help="Prefix new run names when reproducing an existing matrix")
    args = parser.parse_args()
    repo = Path(__file__).resolve().parents[2]
    jobs = json.loads(args.jobs.read_text())
    status_path = args.jobs.with_name(args.name_prefix + args.jobs.stem + ".status.json")
    statuses = []
    child = None

    def interrupt(signum, frame):
        if child is not None and child.poll() is None:
            child.terminate()  # runner owns and cleans up its server process group
            child.wait(timeout=60)
        raise KeyboardInterrupt

    signal.signal(signal.SIGTERM, interrupt)
    signal.signal(signal.SIGINT, interrupt)
    try:
        for job in jobs:
            argv = list(job["args"])
            argv[argv.index("--name") + 1] = args.name_prefix + argv[argv.index("--name") + 1]
            name = argv[argv.index("--name") + 1]
            command = [sys.executable, str(repo / "scripts/a100/run_experiment.py"), *argv]
            status = {"name": name, "command": command, "started_at": time.time()}
            statuses.append(status)
            print(f"START {name}", flush=True)
            with (repo / "run/a100" / f"{name}-driver.log").open("w") as log:
                child = subprocess.Popen(command, cwd=repo, stdout=log, stderr=subprocess.STDOUT)
                status["exit_code"] = child.wait()
            status["ended_at"] = time.time()
            status_path.write_text(json.dumps(statuses, indent=2) + "\n")
            print(f"DONE {name} exit={status['exit_code']}", flush=True)
            if status["exit_code"]:
                raise SystemExit(status["exit_code"])
    finally:
        status_path.write_text(json.dumps(statuses, indent=2) + "\n")


if __name__ == "__main__":
    main()
