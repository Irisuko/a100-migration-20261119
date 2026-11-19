#!/usr/bin/env python3
"""One durable notification attempt. Never relaunch a started invocation."""
import argparse
import os
from pathlib import Path
import subprocess
import time

from eventd import (OwnerBusy, atomic_json, identity, owner_lock, read_json, validate_ack)


def expand(argv, replacements):
    return [replace(arg, replacements) for arg in argv]


def replace(arg, replacements):
    for key, value in replacements.items():
        arg = arg.replace('{' + key + '}', str(value))
    return arg


def run(job_path):
    job = read_json(job_path)
    folder = Path(job['folder'])
    try:
        lock = owner_lock(folder/'worker.lock')
    except OwnerBusy:
        return 0
    with lock:
        if (folder/'result.json').exists() or (folder/'started.json').exists():
            return 0
        if Path(job['hold_path']).exists():
            return 0
        batch = read_json(folder/'events.json')
        if batch['batch_id'] != job['batch_id']:
            raise ValueError('job and batch mismatch')
        # Commit before launching any adapter or postprocess. A crash between
        # this marker and Popen is conservatively unknown, never a blind retry.
        atomic_json(folder/'started.json', {'batch_id': batch['batch_id'],
                    'worker': identity(os.getpid()), 'started_unix': time.time()})
        replacements = {'event_file': folder/'events.json', 'ack_file': folder/'ack.json',
                        'batch_id': batch['batch_id'], 'batch_dir': folder}
        result = {'batch_id': batch['batch_id'], 'status': 'unknown', 'reason': None}
        try:
            for i, post in enumerate(job.get('postprocess', [])):
                record = {'id': post['id'], 'argv': expand(post['argv'], replacements)}
                try:
                    with open(folder/f'postprocess-{i}.log', 'ab') as log:
                        p = subprocess.Popen(record['argv'], stdin=subprocess.DEVNULL, stdout=log, stderr=log,
                                             env={**os.environ, 'CUDA_VISIBLE_DEVICES': ''})
                        try:
                            record['returncode'] = p.wait(timeout=post['timeout_sec'])
                        except subprocess.TimeoutExpired:
                            # Only this directly-owned child, never an experiment
                            # group, is terminated. Its contract forbids daemons.
                            p.terminate()
                            try:
                                p.wait(timeout=5)
                            except subprocess.TimeoutExpired:
                                p.kill(); p.wait()
                            record['error'] = 'timeout'
                except Exception as exc:
                    record['error'] = f'{type(exc).__name__}: {exc}'
                atomic_json(folder/f'postprocess-{i}.json', record)
            argv = expand(job['transport']['argv'], replacements)
            atomic_json(folder/'command.json', {'argv': argv, 'cwd': os.getcwd(), 'CUDA_VISIBLE_DEVICES': ''})
            with open(folder/'transport.log', 'ab') as log:
                p = subprocess.Popen(argv, stdin=subprocess.DEVNULL, stdout=log, stderr=log,
                                     env={**os.environ, 'CUDA_VISIBLE_DEVICES': ''})
                atomic_json(folder/'transport-process.json', {'identity': identity(p.pid)})
                try:
                    result['returncode'] = p.wait(timeout=job['transport'].get('timeout_sec', 1815))
                except subprocess.TimeoutExpired:
                    p.terminate()
                    try:
                        p.wait(timeout=5)
                    except subprocess.TimeoutExpired:
                        p.kill(); p.wait()
                    result['reason'] = 'transport timeout; external outcome may be unknown'
                else:
                    if result['returncode'] != 0:
                        result.update(status='failed', reason=f'transport exit {result["returncode"]}; no automatic replay')
                    else:
                        validate_ack(read_json(folder/'ack.json'), batch)
                        result.update(status='acked', reason=None)
        except Exception as exc:
            result['reason'] = f'{type(exc).__name__}: {exc}'
        result['ended_unix'] = time.time()
        atomic_json(folder/'result.json', result)
    return 0


if __name__ == '__main__':
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--job', required=True)
    args = p.parse_args()
    raise SystemExit(run(args.job))
