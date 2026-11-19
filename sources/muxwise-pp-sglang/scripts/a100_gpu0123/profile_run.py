#!/usr/bin/env python3
"""Diagnostic-only profiling around unchanged server_run.py.

Run under control.py. Example (all runner options precede its --extra):
  profile_run.py --out ROOT/diagnostics/NAME --seconds 30 -- \
    --model 8b --family muxpp --tp 1 --pp 4 --trace DEV_TRACE

The client starts only after profiler acknowledgements and per-rank start-entry
logs. These logs precede initialization: they are not a distributed ready barrier.
The profile includes client warmups. No diagnostic metric is performance eligible.
"""
import argparse
import ast
from concurrent.futures import ThreadPoolExecutor, as_completed
import gzip
import hashlib
import json
import math
import os
from pathlib import Path
import re
import shutil
import signal
import subprocess
import sys
import time
from urllib.request import Request, ProxyHandler, build_opener

REPO = Path(__file__).resolve().parents[2]
HERE = Path(__file__).resolve()
PYTHON = REPO / '.venv/bin/python'
BENCH = HERE.parent / 'bench/bench_client.py'
HTTP = build_opener(ProxyHandler({}))


def atomic_json(path, value):
    path = Path(path)
    temporary = path.with_name(path.name + f'.{os.getpid()}.tmp')
    temporary.write_text(json.dumps(value, indent=2, allow_nan=False) + '\n')
    os.replace(temporary, path)


def one_value(argv, name, default=None):
    values = []
    for index, arg in enumerate(argv):
        if arg == name:
            if index + 1 == len(argv):
                raise ValueError(f'missing value for {name}')
            values.append(argv[index + 1])
        elif arg.startswith(name + '='):
            values.append(arg.split('=', 1)[1])
    if len(values) > 1:
        raise ValueError(f'duplicate {name}')
    return values[0] if values else default


def make_plan(argv, out, seconds):
    import server_run
    if not math.isfinite(seconds) or not 0 < seconds <= 60:
        raise ValueError('profile seconds must be positive and at most60')
    for arg in argv:
        if arg.split('=', 1)[0] in ('--out', '--suite', '--pdmux-pp-request-trace'):
            raise ValueError('output, suite and semantic paths are owned by profiler')
    split = argv.index('--extra') if '--extra' in argv else len(argv)
    before, extra = argv[:split], argv[split:]
    parser = argparse.ArgumentParser(add_help=False, allow_abbrev=False)
    parser.add_argument('--model', required=True, choices=['8b', '70b'])
    parser.add_argument('--family', required=True, choices=['plain', 'chunked', 'muxwise', 'muxpp', 'pd'])
    parser.add_argument('--trace', required=True)
    for flag, default in [('tp', 1), ('pp', 4), ('replicas', 1), ('prefill-replicas', 0), ('port', 32100)]:
        parser.add_argument('--' + flag, type=int, default=default)
    known, _ = parser.parse_known_args(before)
    for name in ('--model', '--family', '--trace', '--tp', '--pp', '--replicas', '--prefill-replicas', '--port'):
        one_value(before, name)
    workers = server_run.worker_layout(known.family, known.tp, known.pp, known.replicas,
                                      known.prefill_replicas, known.port)
    out = Path(out).resolve()
    return dict(out=str(out), family=known.family, tp=known.tp, pp=known.pp, workers=workers,
                seconds=seconds, trace=str(Path(known.trace).resolve()),
                server_args=before + ['--suite', 'trace', '--out', str(out / 'serving')] + extra,
                diagnostic_only=True, measurement_eligible=False)


def worker_command(argv, env, plan):
    if argv[:2] != ['-m', 'sglang.launch_server']:
        raise ValueError('unexpected worker command')
    port = int(one_value(argv, '--port'))
    worker = next((w for w in plan['workers'] if w['port'] == port), None)
    if worker is None:
        raise ValueError('worker port is outside assigned layout')
    if env.get('CUDA_VISIBLE_DEVICES') != ','.join(worker['gpus']):
        raise ValueError('worker GPU mask differs from assigned subset')
    if int(one_value(argv, '--tp-size')) != plan['tp'] or int(one_value(argv, '--pp-size')) != plan['pp']:
        raise ValueError('worker topology differs from diagnostic plan')
    if any(a.split('=', 1)[0] == '--pdmux-pp-request-trace' for a in argv):
        raise ValueError('semantic output is owned by profiler')
    folder = Path(plan['out']) / f'worker{worker["index"]}'
    folder.mkdir(parents=True, exist_ok=True)
    actual, environment = list(argv), dict(env)
    for key in list(environment):
        if key.startswith('SGLANG_PDMUX_PP_') and ('TRACE' in key or 'PROFILE' in key):
            environment.pop(key)
    if plan['family'] == 'muxpp':
        environment.update(SGLANG_PDMUX_PP_PROFILE_RANGES='1',
                           SGLANG_PDMUX_PP_TIMING_TRACE=str(folder / 'timing'),
                           SGLANG_PDMUX_PP_ROUND_TRACE=str(folder / 'rounds.jsonl'))
        actual += ['--pdmux-pp-request-trace', str(folder / 'requests')]
    atomic_json(folder / 'actual-launch.json', dict(argv=[str(PYTHON), *actual],
                environment={k: v for k, v in environment.items()
                             if k == 'CUDA_VISIBLE_DEVICES' or k.startswith('SGLANG_PDMUX_PP_')},
                port=port, role=worker['role'], diagnostic_only=True))
    return actual, environment


def all_ranks_started(plan):
    for worker in plan['workers']:
        path = Path(plan['out']) / 'serving' / f'server-{worker["index"]}.log'
        if not path.exists():
            return False
        destination = str(Path(plan['out']) / f'worker{worker["index"]}' / 'profile')
        found = set()
        for line in path.read_text(errors='replace').splitlines():
            if 'Profiling starts' not in line or f'saved to: {destination}' not in line:
                continue
            prefix = line.split(']', 1)[0]
            pp, tp = re.search(r'\bPP(\d+)\b', prefix), re.search(r'\bTP(\d+)\b', prefix)
            if (plan['pp'] > 1 and not pp) or (plan['tp'] > 1 and not tp):
                continue
            found.add((int(pp.group(1)) if pp else 0, int(tp.group(1)) if tp else 0))
        expected = {(pp, tp) for pp in range(plan['pp']) for tp in range(plan['tp'])}
        if found != expected:
            return False
    return True


def inspect_exports(plan):
    exports = []
    for worker in plan['workers']:
        folder = Path(plan['out']) / f'worker{worker["index"]}' / 'profile'
        ids = set()
        for pp in range(plan['pp']):
            for tp in range(plan['tp']):
                suffix = (f'-PP-{pp}' if plan['pp'] > 1 else '') + f'-TP-{tp}.trace.json.gz'
                files = list(folder.glob('*' + suffix))
                if len(files) != 1:
                    raise ValueError(f'expected one export for worker{worker["index"]} rank PP{pp} TP{tp}, got{len(files)}')
                path = files[0]
                ids.add(path.name[:-len(suffix)])
                with gzip.open(path, 'rt') as stream:
                    payload = json.load(stream)
                events = payload.get('traceEvents', [])
                kernels = sum(e.get('cat') == 'kernel' and e.get('dur', 0) > 0 for e in events)
                label = f'pdmux_pp|pp={pp}|tp={tp}|'
                ranges = sum(str(e.get('name', '')).startswith(label) for e in events)
                if not kernels or (plan['family'] == 'muxpp' and not ranges):
                    raise ValueError(f'missing GPU kernels or semantic ranges in {path}')
                exports.append(dict(worker=worker['index'], pp=pp, tp=tp, path=str(path),
                                    bytes=path.stat().st_size, kernels=kernels, semantic_ranges=ranges,
                                    sha256=hashlib.sha256(path.read_bytes()).hexdigest()))
        if len(ids) != 1:
            raise ValueError('rank exports have different profile session IDs')
    if len(exports) != 4:
        raise ValueError('diagnostic must cover exactly four GPU ranks')
    return exports


def post(port, endpoint, body, timeout):
    started = time.time()
    request = Request(f'http://127.0.0.1:{port}{endpoint}',
                      data=json.dumps(body).encode(), headers={'Content-Type': 'application/json'})
    with HTTP.open(request, timeout=timeout) as response:
        text = response.read().decode(errors='replace')
        return dict(port=port, endpoint=endpoint, started_unix=started, ended_unix=time.time(),
                    status=response.status, response=text)


def capture(plan, child, seconds, startup_timeout, export_timeout, poll=.1, delay=0):
    out = Path(plan['out'])
    report = dict(status='running', diagnostic_only=True, measurement_eligible=False,
                  errors=[], endpoints=[], exports=[], start_unix=time.time(),
                  requested_seconds=seconds, requested_delay_seconds=delay,
                  includes_client_warmups=delay == 0,
                  delay_anchor_uncertainty='Loaded marker is polled after its print, just before client origin; scheduling/read delay is reconciled with final exact client origin.',
                  marker_poll_interval_seconds=poll,
                  start_evidence='All rank start-entry logs; not an initialized barrier.')
    attempted, stopped = False, False
    def persist():
        atomic_json(out / 'profile.json', report)
    def alive():
        if child.poll() is not None:
            raise RuntimeError(f'serving runner exited {child.returncode}')
    def wait_for(predicate, timeout, name):
        end = time.monotonic() + timeout
        while True:
            alive()
            if predicate():
                return
            if time.monotonic() >= end:
                raise TimeoutError(name)
            time.sleep(poll)
    def endpoints(name):
        failures = []
        with ThreadPoolExecutor(max_workers=len(plan['workers'])) as pool:
            jobs = {}
            for worker in plan['workers']:
                body = {}
                if name == '/start_profile':
                    folder = out / f'worker{worker["index"]}' / 'profile'
                    folder.mkdir(parents=True, exist_ok=True)
                    body = dict(output_dir=str(folder), activities=['CPU', 'GPU'],
                                with_stack=False, record_shapes=False, profile_by_stage=False)
                future = pool.submit(post, worker['port'], name, body, min(export_timeout, 60))
                jobs[future] = worker['port']
            for future in as_completed(jobs):
                try:
                    report['endpoints'].append(future.result())
                except Exception as exc:
                    message = f'{name} port{jobs[future]}: {exc!r}'
                    failures.append(message)
                    report['endpoints'].append(dict(port=jobs[future], endpoint=name, error=repr(exc)))
                persist()
        if failures:
            raise RuntimeError('; '.join(failures))
    try:
        persist()
        wait_for(lambda: (out / 'client-ready.json').exists(), startup_timeout, 'client readiness timeout')
        if delay:
            atomic_json(out / 'client-go.json', dict(at_unix=time.time()))
            report['client_released_unix'] = time.time()
            def loaded():
                if (out / 'client-done.json').exists():
                    raise RuntimeError('client completed before delayed profile window')
                path = out / 'serving/client.log'
                return path.exists() and any(line.startswith('Loaded ') for line in path.read_text(errors='replace').splitlines())
            wait_for(loaded, startup_timeout, 'post-warmup client origin marker missing')
            report['loaded_marker_observed_unix'] = time.time()
            target = time.monotonic() + delay
            persist()
            while time.monotonic() < target:
                alive()
                if (out / 'client-done.json').exists():
                    raise RuntimeError('client completed before delayed profile window')
                time.sleep(min(poll, max(0, target - time.monotonic())))
            if (out / 'client-done.json').exists():
                raise RuntimeError('client completed before delayed profile window')
        attempted = True
        endpoints('/start_profile')
        wait_for(lambda: all_ranks_started(plan), export_timeout, 'all-rank profile start entries missing')
        report['all_rank_start_entries_unix'] = time.time()
        report['summary_before'] = summary_snapshot(plan)
        if not delay:
            atomic_json(out / 'client-go.json', dict(at_unix=time.time()))
            report['client_released_unix'] = time.time()
        persist()
        end = time.monotonic() + seconds
        while time.monotonic() < end and not (out / 'client-done.json').exists():
            alive()
            time.sleep(min(poll, max(0, end - time.monotonic())))
        report['window_end_reason'] = 'client_completed' if (out / 'client-done.json').exists() else 'duration'
        report['stop_requested_unix'] = time.time()
        report['summary_after'] = summary_snapshot(plan)
        report['summary_counter_deltas'] = summary_deltas(report['summary_before'], report['summary_after'])
        stopped = True
        endpoints('/stop_profile')
        last_error = None
        def have_exports():
            nonlocal last_error
            try:
                report['exports'] = inspect_exports(plan)
                return True
            except (ValueError, OSError, EOFError) as exc:
                last_error = repr(exc)
                return False
        try:
            wait_for(have_exports, export_timeout, 'incomplete rank exports')
        except TimeoutError as exc:
            raise RuntimeError(f'{exc}: {last_error}') from exc
        report['exports_complete_unix'] = time.time()
        atomic_json(out / 'client-release.json', dict(at_unix=time.time()))
        report['status'] = 'completed'
    except BaseException as exc:
        report['status'] = 'failed'
        report['errors'].append(repr(exc))
        atomic_json(out / 'ABORT.json', dict(error=repr(exc), at_unix=time.time()))
    finally:
        if attempted and not stopped:
            try:
                endpoints('/stop_profile')
            except BaseException as exc:
                report['errors'].append('cleanup: ' + repr(exc))
        report['end_unix'] = time.time()
        persist()
    return report


def stop_process_group(child, grace=20):
    # Signal verified identities via pidfd, including descendants of a dead leader.
    from control import TOKEN, has_token, identity, signal_process
    token = os.environ.get(TOKEN)
    def members():
        leader = identity(child.pid)
        original = getattr(child, 'profile_identity', None)
        if leader and original and leader['starttime'] != original['starttime']:
            raise RuntimeError('refusing cleanup of reused process group identity')
        live = []
        for path in Path('/proc').iterdir():
            if path.name.isdigit():
                entry = identity(int(path.name))
                if entry and entry['pgid'] == child.pid and entry['session'] == child.pid and entry['uid'] == os.getuid():
                    if not original:
                        raise RuntimeError('missing initial diagnostic process identity')
                    if not token or not has_token(entry['pid'], token):
                        if identity(entry['pid']) is None:
                            continue
                        raise RuntimeError(f'refusing cleanup without ownership evidence for PID{entry["pid"]}')
                    live.append(entry)
        child.poll()
        return live
    for sig, timeout in [(signal.SIGINT, grace), (signal.SIGTERM, 5), (signal.SIGKILL, 5)]:
        current = members()
        if not current:
            return
        for entry in current:
            signal_process(entry, sig)
        end = time.monotonic() + timeout
        while time.monotonic() < end:
            if not members():
                return
            time.sleep(.05)
    raise RuntimeError('diagnostic child group did not exit')


def mark_diagnostic(out):
    for name in ('run.json', 'trace_summary.json'):
        path = Path(out) / 'serving' / name
        if not path.exists():
            continue
        raw = path.with_name(path.stem + '.raw-diagnostic.json')
        if raw.exists():
            data = json.loads(path.read_text())
            if data.get('diagnostic_only') is True and data.get('measurement_eligible') is False:
                continue
            raise FileExistsError(f'raw evidence exists but ordinary artifact is not marked: {raw}')
        raw.write_bytes(path.read_bytes())
        data = json.loads(path.read_text())
        data.update(diagnostic_only=True, measurement_eligible=False,
                    profile_manifest=str(Path(out) / 'profile.json'))
        atomic_json(path, data)


def summary_snapshot(plan):
    result = dict(observed_unix=time.time(), observed_monotonic=time.monotonic(), workers={})
    for worker in plan['workers']:
        path = Path(plan['out']) / 'serving' / f'server-{worker["index"]}.log'
        if not path.exists():
            continue
        for line in reversed(path.read_text(errors='replace').splitlines()):
            if 'MuxWise-PP summary stages=' not in line:
                continue
            try:
                stages = ast.literal_eval(line.split('MuxWise-PP summary stages=', 1)[1].split(' mbs=', 1)[0])
                result['workers'][str(worker['index'])] = dict(
                    raw_line=line, stages={str(k): v for k, v in stages.items()},
                    stage_sample_age_seconds={str(k): result['observed_monotonic'] - v['at'] for k, v in stages.items()})
                break
            except (ValueError, SyntaxError, KeyError, TypeError):
                continue
    return result


def summary_deltas(before, after):
    def difference(first, last):
        if isinstance(last, dict):
            return {str(k): difference(first.get(k, {} if isinstance(v, dict) else [] if isinstance(v, list) else 0), v)
                    for k, v in last.items()}
        if isinstance(last, list):
            return [difference(first[i] if i < len(first) else 0, v) for i, v in enumerate(last)]
        value = last - first
        if value < -1e-6:
            raise ValueError('counter reset or inconsistent summary ordering')
        return value
    result = dict(workers={}, gpu_bubble_seconds=None,
                  interpretation='Sampled counter deltas only; per-lane times overlap and are not summed. Host wait is not CUDA communication or GPU idle.',
                  missing_evidence=[])
    for index, end in after['workers'].items():
        start = before['workers'].get(index)
        if not start:
            result['missing_evidence'].append(f'worker{index} has no pre-window summary')
            continue
        result['workers'][index] = {}
        for stage, final in end['stages'].items():
            initial = start['stages'].get(stage)
            if not initial or final['at'] <= initial['at']:
                result['missing_evidence'].append(f'worker{index} stage{stage} has no distinct boundary samples')
                continue
            try:
                row = dict(sample_interval_seconds=final['at'] - initial['at'],
                           before_sample_age_seconds=start['stage_sample_age_seconds'][stage],
                           after_sample_age_seconds=end['stage_sample_age_seconds'][stage],
                           timing=difference(initial['timing'], final['timing']), admission={})
                if initial.get('admission') and final.get('admission'):
                    for name in ('decisions', 'blocked', 'stops', 'wall'):
                        row['admission'][name] = difference(initial['admission'][name], final['admission'][name])
                result['workers'][index][stage] = row
            except (ValueError, TypeError, KeyError) as exc:
                result['missing_evidence'].append(f'worker{index} stage{stage}: {exc}')
    return result


def check_parent(plan):
    from control import identity
    out = Path(plan['out'])
    if (out / 'ABORT.json').exists():
        raise RuntimeError('diagnostic parent aborted')
    parent = identity(plan['parent_pid'])
    if not parent or (plan.get('parent_starttime') is not None and parent['starttime'] != plan['parent_starttime']):
        raise RuntimeError('diagnostic parent exited or identity changed')


def wait_gate(plan, filename):
    out = Path(plan['out'])
    end = time.monotonic() + plan['gate_timeout']
    while True:
        check_parent(plan)
        if (out / filename).exists():
            return
        if time.monotonic() >= end:
            raise TimeoutError(f'diagnostic handshake timeout: {filename}')
        time.sleep(.05)


def client_gate(command, plan):
    out = Path(plan['out'])
    atomic_json(out / 'client-ready.json', dict(pid=os.getpid(), argv=command, at_unix=time.time()))
    wait_gate(plan, 'client-go.json')
    started = time.time()
    child = subprocess.Popen(command)
    error = None
    try:
        while child.poll() is None:
            check_parent(plan)
            time.sleep(.05)
    except BaseException as exc:
        error = exc
    finally:
        if child.poll() is None:
            child.send_signal(signal.SIGINT)
            try:
                child.wait(timeout=10)
            except subprocess.TimeoutExpired:
                child.kill()
                child.wait()
        atomic_json(out / 'client-done.json', dict(returncode=child.returncode, started_unix=started,
                    finished_unix=time.time(), argv=command, interrupted=error is not None,
                    error=repr(error) if error else None))
    if error:
        raise error
    wait_gate(plan, 'client-release.json')
    return child.returncode


def client_command(argv, plan):
    if not argv or Path(argv[0]).resolve() != BENCH:
        raise ValueError('unexpected client entry point')
    old = one_value(argv, '--out-dir')
    if old is None or Path(old).resolve() != Path(plan['out']) / 'serving':
        raise ValueError('client output differs from owned serving directory')
    actual = list(argv)
    for index, value in enumerate(actual):
        if value == '--out-dir':
            actual[index + 1] = str(Path(plan['out']) / 'client-raw')
        elif value.startswith('--out-dir='):
            actual[index] = '--out-dir=' + str(Path(plan['out']) / 'client-raw')
    return [str(PYTHON), *actual]


def publish_client_artifacts(plan):
    out = Path(plan['out'])
    raw = out / 'client-raw'
    if not raw.exists():
        return
    serving = out / 'serving'
    for source in raw.iterdir():
        if source.name == 'trace_summary.json':
            continue
        destination = serving / source.name
        if destination.exists():
            raise FileExistsError(destination)
        shutil.copyfile(source, destination)
    source = raw / 'trace_summary.json'
    if source.exists():
        original = serving / 'trace_summary.raw-diagnostic.json'
        if original.exists() or (serving / 'trace_summary.json').exists():
            raise FileExistsError('diagnostic client summary already published')
        shutil.copyfile(source, original)
        data = json.loads(source.read_text())
        data.update(diagnostic_only=True, measurement_eligible=False,
                    profile_manifest=str(out / 'profile.json'))
        atomic_json(serving / 'trace_summary.json', data)


def exec_child(plan_path, argv):
    plan = json.loads(Path(plan_path).read_text())
    if argv[:2] == ['-m', 'sglang.launch_server']:
        actual, env = worker_command(argv, dict(os.environ), plan)
        os.execve(str(PYTHON), [str(PYTHON), *actual], env)
    elif argv[:2] == ['-m', 'sglang_router.launch_router']:
        os.execve(str(PYTHON), [str(PYTHON), *argv], dict(os.environ))
    elif argv and Path(argv[0]).resolve() == BENCH:
        try:
            return client_gate(client_command(argv, plan), plan)
        finally:
            publish_client_artifacts(plan)
    raise ValueError('diagnostic shim received an unexpected Python entry point')


def runner_child(plan_path):
    import server_run
    plan = json.loads(Path(plan_path).read_text())
    server_run.PYTHON = Path(plan['shim'])
    sys.argv = [str(HERE.parent / 'server_run.py'), *plan['server_args']]
    signal.signal(signal.SIGTERM, lambda *_: (_ for _ in ()).throw(KeyboardInterrupt()))
    server_run.main()


def main(argv=None):
    argv = list(sys.argv[1:] if argv is None else argv)
    if argv[:1] == ['_runner']:
        runner_child(argv[1])
        return 0
    parser = argparse.ArgumentParser(description=__doc__, allow_abbrev=False)
    parser.add_argument('--out', type=Path, required=True)
    parser.add_argument('--seconds', type=float, default=30)
    parser.add_argument('--profile-delay-sec', type=float, default=0,
                        help='Delay after the observed post-warmup client start marker; default0 profiles warmups too')
    parser.add_argument('--startup-timeout', type=float, default=2400)
    parser.add_argument('--export-timeout', type=float, default=180)
    parser.add_argument('--runner-timeout', type=float, default=3600)
    parser.add_argument('server_args', nargs=argparse.REMAINDER)
    args = parser.parse_args(argv)
    server_args = args.server_args[1:] if args.server_args[:1] == ['--'] else args.server_args
    plan = make_plan(server_args, args.out, args.seconds)
    if not math.isfinite(args.profile_delay_sec) or args.profile_delay_sec < 0:
        raise ValueError('profile delay must be finite and nonnegative')
    plan['profile_delay_sec'] = args.profile_delay_sec
    for value in (args.startup_timeout, args.export_timeout, args.runner_timeout):
        if not math.isfinite(value) or value <= 0:
            raise ValueError('timeouts must be finite and positive')
    import server_run
    from control import identity
    server_run.require_active_controller()
    if os.environ.get('CUDA_VISIBLE_DEVICES') != ','.join(server_run.UUIDS):
        raise RuntimeError('fixed four-GPU controller mask required')
    trace = Path(plan['trace'])
    if not trace.is_file():
        raise FileNotFoundError(trace)
    out = Path(plan['out'])
    out.mkdir(parents=True, exist_ok=False)
    parent = identity(os.getpid())
    plan.update(parent_pid=os.getpid(), parent_starttime=parent['starttime'],
                gate_timeout=args.startup_timeout + args.export_timeout + args.runner_timeout,
                shim=str(out / 'python-shim'), trace_sha256=hashlib.sha256(trace.read_bytes()).hexdigest(),
                wrapper_sha256=hashlib.sha256(HERE.read_bytes()).hexdigest(),
                server_run_sha256=hashlib.sha256((HERE.parent / 'server_run.py').read_bytes()).hexdigest())
    plan_path = out / 'profile-plan.json'
    shim = Path(plan['shim'])
    shim.write_text(f'#!{PYTHON}\nimport sys\nsys.path.insert(0, {str(HERE.parent)!r})\n'
                    f'from profile_run import exec_child\nraise SystemExit(exec_child({str(plan_path)!r}, sys.argv[1:]))\n')
    shim.chmod(0o700)
    plan['shim_sha256'] = hashlib.sha256(shim.read_bytes()).hexdigest()
    atomic_json(plan_path, plan)
    child = None
    report = dict(status='failed', diagnostic_only=True, measurement_eligible=False, errors=[])
    previous = signal.signal(signal.SIGTERM, lambda *_: (_ for _ in ()).throw(KeyboardInterrupt()))
    try:
        with (out / 'runner.log').open('x') as log:
            child = subprocess.Popen([str(PYTHON), str(HERE), '_runner', str(plan_path)],
                                     stdout=log, stderr=subprocess.STDOUT, start_new_session=True,
                                     cwd=REPO, env=dict(os.environ))
            child.profile_identity = identity(child.pid)
            report = capture(plan, child, args.seconds, args.startup_timeout, args.export_timeout,
                             delay=args.profile_delay_sec)
            if report['status'] == 'completed':
                code = child.wait(timeout=args.runner_timeout)
                report['runner_returncode'] = code
                if code:
                    raise RuntimeError(f'diagnostic serving/client lifecycle failed: {code}')
                final = json.loads((out / 'serving/run.json').read_text())
                if final.get('status') != 'execution_complete':
                    raise RuntimeError('runner did not complete final recovery')
    except BaseException as exc:
        report['status'] = 'failed'
        report['errors'].append(repr(exc))
    finally:
        if report['status'] != 'completed':
            atomic_json(out / 'ABORT.json', dict(errors=report['errors'], at_unix=time.time()))
        if child:
            try:
                stop_process_group(child)
            except BaseException as exc:
                report['status'] = 'failed'
                report['errors'].append('cleanup: ' + repr(exc))
        try:
            mark_diagnostic(out)
        except Exception as exc:
            report['status'] = 'failed'
            report['errors'].append('diagnostic marking: ' + repr(exc))
        summary = out / 'serving/trace_summary.json'
        if summary.exists():
            try:
                origin = json.loads(summary.read_text()).get('run_started_unix')
                if origin is not None:
                    report['actual_trace_origin_unix'] = origin
                    observed = report.get('loaded_marker_observed_unix')
                    report['loaded_marker_observation_minus_origin_seconds'] = observed - origin if observed else None
                    report['profile_start_offsets_from_trace_origin'] = [
                        dict(port=e['port'], request_seconds=e['started_unix'] - origin,
                             acknowledgement_seconds=e['ended_unix'] - origin)
                        for e in report.get('endpoints', [])
                        if e['endpoint'] == '/start_profile' and 'started_unix' in e]
            except Exception as exc:
                report['status'] = 'failed'
                report['errors'].append('trace-origin reconciliation: ' + repr(exc))
        report['finished_unix'] = time.time()
        report['analysis_commands'] = [[str(PYTHON), str(REPO / 'scripts/a100/analyze_gpu_trace.py'),
                                       '--trace', entry['path'], '--output',
                                       str(out / f'worker{entry["worker"]}' / f'analysis-pp{entry["pp"]}-tp{entry["tp"]}.json')]
                                      for entry in report.get('exports', [])]
        atomic_json(out / 'profile.json', report)
        signal.signal(signal.SIGTERM, previous)
    print(json.dumps(dict(status=report['status'], diagnostic_only=True,
                          profile_manifest=str(out / 'profile.json'))), flush=True)
    return 0 if report['status'] == 'completed' else 1


if __name__ == '__main__':
    raise SystemExit(main())
