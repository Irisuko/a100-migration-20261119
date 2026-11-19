#!/usr/bin/env python3
"""Persistent, CPU-only event observer. It has no experiment execution API."""
from __future__ import annotations

import argparse
import fcntl
import glob
import hashlib
import json
import math
import os
from pathlib import Path
import signal
import sqlite3
import subprocess
import sys
import tempfile
import time

HERE = Path(__file__).resolve().parent
FAIL = {'failed', 'failure', 'error', 'exception', 'timeout', 'timed_out', 'cancelled', 'canceled', 'aborted', 'stopped', 'interrupted', 'finished_with_failures'}
TERMINAL = FAIL | {'completed', 'complete', 'done', 'succeeded'}
DECISION = {'docked', 'blocked', 'decision', 'needs_decision', 'waiting_decision', 'paused', 'suspended', 'waiting_release'}
KINDS = {'queue', 'outer', 'controller', 'outcome', 'dock', 'decision'}


def canonical(value):
    return json.dumps(value, sort_keys=True, ensure_ascii=False, separators=(',', ':'), allow_nan=False)


def digest(value):
    return hashlib.sha256(canonical(value).encode()).hexdigest()


def atomic_json(path, data):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, tmp = tempfile.mkstemp(prefix='.' + path.name + '-', dir=path.parent)
    try:
        with os.fdopen(fd, 'w') as out:
            out.write(canonical(data) + '\n')
            out.flush()
            os.fsync(out.fileno())
        os.replace(tmp, path)
        fsync_dir(path.parent)
    finally:
        if os.path.exists(tmp):
            os.unlink(tmp)


def fsync_dir(path):
    fd = os.open(path, os.O_RDONLY | os.O_DIRECTORY)
    try:
        os.fsync(fd)
    finally:
        os.close(fd)


def read_json(path):
    value = json.loads(Path(path).read_text())
    if not isinstance(value, dict):
        raise ValueError('JSON object required')
    return value


def identity(pid):
    try:
        p = Path('/proc') / str(int(pid))
        fields = (p / 'stat').read_text().rsplit(')', 1)[1].split()
        if fields[0] == 'Z':
            return None
        return {'pid': int(pid), 'starttime': int(fields[19]), 'uid': p.stat().st_uid}
    except (OSError, ValueError, IndexError, TypeError):
        return None


def alive(record):
    return (isinstance(record, dict)
            and all(type(record.get(k)) is int for k in ('pid', 'starttime', 'uid'))
            and identity(record['pid']) == record)


def compact_error(value):
    if value in (None, '', [], {}, False):
        return None
    # Error identity never includes changing progress/counters outside this field.
    text = canonical(value)
    return text[:2000]


def facts(source, path, data):
    """Only facts that warrant a decision, failure response, or batch completion."""
    kind = source['kind']
    state = str(data.get('status', '')).lower()
    out = {}

    def add(key, event_kind, **details):
        out[key] = {'kind': event_kind, 'details': details}

    if kind == 'dock':
        release = path.with_name(source.get('release_file', 'release.json'))
        if not release.exists():
            add('dock:' + str(data.get('stage_id', path.parent.name)), 'dock',
                stage_id=data.get('stage_id'), stage_index=data.get('stage_index'),
                boundary_sha256=hashlib.sha256(path.read_bytes()).hexdigest(),
                release_path=str(release))
        return out
    if kind == 'decision':
        if state not in TERMINAL | {'handled', 'resolved', 'acked', 'dismissed'} and not data.get('resolved', False):
            add('decision:' + str(data.get('id', path.name)), 'decision',
                status=state or 'pending', decision_id=data.get('id', path.name),
                reason=data.get('reason', data.get('summary')), request=data.get('request'))
        return out
    if kind in {'queue', 'outer'} and state in TERMINAL:
        add('terminal', 'batch_terminal', status=state, queue_id=data.get('queue_id'),
            tasks_path=data.get('tasks_path'), plan_sha256=data.get('plan_sha256'))
    dock = data.get('dock') or {}
    if state in DECISION and not (state == 'docked' and dock.get('released') is True):
        add('decision', 'decision', status=state, reason=data.get('reason'),
            stage_index=dock.get('stage_index', data.get('stage_index')),
            stage_id=dock.get('stage_id', data.get('stage_id')),
            dock_path=dock.get('path'), dock_sha256=dock.get('sha256'))
    if kind not in {'queue', 'outer'} and state in FAIL:
        add('failure', 'failure', status=state, returncode=data.get('returncode'),
            error=compact_error(data.get('error', data.get('exception'))))
    for field in ('error', 'exception', 'errors'):
        error = compact_error(data.get(field))
        if error:
            add('error:' + field, 'failure', status=state, field=field, error=error)
    if kind == 'queue':
        tasks = data.get('tasks', {})
        if not isinstance(tasks, dict):
            raise ValueError('queue.tasks must be an object')
        for label, task in tasks.items():
            status = str(task.get('status', '')).lower()
            if status in FAIL:
                add('task:' + label, 'task_failure', label=label, status=status,
                    outcome_path=task.get('outcome_path'),
                    request_summary_path=task.get('request_summary_path'),
                    returncode=task.get('controller_returncode'),
                    error=compact_error(task.get('error', task.get('exception'))))
    if kind == 'outer':
        for i, step in enumerate(data.get('steps', [])):
            status = str(step.get('status', '')).lower()
            if status in FAIL:
                add('step:' + str(step.get('id', i)), 'task_failure',
                    label=step.get('id', str(i)), status=status, returncode=step.get('returncode'))
    return out


class OwnerBusy(RuntimeError):
    pass


def owner_lock(path):
    f = open(path, 'a+')
    try:
        fcntl.flock(f, fcntl.LOCK_EX | fcntl.LOCK_NB)
    except BlockingIOError:
        f.close()
        raise OwnerBusy(f'already owned: {path}') from None
    return f


def validate_config(config):
    if config.get('schema_version') != 1:
        raise ValueError('schema_version must be 1')
    if not Path(config['state_dir']).is_absolute():
        raise ValueError('state_dir must be absolute')
    ids = set()
    for s in config['sources']:
        if not s.get('id') or s['id'] in ids or s.get('kind') not in KINDS:
            raise ValueError('source id must be unique and kind supported')
        ids.add(s['id'])
        paths = [s[k] for k in ('path', 'glob') if k in s]
        if len(paths) != 1 or not os.path.isabs(paths[0]):
            raise ValueError('each source requires exactly one absolute path or glob')
        if 'release_file' in s and Path(s['release_file']).name != s['release_file']:
            raise ValueError('release_file must be a basename')
    for name, default, positive in [('poll_seconds', 5, True), ('coalesce_seconds', 10, False),
                                     ('read_error_grace_sec', 30, False), ('process_grace_sec', 30, False)]:
        value = config.get(name, default)
        if not isinstance(value, (int, float)) or not math.isfinite(value) or value < 0 or (positive and value == 0):
            raise ValueError(f'invalid {name}')
    transport = config['transport']
    argv = transport.get('argv')
    if not isinstance(argv, list) or not argv or not all(isinstance(x, str) and x for x in argv):
        raise ValueError('transport.argv requires nonempty string list')
    timeout = transport.get('timeout_sec', 1815)
    if not isinstance(timeout, (int, float)) or not math.isfinite(timeout) or timeout <= 0:
        raise ValueError('transport.timeout_sec must be finite and positive')
    if type(config.get('max_events_per_batch', 128)) is not int or config.get('max_events_per_batch', 128) < 1:
        raise ValueError('max_events_per_batch must be a positive integer')
    posts = config.get('postprocess', [])
    if not isinstance(posts, list) or len(posts) > 16:
        raise ValueError('postprocess must be a finite list of at most 16 jobs')
    ids = set()
    for post in posts:
        if not post.get('id') or post['id'] in ids:
            raise ValueError('postprocess ids must be unique')
        ids.add(post['id'])
        if not isinstance(post.get('argv'), list) or not post['argv'] or not all(isinstance(v, str) and v for v in post['argv']):
            raise ValueError('postprocess argv must be a nonempty string list')
        if not isinstance(post.get('timeout_sec'), (int, float)) or not math.isfinite(post['timeout_sec']) or post['timeout_sec'] <= 0:
            raise ValueError('postprocess timeout_sec must be finite and positive')


class Daemon:
    def __init__(self, config_path):
        self.config_path = Path(config_path).resolve()
        self.config = read_json(self.config_path)
        validate_config(self.config)
        self.config_hash = digest(self.config)
        self.state = Path(self.config['state_dir'])
        self.state.mkdir(mode=0o700, parents=True, exist_ok=True)
        self.lock = owner_lock(self.state / 'owner.lock')
        self.db = None
        self.children = {}
        self.cache = {}
        try:
            self.db = sqlite3.connect(self.state / 'events.sqlite3', timeout=30)
            self.db.row_factory = sqlite3.Row
            self.db.execute('PRAGMA journal_mode=DELETE')
            self.db.execute('PRAGMA synchronous=FULL')
            self.db.executescript('''
                CREATE TABLE IF NOT EXISTS meta(key TEXT PRIMARY KEY,value TEXT NOT NULL);
                CREATE TABLE IF NOT EXISTS observations(source TEXT PRIMARY KEY,facts TEXT NOT NULL);
                CREATE TABLE IF NOT EXISTS problems(source TEXT PRIMARY KEY,started REAL NOT NULL,emitted INTEGER NOT NULL,error TEXT NOT NULL);
                CREATE TABLE IF NOT EXISTS events(event_id TEXT PRIMARY KEY,created REAL NOT NULL,payload TEXT NOT NULL,batch_id TEXT,acked INTEGER NOT NULL DEFAULT 0);
                CREATE TABLE IF NOT EXISTS batches(batch_id TEXT PRIMARY KEY,created REAL NOT NULL,status TEXT NOT NULL,payload TEXT NOT NULL,owner TEXT,reason TEXT);
                CREATE TABLE IF NOT EXISTS incidents(id TEXT PRIMARY KEY,created REAL NOT NULL,payload TEXT NOT NULL);
            ''')
            old = self.db.execute("SELECT value FROM meta WHERE key='config_hash'").fetchone()
            if old and old['value'] != self.config_hash:
                raise ValueError('config changed for existing state; use explicit reviewed state migration')
            if not old:
                atomic_json(self.state / 'HOLD', {'reason': 'initial setup; explicit arm required'})
                with self.db:
                    self.db.execute("INSERT INTO meta VALUES('config_hash',?)", (self.config_hash,))
                atomic_json(self.state / 'config.snapshot.json', self.config)
        except BaseException:
            self.close()
            raise

    def close(self):
        if self.db is not None:
            self.db.close()
            self.db = None
        if getattr(self, 'lock', None):
            self.lock.close()
            self.lock = None

    def arm(self):
        (self.state / 'HOLD').unlink(missing_ok=True)
        fsync_dir(self.state)

    def hold(self, reason='operator hold'):
        atomic_json(self.state / 'HOLD', {'reason': reason, 'created_unix': time.time()})

    def emit(self, source, path, key, fact, now, episode=None):
        payload = {'source_id': source['id'], 'path': str(path), 'key': key, **fact}
        event_id = digest([payload, episode])
        payload['event_id'] = event_id
        self.db.execute('INSERT OR IGNORE INTO events(event_id,created,payload) VALUES(?,?,?)',
                        (event_id, now, canonical(payload)))

    def problem(self, source, path, key, error, now, grace, kind='source_exception'):
        source_key = canonical([source['id'], str(path), key])
        row = self.db.execute('SELECT * FROM problems WHERE source=?', (source_key,)).fetchone()
        if row is None:
            self.db.execute('INSERT INTO problems VALUES(?,?,0,?)', (source_key, now, error))
            started, emitted = now, False
        else:
            started, emitted = row['started'], row['emitted']
        if not emitted and now - started >= grace:
            self.emit(source, path, key, {'kind': kind, 'details': {'error': error, 'first_seen_unix': started}}, now, started)
            self.db.execute('UPDATE problems SET emitted=1 WHERE source=?', (source_key,))

    def clear_problem(self, source, path, key):
        self.db.execute('DELETE FROM problems WHERE source=?', (canonical([source['id'], str(path), key]),))

    def check_process(self, source, path, data, now):
        if not source.get('monitor_process', False):
            return
        record = None
        expected = False
        if source['kind'] == 'controller':
            active, holder = data.get('active'), data.get('holder')
            if active is not None:
                expected = True
                record = active.get('controller') if isinstance(active, dict) else None
            elif holder is not None:
                expected = True
                record = holder.get('identity') if isinstance(holder, dict) else None
        elif str(data.get('status', '')).lower() not in TERMINAL:
            expected = True
            # A plain PID has no historical identity. Pin its first live identity,
            # and never accept a reused PID on subsequent observations.
            pid = data.get('pid')
            k = canonical(['pid', source['id'], str(path), pid, data.get('started_unix')])
            saved = self.db.execute('SELECT value FROM meta WHERE key=?', (k,)).fetchone()
            record = json.loads(saved['value']) if saved else identity(pid)
            if record and not saved:
                self.db.execute('INSERT INTO meta VALUES(?,?)', (k, canonical(record)))
            if record is None:
                record = {'pid': pid, 'starttime': -1, 'uid': os.getuid()}
        if isinstance(record, dict):
            record = {k: record.get(k) for k in ('pid', 'starttime', 'uid')}
        valid = isinstance(record, dict) and all(type(record.get(k)) is int for k in ('pid', 'starttime', 'uid'))
        if expected and (not valid or not alive(record)):
            reason = 'expected process identity missing or invalid' if not valid else f'expected process absent or identity changed: {canonical(record)}'
            self.problem(source, path, 'process', reason,
                         now, self.config.get('process_grace_sec', 30), 'process_lost')
        else:
            self.clear_problem(source, path, 'process')

    def scan(self, now=None):
        now = time.time() if now is None else now
        with self.db:
            baseline = self.db.execute("SELECT value FROM meta WHERE key='baseline_done'").fetchone() is None
            for source in self.config['sources']:
                paths = [Path(source['path'])] if 'path' in source else [Path(x) for x in sorted(glob.glob(source['glob'], recursive=True))]
                for path in paths:
                    if not path.exists() and not source.get('required', False):
                        continue
                    source_key = canonical([source['id'], str(path)])
                    try:
                        stat = path.stat()
                        stamp = (stat.st_ino, stat.st_size, stat.st_mtime_ns)
                        if source['kind'] == 'dock':
                            release = path.with_name(source.get('release_file', 'release.json'))
                            stamp += (release.exists(),)
                        cached = self.cache.get(source_key)
                        if cached and cached[0] == stamp:
                            self.check_process(source, path, cached[1], now)
                            continue
                        data = read_json(path)
                        new_facts = facts(source, path, data)
                    except (OSError, ValueError, TypeError, AttributeError) as exc:
                        self.problem(source, path, 'read', f'{type(exc).__name__}: {str(exc)[:1000]}', now,
                                     self.config.get('read_error_grace_sec', 30))
                        continue
                    self.clear_problem(source, path, 'read')
                    previous = self.db.execute('SELECT facts FROM observations WHERE source=?', (source_key,)).fetchone()
                    old_facts = json.loads(previous['facts']) if previous else {}
                    for key, fact in new_facts.items():
                        catchup = fact['kind'] in {'decision', 'dock'} and source.get('catch_up', True)
                        if old_facts.get(key) != fact and (not baseline or catchup):
                            self.emit(source, path, key, fact, now)
                    self.db.execute('INSERT OR REPLACE INTO observations VALUES(?,?)', (source_key, canonical(new_facts)))
                    self.check_process(source, path, data, now)
                    # Keep only process-monitor fields, not huge source manifests.
                    self.cache[source_key] = (stamp, {'status': data.get('status'),
                        'pid': data.get('pid'), 'started_unix': data.get('started_unix'),
                        'active': data.get('active'), 'holder': data.get('holder')})
            self.db.execute("INSERT OR REPLACE INTO meta VALUES('baseline_done',?)", (str(now),))

    def incident(self, batch_id, reason):
        payload = {'batch_id': batch_id, 'reason': reason}
        self.db.execute('INSERT OR IGNORE INTO incidents VALUES(?,?,?)',
                        (digest(payload), time.time(), canonical(payload)))

    def prepare_batch(self, now=None):
        now = time.time() if now is None else now
        if (self.state / 'HOLD').exists():
            return None
        with self.db:
            # Unknown/failed dispatch needs explicit resolution. Never spin the
            # model on its own failure, or overlap an uncertain old invocation.
            if self.db.execute("SELECT 1 FROM batches WHERE status NOT IN ('acked','resolved') LIMIT 1").fetchone():
                return None
            rows = self.db.execute('SELECT * FROM events WHERE batch_id IS NULL ORDER BY created,event_id').fetchall()
            if not rows or now - rows[0]['created'] < self.config.get('coalesce_seconds', 10):
                return None
            rows = rows[:self.config.get('max_events_per_batch', 128)]
            event_ids = [r['event_id'] for r in rows]
            batch_id = digest([self.config_hash, event_ids])
            payload = {'schema_version': 1, 'batch_id': batch_id, 'event_ids': event_ids,
                       'events': [json.loads(r['payload']) for r in rows],
                       'created_unix': now, 'reports': []}
            self.db.execute('INSERT INTO batches(batch_id,created,status,payload) VALUES(?,?,?,?)',
                            (batch_id, now, 'prepared', canonical(payload)))
            self.db.executemany('UPDATE events SET batch_id=? WHERE event_id=?', [(batch_id, e) for e in event_ids])
        return payload

    def materialize(self, batch_id):
        row = self.db.execute('SELECT * FROM batches WHERE batch_id=?', (batch_id,)).fetchone()
        if row is None:
            raise ValueError('unknown batch')
        folder = self.state / 'batches' / batch_id
        folder.mkdir(parents=True, exist_ok=True)
        payload = json.loads(row['payload'])
        if not (folder / 'events.json').exists():
            from reports import batch_reports
            payload['reports'] = batch_reports(payload, folder)
            atomic_json(folder / 'events.json', payload)
        if not (folder / 'job.json').exists():
            atomic_json(folder / 'job.json', {'batch_id': batch_id, 'folder': str(folder),
                        'transport': self.config['transport'], 'hold_path': str(self.state / 'HOLD'),
                        'postprocess': self.config.get('postprocess', [])})
        return folder

    def launch(self, batch_id):
        if (self.state / 'HOLD').exists():
            return
        row = self.db.execute('SELECT * FROM batches WHERE batch_id=?', (batch_id,)).fetchone()
        if row is None or row['status'] not in {'prepared', 'launched'}:
            return
        folder = self.materialize(batch_id)
        # Never execute the adapter here: the persistent worker owns dispatch.
        if (folder / 'started.json').exists() or (folder / 'result.json').exists():
            return
        if row['owner'] and alive(json.loads(row['owner'])):
            return
        with self.db:
            self.db.execute("UPDATE batches SET status='launched' WHERE batch_id=?", (batch_id,))
        with open(folder / 'worker.log', 'ab') as log:
            proc = subprocess.Popen([sys.executable, str(HERE / 'dispatch_worker.py'), '--job', str(folder / 'job.json')],
                                    stdin=subprocess.DEVNULL, stdout=log, stderr=log, start_new_session=True,
                                    env={**os.environ, 'CUDA_VISIBLE_DEVICES': ''})
        self.children[batch_id] = proc
        with self.db:
            self.db.execute('UPDATE batches SET owner=? WHERE batch_id=?', (canonical(identity(proc.pid)), batch_id))

    def reconcile(self):
        for key, proc in list(self.children.items()):
            if proc.poll() is not None:
                del self.children[key]
        with self.db:
            rows = self.db.execute("SELECT * FROM batches WHERE status IN ('prepared','launched')").fetchall()
            for row in rows:
                folder = self.state / 'batches' / row['batch_id']
                if (folder / 'result.json').exists():
                    try:
                        result = read_json(folder / 'result.json')
                    except (OSError, ValueError) as exc:
                        result = {'status': 'unknown', 'reason': f'unreadable worker result: {exc}'}
                    status = result.get('status', 'unknown')
                    reason = result.get('reason')
                    if reason is not None and not isinstance(reason, str):
                        status, reason = 'unknown', 'invalid worker result reason type'
                    if (result.get('batch_id') != row['batch_id'] or not isinstance(status, str)
                            or status not in {'acked', 'failed', 'unknown'}):
                        status, reason = 'unknown', 'invalid worker result'
                    if status == 'acked':
                        try:
                            if type(result.get('returncode')) is not int or result['returncode'] != 0:
                                raise ValueError('successful worker result requires returncode 0')
                            validate_ack(read_json(folder/'ack.json'), json.loads(row['payload']))
                        except (OSError, ValueError) as exc:
                            status, reason = 'unknown', f'invalid acknowledgement: {exc}'
                    self.db.execute('UPDATE batches SET status=?,reason=? WHERE batch_id=?', (status, reason, row['batch_id']))
                    if status == 'acked':
                        self.db.execute('UPDATE events SET acked=1 WHERE batch_id=?', (row['batch_id'],))
                    else:
                        self.incident(row['batch_id'], reason or status)
                elif (folder / 'started.json').exists():
                    try:
                        started = read_json(folder / 'started.json')
                    except (OSError, ValueError):
                        started = {}
                    if started.get('batch_id') != row['batch_id']:
                        started = {}
                    if not alive(started.get('worker')):
                        reason = 'invocation started, worker absent, no durable result; no automatic replay'
                        self.db.execute("UPDATE batches SET status='unknown',reason=? WHERE batch_id=?", (reason, row['batch_id']))
                        self.incident(row['batch_id'], reason)
                elif row['status'] == 'launched' and not (row['owner'] and alive(json.loads(row['owner']))):
                    # No invocation marker means the helper cannot have invoked
                    # transport. Its per-batch flock resolves competing helpers.
                    self.db.execute("UPDATE batches SET status='prepared',owner=NULL WHERE batch_id=?", (row['batch_id'],))

    def tick(self, now=None):
        self.scan(now)
        self.reconcile()
        self.prepare_batch(now)
        if not (self.state / 'HOLD').exists():
            row = self.db.execute("SELECT batch_id FROM batches WHERE status='prepared' ORDER BY created LIMIT 1").fetchone()
            if row:
                self.launch(row['batch_id'])
        atomic_json(self.state / 'heartbeat.json', {'pid': os.getpid(), 'identity': identity(os.getpid()),
                    'observed_unix': time.time(), **self.status()})

    def status(self):
        return {'held': (self.state / 'HOLD').exists(),
                'events': self.db.execute('SELECT count(*) FROM events').fetchone()[0],
                'unacked': self.db.execute('SELECT count(*) FROM events WHERE acked=0').fetchone()[0],
                'batches': [dict(r) for r in self.db.execute('SELECT batch_id,status,reason FROM batches ORDER BY created')],
                'incidents': self.db.execute('SELECT count(*) FROM incidents').fetchone()[0]}


def validate_ack(ack, batch):
    if ack.get('batch_id') != batch['batch_id']:
        raise ValueError('ack batch_id mismatch')
    if not isinstance(ack.get('status'), str) or ack['status'] not in {'handled', 'blocked'}:
        raise ValueError('ack status must be handled or blocked')
    if ack.get('event_ids') != batch['event_ids'] or len(set(ack['event_ids'])) != len(ack['event_ids']):
        raise ValueError('ack event_ids mismatch or duplicate')


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('command', choices=['check', 'run', 'scan', 'arm', 'hold', 'status', 'resolve'])
    p.add_argument('--config', required=True)
    p.add_argument('--reason', default='operator hold')
    p.add_argument('--batch-id')
    p.add_argument('--resolution-file')
    args = p.parse_args(argv)
    config = read_json(args.config)
    validate_config(config)
    if args.command == 'check':
        print(canonical({'valid': True, 'config_sha256': digest(config)}))
        return 0
    # arm/hold touch only the operator control file and work while run owns DB.
    state = Path(config['state_dir'])
    if args.command in {'arm', 'hold'}:
        if not (state/'config.snapshot.json').exists():
            raise ValueError('initialize with scan before arm/hold')
        if digest(read_json(state/'config.snapshot.json')) != digest(config):
            raise ValueError('config mismatch')
        if args.command == 'hold':
            atomic_json(state/'HOLD', {'reason': args.reason, 'created_unix': time.time()})
        else:
            (state/'HOLD').unlink(missing_ok=True)
            fsync_dir(state)
        return 0
    if args.command == 'status':
        # Read-only snapshot avoids a second owner or SQLite writer.
        print(canonical(read_json(state/'heartbeat.json')) if (state/'heartbeat.json').exists() else '{}')
        return 0
    d = Daemon(args.config)
    try:
        if args.command == 'resolve':
            if not args.batch_id or not args.resolution_file:
                raise ValueError('resolve needs --batch-id and --resolution-file; stop daemon first')
            resolution = read_json(args.resolution_file)
            if resolution.get('batch_id') != args.batch_id or not resolution.get('reason'):
                raise ValueError('resolution must identify batch and explain reconciled external state')
            row = d.db.execute('SELECT status FROM batches WHERE batch_id=?', (args.batch_id,)).fetchone()
            if not row or row['status'] not in {'failed', 'unknown'}:
                raise ValueError('only failed/unknown delivery can be manually resolved')
            atomic_json(d.state/'batches'/args.batch_id/'operator-resolution.json', resolution)
            with d.db:
                d.db.execute("UPDATE batches SET status='resolved',reason=? WHERE batch_id=?",
                             (canonical(resolution), args.batch_id))
            # Unacknowledged events stay unacknowledged for audit; no replay.
            return 0
        if args.command == 'scan':
            d.scan()
            d.reconcile()
            atomic_json(d.state/'heartbeat.json', {'observed_unix': time.time(), **d.status()})
            print(canonical(d.status()))
            return 0
        stop = False
        def stop_handler(signum, frame):
            nonlocal stop
            stop = True
        signal.signal(signal.SIGTERM, stop_handler)
        signal.signal(signal.SIGINT, stop_handler)
        while not stop:
            d.tick()
            deadline = time.monotonic() + config.get('poll_seconds', 5)
            while not stop and time.monotonic() < deadline:
                time.sleep(max(0, min(.25, deadline-time.monotonic())))
    finally:
        d.close()
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
