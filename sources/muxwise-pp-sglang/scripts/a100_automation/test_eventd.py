"""CPU-only tests; every source, state, and transport lives in a temporary tree."""
import importlib.util
import json
import os
from pathlib import Path
import signal
import subprocess
import sys
import tempfile
import time
import unittest

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import eventd


def write(path, data):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data))


class EventTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.root = Path(self.tmp.name)
        self.q = self.root / 'sources' / 'q' / 'status.json'
        self.config = {
            'schema_version': 1, 'state_dir': str(self.root / 'state'),
            'poll_seconds': .05, 'coalesce_seconds': 0,
            'read_error_grace_sec': 5,
            'sources': [{'id': 'q', 'kind': 'queue',
                         'glob': str(self.root / 'sources/*/status.json')}],
            'transport': {'argv': [sys.executable, '-c', 'raise SystemExit(99)'],
                          'timeout_sec': 10},
        }
        self.config_path = self.root / 'config.json'
        self.d = None

    def tearDown(self):
        if self.d:
            self.d.close()
        self.tmp.cleanup()

    def open(self):
        if self.d:
            self.d.close()
        write(self.config_path, self.config)
        self.d = eventd.Daemon(self.config_path)
        return self.d

    def events(self):
        return [json.loads(r['payload']) for r in self.d.db.execute('SELECT * FROM events ORDER BY created,event_id')]

    def queue(self, status='running', task='running'):
        write(self.q, {'queue_id': 'q', 'status': status, 'updated_unix': time.time(),
                       'tasks': {'t': {'status': task}}})

    def test_baseline_then_failure_and_terminal_edges(self):
        self.queue('failed', 'failed')
        d = self.open()
        d.scan(now=10)
        self.assertEqual(self.events(), [])
        new = self.root / 'sources/new/status.json'
        write(new, {'status': 'running', 'tasks': {'a': {'status': 'running'}}})
        d.scan(now=11)
        write(new, {'status': 'running', 'tasks': {'a': {'status': 'failed'}}})
        d.scan(now=12)
        d.scan(now=13)
        self.assertEqual([e['kind'] for e in self.events()], ['task_failure'])
        write(new, {'status': 'failed', 'tasks': {'a': {'status': 'failed'}}})
        d.scan(now=14)
        self.assertEqual(sorted(e['kind'] for e in self.events()), ['batch_terminal', 'task_failure'])
        self.open().scan(now=15)
        self.assertEqual(len(self.events()), 2)

    def test_normal_progress_and_task_completion_never_dispatch(self):
        self.queue()
        d = self.open()
        d.arm()
        for i in range(10):
            self.queue(task='completed' if i > 5 else 'running')
            d.tick(now=10+i)
        self.assertEqual(self.events(), [])
        self.assertEqual(d.db.execute('SELECT count(*) FROM batches').fetchone()[0], 0)

    def test_hold_collects_until_arm_and_coalesces(self):
        self.queue()
        d = self.open()
        d.scan(now=10)
        self.queue('failed', 'failed')
        d.tick(now=11)
        self.assertEqual(len(self.events()), 2)
        self.assertEqual(d.db.execute('SELECT count(*) FROM batches').fetchone()[0], 0)
        self.assertTrue((d.state / 'HOLD').exists())
        d.arm()
        batch = d.prepare_batch(now=12)
        self.assertEqual(len(batch['event_ids']), 2)
        self.assertIsNone(d.prepare_batch(now=13))

    def test_dock_and_decision_catchup_but_released_dock_silent(self):
        dock = self.root / 'dock/boundary.json'
        decision = self.root / 'inbox/need.json'
        released = self.root / 'old/boundary.json'
        for p in (dock, released):
            write(p, {'stage_id': 's1', 'stage_index': 1})
        write(released.with_name('release.json'), {'mode': 'continue_parameters'})
        write(decision, {'id': 'choose', 'status': 'pending', 'reason': 'needs decision'})
        self.config['sources'] = [
            {'id': 'dock', 'kind': 'dock', 'glob': str(self.root / '*/boundary.json')},
            {'id': 'inbox', 'kind': 'decision', 'glob': str(self.root / 'inbox/*.json')}]
        d = self.open()
        d.scan(now=10)
        self.assertEqual(sorted(e['kind'] for e in self.events()), ['decision', 'dock'])
        d.scan(now=11)
        self.open().scan(now=12)
        self.assertEqual(len(self.events()), 2)

    def test_partial_json_grace_recovery_and_second_episode(self):
        self.queue()
        d = self.open()
        d.scan(now=10)
        self.q.write_text('{')
        d.scan(now=11)
        d.scan(now=15)
        self.assertEqual(self.events(), [])
        d.scan(now=16)
        d.scan(now=20)
        self.assertEqual(len(self.events()), 1)
        self.queue()
        d.scan(now=21)
        self.q.write_text('{')
        d.scan(now=22)
        d.scan(now=27)
        self.assertEqual(len(self.events()), 2)

    def test_flock_denies_second_owner_and_config_drift(self):
        d = self.open()
        with self.assertRaises(eventd.OwnerBusy):
            eventd.Daemon(self.config_path)
        d.close()
        self.d = None
        self.config['coalesce_seconds'] = 42
        write(self.config_path, self.config)
        with self.assertRaisesRegex(ValueError, 'config'):
            eventd.Daemon(self.config_path)

    def test_delete_journal_full_sync(self):
        d = self.open()
        self.assertEqual(d.db.execute('PRAGMA journal_mode').fetchone()[0], 'delete')
        self.assertEqual(d.db.execute('PRAGMA synchronous').fetchone()[0], 2)

    def fake_transport(self, mode='ack', delay=0):
        fake = self.root / 'fake.py'
        fake.write_text('''import json,sys,time,os
from pathlib import Path
event,ack,log,mode,delay = sys.argv[1:]
e=json.loads(Path(event).read_text())
with open(log,'a') as f:
 f.write(e['batch_id']+'\\n'); f.flush(); os.fsync(f.fileno())
time.sleep(float(delay))
if mode=='nonzero': sys.exit(3)
if mode=='ack':
 p=Path(ack); p.write_text(json.dumps({'batch_id':e['batch_id'],'event_ids':e['event_ids'],'status':'handled'}))
''')
        self.config['transport']['argv'] = [sys.executable, str(fake), '{event_file}', '{ack_file}',
                                           str(self.root/'calls'), mode, str(delay)]

    def prepared(self):
        self.queue()
        d = self.open()
        d.scan(now=time.time()-2)
        self.queue('completed', 'completed')
        d.scan(now=time.time()-1)
        d.arm()
        return d, d.prepare_batch(now=time.time())

    def wait_batch(self, d, timeout=5):
        end = time.monotonic()+timeout
        while time.monotonic() < end:
            d.reconcile()
            r = d.db.execute('SELECT * FROM batches').fetchone()
            if r['status'] in ('acked', 'unknown', 'failed'):
                return dict(r)
            time.sleep(.03)
        self.fail('worker did not terminate')

    def test_fake_ack_and_restarts_do_not_repeat(self):
        self.fake_transport()
        d, batch = self.prepared()
        d.launch(batch['batch_id'])
        self.assertEqual(self.wait_batch(d)['status'], 'acked')
        self.open().tick()
        self.assertEqual((self.root/'calls').read_text().splitlines(), [batch['batch_id']])
        self.assertEqual(self.d.db.execute("SELECT count(*) FROM events WHERE acked=1").fetchone()[0], 1)

    def test_live_worker_survives_daemon_close_without_duplicate(self):
        self.fake_transport(delay=.3)
        d, batch = self.prepared()
        d.launch(batch['batch_id'])
        original = d.children[batch['batch_id']]
        d = self.open()
        for _ in range(3):
            d.tick()
            time.sleep(.03)
        self.assertEqual(self.wait_batch(d)['status'], 'acked')
        original.wait(timeout=3)
        self.assertEqual((self.root/'calls').read_text().splitlines(), [batch['batch_id']])

    def test_missing_ack_unknown_does_not_reinvoke_and_blocks_new_dispatch(self):
        self.fake_transport(mode='noack')
        d, batch = self.prepared()
        d.launch(batch['batch_id'])
        self.assertEqual(self.wait_batch(d)['status'], 'unknown')
        self.queue('failed', 'failed')
        for _ in range(3):
            d.tick()
        self.assertEqual((self.root/'calls').read_text().splitlines(), [batch['batch_id']])
        self.assertTrue(d.db.execute("SELECT count(*) FROM events WHERE batch_id IS NULL").fetchone()[0])

    def test_started_without_live_worker_is_unknown_not_replayed(self):
        self.fake_transport()
        d, batch = self.prepared()
        folder = d.state / 'batches' / batch['batch_id']
        write(folder/'started.json', {'batch_id': batch['batch_id'], 'worker': {'pid': 999999999, 'starttime': 0}})
        d.reconcile()
        d.tick()
        self.assertEqual(d.db.execute('SELECT status FROM batches').fetchone()[0], 'unknown')
        self.assertFalse((self.root/'calls').exists())

    def test_nonzero_transport_preserves_failed_events(self):
        self.fake_transport(mode='nonzero')
        d,b = self.prepared()
        d.launch(b['batch_id'])
        self.assertEqual(self.wait_batch(d)['status'], 'failed')
        d.tick()
        self.assertEqual(d.db.execute('SELECT acked FROM events').fetchone()[0], 0)
        self.assertEqual(len((self.root/'calls').read_text().splitlines()), 1)

    def test_two_helpers_same_batch_call_transport_once(self):
        self.fake_transport(delay=.2)
        d, batch = self.prepared()
        folder = d.materialize(batch['batch_id'])
        argv = [sys.executable, str(HERE/'dispatch_worker.py'), '--job', str(folder/'job.json')]
        helpers = [subprocess.Popen(argv) for _ in range(2)]
        for p in helpers:
            self.assertEqual(p.wait(timeout=3), 0)
        self.assertEqual(self.wait_batch(d)['status'], 'acked')
        self.assertEqual((self.root/'calls').read_text().splitlines(), [batch['batch_id']])

    def test_prepared_crash_window_recovers_only_unstarted_helper(self):
        self.fake_transport()
        d, batch = self.prepared()
        # The daemon committed launch intent but died before spawning a helper.
        with d.db:
            d.db.execute("UPDATE batches SET status='launched',owner=NULL")
        d = self.open()
        d.tick()
        self.assertEqual(self.wait_batch(d)['status'], 'acked')
        self.assertEqual((self.root/'calls').read_text().splitlines(), [batch['batch_id']])

    def test_hold_rechecked_by_worker_after_batch_prepared(self):
        self.fake_transport()
        d, batch = self.prepared()
        folder = d.materialize(batch['batch_id'])
        d.hold()
        completed = subprocess.run([sys.executable, str(HERE/'dispatch_worker.py'), '--job', str(folder/'job.json')], timeout=3)
        self.assertEqual(completed.returncode, 0)
        self.assertFalse((folder/'started.json').exists())
        self.assertFalse((self.root/'calls').exists())
        d.arm()
        d.tick()
        self.assertEqual(self.wait_batch(d)['status'], 'acked')

    def test_transport_timeout_unknown_no_repeat(self):
        self.fake_transport(delay=1)
        self.config['transport']['timeout_sec'] = .05
        d, b = self.prepared()
        d.launch(b['batch_id'])
        self.assertEqual(self.wait_batch(d)['status'], 'unknown')
        d.tick()
        self.assertEqual(len((self.root/'calls').read_text().splitlines()), 1)

    def test_worker_death_after_start_keeps_unknown_even_if_transport_later_acks(self):
        self.fake_transport(delay=.2)
        d, b = self.prepared()
        d.launch(b['batch_id'])
        original = d.children[b['batch_id']]
        deadline = time.monotonic()+3
        while not (self.root/'calls').exists() and time.monotonic() < deadline:
            time.sleep(.01)
        self.assertTrue((self.root/'calls').exists())
        # Signal only the fake notification worker launched inside this test.
        original.kill()
        original.wait(timeout=3)
        d.reconcile()
        self.assertEqual(d.db.execute('SELECT status FROM batches').fetchone()[0], 'unknown')
        time.sleep(.25)
        d.tick()
        self.assertEqual(d.db.execute('SELECT status FROM batches').fetchone()[0], 'unknown')
        self.assertEqual(len((self.root/'calls').read_text().splitlines()), 1)

    def test_corrupt_result_retained_unknown_without_crashing_poll(self):
        self.fake_transport()
        d, b = self.prepared()
        folder = d.materialize(b['batch_id'])
        (folder/'result.json').write_text('{')
        d.tick()
        self.assertEqual(d.db.execute('SELECT status FROM batches').fetchone()[0], 'unknown')
        self.assertFalse((self.root/'calls').exists())

    def test_bad_ack_batch_or_ids_rejected(self):
        batch = {'batch_id': 'a', 'event_ids': ['1', '2']}
        for ack in [{'batch_id': 'b', 'event_ids': ['1', '2'], 'status': 'handled'},
                    {'batch_id': 'a', 'event_ids': ['1'], 'status': 'handled'},
                    {'batch_id': 'a', 'event_ids': ['2', '1'], 'status': 'handled'},
                    {'batch_id': 'a', 'event_ids': ['1', '2'], 'status': 'running'}]:
            with self.assertRaises(ValueError):
                eventd.validate_ack(ack, batch)

    def test_process_loss_detected_despite_unchanged_file_and_pid_reuse(self):
        self.config['sources'] = [{'id': 'outer', 'kind': 'outer', 'path': str(self.q), 'monitor_process': True}]
        self.config['process_grace_sec'] = 5
        write(self.q, {'status': 'running', 'pid': os.getpid(), 'started_unix': 1})
        d = self.open()
        d.scan(now=10)
        self.assertEqual(self.events(), [])
        from unittest.mock import patch
        with patch.object(eventd, 'identity', return_value={'pid': os.getpid(), 'starttime': 1, 'uid': os.getuid()}):
            d.scan(now=11)
            d.scan(now=16)
        self.assertEqual([e['kind'] for e in self.events()], ['process_lost'])

    def test_missing_required_process_identity_raises_once_but_idle_does_not(self):
        self.config['sources'] = [{'id': 'controller', 'kind': 'controller', 'path': str(self.q), 'monitor_process': True}]
        self.config['process_grace_sec'] = 0
        write(self.q, {'active': None, 'holder': None})
        d = self.open()
        d.scan(now=10)
        self.assertEqual(self.events(), [])
        write(self.q, {'active': {'label': 'run'}, 'holder': None})
        d.scan(now=11)
        d.scan(now=12)
        self.assertEqual([e['kind'] for e in self.events()], ['process_lost'])

    def test_outer_missing_pid_is_not_silently_healthy(self):
        self.config['sources'] = [{'id': 'outer', 'kind': 'outer', 'path': str(self.q), 'monitor_process': True}]
        self.config['process_grace_sec'] = 0
        write(self.q, {'status': 'running', 'steps': []})
        d = self.open()
        d.scan(now=10)
        self.assertEqual([e['kind'] for e in self.events()], ['process_lost'])

    def test_idle_holder_loss_detected_from_unchanged_controller(self):
        self.config['sources'] = [{'id': 'controller', 'kind': 'controller', 'path': str(self.q), 'monitor_process': True}]
        self.config['process_grace_sec'] = 0
        ident = eventd.identity(os.getpid())
        write(self.q, {'active': None, 'holder': {'identity': ident}})
        d = self.open()
        d.scan(now=10)
        self.assertEqual(self.events(), [])
        from unittest.mock import patch
        with patch.object(eventd, 'identity', return_value=None):
            d.scan(now=11)
        self.assertEqual([e['kind'] for e in self.events()], ['process_lost'])

    def test_released_nested_outer_dock_suppressed_and_new_stage_identified(self):
        self.config['sources'] = [{'id': 'outer', 'kind': 'outer', 'path': str(self.q)}]
        write(self.q, {'status': 'docked', 'dock': {'path': '/dock1', 'sha256': 'one', 'stage_index': 1, 'released': True}})
        d = self.open()
        d.scan(now=10)
        self.assertEqual(self.events(), [])
        write(self.q, {'status': 'docked', 'dock': {'path': '/dock2', 'sha256': 'two', 'stage_index': 2}})
        d.scan(now=11)
        event = self.events()[0]
        self.assertEqual(event['details']['stage_index'], 2)
        self.assertEqual(event['details']['dock_path'], '/dock2')
        self.assertEqual(event['details']['dock_sha256'], 'two')

    def test_native_interrupted_and_outer_finished_with_failures(self):
        self.queue()
        d = self.open()
        d.scan(now=10)
        self.queue('failed', 'interrupted')
        d.scan(now=11)
        self.assertEqual(sorted(e['kind'] for e in self.events()), ['batch_terminal', 'task_failure'])
        self.assertIn('terminal', eventd.facts({'kind': 'outer'}, self.q, {'status': 'finished_with_failures'}))

    def test_dock_release_changes_without_boundary_rewrite(self):
        self.config['sources'] = [{'id': 'dock', 'kind': 'dock', 'path': str(self.q)}]
        write(self.q, {'stage_id': 's'})
        d = self.open()
        d.scan(now=10)
        write(self.q.with_name('release.json'), {'mode': 'continue'})
        d.scan(now=11)
        self.assertEqual(json.loads(d.db.execute('SELECT facts FROM observations').fetchone()[0]), {})

    def test_finite_postprocess_failure_not_recursive_and_adapter_still_notified(self):
        self.fake_transport()
        self.config['postprocess'] = [{'id': 'bad', 'argv': [sys.executable, '-c', 'raise SystemExit(3)'], 'timeout_sec': 3}]
        d,b = self.prepared()
        d.launch(b['batch_id'])
        self.assertEqual(self.wait_batch(d)['status'], 'acked')
        folder = d.state/'batches'/b['batch_id']
        self.assertEqual(json.loads((folder/'postprocess-0.json').read_text())['returncode'], 3)
        self.assertEqual(len(self.events()), 1)

    def test_argv_literals_not_interpreted_by_shell(self):
        from dispatch_worker import expand
        value = '$(touch /tmp/NOT_CREATED) ; `echo hello`'
        self.assertEqual(expand(['--key', value, '{batch_id}'], {'batch_id': 'b'}), ['--key', value, 'b'])


if __name__ == '__main__':
    unittest.main()
