"""CPU-only pool boundary and exact saved-evidence regressions."""
import copy
import importlib.util
import json
from pathlib import Path
import sys
import unittest
from unittest.mock import patch

HERE=Path(__file__).resolve().parent
sys.path.insert(0,str(HERE))
def load(name):
    spec=importlib.util.spec_from_file_location('_pool_test_'+name,HERE/(name+'.py'))
    value=importlib.util.module_from_spec(spec);spec.loader.exec_module(value);return value

class PoolReaders(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.e=load('pool_evidence');cls.q=load('queue');cls.launch=load('run_campaign')
        cls.fixture_run=cls.e.ROOT/'preflight/pool-functional-8b-chunked-tp4-customar-v1'
        cls.metadata=cls.e.read(cls.fixture_run/'run.json')
        cls.command=cls.e.read(cls.e.ROOT/'runs'/cls.fixture_run.name/'command.json')
    def test_exact622_source_and_pool_metadata(self):
        self.assertEqual(len(self.e.current_sources()),622)
        self.e.validate_metadata(self.fixture_run,self.metadata)
    def test_main_runner_alias_is_not_pool_runner(self):
        m=copy.deepcopy(self.metadata)
        src=m['source_sha256'];key='scripts/a100_gpu4567/server_run.py'
        src['scripts/a100_gpu0123/server_run.py']=src.pop(key)
        with self.assertRaisesRegex(ValueError,'source'):self.e.validate_metadata(self.fixture_run,m)
    def test_changed_runtime_hash_rejected(self):
        m=copy.deepcopy(self.metadata);m['source_sha256'][next(iter(m['source_sha256']))]='0'*64
        with self.assertRaisesRegex(ValueError,'source'):self.e.validate_metadata(self.fixture_run,m)
    def test_conflicting_path_alias_rejected(self):
        m=copy.deepcopy(self.metadata);key=next(iter(m['source_sha256']))
        m['source_sha256'][str(self.e.REPO/key)]='0'*64
        with self.assertRaisesRegex(ValueError,'alias'):self.e.validate_metadata(self.fixture_run,m)
    def test_wrong_physical_pool_rejected(self):
        for field,value in [('indices',[0,1,2,3]),('uuids',list(reversed(self.e.UUIDS))),('numa_node',0),('controller_root',str(self.e.ROOT.parent)),('cpu_affinity',[0])]:
            with self.subTest(field=field):
                m=copy.deepcopy(self.metadata);m['physical_pool'][field]=value
                with self.assertRaises(ValueError):self.e.validate_metadata(self.fixture_run,m)
    def test_wrong_worker_mask_or_port_rejected(self):
        for field,value in [('gpus',[self.e.UUIDS[0]]),('port',32100),('nccl_port',32200)]:
            m=copy.deepcopy(self.metadata);m['workers'][0][field]=value
            with self.assertRaises(ValueError):self.e.validate_metadata(self.fixture_run,m)
    def test_command_single_numa1_prefix(self):
        t={'id':'example','timeout':100,'argv':['--out',str(self.e.ROOT/'preflight/example')]}
        controller,actual=self.q.commands(t,self.e.ROOT)
        self.assertEqual(actual[:3],['taskset','-c','32-63,96-127'])
        self.assertEqual(controller[controller.index('--')+1:],actual[3:])
        self.e.validate_command(self.fixture_run,self.metadata,self.command)
    def test_foreign_command_or_duplicate_prefix_rejected(self):
        for prefix in [['taskset','-c','0-31,64-95'],['taskset','-c','32-63,96-127','taskset','-c','32-63,96-127']]:
            c=copy.deepcopy(self.command);c['argv']=prefix+c['argv'][3:]
            with self.assertRaises(ValueError):self.e.validate_command(self.fixture_run,self.metadata,c)
    def test_cross_root_queue_refused_before_task_checks(self):
        with self.assertRaisesRegex(ValueError,'root'):self.q.validate_plan({},self.e.ROOT.parent)
        with self.assertRaisesRegex(ValueError,'root'):self.launch.prepare('/absent','x',root=self.e.ROOT.parent)
    def test_queue_needs_actual_current_serving_source(self):
        task={'id':'x','phase':'development','argv':['--out',str(self.e.ROOT/'x')]}
        with self.assertRaisesRegex(ValueError,'source'):self.q.check_task(task,{'source_sha256':{}},None)
    def test_actual_completed_full_gate_preserves_numeric_scope(self):
        report=self.e.audit_full_gate(self.fixture_run)
        self.assertEqual(report['status'],'passed',report.get('errors'))
        self.assertEqual(report['correctness_verdict'],'not_compared')
        self.assertFalse(report['cross_system_numerical_equivalence'])
        self.assertEqual(report['rank_count'],4)
        self.assertEqual(report['raw_coverage']['folders'][0]['stress200'],200)
        self.assertEqual(report['custom_allreduce']['positive_rank_markers'],4)
    def test_incomplete_full_gate_never_passes(self):
        original=self.e.read
        def changed(path):
            d=original(path)
            if Path(path).name=='outcome.json':d={**d,'status':'failed','returncode':1}
            return d
        with patch.object(self.e,'read',side_effect=changed):
            report=self.e.audit_full_gate(self.fixture_run)
        self.assertEqual(report['status'],'failed')

if __name__=='__main__':unittest.main()

class PoolLaunch(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.e=load('pool_evidence');cls.q=load('queue');cls.launch=load('run_campaign')
    def test_cpu_prepare_new_pool_platform_and_commands(self):
        import tempfile
        with tempfile.TemporaryDirectory() as directory:
            path=Path(directory)/'tasks.json'
            task=dict(id='cpu-only-unlaunched',phase='correctness',timeout=100,argv=['--model','8b','--family','chunked','--out',str(self.e.ROOT/'preflight/cpu-only-unlaunched')])
            path.write_text(json.dumps(dict(tasks=[task],source_sha256=self.e.current_sources())))
            manifest,env=self.launch.prepare(path,'cpu-only-unlaunched',base_env={'MC_INTRANODE_NVLINK':'0'})
            self.assertEqual(manifest['physical_pool'],self.e.pool_identity())
            self.assertNotIn('MC_INTRANODE_NVLINK',env)
            self.assertIn(str(HERE/'queue.py'),manifest['command'])
            self.assertEqual(manifest['campaign_root'],str(self.e.ROOT))
            self.assertFalse((self.e.ROOT/'queues/cpu-only-unlaunched').exists())
            nvlink,_=self.launch.prepare(path,'cpu-only-unlaunched',base_env={},mooncake_transport='nvlink')
            self.assertEqual(nvlink['public_environment']['MC_INTRANODE_NVLINK'],'1')
    def test_pd_actual_completed_full_gate(self):
        path=self.e.ROOT/'preflight/pool-functional-8b-pd-3p1d-nvlink-v1'
        report=self.e.audit_full_gate(path)
        self.assertEqual(report['status'],'passed',report['errors'])
        self.assertEqual(len(report['raw_coverage']['folders']),3)
        self.assertEqual(report['mooncake_transport'],'nvlink')
        self.assertEqual([v['role'] for v in report['worker_api']],['prefill','prefill','prefill','decode'])
    def test_actual_gates_never_depend_on_main_serving_client(self):
        report=self.e.audit_full_gate(self.e.ROOT/'preflight/pool-functional-8b-chunked-tp4-customar-v1')
        self.assertEqual(report['status'],'passed',report['errors'])
        self.assertFalse(any('/scripts/a100_gpu0123/' in path for path in report['evidence_sha256']))

class PoolSelector(unittest.TestCase):
    @classmethod
    def setUpClass(cls):cls.s=load('select_development');cls.e=load('pool_evidence')
    def test_main_pool_task_rejected_before_reads(self):
        with self.assertRaisesRegex(ValueError,'outside fixed pool'):
            self.s.read_measurement(self.e.ROOT.parent/'runs/some-main-task',{})
    def test_preserves_actual_source_identity_and_fixed_ranking(self):
        self.assertEqual(self.s.rank_key(dict(historical_main_slo=.7,ttft_p95_sec=2,output_tokens_per_sec=10,configuration_fingerprint='a')),(-.7,2,-10,'a'))
        self.assertEqual(self.s.metrics.__file__,str(HERE/'bench/metrics.py'))
        self.assertEqual(self.s._common.__file__,str(HERE/'bench/common.py'))
    def test_cached_foreign_common_cannot_contaminate_pool_metrics(self):
        import types
        foreign=types.ModuleType('common')
        with patch.dict(sys.modules,{'common':foreign}):
            s=load('select_development')
            self.assertIs(sys.modules['common'],foreign)
            self.assertIsNot(s._common,foreign)
    def _wrapped(self,*,resource_error=False,source_error=False,outcome_error=False):
        e=self.s._pool;taskdir=e.ROOT/'runs/synthetic-pool-finalist';out=e.ROOT/'development/synthetic-pool-finalist'
        source={'actual-pool-source':'f'*64}
        row=dict(label='synthetic-pool-finalist',measurement_directory=str(out),measurement_available=True,
                 selection_eligible=True,exclusion_reasons=[],platform_provenance={'status':'verified'},
                 source_sha256=source,source_fingerprint='actual-fingerprint',failed=0,unfinished=0,unsent=0)
        provenance={'task':{'id':'synthetic-pool-finalist','argv':[],'physical_pool_id':'gpu4567-numa1','canonical_slot_id':'original-slot'}}
        def read(path):
            if Path(path).name=='outcome.json':return dict(status='completed',returncode=1 if outcome_error else 0,reservation='held')
            if Path(path).name=='run.json':return dict(status='execution_complete')
            return {}
        with patch.object(self.s,'_read_measurement',return_value=row),patch.object(Path,'read_text',return_value=json.dumps(provenance)),patch.object(e,'pin',side_effect=lambda pins,p,*a,**k:Path(p)),patch.object(e,'read',side_effect=read),patch.object(e,'validate_metadata',side_effect=ValueError('changed source') if source_error else None),patch.object(e,'validate_command'),patch.object(e,'worker_api',return_value=[]),patch.object(e,'resources',side_effect=ValueError('rank missing') if resource_error else None),patch.object(e,'custom_allreduce',return_value={}),patch.object(e,'transport_evidence',return_value={}):
            result=self.s.read_measurement(taskdir,{})
        return result,source
    def test_wrapper_keeps_real_source_and_canonical_slot(self):
        row,source=self._wrapped();self.assertTrue(row['selection_eligible']);self.assertIs(row['source_sha256'],source)
        self.assertEqual(row['source_fingerprint'],'actual-fingerprint');self.assertEqual(row['canonical_slot_id'],'original-slot')
    def test_missing_rank_source_or_failed_controller_never_selects(self):
        for field in ('resource_error','source_error','outcome_error'):
            with self.subTest(field=field):
                row,_=self._wrapped(**{field:True});self.assertFalse(row['selection_eligible']);self.assertIn('pool_evidence_gate_failed',row['exclusion_reasons'])
    def test_tp2_pd_actual_gate(self):
        report=self.e.audit_full_gate(self.e.ROOT/'preflight/pool-functional-8b-pd-1p1d-tp2-nvlink-v1')
        self.assertEqual(report['status'],'passed',report['errors']);self.assertEqual(report['rank_count'],4)
        self.assertEqual(len(report['raw_coverage']['folders']),1)
        self.assertEqual(report['custom_allreduce']['positive_rank_markers'],4)

class RawEvidence(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.e=load('pool_evidence');cls.raw=load('fullgate_evidence')
        cls.run_path=cls.e.ROOT/'preflight/pool-functional-8b-pd-1p1d-tp2-nvlink-v1'
        cls.rows=cls.e.read(cls.run_path/'pair0/cancellation.json');cls.pair=cls.e.read(cls.run_path/'pair0/pair.json')
        cls.classifier=staticmethod(cls.raw.pure_functions(HERE/'server_run.py',{'_cancellation_response_aborted','cancellation_confirmed'},{},cls.e.REPO)['cancellation_confirmed'])
    def test_raw_dualrole_capture_mutations_fail(self):
        for change in ('missing','order','rid','port','room','base64','body','role'):
            with self.subTest(change=change):
                rows=copy.deepcopy(self.rows)
                if change=='missing':rows.pop()
                elif change=='order':rows[0],rows[1]=rows[1],rows[0]
                elif change=='rid':rows[0]['requested_rid']='foreign-rid'
                elif change=='port':rows[0]['responses']['decode']['port']=32100
                elif change=='room':rows[0]['bootstrap_room']=-1
                elif change=='base64':rows[0]['responses']['decode']['raw_body_base64']='invalid'
                elif change=='body':rows[0]['responses']['decode']['response']={}
                elif change=='role':del rows[0]['responses']['prefill']
                with self.assertRaises((ValueError,KeyError)):self.e.validate_pd_capture_records(rows,self.pair,self.classifier)
    def test_truncated_stress_or_nonmatching_recovery_fail(self):
        actual=self.raw.take
        for which in ('stress200.json','after-cancel.json'):
            def changed(path,pins,repo):
                d=actual(path,pins,repo)
                if Path(path).name==which:
                    d=copy.deepcopy(d)
                    if which=='stress200.json':d.pop()
                    else:d['output_ids'][0]=(d['output_ids'][0]+1)%128000
                return d
            with self.subTest(which=which),patch.object(self.raw,'take',side_effect=changed):
                with self.assertRaises((ValueError,AssertionError)):
                    self.raw.audit_full_gate(self.run_path,self.e.read(self.run_path/'run.json'),{},self.e.REPO)

class PoolReport(unittest.TestCase):
    @classmethod
    def setUpClass(cls):cls.r=load('report')
    def fixture(self,directory):
        p=Path(directory);evidence=p/'evidence';evidence.write_text('immutable-reviewed-budget-fixture')
        tasks=[];slots=[]
        for group in range(10):
            for slot in range(1,5):
                label=f'group{group}-slot{slot}'
                tasks.append(dict(id=label,canonical_slot_id='canonical-'+label,profile_slot=slot,physical_pool_id='gpu4567-numa1',seed_id=f'seed{group}',candidate_id=f'candidate{group}',workload='toolagent',finalist_role='main'))
                slots.append(dict(execution_task_id=label,canonical_slot_id='canonical-'+label,profile_slot=slot,state='unstarted',consumed=False,definitely_unstarted=True,terminal_consistent=False))
        plan=p/'tasks.json';plan.write_text(json.dumps({'tasks':tasks}))
        budget=p/'budget.json';data=dict(total_original_slots=40,additional_slots=0,status='valid_budget_snapshot',slots=slots,evidence_sha256={str(evidence):self.r.evidence.sha(evidence)})
        budget.write_text(json.dumps(data));return plan,budget,data
    def test_pending_all40_retained_without_draft_winner(self):
        import tempfile
        with tempfile.TemporaryDirectory() as tmp:
            plan,budget,_=self.fixture(tmp);result=self.r.summarize([plan],budget,self.r.evidence.sha(budget))
            self.assertEqual(len(result['rows']),40);self.assertEqual(len(result['groups']),10)
            self.assertTrue(all(g['status']=='withheld' and g['draft_best_label'] is None for g in result['groups']))
    def test_budget_unknown_and_missing_never_disappear(self):
        import tempfile
        with tempfile.TemporaryDirectory() as tmp:
            plan,budget,data=self.fixture(tmp);data['status']='review_required';data['slots'][0].update(state='unknown',consumed=True)
            budget.write_text(json.dumps(data));result=self.r.summarize([plan],budget,self.r.evidence.sha(budget))
            self.assertEqual(result['rows'][0]['budget_state'],'unknown');self.assertTrue(result['rows'][0]['consumed'])
            self.assertFalse(any(r['selection_eligible'] for r in result['rows']))
    def test_fifth_duplicate_slot_or_wrong_pool_rejected(self):
        import tempfile
        for mutation in ('fifth','duplicate','pool'):
            with self.subTest(mutation=mutation),tempfile.TemporaryDirectory() as tmp:
                plan,budget,_=self.fixture(tmp);data=json.loads(plan.read_text())
                if mutation=='fifth':data['tasks'][0]['profile_slot']=5
                elif mutation=='duplicate':data['tasks'][0]['canonical_slot_id']=data['tasks'][1]['canonical_slot_id']
                else:data['tasks'][0]['physical_pool_id']='gpu0123'
                plan.write_text(json.dumps(data))
                with self.assertRaises(ValueError):self.r.summarize([plan],budget,self.r.evidence.sha(budget))
    def test_budget_evidence_mutation_fails_closed(self):
        import tempfile
        with tempfile.TemporaryDirectory() as tmp:
            plan,budget,_=self.fixture(tmp);(Path(tmp)/'evidence').write_text('changed')
            with self.assertRaises(ValueError):self.r.summarize([plan],budget,self.r.evidence.sha(budget))
