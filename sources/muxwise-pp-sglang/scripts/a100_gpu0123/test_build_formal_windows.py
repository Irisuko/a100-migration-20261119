"""CPU tests use synthetic source rows; no campaign formal window is selected."""
import hashlib
import importlib.util
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

HERE=Path(__file__).resolve().parent
PATH=HERE/'build_formal_windows.py'
target=None
if PATH.is_file():
    spec=importlib.util.spec_from_file_location('_formal_windows_target',PATH)
    target=importlib.util.module_from_spec(spec);spec.loader.exec_module(target)


def row(timestamp,input_length=8,output_length=2,hash_ids=None):
    return dict(timestamp=timestamp,input_length=input_length,output_length=output_length,
                hash_ids=[11] if hash_ids is None else hash_ids)


def write(path,data):path.write_text(json.dumps(data,indent=2)+'\n')
def sha(path):return hashlib.sha256(path.read_bytes()).hexdigest()


class FormalWindowTests(unittest.TestCase):
    def setUp(self):self.assertIsNotNone(target,'Deterministic formal-window helper is missing')

    def test_earliest_three_second_grid_window_and_rate_math(self):
        rows=list(enumerate([row(1440000),row(1443000),row(1446000),row(1449000)]))
        result=target.choose_window(rows,.01)
        self.assertEqual(result['source_end_exclusive_sec'],1449)
        self.assertEqual(result['requests'],3)
        self.assertEqual(result['arrival_window_sec'],300)
        self.assertEqual(result['active_arrival_span_sec'],200)
        self.assertEqual(result['time_scale'],300/22.5)
        self.assertEqual(result['source_indices'],[0,1,2])
        self.assertTrue(all(not x['qualifies'] for x in result['search'][:-1]))

    def test_nominal_duration_alone_cannot_qualify_a_short_burst(self):
        rows=list(enumerate([row(1440000),row(1440001),row(1440002),row(1452000)]))
        result=target.choose_window(rows,.01)
        self.assertEqual(result['source_end_exclusive_sec'],1455)
        self.assertGreaterEqual(result['active_arrival_span_sec'],180)
        self.assertFalse(result['search'][0]['qualifies'])

    def test_exact_active_span_boundary_is_accepted_without_tolerance(self):
        rows=list(enumerate(row(t) for t in [1440000,1441000,1442000,1443000,1446480]))
        result=target.choose_window(rows,.02)
        self.assertEqual(result['active_arrival_span_sec'],180)
        self.assertEqual(result['source_end_exclusive_sec'],1449)

    def test_preserves_source_indices_order_and_filters_whole_oversized_rows(self):
        rows=[(2,row(1439999)),(4,row(1440000)),(7,row(1443000,32768,1)),
              (8,row(1443000)),(9,row(1446000)),(11,row(1449000)),(15,row(3000000))]
        result=target.choose_window(rows,.01)
        self.assertEqual(result['source_indices'],[4,8,9])
        self.assertEqual(result['filtered_requests_in_window'],1)
        self.assertEqual(result['requests'],3)

    def test_insufficient_rows_or_distinct_timestamps_fail_closed(self):
        for rows,rate in [([(0,row(1440000)),(1,row(1443000)),(2,row(1446000))],1000),
                          ([(0,row(1440000)),(1,row(1440000)),(2,row(1443000))],.001)]:
            with self.assertRaisesRegex(ValueError,'qualifying'):
                target.choose_window(rows,rate)

    def test_invalid_rates_and_duplicate_or_unsorted_source_indices_reject(self):
        rows=[(0,row(1440000)),(1,row(1443000)),(2,row(1446000))]
        for rate in [0,-1,float('inf'),float('nan'),True,'0.01']:
            with self.subTest(rate=rate),self.assertRaises(ValueError):target.choose_window(rows,rate)
        for bad in [[rows[0],rows[0],rows[2]],[rows[1],rows[0],rows[2]]]:
            with self.assertRaises(ValueError):target.choose_window(bad,.01)

    def test_frozen_rates_required_and_record_digest_exact(self):
        record=dict(model='8b',workload='toolagent',absolute_requests_per_sec=.01,role='main')
        good=dict(schema_version=1,status='frozen',records=[record],selection_manifest_sha256='c'*64,
                  evidence_phases=['rate_calibration'],evidence_sha256={'development-evidence.json':'e'*64})
        self.assertEqual(target.validate_rates(good),[record])
        expected=hashlib.sha256(json.dumps(record,sort_keys=True,separators=(',',':'),allow_nan=False).encode()).hexdigest()
        self.assertEqual(target.digest(record),expected)
        for status in ['draft','reviewed',None]:
            with self.assertRaises(ValueError):target.validate_rates({**good,'status':status})
        with self.assertRaises(ValueError):target.validate_rates({**good,'records':[record,record]})
        with self.assertRaises(ValueError):target.validate_rates({**good,'evidence_phases':['formal_main']})

    def test_invalid_freeze_metadata_rejected(self):
        good=dict(schema_version=1,status='frozen',selection_manifest_sha256='c'*64,
                  evidence_phases=['rate_calibration'],evidence_sha256={'evidence.json':'e'*64},
                  records=[dict(model='70b',workload='conversation',absolute_requests_per_sec=.5,role='main')])
        for change in [dict(evidence_sha256={}),dict(selection_manifest_sha256='bad'),dict(evidence_phases=[]),
                       dict(records=[]),dict(records=[{**good['records'][0],'model':'mistral'}]),
                       dict(records=[{**good['records'][0],'role':'pilot'}])]:
            with self.subTest(change=change),self.assertRaises(ValueError):target.validate_rates({**good,**change})

    def test_noninteger_or_out_of_order_raw_fields_rejected(self):
        good=[(0,row(1440000)),(1,row(1443000)),(2,row(1446000))]
        for change in [dict(timestamp=1440000.0),dict(timestamp=-1),dict(input_length=True),
                       dict(input_length=0),dict(output_length=-1),dict(timestamp=1447000)]:
            bad=[(0,{**good[0][1],**change}),*good[1:]]
            with self.subTest(change=change),self.assertRaises(ValueError):target.choose_window(bad,.01)

    def test_synthetic_build_uses_canonical_builder_and_target_tokenizer_without_padding(self):
        repo=HERE.parents[1];canonical=repo.parent/'muxwise-pp-experiments/bench/build_trace.py'
        original=json.loads((repo/'run/a100-gpu0123-20261114/traces/manifest.json').read_text())
        models=original['models']
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp);source=root/'synthetic-source.jsonl'
            raw=[row(1200000),row(1440000),row(1443000,32768,1,[21]*64),
                 row(1443000,12),row(1446000,16),row(1449000,20)]
            source.write_text('\n'.join(json.dumps(r) for r in raw)+'\n')
            parent=root/'synthetic-parent.json'
            parent_data=dict(schema_version=1,status='PASS',builder={'path':str(canonical),'sha256':sha(canonical)},
                sources=[dict(path=str(source),sha256=sha(source),workload='toolagent')],models=models,records=[
                    dict(model=Path(m['path']).name,workload='toolagent',purpose=purpose,source_start_sec=start,source_end_exclusive_sec=end)
                    for m in models for purpose,start,end in [('dev',1200,1440),('heldout',1440,3000)]])
            write(parent,parent_data);rates=root/'rates.json'
            records=[dict(model='8b',workload='toolagent',absolute_requests_per_sec=.01,role=role) for role in ['main','critical']]
            records.append(dict(model='70b',workload='toolagent',absolute_requests_per_sec=.01,role='main'))
            evidence=root/'synthetic-dev-evidence.json';write(evidence,{'phase':'rate_calibration','synthetic_fixture':True})
            write(rates,dict(schema_version=1,status='frozen',records=records,selection_manifest_sha256='c'*64,
                             evidence_phases=['rate_calibration'],evidence_sha256={str(evidence):sha(evidence)}))
            out=root/'generated'
            manifest=target.build_windows(rates,parent,out,expected_parent_sha256=sha(parent))
            self.assertEqual(manifest['status'],'PASS')
            self.assertEqual(len(manifest['records']),3)
            a,b,c=manifest['records'];self.assertEqual(a['path'],b['path'])
            self.assertNotEqual(a['path'],c['path'])
            self.assertEqual(Path(c['tokenizer']['path']).name,'Llama-3.3-70B-Instruct')
            self.assertEqual(a['sha256'],c['sha256'])
            self.assertNotEqual(a['rate_record_sha256'],b['rate_record_sha256'])
            self.assertEqual(a['source_end_exclusive_sec'],1449)
            self.assertEqual(a['requests'],3);self.assertEqual(a['padding_tokens'],0)
            self.assertEqual(a['dev_overlap_source_rows'],0)
            self.assertTrue(a['all_rows_and_hash_blocks_validated'])
            generated=[json.loads(line) for line in Path(a['path']).read_text().splitlines()]
            self.assertEqual([x['idx'] for x in generated],[1,3,4])
            self.assertEqual([x['input_len'] for x in generated],[8,12,16])
            self.assertEqual([x['arrival'] for x in generated],[0,7.5,15])
            self.assertEqual(sha(source),parent_data['sources'][0]['sha256'])
            self.assertEqual(len(list(out.glob('*.jsonl'))),2)
            with self.assertRaises(ValueError):target.build_windows(rates,parent,out,expected_parent_sha256=sha(parent))
            for label,expected_hash in [('untrusted-parent','0'*64)]:
                rejected=root/label
                with self.assertRaisesRegex(ValueError,'hash mismatch'):
                    target.build_windows(rates,parent,rejected,expected_parent_sha256=expected_hash)
                self.assertFalse(rejected.exists())
            evidence_original=evidence.read_bytes();evidence.write_text('{}\n')
            with self.assertRaisesRegex(ValueError,'hash mismatch'):
                target.build_windows(rates,parent,root/'changed-evidence',expected_parent_sha256=sha(parent))
            self.assertFalse((root/'changed-evidence').exists());evidence.write_bytes(evidence_original)
            source_original=source.read_bytes();raw[1]['hash_ids']=[]
            source.write_text('\n'.join(json.dumps(r) for r in raw)+'\n')
            parent_data['sources'][0]['sha256']=sha(source);write(parent,parent_data)
            with self.assertRaisesRegex(ValueError,'Padding'):
                target.build_windows(rates,parent,root/'padding-refused',expected_parent_sha256=sha(parent))
            self.assertFalse((root/'padding-refused').exists())
            source.write_bytes(source_original);parent_data['sources'][0]['sha256']=sha(source);write(parent,parent_data)
            real_run=target.subprocess.run
            def mutate_after_build(*args,**kwargs):
                result=real_run(*args,**kwargs)
                source.write_bytes(source_original+b'\n')
                return result
            with patch.object(target.subprocess,'run',side_effect=mutate_after_build):
                with self.assertRaisesRegex(ValueError,'Input changed during build'):
                    target.build_windows(rates,parent,root/'changed-during-build',expected_parent_sha256=sha(parent))
            self.assertFalse((root/'changed-during-build/manifest.json').exists())


if __name__=='__main__':unittest.main()
