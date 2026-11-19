import unittest
from compare_preflight import compare_rows
class ComparisonTests(unittest.TestCase):
 def test_empty_artifacts_never_pass(self):
  with self.assertRaises(ValueError):compare_rows([],[])
 def test_missing_request_never_passes(self):
  with self.assertRaises(ValueError):compare_rows([dict(input_ids=[1],output_ids=[2])],[])
 def test_changed_prompt_never_compares(self):
  with self.assertRaises(ValueError):compare_rows([dict(input_ids=[1],output_ids=[2])],[dict(input_ids=[3],output_ids=[2])])
 def test_non_exact_tokens_remain_differences(self):
  r=compare_rows([dict(input_ids=[1],output_ids=[2,3])],[dict(input_ids=[1],output_ids=[2,4])])
  self.assertEqual(r['status'],'numerical_differences_require_analysis')
  self.assertEqual(r['rows'][0]['first_mismatch'],1)
if __name__=='__main__':unittest.main()

class PDComparisonTests(unittest.TestCase):
 def test_pd_pairs_compare_with_ordinary_prefill_worker(self):
  import tempfile,json
  from pathlib import Path
  from compare_preflight import compare
  with tempfile.TemporaryDirectory() as directory:
   ref=Path(directory)/'ref';got=Path(directory)/'pd';ref.mkdir();got.mkdir()
   args={'model':'8b','tp':1,'pp':1,'replicas':4,'suite':'correctness'}
   (ref/'run.json').write_text(json.dumps({'args':dict(args,family='plain')}))
   (got/'run.json').write_text(json.dumps({'args':dict(args,family='pd',prefill_replicas=2)}))
   rows=[{'input_ids':[1],'output_ids':[2,3]}]
   for index in range(2):
    (ref/f'worker{index}').mkdir();(got/f'pair{index}').mkdir()
    (got/f'pair{index}'/'pair.json').write_text(json.dumps({'prefill':{'index':index},'decode':{'index':index+2}}))
    for suite in ['serial','cache','concurrent']:
     (ref/f'worker{index}'/(suite+'.json')).write_text(json.dumps(rows))
     (got/f'pair{index}'/(suite+'.json')).write_text(json.dumps(rows))
   self.assertEqual(compare(ref,got)['status'],'exact')
