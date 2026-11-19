#!/usr/bin/env python3
"""Compare same-model/same-topology preflight tokens without tolerances."""
import argparse,json
from pathlib import Path

def compare_rows(expected, actual):
 rows=[]
 if not expected:raise ValueError('Empty comparison artifacts')
 if len(expected)!=len(actual):raise ValueError('Missing or extra request records')
 for i,(ref,got) in enumerate(zip(expected,actual)):
  if ref['input_ids']!=got['input_ids']:raise ValueError(f'Request{i}: different input IDs')
  a,b=ref['output_ids'],got['output_ids']
  if len(a)!=len(b):raise ValueError(f'Request{i}: different output lengths')
  mismatches=[j for j,(x,y) in enumerate(zip(a,b)) if x!=y]
  rows.append({'request':i,'input_tokens':len(ref['input_ids']),'output_tokens':len(a),'exact':not mismatches,'mismatch_count':len(mismatches),'first_mismatch':mismatches[0] if mismatches else None})
 return {'status':'exact' if all(r['exact'] for r in rows) else 'numerical_differences_require_analysis','exact_requests':sum(r['exact'] for r in rows),'total_requests':len(rows),'rows':rows}

def compare(ref,actual):
 r=json.loads((ref/'run.json').read_text());a=json.loads((actual/'run.json').read_text())
 for key in ('model','tp','pp','replicas','suite'):
  if r['args'][key]!=a['args'][key]:raise ValueError(f'Incomparable {key}')
 if r['args']['family']!='plain':raise ValueError('Reference must be ordinary inference')
 if r['args']['suite']!='correctness':raise ValueError('Full correctness preflight is required')
 result={'reference':str(ref),'candidate':str(actual),'parts':{},'status':'incomplete'}
 if a['args']['family']=='pd':
  p=a['args']['prefill_replicas'];d=a['args']['replicas']-p
  pairs=[]
  for index in range(max(p,d)):
   pair_path=Path(f'pair{index}')
   pair=json.loads((actual/pair_path/'pair.json').read_text())
   expected_p=index%p;expected_d=p+index%d
   if pair['prefill']['index']!=expected_p or pair['decode']['index']!=expected_d:
    raise ValueError('PD pairing metadata does not cover expected instances')
   pairs.append((Path(f'worker{expected_p}'),pair_path))
 else:
  pairs=[(Path(f'worker{i}'),Path(f'worker{i}')) for i in range(a['args']['replicas'])]
 for ref_dir,actual_dir in pairs:
  for suite in ('serial','cache','concurrent'):
   file=actual_dir/(suite+'.json')
   part=compare_rows(json.loads((ref/ref_dir/(suite+'.json')).read_text()),json.loads((actual/file).read_text()))
   part['reference_file']=str(ref/ref_dir/(suite+'.json'))
   result['parts'][str(file)]=part
 result['status']='exact' if all(p['status']=='exact' for p in result['parts'].values()) else 'numerical_differences_require_analysis'
 result['limitations']=['Token equality only; cancellation, resource conservation and stress are separate gates.','Any mismatch requires actual-prefix numerical analysis; no post-hoc tolerance converts it to a pass.']
 return result

def main():
 p=argparse.ArgumentParser(description=__doc__);p.add_argument('--reference',type=Path,required=True);p.add_argument('--candidate',type=Path,required=True);p.add_argument('--output',type=Path,required=True);a=p.parse_args()
 if a.output.exists():raise FileExistsError('Use a new output label')
 result=compare(a.reference,a.candidate);a.output.write_text(json.dumps(result,indent=2)+'\n');print(result['status'])
if __name__=='__main__':main()
