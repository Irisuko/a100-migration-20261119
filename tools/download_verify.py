"""Download completed private release assets and verify every byte against local pins."""
from pathlib import Path
import concurrent.futures,hashlib,json,subprocess,time
from migration_tools import digest
R=Path(__file__).resolve().parents[2];repo='Irisuko/a100-migration-20261119';tag='migration-20261119';dest=R/'verification/downloaded';dest.mkdir(exist_ok=True)
meta=json.loads((R/'repository/manifests/release-assets.json').read_text());expected={x['name']:x for x in meta['assets']};p=R/'artifacts/SHA256SUMS';expected[p.name]={'name':p.name,'bytes':p.stat().st_size,'sha256':digest(p)}
def download(name):
 start=time.time();print('DOWNLOAD_START',name,flush=True)
 for attempt in range(1,4):
  p=subprocess.run(['gh','release','download',tag,'--repo',repo,'--pattern',name,'--dir',str(dest),'--clobber'],capture_output=True,text=True)
  target=dest/name
  if p.returncode==0 and target.stat().st_size==expected[name]['bytes'] and digest(target)==expected[name]['sha256']:
   row={'name':name,'bytes':target.stat().st_size,'sha256':expected[name]['sha256'],'status':'download_verified','seconds':time.time()-start};print('DOWNLOAD_DONE',json.dumps(row),flush=True);return row
  print('DOWNLOAD_RETRY',name,attempt,p.stderr[-600:],flush=True);time.sleep(5*attempt)
 raise RuntimeError('Download verification failed '+name)
start=time.time();submitted={};done=[]
with concurrent.futures.ThreadPoolExecutor(max_workers=4) as pool:
 while len(done)<len(expected):
  rels=json.loads(subprocess.check_output(['gh','api',f'repos/{repo}/releases'],text=True));rel=next(x for x in rels if x['tag_name']==tag)
  for asset in rel['assets']:
   name=asset['name']
   if name not in expected or name in submitted or asset['state']!='uploaded':continue
   assert asset['size']==expected[name]['bytes'] and asset.get('digest')=='sha256:'+expected[name]['sha256'],('remote asset digest mismatch',name)
   submitted[name]=pool.submit(download,name)
  done=[]
  for name,future in submitted.items():
   if future.done():done.append(future.result())
  (R/'verification/download-progress.json').write_text(json.dumps({'verified':len(done),'expected':len(expected),'scheduled':len(submitted),'verified_bytes':sum(x['bytes'] for x in done),'time_unix':time.time()},indent=2))
  if len(done)<len(expected):time.sleep(20)
private=json.loads(subprocess.check_output(['gh','repo','view',repo,'--json','isPrivate'],text=True));assert private['isPrivate']
result={'status':'PASS','repository':repo,'private':True,'release_tag':tag,'asset_count':len(done),'bytes':sum(x['bytes'] for x in done),'duration_s':time.time()-start,'all_remote_api_sha256_match':True,'all_download_sha256_match':True,'assets':sorted(done,key=lambda x:x['name'])}
(R/'repository/manifests/remote-roundtrip-verification.json').write_text(json.dumps(result,indent=2)+'\n');print('REMOTE_ROUNDTRIP_PASS',len(done),flush=True)
