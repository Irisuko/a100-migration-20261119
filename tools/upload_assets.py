from pathlib import Path
import concurrent.futures,json,subprocess,time,hashlib
R=Path(__file__).resolve().parents[2];repo='Irisuko/a100-migration-20261119';tag='migration-20261119'
meta=json.loads((R/'repository/manifests/release-assets.json').read_text());names=['SHA256SUMS']+[x['name'] for x in meta['assets']]
assert json.loads(subprocess.check_output(['gh','repo','view',repo,'--json','isPrivate'],text=True))['isPrivate']
releases=json.loads(subprocess.check_output(['gh','api',f'repos/{repo}/releases'],text=True))
release=next(x for x in releases if x['tag_name']==tag)
pages=json.loads(subprocess.check_output(['gh','api',f'repos/{repo}/releases/{release["id"]}/assets?per_page=100','--paginate','--slurp'],text=True))
remote={x['name']:x for page in pages for x in page}
expected={x['name']:x for x in meta['assets']}
expected['SHA256SUMS']={'bytes':(R/'artifacts/SHA256SUMS').stat().st_size,'sha256':hashlib.sha256((R/'artifacts/SHA256SUMS').read_bytes()).hexdigest()}
def send(name):
 old=remote.get(name,{})
 if old.get('state')=='uploaded' and old.get('size')==expected[name]['bytes'] and old.get('digest')=='sha256:'+expected[name]['sha256']:
  print('UPLOAD_SKIP_VERIFIED',name,flush=True);return {'name':name,'status':'already_uploaded_digest_verified'}
 start=time.time();print('UPLOAD_START',name,flush=True)
 for attempt in range(1,4):
  p=subprocess.run(['gh','release','upload',tag,str(R/'artifacts'/name),'--repo',repo,'--clobber'],capture_output=True,text=True)
  if p.returncode==0:
   row={'name':name,'status':'uploaded','seconds':time.time()-start,'attempt':attempt};print('UPLOAD_DONE',json.dumps(row),flush=True);return row
  print('UPLOAD_RETRY',name,attempt,p.stderr[-800:],flush=True);time.sleep(5*attempt)
 raise RuntimeError('Upload failed '+name)
with concurrent.futures.ThreadPoolExecutor(max_workers=32) as pool:rows=list(pool.map(send,names))
(R/'verification/upload-receipts.json').write_text(json.dumps(rows,indent=2)+'\n');print('UPLOAD_ALL_COMPLETE',len(rows),flush=True)
