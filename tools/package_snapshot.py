"""Archive immutable prepared staging and split assets below GitHub limits."""
from pathlib import Path
import hashlib,json,shutil,subprocess,time,tarfile
R=Path(__file__).resolve().parents[2];A=R/'artifacts';S=R/'snapshot'
assert (R/'repository/manifests/workspace-summary.json').exists(),'manifest must finish first'
assert not (A/'workspace.tar.zst').exists(),'do not overwrite an archive'
with (R/'logs/tar-stderr.log').open('w') as err:
 tar=subprocess.Popen(['tar','--sort=name','-cf','-','-C',str(S),'.'],stdout=subprocess.PIPE,stderr=err)
 z=subprocess.run(['zstd','-T4','-3','-o',str(A/'workspace.tar.zst')],stdin=tar.stdout,stdout=subprocess.PIPE,stderr=subprocess.PIPE,text=True)
 tar.stdout.close();rc=tar.wait()
 assert rc==0 and z.returncode==0,(rc,z.returncode,z.stderr)
 print('ARCHIVE_COMPRESSED', (A/'workspace.tar.zst').stat().st_size,flush=True)
parts=[]
with (A/'workspace.tar.zst').open('rb') as f:
 i=0
 while True:
  first=f.read(8*1024**2)
  if not first:break
  p=A/f'workspace.tar.zst.part-{i:03d}';h=hashlib.sha256();n=0
  with p.open('wb') as o:
   b=first
   while b:
    o.write(b);h.update(b);n+=len(b)
    if n>=128*1024**2:break
    b=f.read(min(8*1024**2,128*1024**2-n))
  parts.append({'name':p.name,'bytes':n,'sha256':h.hexdigest()});i+=1
subprocess.run(['zstd','-T2','-6','-k',str(A/'workspace-manifest.jsonl')],check=True,capture_output=True)
for label in ['fair8b-20261117T142324Z','edge8b-20261117T161715Z','pcie-host8b-20261118T025723Z','pd-boundary-mini-20261118T074218Z']:
 p=A/(label+'-migration.tar.gz')
 with tarfile.open(p,'w:gz',compresslevel=6) as tf:
  tf.add(S/'workspace/experiment-reports'/label,arcname=label,filter=lambda x:None if any(k in ['cache','__pycache__'] for k in Path(x.name).parts) else x)
 print('FORMAL_DERIVATIVE',p.name,flush=True)
assets=[A/x['name'] for x in parts]+[A/'workspace-manifest.jsonl.zst']+list(A.glob('*-migration.tar.gz'))
assets=sorted(set(assets));rows=[]
for p in assets:
 h=hashlib.sha256()
 with p.open('rb') as f:
  for b in iter(lambda:f.read(8*1024**2),b''):h.update(b)
 assert p.stat().st_size<2*1024**3
 rows.append({'name':p.name,'bytes':p.stat().st_size,'sha256':h.hexdigest()})
(A/'SHA256SUMS').write_text(''.join(x['sha256']+'  '+x['name']+'\n' for x in rows))
(R/'repository/manifests/release-assets.json').write_text(json.dumps({'tag':'migration-20261119','assets':rows,'split_part_bytes':128*1024**2},indent=2)+'\n')
shutil.copy2(A/'SHA256SUMS',R/'repository/manifests/SHA256SUMS')
print('ASSETS_READY',json.dumps(rows),flush=True)
