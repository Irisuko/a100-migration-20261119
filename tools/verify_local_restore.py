from pathlib import Path
import hashlib,json,subprocess,time
from migration_tools import digest,verify_tree
R=Path(__file__).resolve().parents[2];V=R/'verification';A=R/'artifacts';start=time.time()
joined=V/'reassembled.tar.zst';assert not joined.exists()
h=hashlib.sha256();size=0
with joined.open('wb') as out:
 for p in sorted(A.glob('workspace.tar.zst.part-*')):
  with p.open('rb') as src:
   for b in iter(lambda:src.read(8*1024**2),b''):out.write(b);h.update(b);size+=len(b)
assert size==(A/'workspace.tar.zst').stat().st_size
assert h.hexdigest()==digest(A/'workspace.tar.zst')
print('REASSEMBLY_PASS',size,flush=True)
restore=V/'restored';restore.mkdir()
with (R/'logs/restore-tar.log').open('w') as log:subprocess.run(['tar','--zstd','--same-permissions','-xf',str(joined),'-C',str(restore)],stdout=log,stderr=subprocess.STDOUT,check=True)
print('EXTRACTION_COMPLETE',flush=True)
rows=[json.loads(l) for l in (A/'workspace-manifest.jsonl').read_text().splitlines()];errors=verify_tree(restore,rows)
expected={x['path'] for x in rows};actual={str(p.relative_to(restore)) for p in restore.rglob('*')}
assert expected==actual,{'missing':list(expected-actual)[:10],'extra':list(actual-expected)[:10]}
assert not errors,errors[:20]
git={}
for name in ['muxwise-pp-sglang','muxwise-pp-experiments']:
 p=restore/'workspace'/name
 z=subprocess.run(['git','-C',str(p),'fsck','--full'],capture_output=True,text=True);(R/'logs'/('git-fsck-'+name+'.log')).write_text(z.stdout+z.stderr);assert z.returncode==0,z.stderr[-1000:]
 head=subprocess.check_output(['git','-C',str(p),'rev-parse','HEAD'],text=True).strip();assert head==(R/'repository/environment'/name/'head.txt').read_text().strip();git[name]=head
result={'status':'PASS','entries':len(rows),'files':sum(x['type']=='file' for x in rows),'symlinks':sum(x['type']=='symlink' for x in rows),'sha256_size_mode_symlinks_checked':True,'extra_or_missing_entries':0,'reassembled_archive_sha256':h.hexdigest(),'archive_bytes':size,'git_histories_fsck':git,'duration_s':time.time()-start,'scope':'local generated release split assets reassembled and extracted to empty directory; remote byte identity checked separately; noGPU runtime validation'}
(R/'repository/manifests/local-restoration-verification.json').write_text(json.dumps(result,indent=2)+'\n');print('LOCAL_RESTORE_PASS',json.dumps(result),flush=True)
