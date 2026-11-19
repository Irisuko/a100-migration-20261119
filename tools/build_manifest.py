"""Inventory and hash migration staging, preserving modes and link identity."""
import collections,hashlib,json,os,re,stat,time
from pathlib import Path
from migration_tools import digest,is_70b
R=Path(__file__).resolve().parents[2];S=R/'snapshot';W=S/'workspace';O=Path('/data01/home/mhy/workspace')
counts=collections.Counter();sizes=collections.Counter();rows=[];links=[];secrets=[];changes=[];excluded=[]
secret=re.compile(rb'(?:gh[pousr]_[A-Za-z0-9]{30,}|github_pat_[A-Za-z0-9_]{50,}|hf_[A-Za-z0-9]{30,}|AKIA[0-9A-Z]{16}|-----BEGIN (?:RSA |OPENSSH |EC )?PRIVATE KEY-----)')
# Record exclusions versus original, including the first interrupted filtering pass.
for root,dirs,files in os.walk(O,followlinks=False):
 p=Path(root);rel=p.relative_to(O)
 if p==O:dirs[:]=[d for d in dirs if d!='models']
 dirs[:]=[d for d in dirs if d not in ['.venv','__pycache__','.pytest_cache']]
 for name in list(dirs):
  q=p/name;dest=W/q.relative_to(O)
  if not dest.exists() and not dest.is_symlink():excluded.append({'path':str(q.relative_to(O)),'type':'directory','reason':'70B/mixed70B scope exclusion'});dirs.remove(name)
 for name in files:
  q=p/name;dest=W/q.relative_to(O)
  if not dest.exists() and not dest.is_symlink():excluded.append({'path':str(q.relative_to(O)),'type':'file','reason':'70B/mixed70B/control-state or duplicate-archive exclusion'})
# Compare potentially transformed files against source; restore original permissions.
candidates=set((R/'logs/content-70b-paths.txt').read_text().splitlines())
for rel in sorted(candidates):
 src=O/rel;p=W/rel
 if not p.is_file() or p.is_symlink():continue
 mode=stat.S_IMODE(src.stat().st_mode)
 if stat.S_IMODE(p.stat().st_mode)!=mode:p.chmod(mode)
 a=digest(src);b=digest(p)
 if a!=b:changes.append({'path':rel,'original_sha256':a,'migration_sha256':b,'reason':'70B structured-record filtering; old full-snapshot hash lists no longer apply'})
for root,dirs,files in os.walk(S,followlinks=False):
 p=Path(root)
 for name in sorted(dirs+files):
  f=p/name;rel=f.relative_to(S);st=f.lstat();row={'path':str(rel),'mode':stat.S_IMODE(st.st_mode)}
  if stat.S_ISLNK(st.st_mode):
   row.update(type='symlink',target=os.readlink(f));links.append(row.copy())
  elif stat.S_ISDIR(st.st_mode):row.update(type='directory')
  elif stat.S_ISREG(st.st_mode):
   h=hashlib.sha256();tail=b'';hits=False
   # Full hashing; recognizable credential marker scan on plain-text candidates.
   scan=f.suffix.lower() in ['.json','.jsonl','.yaml','.yml','.txt','.md','.py','.sh','.toml','.ini','.cfg','.log','.patch','.diff'] or f.name in ['config','credentials','.env']
   with f.open('rb') as stream:
    for chunk in iter(lambda:stream.read(8*1024**2),b''):
     h.update(chunk)
     if scan and secret.search(tail+chunk):hits=True
     tail=chunk[-256:] if scan else b''
   row.update(type='file',size=st.st_size,sha256=h.hexdigest());sizes[rel.parts[0]+'/'+rel.parts[1] if len(rel.parts)>1 else rel.parts[0]]+=st.st_size
   if hits:secrets.append({'path':str(rel),'type':'recognizable credential marker; review locally, never print value'})
  else:raise RuntimeError('Unsupported special file '+str(rel))
  counts[row['type']]+=1;rows.append(row)
  if len(rows)%25000==0:print('MANIFEST',len(rows),flush=True)
rows.sort(key=lambda x:x['path'])
manifest=R/'artifacts/workspace-manifest.jsonl';manifest.write_text(''.join(json.dumps(x,ensure_ascii=False)+'\n' for x in rows))
summary={'counts':dict(counts),'bytes_by_tree':dict(sizes),'regular_file_bytes':sum(x.get('size',0) for x in rows),'manifest_sha256':digest(manifest),'exclusions':len(excluded),'transformed_files':len(changes),'secret_marker_candidates':len(secrets),'generated_unix':time.time()}
for name,data in [('workspace-summary.json',summary),('excluded-paths.json',excluded),('transformed-files.json',changes),('symlinks.json',links),('secret-scan.json',secrets)]:
 (R/'repository/manifests'/name).write_text(json.dumps(data,ensure_ascii=False,indent=2)+'\n')
print('MANIFEST_COMPLETE',json.dumps(summary),flush=True)
