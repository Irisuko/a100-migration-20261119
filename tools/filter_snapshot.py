"""Apply user-requested70B exclusions to migration staging only; never original."""
import csv,io,json,os,re,shutil,sys
from pathlib import Path
from migration_tools import is_70b,filter_json,DROP,digest,structured_format
R=Path(__file__).resolve().parents[2];S=R/'snapshot/workspace';ORIG=Path('/data01/home/mhy/workspace')
assert str(S).startswith('/data01/home/mhy/migration-backups/')
changes=[]
protected=json.loads((R/'repository/manifests/protected-8b-run-roots.json').read_text())
def remove(p,why):
 rel=str(p.relative_to(S));size=p.lstat().st_size
 changes.append({'path':rel,'action':'excluded','reason':why,'bytes_if_file':size if p.is_file() and not p.is_symlink() else None})
 if p.is_dir() and not p.is_symlink():shutil.rmtree(p)
 else:p.unlink()
# Names carry explicit model attribution. Preserve original directories untouched.
for root,dirs,files in os.walk(S,topdown=True,followlinks=False):
 p=Path(root)
 dirs[:]=[d for d in dirs if d!='.git']
 for name in list(dirs):
  f=p/name
  if is_70b(name):remove(f,'70B-specific directory');dirs.remove(name)
 for name in files:
  f=p/name
  if is_70b(name) or (f.is_symlink() and is_70b(os.readlink(f))):remove(f,'70B-specific filename or link target')
# Detect70B runs whose directory names do not identify their model.
for p in list(S.rglob('run.json')):
 if not p.is_file() or p.is_symlink():continue
 try:d=json.loads(p.read_text());a=d.get('args',{})
 except (ValueError,AttributeError):continue
 if isinstance(a,dict) and any(is_70b(a.get(k,'')) for k in ['model','model_path','model_name']):remove(p.parent,'run.json declares70B model')
# Parse mixed structured records. Do not alter common implementation code or patches.
for rel in (R/'logs/content-70b-paths.txt').read_text().splitlines():
 p=S/rel
 if any(rel.startswith(x+'/') for x in protected):continue
 if not p.is_file() or p.is_symlink() or '.git' in p.parts:continue
 if p.suffix in ['.py','.sh','.cpp','.cu','.c','.h','.hpp','.rs','.patch','.diff']:continue
 old=p.read_bytes()
 if not is_70b(old.decode(errors="replace")):continue
 oldhash=digest(p)
 oldmode=p.stat().st_mode & 0o7777
 if not oldmode & 0o200:p.chmod(oldmode|0o200)
 try:
  if structured_format(p.name)=='json':
   d=json.loads(old);clean=filter_json(d)
   if clean is DROP:remove(p,'70B-specific structured record');continue
   if clean==d:continue
   p.write_text(json.dumps(clean,ensure_ascii=False,indent=2)+'\n')
  elif structured_format(p.name)=='jsonl':
   rows=[json.loads(l) for l in old.decode().splitlines() if l.strip()];clean=[v for x in rows if (v:=filter_json(x)) is not DROP]
   if clean==rows:continue
   p.write_text(''.join(json.dumps(x,ensure_ascii=False)+'\n' for x in clean))
  elif structured_format(p.name)=='csv':
   reader=csv.DictReader(io.StringIO(old.decode('utf-8-sig')));fields=reader.fieldnames;rows=list(reader);clean=[x for x in rows if not any(is_70b(v) for v in x.values())]
   if clean==rows:continue
   with p.open('w',newline='',encoding='utf-8-sig') as f:w=csv.DictWriter(f,fieldnames=fields);w.writeheader();w.writerows(clean)
  elif p.suffix in ['.md','.txt','.log','.yml','.yaml','.tsv']:
   # Avoid retaining unlabeled numerical70B paragraphs by merely deleting keywords.
   # Generic code notes are recoverable via git history; mixed operational notes are omitted.
   remove(p,'mixed70B narrative/configuration/log omitted;8B raw runs retained');continue
  else:continue
 except (ValueError,UnicodeError,csv.Error) as e:
  raise RuntimeError(f'Could not safely filter {rel}: {type(e).__name__}') from e
 p.chmod(oldmode)
 changes.append({'path':rel,'action':'filtered_structured_records','original_sha256':oldhash,'migration_sha256':digest(p),'reason':'remove70B records/keys/narrative fields; not an untouched historical artifact'})
(R/'repository/manifests/model-exclusions.json').write_text(json.dumps(changes,ensure_ascii=False,indent=2)+'\n')
print(json.dumps({'actions':len(changes),'excluded':sum(x['action']=='excluded' for x in changes),'filtered':sum(x['action']=='filtered_structured_records' for x in changes)}),flush=True)
