"""Conservative model-record filtering and read-only restored-tree verification."""
import argparse,hashlib,json,os,re,stat
from pathlib import Path
PAT=re.compile(r'70[bB]')
DROP=object()
def is_70b(value):
 if not isinstance(value,str):return False
 for m in PAT.finditer(value):
  lo,hi=m.span()
  while lo>0 and value[lo-1].isascii() and value[lo-1].isalnum():lo-=1
  while hi<len(value) and value[hi].isascii() and value[hi].isalnum():hi+=1
  token=value[lo:hi]
  if token.lower()=='70b' or not re.fullmatch(r'[0-9a-f]+',token,re.I):return True
 return False
def structured_format(name):
 suffixes=Path(name).suffixes
 return next((x for x in ['jsonl','json','csv'] if '.'+x in suffixes),None)
def filter_json(value):
 if isinstance(value,dict):
  identity={'model','model_name','model_path','model_id','id','case','case_name','name','path','label','tag','experiment','experiment_id','experiment_name','task_id','task_name','run_id','run_name','queue_id','operation','config_path','condition'}
  def only70(x):return is_70b(x) and not re.search(r'(?i)8b',x)
  if any(k in identity and only70(v) for k,v in value.items()):return DROP
  for k in ['args','config']:
   v=value.get(k)
   if isinstance(v,dict) and any(n in identity and only70(z) for n,z in v.items()):return DROP
  for k in ['command','argv']:
   v=value.get(k)
   if isinstance(v,list) and any(only70(z) for z in v) and not any(isinstance(z,str) and re.search(r'(?i)8b',z) for z in v):return DROP
  out={}
  for k,v in value.items():
   if is_70b(k):
    if re.search(r'(?i)8b',k):k=re.sub(r'(?i)(?:_or_|_and_|[-_/])70b','',k)
    else:continue
   clean=filter_json(v)
   if clean is not DROP:
    if k in out:raise ValueError('70B projection key collision: '+k)
    out[k]=clean
  return out
 if isinstance(value,list):return [clean for item in value if (clean:=filter_json(item)) is not DROP]
 if is_70b(value):return DROP
 return value

def digest(p):
 h=hashlib.sha256()
 with p.open('rb') as f:
  for b in iter(lambda:f.read(8*1024**2),b''):h.update(b)
 return h.hexdigest()

def verify_tree(root,rows):
 root=Path(root).resolve();errors=[]
 for row in rows:
  rel=Path(row['path'])
  if rel.is_absolute() or '..' in rel.parts:errors.append({'path':str(rel),'error':'unsafe manifest path'});continue
  p=root/rel
  if not p.parent.resolve().is_relative_to(root):errors.append({'path':str(rel),'error':'parent symlink escapes root'});continue
  try:
   st=p.lstat();typ=row['type']
   if typ=='symlink':assert stat.S_ISLNK(st.st_mode) and os.readlink(p)==row['target'],'symlink mismatch'
   elif typ=='directory':assert stat.S_ISDIR(st.st_mode),'directory mismatch'
   elif typ=='file':
    assert stat.S_ISREG(st.st_mode),'not a regular file'
    assert st.st_size==row['size'],'size mismatch'
    assert digest(p)==row['sha256'],'sha256 mismatch'
   else:raise ValueError('unknown entry type')
   if typ!='symlink' and 'mode' in row:assert stat.S_IMODE(st.st_mode)==row['mode'],'mode mismatch'
  except (OSError,AssertionError,ValueError) as e:errors.append({'path':str(rel),'error':str(e)})
 return errors
if __name__=='__main__':
 p=argparse.ArgumentParser();p.add_argument('root');p.add_argument('manifest');a=p.parse_args()
 rows=[json.loads(l) for l in Path(a.manifest).read_text().splitlines() if l.strip()]
 errors=verify_tree(a.root,rows);print(json.dumps({'status':'FAIL' if errors else 'PASS','entries':len(rows),'errors':errors},indent=2));raise SystemExit(bool(errors))
