"""Rebuild checksums for clearly labeled scoped8B result directories."""
from pathlib import Path
import hashlib,json,os,stat
from migration_tools import digest,is_70b
R=Path(__file__).resolve().parents[2];S=R/'snapshot/workspace';O=Path('/data01/home/mhy/workspace');checks=[]
labels=['fair8b-20261117T142324Z','edge8b-20261117T161715Z','pcie-host8b-20261118T025723Z','pd-boundary-mini-20261118T074218Z']
for label in labels:
 root=S/'experiment-reports'/label;orig=O/'experiment-reports'/label;n=0
 for p in orig.rglob('*'):
  if not p.is_file() or p.is_symlink():continue
  rel=p.relative_to(orig)
  if (p.name.startswith('trace_') and p.suffix in ['.json','.jsonl']) or (rel.parts[0]=='traces' and p.suffix=='.jsonl') or p.name in ['summary.csv','matrix.csv']:
   dest=root/rel;assert dest.is_file(),str(dest);assert digest(p)==digest(dest),str(rel);n+=1
 # Formal data/runtime pin lists stay scientifically authoritative if present.
 pins=root/'frozen-pins.json';pin_count=0
 if pins.exists():
  for rel,h in json.loads(pins.read_text()).items():assert digest(root/rel)==h,rel;pin_count+=1
 (root/'MIGRATION_NOTICE.md').write_text('# 迁移派生包\n\n按用户要求剔除历史70B专属材料。原正式8B请求、输入和统计文件已与原件逐字节核对；本目录重新生成MANIFEST.sha256。原始完整归档保留在旧服务器，不在此次上传范围。\n\n源目录原有delivery-sha256.json、completion-pins.json等历史全包清单可能仍引用已排除的环境诊断，不能替代此次迁移清单。运行源码/输入冻结清单若存在则另行核验。\n')
 paths=[p for p in sorted(root.rglob('*')) if p.is_file() and not p.is_symlink() and p.name!='MANIFEST.sha256' and not any(k in ['cache','__pycache__'] for k in p.relative_to(root).parts)]
 target=root/'MANIFEST.sha256'
 if target.exists():target.chmod(stat.S_IMODE(target.stat().st_mode)|0o200)
 target.write_text(''.join(digest(p)+'  '+str(p.relative_to(root))+'\n' for p in paths))
 checks.append({'label':label,'raw_input_request_summary_files_verified':n,'frozen_runtime_pins_verified':pin_count,'new_internal_manifest_files':len(paths)})
print('DERIVATIVE_RAW_CHECKS',json.dumps(checks),flush=True)
(R/'repository/manifests/formal8b-raw-unchanged.json').write_text(json.dumps(checks,indent=2)+'\n')
