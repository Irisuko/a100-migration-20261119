"""One review-driven refinement, confined to disposable migration staging."""
from pathlib import Path
import json,os,shutil,re,stat
R=Path(__file__).resolve().parents[2];S=R/'snapshot/workspace';O=Path('/data01/home/mhy/workspace');removed=[]
def delete(p,reason):
 if not p.exists() and not p.is_symlink():return
 removed.append({'path':str(p.relative_to(S)),'reason':reason})
 if p.is_dir() and not p.is_symlink():shutil.rmtree(p)
 else:p.unlink()
# Re-filter ORIGINAL records; otherwise removing an identifier could leave orphan payloads.
for row in json.loads((R/'repository/manifests/transformed-files.json').read_text()):
 p=S/row['path'];src=O/row['path']
 if p.exists():
  p.chmod(stat.S_IMODE(p.stat().st_mode)|0o200);shutil.copy2(src,p)
# Exact review findings: opaque tensor subtrees and duplicate deduplicated archive.
for rel in ['muxwise-pp-sglang/run/a100-gpu0123-20261114/analysis/event-11641eb9-independent-v1','muxwise-pp-sglang/run/a100-gpu0123-20261114/analysis/event-d544a51c-root-v1','muxwise-pp-sglang/run/a100-gpu0123-20261114/pool-gpu4567/rate-calibration/selected-dev600-1x2x4x-v1/content.tar']:
 delete(S/rel,'review-confirmed70B opaque tensors/mixed archive; canonical8B evidence retained')
# Actual historical70B failure records embedded into later8B environment diagnostics.
for p in list(S.rglob('pd-failure-analysis.json')):delete(p,'contains historical70B PD failure rather than this8B measurement')
# Local analysis programs with hardcoded70B-only capture inputs are not shared runtime.
pattern=re.compile(r'diagnostics/(?:attention-decomposition-70b|rootcause-70b|splitkv-workspace-capture-70b)',re.I)
for p in list((S/'muxwise-pp-sglang/run/a100-gpu0123-20261114/analysis').rglob('*.py')):
 if not p.exists() or p.is_symlink():continue
 if pattern.search(p.read_text(errors='replace')):
  event=next((a for a in p.parents if a.name.startswith('event-')),None)
  delete(event or p,'local analysis hardcodes70B capture input; shared runtime retained separately')
for p in list((S/'experiment-reports').glob('*.tar.gz')):delete(p,'original package embeds70B failure; replace with explicitly labeled8B migration derivative')
(R/'repository/manifests/review-scope-exclusions.json').write_text(json.dumps(removed,ensure_ascii=False,indent=2)+'\n')
# Discard only our interrupted compressed staging output; original results untouched.
p=R/'artifacts/workspace.tar.zst'
if p.exists():p.unlink()
(R/'logs/original-manifest-before-review.jsonl').write_bytes((R/'artifacts/workspace-manifest.jsonl').read_bytes())
print('REVIEW_EXCLUSIONS',len(removed),flush=True)
