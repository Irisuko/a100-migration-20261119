#!/usr/bin/env python3
"""Compile Llama BF16 head_dim128 FlashInfer specializations without GPU access."""
import os
from pathlib import Path
import json,hashlib,time
ROOT=Path(__file__).resolve().parents[2]
OUT=ROOT/'run/a100-4gpu-20261114'
os.environ['CUDA_VISIBLE_DEVICES']=''
os.environ['CUDA_HOME']=str(Path.home()/'.local/cuda-12.8')
os.environ['PATH']=str(ROOT/'.venv/bin')+':'+os.environ['CUDA_HOME']+'/bin:'+os.environ.get('PATH','')
os.environ['TORCH_CUDA_ARCH_LIST']='8.0'
os.environ['FLASHINFER_CUDA_ARCH_LIST']='8.0'
os.environ['FLASHINFER_WORKSPACE_BASE']=str(OUT/'cache/flashinfer')
os.environ['MAX_JOBS']='2'
import torch
from flashinfer.jit.attention.pytorch import gen_batch_decode_module,gen_batch_prefill_module
assert not torch.cuda.is_initialized()
specs=[gen_batch_decode_module(torch.bfloat16,torch.bfloat16,torch.bfloat16,torch.int32,128,128,0,False,False),gen_batch_prefill_module('fa2',torch.bfloat16,torch.bfloat16,torch.bfloat16,torch.int32,128,128,0,False,False,False)]
report={'device_execution':False,'architecture':'sm_80','specs':[]}
for spec in specs:
 start=time.time();spec.build(verbose=True)
 binary=spec.jit_library_path
 report['specs'].append({'name':spec.name,'ninja':str(spec.ninja_path),'library':str(binary),'sha256':hashlib.sha256(binary.read_bytes()).hexdigest(),'build_s':time.time()-start})
 (OUT/'environment/attention-build.json').write_text(json.dumps(report,indent=2)+'\n')
assert not torch.cuda.is_initialized()
print(json.dumps(report),flush=True)
