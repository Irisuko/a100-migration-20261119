"""Versioned exact reviewed transitions, never a general source-hash exception."""
import ast
import base64
import hashlib
import json
import re
from pathlib import Path

TARGET='scripts/a100_gpu0123/server_run.py'
BASE='3ecd450eb74d0dd78558e6c916027a1fe125d22c384e8fe9c4907752cbab3d8b'
CANDIDATE='87bf20e9f45f3ef7fe92f06946cd016aecef9782ce222a050c6270549e412521'
PATCH='d657e1cdf0b6bda97d00a4aa41778662d7bd24d3b9e3d1644ba62d82f805ccd0'
RUNTIME='4fd479dd234bf1b98cd8551791dc8d534bd20e42a4d8ee8cf58f64555c0a281b'
SCHEDULER='python/sglang/srt/managers/scheduler.py'
SCHEDULER_BASE='df664cd14626ee84e70b5b871ffed794f142b3d65e560af9f4cfef11d0681f27'
SCHEDULER_CANDIDATE='a1d9decd0fd22343a96d3b4865fb60312a3128a0d6d06d15be53dc635f21caab'
SCHEDULER_PATCH='c3347dd6c047b11235a620ed6dac484f60f82bb4183f34cc8a6e3be88ff81428'
CANDIDATE_RUNTIME='d6bfcfc8083bf5b7e43c7191e3d89d85891f9647e151649a4abbb7301fc14c8e'
V1_ARTIFACT='5e52cc885ad3b623d837fe7cbdde479e55f0a92ce4ae24720c0d9880a8e8571c'
V2_ARTIFACT='a62cfd4d520c5a74fd74abe8a438fe3a5cea1209b2c10e1c9f7711e0d82f94c9'
CLIENT='python/sglang/srt/managers/tp_worker_overlap_thread.py'
CLIENT_BASE='bfc0e817bbfd46e3ea47c36b036c52cf8ae599ae08778f78653bb237fbfcf551'
CLIENT_CANDIDATE='59fa602db9960a4cb3a4f6f17b83b7654f901b80c379ccfdcae133cd4f8577f3'
CLIENT_PATCH='072fc63549a52168d03e1172d707f4975d01598e3de5a58388a4ff36f59b7e20'
PD='python/sglang/srt/disaggregation/mooncake/conn.py'
PD_BASE='02ce2e35a0c37179e09197c7d48554699cbea6cdd80b891dc068024696310ff7'
PD_CANDIDATE='bf12fa235bb22860c507bdc62742211a8cdc481143a83735d141e85d307a690c'
PD_PATCH='77ec6627c47489ab4d9c10ae13966511a7576656e7318528f0ea2fbd89fa2091'
COMBINED_RUNTIME='1bab3148e9f3ceb49d77b370a4525e623df6c61af8a5b8669958627c453b79f2'
V3_ARTIFACT='c3dda2d55a250b3b99ebb19ab201025823313aa5ea94393abe2484c95d5a2e01'
CANCELLATION_CANDIDATE='464709fe3f61a69bc9fc446ac2f4377ecff4b8f57617fa41565d79073f04a2b0'
CANCELLATION_PATCH='7d7facfba9af32de830dad1ad811b67a5b9745cd32211ce9fb72832f40e33bfc'
NORMALIZED_CANCELLATION_BASE='e5c41be36ded5c893011f00be0a07af84a7e75ec1f07ca624ef81fdd2f4cad2f'
PD_STAGE_PATCH='372d04ba5d4d3a03cdd9bddc1ef8dfc717f8932e28a9bcd8dafc09c09e96e257'
PD_CAPTURE_EVIDENCE='3e6534fdc8f5222e244c1f3feb15fa61998134eeb818e7f8194b6c0e51552b54'
POSTFLIGHT_CANDIDATE='96551371daef7767a99a73965fc6acef7afce62232f8113bc1132a005afcc283'
POSTFLIGHT_PATCH='3ff901f260826b72e709ff02815947d56d9fa9d920654ae722d33395b9858bde'
POSTFLIGHT_PRIOR='5228bfb2b950a9158d1148e24d4736c562570936246c87373d76479251316e77'
POSTFLIGHT_HELPER='scripts/a100_gpu0123/serving_compatibility.py'
POSTFLIGHT_HELPER_BASE='2840cb8d56aa44e186f28e809d6691d01e75bc7b0a2e1d98c5dc933ed1b37dee'
POSTFLIGHT_RENDERER='scripts/a100_gpu0123/render_finalists.py'
POSTFLIGHT_RENDERER_BASE='dd9c3ec7a7ad3b66534865c244c283386615274f23185f5c30ef7b3c0c7d227f'


def require(condition,message):
    if not condition:raise ValueError(message)


def digest(value):
    return hashlib.sha256(json.dumps(value,sort_keys=True,separators=(',',':')).encode()).hexdigest()


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def prove_default_off(base,candidate):
    """Remove only additive policy scaffolding; restore the exact old flag AST."""
    counts={key:0 for key in ('helper','parser','policy','extra_guard','metadata','command_splice')}
    splice=ast.dump(ast.parse("[] if overlap['enabled'] else ['--disable-overlap-schedule']",mode='eval').body)
    class Restore(ast.NodeTransformer):
        def visit_FunctionDef(self,node):
            if node.name=='overlap_policy':counts['helper']+=1;return None
            return self.generic_visit(node)
        def visit_Expr(self,node):
            call=node.value
            if (isinstance(call,ast.Call) and isinstance(call.func,ast.Attribute)
                    and call.func.attr=='add_argument' and call.args
                    and isinstance(call.args[0],ast.Constant) and call.args[0].value=='--overlap'):
                counts['parser']+=1;return None
            return self.generic_visit(node)
        def visit_Assign(self,node):
            if len(node.targets)==1:
                target=node.targets[0]
                if isinstance(target,ast.Name) and target.id=='overlap':
                    counts['policy']+=1;return None
                if (isinstance(target,ast.Subscript) and isinstance(target.value,ast.Name)
                        and target.value.id=='metadata' and isinstance(target.slice,ast.Constant)
                        and target.slice.value=='overlap_policy'):
                    counts['metadata']+=1;return None
            return self.generic_visit(node)
        def visit_If(self,node):
            if (len(node.body)==1 and isinstance(node.body[0],ast.Raise)
                    and isinstance(node.body[0].exc,ast.Call) and node.body[0].exc.args
                    and isinstance(node.body[0].exc.args[0],ast.Constant)
                    and node.body[0].exc.args[0].value=='--extra cannot override native overlap; use --overlap'):
                counts['extra_guard']+=1;return None
            return self.generic_visit(node)
        def visit_Starred(self,node):
            if ast.dump(node.value)==splice:
                counts['command_splice']+=1
                return ast.Constant(value='--disable-overlap-schedule')
            return self.generic_visit(node)
    original=ast.dump(ast.parse(base));restored=ast.dump(Restore().visit(ast.parse(candidate)))
    require(all(count==1 for count in counts.values()) and original==restored,
            'Candidate contains changes beyond the exact default-off policy scaffolding')
    return dict(restored_ast_sha256=hashlib.sha256(restored.encode()).hexdigest(),
                removed_nodes=counts,default_off_backend_command_tokens_identical=True)


def load(path,current,repo):
    path=Path(path).resolve();data=json.loads(path.read_text());repo=Path(repo).resolve()
    if data.get('schema_version')==5:return load_postflight(path,data,current,repo)
    if data.get('schema_version')==4:return load_v4(path,data,current,repo)
    if data.get('schema_version')==3:return load_v3(path,data,current,repo)
    if data.get('schema_version')==2:return load_v2(path,data,current,repo)
    require(data.get('schema_version')==1 and data.get('target')==TARGET
            and data.get('base_sha256')==BASE and data.get('candidate_sha256')==CANDIDATE
            and data.get('patch_sha256')==PATCH and data.get('runtime_fingerprint')==RUNTIME,
            'Unknown compatibility path or hash transition')
    require(current.get(TARGET)==CANDIDATE,'Reviewed overlap runner must be applied before compatibility generation')
    runtime={key:value for key,value in current.items() if key.startswith('python/sglang/')}
    require(digest(runtime)==RUNTIME and runtime==data.get('runtime_source_sha256'),
            'Neural-network/runtime source changed outside the reviewed runner transition')
    artifacts={str(path):sha(path)}
    contents={}
    for field,expected in [('base_path',BASE),('candidate_path',CANDIDATE),('patch_path',PATCH)]:
        file=Path(data[field]);file=file if file.is_absolute() else repo/file
        require(sha(file)==expected,f'Compatibility artifact hash changed: {field}')
        artifacts[str(file.resolve())]=expected;contents[field]=file.read_text()
    proof=prove_default_off(contents['base_path'],contents['candidate_path'])
    require(proof==data.get('ast_proof'),'Compatibility AST proof differs')
    return dict(path=str(path),sha256=sha(path),target=TARGET,base_sha256=BASE,
                candidate_sha256=CANDIDATE,runtime_fingerprint=RUNTIME,ast_proof=proof,
                artifacts_sha256=artifacts)


def prove_scheduler_scope(base,candidate):
    """Remove exactly the two original-MuxWise guards; recover the whole AST."""
    snippets={
        'flush_guard':'if self.enable_pdmux and self.pp_size == 1 and self.split_prefill_batch is not None:\n return False',
        'abort_enumeration':'if self.enable_pdmux and self.pp_size == 1:\n batches.append(self.split_prefill_batch)',
    }
    expected={name:ast.dump(ast.parse(code).body[0]) for name,code in snippets.items()}
    owners={'flush_guard':'flush_cache','abort_enumeration':'_abortable_running_requests'}
    counts={name:0 for name in snippets}
    class Restore(ast.NodeTransformer):
        owner=None
        def visit_FunctionDef(self,node):
            previous=self.owner;self.owner=node.name
            result=self.generic_visit(node);self.owner=previous
            return result
        def visit_If(self,node):
            for name,pattern in expected.items():
                if self.owner==owners[name] and ast.dump(node)==pattern:
                    counts[name]+=1;return None
            return self.generic_visit(node)
    restored=ast.dump(Restore().visit(ast.parse(candidate)))
    require(counts=={'flush_guard':1,'abort_enumeration':1} and restored==ast.dump(ast.parse(base)),
            'Scheduler contains changes beyond the exact original-MuxWise guarded nodes')
    return dict(restored_ast_sha256=hashlib.sha256(restored.encode()).hexdigest(),removed_nodes=counts,
                affected_predicate='enable_pdmux and pp_size == 1',
                unaffected_predicate='not enable_pdmux or pp_size > 1')


def load_v2(path,data,current,repo,*,_cancellation_runner=False,_postflight_runner=False):
    transition=data.get('scheduler_transition',{})
    require(transition.get('target')==SCHEDULER and transition.get('base_sha256')==SCHEDULER_BASE
            and transition.get('candidate_sha256')==SCHEDULER_CANDIDATE
            and transition.get('patch_sha256')==SCHEDULER_PATCH
            and data.get('base_runtime_fingerprint')==RUNTIME
            and data.get('candidate_runtime_fingerprint')==CANDIDATE_RUNTIME,
            'Unknown v2 compatibility path or hash transition')
    require(current.get(SCHEDULER)==SCHEDULER_CANDIDATE and current.get(TARGET)==CANDIDATE
            and sha(repo/SCHEDULER)==SCHEDULER_CANDIDATE
            and sha(repo/TARGET)==(POSTFLIGHT_CANDIDATE if _postflight_runner else CANCELLATION_CANDIDATE if _cancellation_runner else CANDIDATE),
            'Both reviewed candidates must be applied to the live repository before v2 generation')
    runtime={key:value for key,value in current.items() if key.startswith('python/sglang/')}
    old_runtime={**runtime,SCHEDULER:SCHEDULER_BASE}
    require(digest(runtime)==CANDIDATE_RUNTIME and digest(old_runtime)==RUNTIME
            and runtime==data.get('candidate_runtime_source_sha256')
            and old_runtime==data.get('base_runtime_source_sha256'),
            'Runtime keyset/hash changed beyond the exact scheduler transition')
    v1=Path(data['runner_compatibility_path']);v1=v1 if v1.is_absolute() else repo/v1
    require(data.get('runner_compatibility_sha256')==V1_ARTIFACT and sha(v1)==V1_ARTIFACT,
            'Original reviewed runner compatibility artifact changed')
    runner=load(v1,{**current,SCHEDULER:SCHEDULER_BASE},repo)
    artifacts={**runner['artifacts_sha256'],str(path):sha(path)};contents={}
    for field,expected in [('base_path',SCHEDULER_BASE),('candidate_path',SCHEDULER_CANDIDATE),('patch_path',SCHEDULER_PATCH)]:
        file=Path(transition[field]);file=file if file.is_absolute() else repo/file
        require(sha(file)==expected,f'Scheduler compatibility artifact hash changed: {field}')
        artifacts[str(file.resolve())]=expected;contents[field]=file.read_text()
    proof=prove_scheduler_scope(contents['base_path'],contents['candidate_path'])
    require(proof==transition.get('ast_proof'),'Scheduler compatibility AST proof differs')
    return dict(schema_version=2,path=str(path),sha256=sha(path),target=TARGET,base_sha256=BASE,
                candidate_sha256=CANDIDATE,runtime_fingerprint=CANDIDATE_RUNTIME,
                base_runtime_fingerprint=RUNTIME,ast_proof=runner['ast_proof'],
                scheduler_transition={key:transition[key] for key in ('target','base_sha256','candidate_sha256','patch_sha256','ast_proof')},
                artifacts_sha256=artifacts)


def source_transition_allowed(target,old,new,compatibility):
    if not compatibility:return False
    if compatibility.get('schema_version')==5:
        if target==TARGET:
            return new==POSTFLIGHT_CANDIDATE and (old==CANCELLATION_CANDIDATE or
                source_transition_allowed(target,old,CANCELLATION_CANDIDATE,compatibility['prior_compatibility']))
        return source_transition_allowed(target,old,new,compatibility['prior_compatibility'])
    if target==TARGET:
        if compatibility.get('schema_version')==4:
            normalized=(old==NORMALIZED_CANCELLATION_BASE
                        and compatibility.get('normalized_pd_transition',{}).get('base_sha256')==NORMALIZED_CANCELLATION_BASE)
            return (old in (BASE,CANDIDATE) or normalized) and new==CANCELLATION_CANDIDATE
        return old==BASE and new==CANDIDATE
    if compatibility.get('schema_version') in (2,3,4) and target==SCHEDULER:
        return old==SCHEDULER_BASE and new==SCHEDULER_CANDIDATE
    if compatibility.get('schema_version') in (3,4):
        return ((target==CLIENT and old==CLIENT_BASE and new==CLIENT_CANDIDATE)
                or (target==PD and old==PD_BASE and new==PD_CANDIDATE))
    return False


def prove_overlap_client_scope(base,candidate):
    original=ast.parse(base);updated=ast.parse(candidate)
    owner=next(n for n in updated.body if isinstance(n,ast.ClassDef) and n.name=='TpModelWorkerClient')
    method=next(n for n in owner.body if isinstance(n,ast.FunctionDef) and n.name=='forward_batch_generation')
    guard=ast.parse('if pp_proxy_tensors is not None:\n raise ValueError("Native overlap is PP1-only; pipeline proxy tensors are unsupported")').body[0]
    require(len(method.args.kwonlyargs)==len(method.args.kw_defaults)==1
            and method.args.kwonlyargs[0].arg=='pp_proxy_tensors'
            and isinstance(method.args.kw_defaults[0],ast.Constant) and method.args.kw_defaults[0].value is None
            and ast.dump(method.body[0])==ast.dump(guard),'Unknown overlap client parameter/guard change')
    method.args.kwonlyargs=[];method.args.kw_defaults=[];method.body.pop(0)
    restored=ast.dump(updated)
    require(restored==ast.dump(original),'Overlap client changed outside the exact signature/initial guard')
    return dict(restored_ast_sha256=hashlib.sha256(restored.encode()).hexdigest(),
                unchanged_existing_function_body=True,removed_parameter='pp_proxy_tensors=None',
                removed_initial_guard_count=1,unaffected_predicate='disable_overlap_schedule is True')


def prove_pd_scope(base,candidate):
    # Byte pins also constrain the new connection method's complete body.
    require(hashlib.sha256(base.encode()).hexdigest()==PD_BASE
            and hashlib.sha256(candidate.encode()).hexdigest()==PD_CANDIDATE,
            'Unknown PD transport source bytes')
    original=ast.parse(base);updated=ast.parse(candidate)
    def owner(tree):return next(n for n in tree.body if isinstance(n,ast.ClassDef) and n.name=='MooncakeKVManager')
    old=owner(original);new=owner(updated)
    counts={name:0 for name in ('_intranode_nvlink','_nvlink_connections','_connect','register_buffer_to_engine_condition','send_aux_condition')}
    retained=[]
    for method in new.body:
        if isinstance(method,ast.FunctionDef):
            if method.name=='_connect':counts['_connect']+=1;continue
            if method.name=='__init__':
                body=[]
                for node in method.body:
                    if (isinstance(node,ast.Assign) and len(node.targets)==1 and isinstance(node.targets[0],ast.Attribute)
                            and isinstance(node.targets[0].value,ast.Name) and node.targets[0].value.id=='self'
                            and node.targets[0].attr in ('_intranode_nvlink','_nvlink_connections')):
                        counts[node.targets[0].attr]+=1
                    else:body.append(node)
                method.body=body
            for name,index in (('register_buffer_to_engine',1),('send_aux',0)):
                if method.name==name:
                    prior=next(n for n in old.body if isinstance(n,ast.FunctionDef) and n.name==name)
                    require(isinstance(method.body[index],ast.If) and isinstance(prior.body[index],ast.If),'Unknown PD guard structure')
                    method.body[index].test=prior.body[index].test;counts[name+'_condition']+=1
        retained.append(method)
    new.body=retained;restored=ast.dump(updated)
    require(all(value==1 for value in counts.values()) and restored==ast.dump(original),
            'PD code changed outside the exact reviewed constructor/connection/aux guards')
    return dict(restored_ast_sha256=hashlib.sha256(restored.encode()).hexdigest(),removed_nodes=counts,
                whole_module_ast_equal=True,unaffected_predicate="disaggregation_mode == 'null'",
                old_pd_evidence='always rejected across this transition')


def load_v3(path,data,current,repo,*,_cancellation_runner=False,_postflight_runner=False):
    require(data.get('base_runtime_fingerprint')==RUNTIME and data.get('candidate_runtime_fingerprint')==COMBINED_RUNTIME,
            'Unknown combined runtime fingerprint')
    runtime={key:value for key,value in current.items() if key.startswith('python/sglang/')}
    original={**runtime,SCHEDULER:SCHEDULER_BASE,CLIENT:CLIENT_BASE,PD:PD_BASE}
    require(digest(runtime)==COMBINED_RUNTIME and digest(original)==RUNTIME
            and runtime==data.get('candidate_runtime_source_sha256') and original==data.get('base_runtime_source_sha256'),
            'Combined runtime keys/hashes differ outside the exact three reviewed files')
    require(all(current.get(key)==expected and sha(repo/key)==(POSTFLIGHT_CANDIDATE if _postflight_runner and key==TARGET else CANCELLATION_CANDIDATE if _cancellation_runner and key==TARGET else expected)
                for key,expected in ((TARGET,CANDIDATE),(SCHEDULER,SCHEDULER_CANDIDATE),(CLIENT,CLIENT_CANDIDATE),(PD,PD_CANDIDATE))),
            'All reviewed candidates must be applied to the live repository before v3 generation')
    v2=Path(data['scheduler_compatibility_path']);v2=v2 if v2.is_absolute() else repo/v2
    require(data.get('scheduler_compatibility_sha256')==V2_ARTIFACT and sha(v2)==V2_ARTIFACT,
            'Original scheduler/runner proof artifact changed')
    prior=load_v2(v2,json.loads(v2.read_text()),{**current,CLIENT:CLIENT_BASE,PD:PD_BASE},repo,
                  _cancellation_runner=_cancellation_runner,_postflight_runner=_postflight_runner)
    artifacts={**prior['artifacts_sha256'],str(path):sha(path)};transitions={}
    for name,target,base,candidate,patch,prover in (
            ('client_transition',CLIENT,CLIENT_BASE,CLIENT_CANDIDATE,CLIENT_PATCH,prove_overlap_client_scope),
            ('pd_transition',PD,PD_BASE,PD_CANDIDATE,PD_PATCH,prove_pd_scope)):
        transition=data.get(name,{})
        require(all(transition.get(key)==value for key,value in
                    [('target',target),('base_sha256',base),('candidate_sha256',candidate),('patch_sha256',patch)]),
                f'Unknown combined source transition: {name}')
        contents={}
        for field,expected in [('base_path',base),('candidate_path',candidate),('patch_path',patch)]:
            file=Path(transition[field]);file=file if file.is_absolute() else repo/file
            require(sha(file)==expected,f'Combined compatibility artifact changed: {name}/{field}')
            artifacts[str(file.resolve())]=expected;contents[field]=file.read_text()
        proof=prover(contents['base_path'],contents['candidate_path'])
        require(proof==transition.get('ast_proof'),f'Combined AST proof differs: {name}')
        transitions[name]={key:transition[key] for key in ('target','base_sha256','candidate_sha256','patch_sha256','ast_proof')}
    return dict(prior,schema_version=3,path=str(path),sha256=sha(path),runtime_fingerprint=COMBINED_RUNTIME,
                artifacts_sha256=artifacts,**transitions)


def seed_request(family):
    return 'auto' if family in ('plain','chunked','pd') else 'off'


def prove_cancellation_scope(base,candidate):
    """Restore only named HTTP-cancellation helpers/wiring, then the entire AST."""
    require(hashlib.sha256(base.encode()).hexdigest()==CANDIDATE
            and hashlib.sha256(candidate.encode()).hexdigest()==CANCELLATION_CANDIDATE,
            'Unknown cancellation runner source bytes')
    old=ast.parse(base);new=ast.parse(candidate)
    old_functions={n.name:n for n in old.body if isinstance(n,ast.FunctionDef)}
    helpers={'cancellation_http_request','paired_cancellation_request','_cancellation_response_aborted','cancellation_confirmed'}
    counts={key:0 for key in (*sorted(helpers),'signature','cancel_req','abort_predicate','pd_callback','pd_call_keyword')}
    callback=ast.dump(ast.parse('cancel_transport=lambda payload,prefill_worker=prefill_worker,decode_worker=decode_worker:paired_cancellation_request(prefill_worker,decode_worker,payload)').body[0])
    new_guard=ast.dump(ast.parse("if not all(cancellation_confirmed(x,prefix+str(i)) for i,x in enumerate(aborted)):\n raise RuntimeError('not all cancellation requests confirmed abort')").body[0])
    old_pre=old_functions['preflight']
    old_cancel=next(n for n in ast.walk(old_pre) if isinstance(n,ast.FunctionDef) and n.name=='cancel_req')
    old_guard=ast.parse("if not all('abort' in json.dumps(x).lower() for x in aborted):\n raise RuntimeError('not all cancellation requests confirmed abort')").body[0]
    class Restore(ast.NodeTransformer):
        owner=None
        def visit_FunctionDef(self,node):
            if self.owner is None and node.name in helpers:counts[node.name]+=1;return None
            prior=self.owner
            if node.name=='cancel_req' and self.owner=='preflight':
                require(ast.dump(node.body[0])==ast.dump(old_cancel.body[0]),'Cancellation request payload changed')
                counts['cancel_req']+=1;return old_cancel
            self.owner=node.name
            if node.name=='preflight':
                require(node.args.args[-1].arg=='cancel_transport' and isinstance(node.args.defaults[-1],ast.Constant)
                        and node.args.defaults[-1].value is None,'Unknown cancellation signature change')
                node.args.args.pop();node.args.defaults.pop();counts['signature']+=1
            result=self.generic_visit(node);self.owner=prior;return result
        def visit_If(self,node):
            if self.owner=='preflight' and ast.dump(node)==new_guard:
                counts['abort_predicate']+=1;return old_guard
            return self.generic_visit(node)
        def visit_Assign(self,node):
            if self.owner=='main' and ast.dump(node)==callback:counts['pd_callback']+=1;return None
            return self.generic_visit(node)
        def visit_Call(self,node):
            if self.owner=='main' and isinstance(node.func,ast.Name) and node.func.id=='preflight':
                retained=[]
                for keyword in node.keywords:
                    if keyword.arg=='cancel_transport':
                        require(isinstance(keyword.value,ast.Name) and keyword.value.id=='cancel_transport','Unknown PD callback argument')
                        counts['pd_call_keyword']+=1
                    else:retained.append(keyword)
                node.keywords=retained
            return self.generic_visit(node)
    restored=ast.dump(Restore().visit(new))
    require(all(v==1 for v in counts.values()) and restored==ast.dump(old),'Runner changed outside exact cancellation helpers, branch and pure callback wiring')
    return dict(restored_ast_sha256=hashlib.sha256(restored.encode()).hexdigest(),restored_nodes=counts,
                whole_module_ast_equal=True,changed_execution_scope="preflight suite == 'correctness' cancellation branch",
                unchanged_request_payloads=True,smoke_and_trace_http_paths_unchanged=True,
                legacy_full_reuse='every expected worker/pair has16 matching normal abort responses; no HTTP error reconstruction')


def load_v4(path,data,current,repo,*,_postflight_runner=False):
    require(data.get('schema_version')==4 and data.get('target')==TARGET
            and data.get('base_sha256')==CANDIDATE and data.get('candidate_sha256')==CANCELLATION_CANDIDATE
            and data.get('patch_sha256')==CANCELLATION_PATCH,'Unknown cancellation compatibility transition')
    runtime={k:v for k,v in current.items() if k.startswith('python/sglang/')}
    require(digest(runtime)==COMBINED_RUNTIME and runtime==data.get('runtime_source_sha256'),
            'Frozen618 runtime keyset/hashes changed')
    require(current.get(TARGET)==CANCELLATION_CANDIDATE and sha(repo/TARGET)==(POSTFLIGHT_CANDIDATE if _postflight_runner else CANCELLATION_CANDIDATE),
            'Exact cancellation runner must be applied at the idle boundary before v4 generation')
    prior_path=Path(data['prior_compatibility_path']);prior_path=prior_path if prior_path.is_absolute() else repo/prior_path
    require(data.get('prior_compatibility_sha256')==V3_ARTIFACT and sha(prior_path)==V3_ARTIFACT,
            'Prior combined runtime proof changed')
    # Only this validated schema4 entry invokes the exact, hard-coded new-runner
    # override while checking the immutable old proof chain. Schema3 keeps its
    # original live-runner requirement; no caller-supplied hash is accepted.
    prior=load_v3(prior_path,json.loads(prior_path.read_text()),{**current,TARGET:CANDIDATE},repo,_cancellation_runner=True,_postflight_runner=_postflight_runner)
    artifacts={**prior['artifacts_sha256'],str(path):sha(path)};contents={}
    for field,expected in [('base_path',CANDIDATE),('candidate_path',CANCELLATION_CANDIDATE),('patch_path',CANCELLATION_PATCH)]:
        file=Path(data[field]);file=file if file.is_absolute() else repo/file
        require(sha(file)==expected,f'Cancellation proof artifact changed: {field}')
        artifacts[str(file.resolve())]=expected;contents[field]=file.read_text()
    proof=prove_cancellation_scope(contents['base_path'],contents['candidate_path'])
    require(proof==data.get('ast_proof'),'Cancellation AST proof differs')
    stage=data.get('normalized_pd_transition',{})
    require(stage.get('base_sha256')==NORMALIZED_CANCELLATION_BASE
            and stage.get('patch_sha256')==PD_STAGE_PATCH and stage.get('evidence_sha256')==PD_CAPTURE_EVIDENCE,
            'Unknown normalized PD source/evidence transition')
    stage_contents={}
    for field,expected in [('base_path',NORMALIZED_CANCELLATION_BASE),('patch_path',PD_STAGE_PATCH),('evidence_path',PD_CAPTURE_EVIDENCE)]:
        file=Path(stage[field]);file=file if file.is_absolute() else repo/file
        require(sha(file)==expected,f'Normalized PD proof artifact changed: {field}')
        artifacts[str(file.resolve())]=expected;stage_contents[field]=file.read_text()
    stage_proof=prove_pd_stage_scope(stage_contents['base_path'],contents['candidate_path'])
    require(stage_proof==stage.get('ast_proof'),'PD stage AST proof differs')
    return dict(prior,schema_version=4,path=str(path),sha256=sha(path),candidate_sha256=CANCELLATION_CANDIDATE,
                cancellation_transition=dict(target=TARGET,base_sha256=CANDIDATE,candidate_sha256=CANCELLATION_CANDIDATE,
                                             patch_sha256=CANCELLATION_PATCH,ast_proof=proof),
                normalized_pd_transition=dict(stage,candidate_path=str(Path(data['candidate_path']).resolve())),artifacts_sha256=artifacts)


def effective(family,pp,requested):
    require(requested in ('off','auto','on'),'Unknown requested overlap mode')
    supported=family in ('plain','chunked','pd') and pp==1
    require(requested!='on' or supported,'Unsupported native overlap mode')
    return 'on' if supported and requested!='off' else 'off'


def check_support(metadata,current,compatibility,requested,run=None):
    if compatibility.get('schema_version')==5:
        return check_postflight_support(metadata,current,compatibility,requested,run)
    old=metadata.get('source_sha256',{});args=metadata.get('args',{})
    require(set(old)==set(current),'Source path set differs outside the reviewed runner transition')
    differences=sorted(key for key in current if old[key]!=current[key])
    require(all(source_transition_allowed(key,old[key],current[key],compatibility) for key in differences),
            'Unknown source hash change; no compatibility exception applies')
    scheduler_changed=SCHEDULER in differences
    client_changed=CLIENT in differences;pd_changed=PD in differences
    if pd_changed:
        require(args.get('family') in ('plain','chunked','muxwise','muxpp'),
                'Old PD evidence cannot cross the aux transport fix; fresh preflight required')
    if scheduler_changed:
        require(args.get('family') in ('plain','chunked','pd')
                or (args.get('family')=='muxpp' and args.get('pp',0)>1),
                'Old original MuxWise evidence cannot cross the scheduler abort/flush fix; new preflight required')
    cancellation_changed=TARGET in differences and current.get(TARGET)==CANCELLATION_CANDIDATE
    if TARGET in differences and old.get(TARGET)==BASE:
        require(args.get('overlap','off')=='off','Old runner evidence cannot claim a different native overlap mode')
        observed='off';kind='reviewed_default_off_compatibility'
    else:
        expected_runner=CANCELLATION_CANDIDATE if compatibility.get('schema_version')==4 else CANDIDATE
        require(current.get(TARGET)==expected_runner,'Current runner is not the reviewed candidate')
        recorded=args.get('overlap')
        observed=effective(args['family'],args['pp'],recorded)
        expected=dict(requested=recorded,effective=observed,enabled=observed=='on',
                      supported=args['family'] in ('plain','chunked','pd') and args['pp']==1)
        require(metadata.get('overlap_policy')==expected,'Missing or contradictory explicit overlap metadata')
        kind='exact_current_source'
    if scheduler_changed:
        kind='reviewed_runner_and_scheduler_compatibility' if TARGET in differences else 'reviewed_unaffected_scheduler_compatibility'
    desired=effective(args['family'],args['pp'],requested)
    require(desired==observed,'Overlap mode changes require a new preflight; old off-mode evidence cannot qualify auto-on')
    if client_changed:
        require(observed=='off','Old native-overlap-on evidence cannot cross the dispatch fix; fresh preflight required')
    if client_changed or pd_changed:kind='reviewed_combined_inactive_modes_compatibility'
    commands=metadata.get('commands',[])
    require(len(commands)==args.get('replicas'),'Missing backend command coverage')
    for entry in commands:
        command=entry.get('command',[])
        require(command[1:3]==['-m','sglang.launch_server']
                and command.count('--disable-overlap-schedule')==int(observed=='off'),
                'Backend command contradicts recorded effective overlap mode')
        if scheduler_changed:
            require(command.count('--enable-pdmux')==int(args['family']=='muxpp'),
                    'Backend command contradicts unaffected scheduler family')
    require(not any('--disable-overlap-schedule'.startswith(flag.split('=',1)[0])
                    for flag in args.get('extra',[]) if flag.startswith('--')),
            'Historical extra overlap override is outside the command-equivalence proof')
    cancellation_pins={}
    if cancellation_changed:
        require(run is not None,'Cancellation runner transition requires the bound run directory')
        if old.get(TARGET)==NORMALIZED_CANCELLATION_BASE:
            cancellation_pins=check_normalized_pd_cancellation(run,metadata,compatibility)
        else:
            cancellation_pins=check_saved_cancellation(run,metadata)
        kind='reviewed_cancellation_runner_compatibility' if kind=='exact_current_source' else kind+'+cancellation_runner'
    return dict(kind=kind,source_differences=differences,effective_overlap=desired,
                cancellation_runner_transition=cancellation_changed,cancellation_evidence_sha256=cancellation_pins,
                scheduler_transition=scheduler_changed,
                native_overlap_client_transition=client_changed,pd_transport_transition=pd_changed,
                requested_overlap=requested,compatibility_path=compatibility['path'],
                compatibility_sha256=compatibility['sha256'])


def check_actual_scheduler_scope(run,metadata):
    """Check the guarded predicate on each actual worker, beyond family labels."""
    result={};args=metadata['args'];expected_mux=args['family']=='muxpp'
    for index in range(args['replicas']):
        path=Path(run)/f'worker{index}/server-final.json'
        require(path.is_file(),f'Missing actual scheduler scope evidence: {path}')
        snapshot=json.loads(path.read_text())
        require(snapshot.get('enable_pdmux') is expected_mux and type(snapshot.get('pp_size')) is int
                and snapshot['pp_size']==args['pp']
                and (snapshot['enable_pdmux'] is False or snapshot['pp_size']>1),
                f'Actual scheduler is outside the unaffected transition scope: {path}')
        result[str(path.resolve())]=sha(path)
    return result


def check_actual_non_pd_scope(run,metadata):
    require(metadata['args']['family'] in ('plain','chunked','muxwise','muxpp'),'Old PD evidence requires new preflight')
    result={}
    for index in range(metadata['args']['replicas']):
        path=Path(run)/f'worker{index}/server-final.json'
        require(path.is_file(),f'Missing actual non-PD scope evidence: {path}')
        snapshot=json.loads(path.read_text())
        require(snapshot.get('disaggregation_mode')=='null',f'Actual worker is not proven non-PD: {path}')
        result[str(path.resolve())]=sha(path)
    return result


def check_actual_overlap(run,metadata,expected):
    """ServerArgs can force overlap off; check every actual post-preflight API."""
    result={}
    for index in range(metadata['args']['replicas']):
        path=Path(run)/f'worker{index}/server-final.json'
        require(path.is_file(),f'Missing actual overlap evidence: {path}')
        snapshot=json.loads(path.read_text())
        require(isinstance(snapshot,dict) and snapshot.get('disable_overlap_schedule') is (expected=='off'),
                f'Server actual overlap mode differs from requested seed: {path}')
        result[str(path.resolve())]=sha(path)
    return result


def check_saved_cancellation(run,metadata):
    """Legacy full runs must contain all normal abort bodies; logs are not substitutes."""
    args=metadata.get('args',{});suite=args.get('suite')
    require(suite in ('smoke','trace','correctness'),'Unknown legacy preflight suite')
    if suite in ('smoke','trace'):return {}
    run=Path(run).resolve()
    require(metadata.get('status')=='execution_complete' and Path(args.get('out','')).resolve()==run,
            'Legacy full cancellation requires the completed bound run')
    replicas=args.get('replicas');workers=metadata.get('workers',[])
    require(type(replicas) is int and replicas>0 and len(workers)==replicas
            and [w.get('index') for w in workers]==list(range(replicas)),
            'Missing legacy worker coverage')
    pd=args.get('family')=='pd';pins={};folders=[]
    if pd:
        prefills=args.get('prefill_replicas')
        require(type(prefills) is int and 0<prefills<replicas,'Invalid PD role layout')
        require(all(w.get('role')==('prefill' if i<prefills else 'decode') for i,w in enumerate(workers)),
                'Saved workers contradict PD role layout')
        pairs=[(workers[i%prefills],workers[prefills+i%(replicas-prefills)]) for i in range(max(prefills,replicas-prefills))]
        for index,(prefill,decode) in enumerate(pairs):
            folder=run/f'pair{index}';path=folder/'pair.json'
            require(path.is_file() and json.loads(path.read_text())==dict(prefill=prefill,decode=decode),
                    f'Missing or inconsistent saved PD pair: {path}')
            pins[str(path)]=sha(path);folders.append(folder)
    else:
        require(all(w.get('role')=='model' for w in workers),'Non-PD worker roles differ')
        folders=[run/f'worker{i}' for i in range(replicas)]
    def normal_abort(meta,rid):
        finish=meta.get('finish_reason',{}) if isinstance(meta,dict) else {}
        return (isinstance(meta,dict) and isinstance(finish,dict) and meta.get('id')==rid and finish.get('type')=='abort'
                and finish.get('message') in ('Aborted','Abort before prefill','Aborted before prefill','Aborted by AbortReq.')
                and finish.get('status_code') is None and finish.get('err_type') is None)
    for folder in folders:
        path=folder/'cancellation.json';require(path.is_file(),f'Missing legacy cancellation evidence: {path}')
        values=json.loads(path.read_text())
        require(isinstance(values,list) and len(values)==16,f'All16 legacy abort responses required: {path}')
        prefix=None
        for index,row in enumerate(values):
            require(isinstance(row,dict) and 'error' not in row,'Transport errors cannot reuse the old cancellation classifier')
            meta=row.get('meta_info',{});rid=meta.get('id') if isinstance(meta,dict) else None
            require(isinstance(rid,str) and re.fullmatch(r'a100-cancel-\d+-\d+',rid),'Missing normal abort request ID')
            if prefix is None:prefix=rid.rsplit('-',1)[0]+'-'
            require(rid==prefix+str(index) and normal_abort(meta,rid),'Legacy response is not the matching normal AbortReq terminal state')
            tokens=row.get('output_ids');count=meta.get('completion_tokens')
            empty_waiting=(tokens is None and row.get('text')=='' and count==0
                           and meta['finish_reason'].get('message')=='Abort before prefill')
            require(type(count) is int and (empty_waiting or (isinstance(tokens,list) and count==len(tokens)<1000)),
                    'Legacy abort lacks a consistent interrupted output count')
            if pd:
                pre=row.get('_prefill_meta_info',{});finish=pre.get('finish_reason',{}) if isinstance(pre,dict) else {}
                pre_success=(isinstance(pre,dict) and pre.get('id')==rid and isinstance(finish,dict)
                             and finish.get('type')=='length' and finish.get('length')==0
                             and finish.get('status_code') is None and finish.get('err_type') is None)
                require(normal_abort(pre,rid) or pre_success,'PD legacy prefill is missing or failed')
        pins[str(path)]=sha(path)
    return pins


def prove_pd_stage_scope(base,candidate):
    require(hashlib.sha256(base.encode()).hexdigest()==NORMALIZED_CANCELLATION_BASE
            and hashlib.sha256(candidate.encode()).hexdigest()==CANCELLATION_CANDIDATE,
            'Unknown PD stage source bytes')
    original=ast.parse(base);updated=ast.parse(candidate)
    def assignment(tree):
        owner=next(n for n in tree.body if isinstance(n,ast.FunctionDef) and n.name=='_cancellation_response_aborted')
        matches=[(i,n) for i,n in enumerate(owner.body) if isinstance(n,ast.Assign)
                 and len(n.targets)==1 and isinstance(n.targets[0],ast.Name) and n.targets[0].id=='prefixes']
        require(len(matches)==1,'Unknown PD prefix assignment');return owner,matches[0]
    _,(_,before)=assignment(original);owner,(index,_)=assignment(updated);owner.body[index]=before
    restored=ast.dump(updated);require(restored==ast.dump(original),'PD stage change extends beyond the prefix assignment')
    return dict(restored_ast_sha256=hashlib.sha256(restored.encode()).hexdigest(),
                whole_module_ast_equal=True,changed_assignment='_cancellation_response_aborted.prefixes',
                paired_request_payload_and_room_unchanged=True,ordinary_none_role_unchanged=True)


def pd_capture_classifier(path):
    """Execute only the two pure, exact-hash classifier functions; no runner import."""
    require(sha(path)==CANCELLATION_CANDIDATE,'PD replay classifier source changed')
    names={'_cancellation_response_aborted','cancellation_confirmed'}
    body=[n for n in ast.parse(Path(path).read_text()).body if isinstance(n,ast.FunctionDef) and n.name in names]
    require(len(body)==2,'Missing PD classifier functions')
    namespace={};exec(compile(ast.Module(body=body,type_ignores=[]),str(path),'exec'),namespace)
    return namespace['cancellation_confirmed']


def validate_pd_capture_records(rows,pair,classifier):
    require(isinstance(rows,list) and len(rows)==16,'Every PD pair requires16 normalized captures')
    require(isinstance(pair,dict) and set(pair)=={'prefill','decode'}
            and all(isinstance(pair[role],dict) and pair[role].get('role')==role for role in pair),
            'Invalid saved PD pair roles')
    prefix=None
    for index,row in enumerate(rows):
        require(isinstance(row,dict),'Missing normalized PD capture')
        rid=row.get('requested_rid');room=row.get('bootstrap_room');responses=row.get('responses')
        require(isinstance(rid,str) and re.fullmatch(r'a100-cancel-\d+-\d+',rid),'Missing normalized cancellation RID')
        if prefix is None:prefix=rid.rsplit('-',1)[0]+'-'
        require(rid==prefix+str(index),'Missing or reordered cancellation RID')
        require(type(room) is int and 0<=room<2**63,'Missing shared bootstrap room')
        require(isinstance(responses,dict) and set(responses)=={'prefill','decode'},'Both PD role captures required')
        for role,response in responses.items():
            require(isinstance(response,dict) and 'error' not in response
                    and response.get('requested_rid')==rid and response.get('port')==pair[role]['port'],
                    'PD capture RID/port differs from saved pair')
            require(type(response.get('http_status')) is int and response['http_status'] in (200,500),
                    'Unexpected PD cancellation HTTP status')
            raw=response.get('raw_body');encoded=response.get('raw_body_base64')
            require(isinstance(raw,str) and isinstance(encoded,str),'Missing raw PD response bytes')
            require(base64.b64decode(encoded,validate=True).decode('utf-8',errors='replace')==raw,
                    'PD raw body/base64 mismatch')
            require(json.loads(raw)==response.get('response'),'PD raw body/parsed JSON mismatch')
            body=response['response'];require(isinstance(body,dict),'PD response body must be JSON object')
            if response['http_status']==500:
                rooms=re.findall(r'bootstrap_room=(\d+)',body.get('message',''))
                require(rooms and all(int(value)==room for value in rooms),'PD error body bootstrap room differs')
            else:
                meta=body.get('meta_info',{})
                require(isinstance(meta,dict) and meta.get('id')==rid,'PD normal body RID differs')
                # Normal bodies do not echo the room. The exact paired-call source
                # sends the same payload/outer room to both roles; validate any echo.
                require('bootstrap_room' not in meta or meta['bootstrap_room']==room,'PD normal body room differs')
        require(classifier(row,rid),'New exact role/stage cancellation predicate rejected saved capture')


def check_normalized_pd_cancellation(run,metadata,compatibility):
    """Reuse only the four reviewed successful e5 full runs; never infer full success."""
    run=Path(run).resolve();stage=compatibility.get('normalized_pd_transition',{})
    path=Path(stage['evidence_path']);require(sha(path)==PD_CAPTURE_EVIDENCE,'Reviewed PD evidence manifest changed')
    manifest=json.loads(path.read_text());record=manifest.get('runs',{}).get(str(run))
    require(manifest.get('source_sha256')==NORMALIZED_CANCELLATION_BASE and record is not None,
            'Run is outside the four reviewed successful normalized PD full runs')
    pins={str(path.resolve()):PD_CAPTURE_EVIDENCE,**record['files_sha256']}
    for filename,expected in pins.items():require(sha(filename)==expected,f'Reviewed PD evidence changed: {filename}')
    require(metadata==json.loads(Path(record['metadata_path']).read_text()),'PD metadata differs from bound saved run')
    args=metadata['args'];workers=metadata['workers'];topology=record['topology']
    require(all(args.get(key)==value for key,value in topology.items()) and args.get('model')=='8b'
            and args.get('family')=='pd' and args.get('suite')=='correctness' and args.get('stress200') is True
            and Path(args['out']).resolve()==run and metadata.get('status')=='execution_complete',
            'Normalized PD reuse requires the completed reviewed full configuration')
    outcome=json.loads(Path(record['outcome_path']).read_text())
    require(outcome.get('status')=='completed' and outcome.get('returncode')==0,'PD controller did not complete')
    post=json.loads(Path(record['postflight_path']).read_text());recovery=json.loads(Path(record['recovery_path']).read_text())
    resource=json.loads(Path(record['resource_path']).read_text())
    require(post.get('status')=='passed' and not post.get('errors') and post.get('all_rank_resource_status')=='passed'
            and post.get('recovery_status')=='passed' and recovery.get('status')=='passed' and not recovery.get('errors')
            and resource.get('status')=='passed' and not resource.get('errors') and not resource.get('missing_evidence'),
            'PD postflight/recovery/resource gate did not pass')
    prefills=args['prefill_replicas'];replicas=args['replicas'];pairs=max(prefills,replicas-prefills)
    require(len(workers)==replicas and [w['index'] for w in workers]==list(range(replicas))
            and all(w['role']==('prefill' if i<prefills else 'decode') for i,w in enumerate(workers)),
            'PD worker/role coverage differs')
    require([p['pair'] for p in post['pairs']]==list(range(pairs))
            and [p['pair'] for p in recovery['pairs']]==list(range(pairs)), 'PD postflight pair coverage differs')
    classifier=pd_capture_classifier(stage['candidate_path'])
    for index in range(pairs):
        p=post['pairs'][index];r=recovery['pairs'][index]
        require(all(p.get(k)==v for k,v in [('serial',14),('cache',2),('concurrent',32),('cancel_confirmed',16),('stress200',200)])
                and r.get('status')=='passed' and r.get('compared_tokens')==16 and r.get('first_difference') is None
                and len(r.get('recovery_output_ids',[]))==16 and r['recovery_output_ids']==r.get('serial_prefix_ids'),
                'PD full cases/stress/exact recovery incomplete')
        folder=run/f'pair{index}';pair=json.loads((folder/'pair.json').read_text())
        require(pair==dict(prefill=workers[index%prefills],decode=workers[prefills+index%(replicas-prefills)]),
                'PD pair differs from saved workers')
        validate_pd_capture_records(json.loads((folder/'cancellation.json').read_text()),pair,classifier)
    return pins



def prove_postflight_scope(base,candidate):
    """Exact postmeasurement-only source; no command, model or resource relaxation."""
    require(hashlib.sha256(base.encode()).hexdigest()==CANCELLATION_CANDIDATE
            and hashlib.sha256(candidate.encode()).hexdigest()==POSTFLIGHT_CANDIDATE,
            'Unknown postflight runner source bytes')
    original=ast.parse(base);updated=ast.parse(candidate)
    helpers=[n for n in updated.body if isinstance(n,ast.FunctionDef) and n.name=='quiesce_router']
    require(len(helpers)==1,'Expected exactly one quiesce helper');updated.body.remove(helpers[0])
    main=next(n for n in updated.body if isinstance(n,ast.FunctionDef) and n.name=='main')
    block=next(n for n in main.body if isinstance(n,ast.Try))
    call=ast.dump(ast.parse('quiesce_router(router if a.replicas>1 else None,services,metadata)').body[0])
    indices=[i for i,n in enumerate(block.body) if ast.dump(n)==call]
    require(len(indices)==1,'Expected one final quiesce call');index=indices[0]
    require(ast.dump(block.body[index+1])==ast.dump(ast.parse("save(a.out/'run.json',metadata)").body[0]),
            'Expected adjacent planned-shutdown metadata save')
    require(isinstance(block.body[index-1],ast.If)
            and ast.dump(block.body[index-1].test)==ast.dump(ast.parse('a.oracle_cases',mode='eval').body)
            and isinstance(block.body[index+2],ast.For)
            and ast.dump(block.body[index+2].iter)==ast.dump(ast.parse('workers',mode='eval').body),
            'Quiescence moved outside the completed-client/oracle and final-resource boundary')
    del block.body[index:index+2]
    restored=ast.dump(updated)
    require(restored==ast.dump(original),'Postflight source differs beyond helper and two final statements')
    return dict(whole_module_ast_equal=True,removed_nodes=dict(helper=1,call=1,metadata_save=1),
                restored_ast_sha256=hashlib.sha256(restored.encode()).hexdigest(),
                measurement_and_preflight_paths_unchanged=True,strict_resource_phase_unchanged=True)


def load_postflight(path,data,current,repo):
    require(data.get('schema_version')==5 and data.get('target')==TARGET
            and data.get('base_sha256')==CANCELLATION_CANDIDATE
            and data.get('candidate_sha256')==POSTFLIGHT_CANDIDATE
            and data.get('patch_sha256')==POSTFLIGHT_PATCH,'Unknown postflight-only transition')
    require(current.get(TARGET)==POSTFLIGHT_CANDIDATE and sha(repo/TARGET)==POSTFLIGHT_CANDIDATE,
            'Exact postflight runner must be applied before generation')
    runtime={k:v for k,v in current.items() if k.startswith('python/sglang/')}
    require(digest(runtime)==COMBINED_RUNTIME and runtime==data.get('runtime_source_sha256'),
            'Postflight runtime618 keyset/hash mismatch')
    prior_path=Path(data['prior_compatibility_path']).resolve()
    require(data.get('prior_compatibility_sha256')==POSTFLIGHT_PRIOR and sha(prior_path)==POSTFLIGHT_PRIOR,
            'Prior cancellation/full evidence compatibility changed')
    prior=load_v4(prior_path,json.loads(prior_path.read_text()),{**current,TARGET:CANCELLATION_CANDIDATE},
                  repo,_postflight_runner=True)
    artifacts={**prior['artifacts_sha256'],str(path):sha(path)};contents={}
    for field,expected in [('base_path',CANCELLATION_CANDIDATE),('candidate_path',POSTFLIGHT_CANDIDATE),('patch_path',POSTFLIGHT_PATCH)]:
        file=Path(data[field]).resolve();require(sha(file)==expected,f'Postflight artifact changed: {field}')
        artifacts[str(file)]=expected;contents[field]=file.read_text()
    proof=prove_postflight_scope(contents['base_path'],contents['candidate_path'])
    require(proof==data.get('ast_proof'),'Postflight AST proof differs')
    expected={TARGET:CANCELLATION_CANDIDATE,POSTFLIGHT_HELPER:POSTFLIGHT_HELPER_BASE,
              POSTFLIGHT_RENDERER:POSTFLIGHT_RENDERER_BASE}
    archive=data.get('archive',{})
    require(set(archive)==set(expected),'Unknown/missing historical archive path')
    for key,old in expected.items():
        record=archive[key];file=Path(record['archive_path']).resolve()
        require(record.get('base_sha256')==old and sha(file)==old,'Historical source archive changed')
        require(record.get('candidate_sha256')==sha(repo/key),'Reviewed new helper/runner/renderer bytes differ')
        artifacts[str(file)]=old
    require(archive[TARGET]['candidate_sha256']==POSTFLIGHT_CANDIDATE
            and archive[POSTFLIGHT_HELPER]['candidate_sha256']==sha(__file__),
            'Executing helper does not match the exact reviewed artifact')
    return dict(prior,schema_version=5,path=str(path),sha256=sha(path),candidate_sha256=POSTFLIGHT_CANDIDATE,
                prior_compatibility=prior,archive=archive,postflight_ast_proof=proof,artifacts_sha256=artifacts)


def check_postflight_support(metadata,current,compatibility,requested,run):
    require(run is not None and metadata.get('status')=='execution_complete'
            and Path(metadata.get('args',{}).get('out','')).resolve()==Path(run).resolve(),
            'Postflight reuse requires the actual completed bound run')
    original=metadata.get('source_sha256',{})
    require(set(original)==set(current),'Postflight serving source keyset differs')
    require(current.get(TARGET)==POSTFLIGHT_CANDIDATE,'Wrong postflight execution source')
    # Only the source view passed into unchanged prior gates is adapted; the
    # historical metadata, files and returned actual differences remain intact.
    predecessor={**current,TARGET:CANCELLATION_CANDIDATE}
    checked=metadata
    if original.get(TARGET)==POSTFLIGHT_CANDIDATE:
        checked={**metadata,'source_sha256':{**original,TARGET:CANCELLATION_CANDIDATE}}
    result=check_support(checked,predecessor,compatibility['prior_compatibility'],requested,run=run)
    return dict(result,kind=result['kind']+'+reviewed_postmeasurement_only',
                source_differences=sorted(k for k in current if original[k]!=current[k]),
                postflight_only_transition=original.get(TARGET)!=POSTFLIGHT_CANDIDATE,
                compatibility_path=compatibility['path'],compatibility_sha256=compatibility['sha256'])


def resolve_postflight_pins(pins,repo,compatibility):
    """Verify all old pins; replace exactly reviewed live paths with archived bytes."""
    require(compatibility.get('schema_version')==5 and isinstance(pins,dict) and pins,
            'Explicit postflight proof and nonempty historical pins required')
    repo=Path(repo).resolve();result={};seen={}
    expected={TARGET:CANCELLATION_CANDIDATE,POSTFLIGHT_HELPER:POSTFLIGHT_HELPER_BASE,
              POSTFLIGHT_RENDERER:POSTFLIGHT_RENDERER_BASE}
    for filename,old in pins.items():
        file=Path(filename);file=(file if file.is_absolute() else repo/file).resolve()
        require(file.is_file() and isinstance(old,str) and re.fullmatch(r'[0-9a-f]{64}',old),
                f'Historical pin missing or invalid: {file}')
        require(str(file) not in seen or seen[str(file)]==old,'Conflicting historical aliases')
        seen[str(file)]=old;actual=sha(file)
        if actual==old:result[str(file)]=old;continue
        key=str(file.relative_to(repo)) if file.is_relative_to(repo) else None
        require(key in expected and old==expected[key],f'Unreviewed historical pin change: {file}')
        record=compatibility['archive'][key];archive=Path(record['archive_path']).resolve()
        require(record['base_sha256']==old and sha(archive)==old
                and record['candidate_sha256']==actual,'Exact archived/current source pair changed')
        result[str(archive)]=old;result[str(file)]=actual
    result.update(compatibility['artifacts_sha256'])
    return result


def postflight_measurement_identity(measurement,current,compatibility):
    require(compatibility.get('schema_version')==5 and measurement.get('selection_eligible') is True,
            'Only complete eligible architecture measurements can reuse the postflight proof')
    old=measurement.get('source_sha256',{})
    require(set(old)==set(current) and current.get(TARGET)==POSTFLIGHT_CANDIDATE,
            'Measured source path set or execution runner differs')
    require(old.get(TARGET) in (CANCELLATION_CANDIDATE,POSTFLIGHT_CANDIDATE)
            and all(old[k]==current[k] for k in current if k!=TARGET),
            'Measured source differs outside the one postmeasurement runner transition')
    require(digest({k:v for k,v in current.items() if k.startswith('python/sglang/')})==COMBINED_RUNTIME,
            'Measured runtime618 differs')
    software=measurement.get('platform_software')
    require(isinstance(software,dict) and measurement.get('source_fingerprint')==digest(dict(runtime=old,platform_software=software)),
            'Measured actual source/platform fingerprint differs')
    return dict(actual_source_fingerprint=measurement['source_fingerprint'],actual_source_sha256=old,
                execution_source_sha256=current,compatibility_path=compatibility['path'],
                compatibility_sha256=compatibility['sha256'],
                comparison_fingerprint=digest(dict(runtime=current,platform_software=software,
                                                   postmeasurement_proof=compatibility['sha256'])))
