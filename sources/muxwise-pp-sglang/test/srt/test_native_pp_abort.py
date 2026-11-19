"""Exercise native Scheduler PP abort paths without CUDA/framework imports."""
import ast
import logging
from pathlib import Path
from types import SimpleNamespace

import pytest


SOURCE=Path(__file__).resolve().parents[2]/'python/sglang/srt/managers/scheduler.py'


def scheduler_class():
    tree=ast.parse(SOURCE.read_text())
    original=next(node for node in tree.body if isinstance(node,ast.ClassDef) and node.name=='Scheduler')
    methods=[]
    for node in original.body:
        if isinstance(node,ast.FunctionDef) and node.name in {'abort_request','event_loop_pp','_abortable_running_requests'}:
            node.decorator_list=[]
            methods.append(node)
    cls=ast.ClassDef(name='Scheduler',bases=[],keywords=[],body=methods,decorator_list=[])
    module=ast.fix_missing_locations(ast.Module(body=[cls],type_ignores=[]))
    namespace={'AbortReq':SimpleNamespace,'DisaggregationMode':SimpleNamespace(PREFILL='prefill',DECODE='decode'),
               'logger':logging.getLogger('native-pp-abort-test'),'ScheduleBatch':lambda **kw:SimpleNamespace(**kw)}
    exec(compile(module,str(SOURCE),'exec'),namespace)
    return namespace['Scheduler']


class Request:
    def __init__(self,rid,finished=False):
        self.rid=rid;self.done=finished;self.abort_writes=0;self._abort=False
    def finished(self):return self.done
    @property
    def to_abort(self):return self._abort
    @to_abort.setter
    def to_abort(self,value):self.abort_writes+=1;self._abort=value


def batch(*requests):return SimpleNamespace(reqs=list(requests))


def scheduler(pp=4,mux=False):
    s=scheduler_class()()
    s.pp_size=pp;s.enable_pdmux=mux;s.waiting_queue=[];s.grammar_queue=[]
    s.disaggregation_mode=None;s.running_batch=batch();s.cur_batch=None;s.last_batch=None
    return s


@pytest.mark.parametrize('location',['running_mbs','_pp_native_mbs','_pp_native_last_mbs'])
def test_abort_reaches_live_request_outside_selected_native_microbatch(location):
    s=scheduler();hidden=Request('cancel-hidden')
    setattr(s,location,[None,batch(hidden),None,batch()])
    s.abort_request(SimpleNamespace(rid='cancel-',abort_all=False))
    assert hidden.to_abort


def test_native_pipeline_exposes_the_actual_active_and_last_arrays_before_receiving_abort():
    s=scheduler()
    class StopLoop(Exception):pass
    def receive():raise StopLoop()
    s.recv_requests=receive
    with pytest.raises(StopLoop):s.event_loop_pp()
    assert len(s._pp_native_mbs)==4
    assert len(s._pp_native_last_mbs)==4
    hidden=Request('during-prefill')
    s._pp_native_mbs[2]=batch(hidden)
    s.abort_request(SimpleNamespace(rid='during-',abort_all=False))
    assert hidden.to_abort


def test_shared_request_is_marked_once_and_finished_or_nonmatching_requests_are_preserved():
    s=scheduler();live=Request('cancel-1');finished=Request('cancel-done',True);other=Request('keep-1')
    s.running_batch=batch(live,finished);s.cur_batch=batch(live);s.last_batch=batch(live)
    s.running_mbs=[s.running_batch,batch(other)]
    s._pp_native_mbs=[s.cur_batch];s._pp_native_last_mbs=[s.last_batch]
    s.abort_request(SimpleNamespace(rid='cancel-',abort_all=False))
    assert live.abort_writes==1
    assert not finished.to_abort
    assert not other.to_abort


def test_abort_all_reaches_every_native_microbatch_without_matching_prefix():
    s=scheduler();requests=[Request(str(i)) for i in range(4)]
    s.running_mbs=[batch(req) for req in requests]
    s.abort_request(SimpleNamespace(rid='absent',abort_all=True))
    assert all(req.to_abort for req in requests)


@pytest.mark.parametrize('pp,mux',[(1,False),(4,True)])
def test_pp1_and_specialized_mux_keep_existing_current_and_running_scope(pp,mux):
    s=scheduler(pp,mux);running=Request('r');current=Request('c');hidden=Request('h')
    s.running_batch=batch(running);s.cur_batch=batch(current);s.running_mbs=[batch(hidden)]
    s._pp_native_mbs=[batch(hidden)];s._pp_native_last_mbs=[batch(hidden)]
    s.abort_request(SimpleNamespace(rid='',abort_all=True))
    assert running.to_abort and current.to_abort
    assert not hidden.to_abort
