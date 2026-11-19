"""CPU regressions for the explicitly assigned physical GPUs 0–3."""
import importlib.util
import os
from pathlib import Path
import sys
from unittest import mock

import pytest

import control
import platform_probe
import server_run
import campaign

EXPECTED = [
    'GPU-01a10c02-8860-b872-3eeb-59004ed80ce2',
    'GPU-1c7d2944-1c3a-07a4-5810-ee0b6f269aaa',
    'GPU-dc982fa8-613f-9bb2-af81-fd679a5ac64a',
    'GPU-8bcee978-642e-88df-687f-bf5530de7edb',
]

def test_environment_and_workers_only_address_assigned_gpus(tmp_path):
    env = control.environment(tmp_path, {'CUDA_VISIBLE_DEVICES': '4,5,6,7'})
    assert env['CUDA_VISIBLE_DEVICES'].split(',') == EXPECTED
    workers = server_run.worker_layout('chunked', 2, 1, 2, 0, 32100)
    assert workers[0]['gpus'] == EXPECTED[:2]
    assert workers[1]['gpus'] == EXPECTED[2:]
    assert campaign.create_manifest()['physical_gpus'] == [0, 1, 2, 3]

def test_gpu_queries_cannot_touch_unassigned_indices():
    def query(argv, **kwargs):
        assert argv[argv.index('-i') + 1] == '0,1,2,3'
        if '--query-gpu=index,uuid' in argv:
            return ''.join(f'{i}, {uuid}\n' for i, uuid in enumerate(EXPECTED))
        return ''
    with mock.patch.object(control.subprocess, 'check_output', side_effect=query):
        assert control.gpu_processes() == []

def test_probe_rejects_a_valid_four_gpu_mask_outside_assignment():
    env = dict(CUDA_VISIBLE_DEVICES=','.join(['GPU-other0','GPU-other1','GPU-other2','GPU-other3']),
               RANK='0',LOCAL_RANK='0',WORLD_SIZE='4',LOCAL_WORLD_SIZE='4')
    with pytest.raises(ValueError, match='assigned'):
        platform_probe.launch_config(env)
    env['CUDA_VISIBLE_DEVICES'] = ','.join(EXPECTED)
    assert platform_probe.launch_config(env)[3] == EXPECTED

def test_old_campaign_suspension_does_not_block_new_campaign(tmp_path):
    old = control.REPO / 'run/a100-4gpu-20261114'
    assert control.DEFAULT_ROOT != old
    assert control.GLOBAL_SUSPENSION.parent != old
    control.Controller(tmp_path).assert_enabled()

@pytest.mark.parametrize('flag', ['--base-gpu-id','--gpu-id-step','--tp','--tp-size',
                                  '--pp-size','--pp','--dp-size','--nnodes','--node-rank'])
def test_server_extra_cannot_override_assignment_or_four_card_budget(flag):
    argv=['server_run.py','--model','8b','--family','plain','--tp','4','--pp','1',
          '--out','/never-created','--extra',flag,'7']
    with mock.patch.object(sys,'argv',argv), mock.patch.object(server_run,'require_active_controller',
          side_effect=AssertionError('argument validation must precede controller access')):
        with pytest.raises(ValueError,match='override'):
            server_run.main()

def test_holder_rejects_unassigned_mask_before_cuda_initialization(tmp_path):
    with mock.patch.dict(os.environ, {'CUDA_VISIBLE_DEVICES':'4,5,6,7'}), \
         mock.patch.object(control,'assert_clear',side_effect=AssertionError('must reject mask first')):
        with pytest.raises(RuntimeError,match='assigned'):
            control.holder(tmp_path,tmp_path/'ready.json')
