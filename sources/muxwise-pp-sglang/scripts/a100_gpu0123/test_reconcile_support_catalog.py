"""Saved support catalogs cannot qualify a failed or invented attempt."""
import importlib.util
from pathlib import Path

import pytest


@pytest.fixture
def module():
    path = Path(__file__).with_name('reconcile_support_catalog.py')
    assert path.exists(), 'Reconciliation implementation is missing'
    spec = importlib.util.spec_from_file_location('catalog_reconcile_test', path)
    result = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(result)
    return result


def attempt(label, ended, eligible=True, outcome='completed'):
    return dict(label=label, completed_unix=ended, eligible=eligible,
                controller_status=outcome, controller_returncode=0 if outcome == 'completed' else 1,
                runtime_status='execution_complete' if outcome == 'completed' else 'failed')


def test_selects_completion_time_without_lexical_version_bias(module):
    rows = [attempt('old-v99', 10), attempt('new-v2', 20), attempt('failed-v100', 30, False, 'failed')]
    assert module.select_latest(rows)['label'] == 'new-v2'
    assert len(rows) == 3


def test_finished_but_rejected_full_or_source_gate_cannot_qualify(module):
    rows = [attempt('accepted', 10), attempt('source-or-full-gate-rejected', 20, False)]
    assert module.select_latest(rows)['label'] == 'accepted'
    assert module.select_latest([rows[1]]) is None


def test_eligibility_boolean_cannot_override_incomplete_or_failed_execution(module):
    rows = [attempt('running', 20, True, 'running'), attempt('failed', 30, True, 'failed')]
    rows.append({**attempt('bad-runtime', 40), 'runtime_status': 'failed'})
    assert module.select_latest(rows) is None


def test_missing_and_nonfinite_completion_times_never_rank(module):
    assert module.select_latest([attempt('missing', None), attempt('nan', float('nan'))]) is None


def test_inventory_requires_explicit_existing_candidate_and_unique_label_binding(module):
    candidates = [{'id': 'a'}, {'id': 'b'}]
    valid = [{'candidate_id': 'a', 'label': 'one', 'measurement_directory': '/tmp/one'}]
    assert module.validate_inventory(valid, candidates) == valid
    for rows in [valid + [dict(valid[0], candidate_id='b')], [dict(valid[0], candidate_id='invented')],
                 [dict(valid[0], label='../escape')], valid + [dict(valid[0], label='two')]]:
        with pytest.raises(ValueError):
            module.validate_inventory(rows, candidates)
