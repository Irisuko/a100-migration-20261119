"""Read existing terminal summaries; preserve missing and failed planned work."""
import csv
import hashlib
import io
import json
import math
from pathlib import Path

from eventd import atomic_json, digest, fsync_dir, read_json

COUNTS = ['planned_requests', 'sent_requests', 'completed_requests', 'failed_requests',
          'unfinished_requests', 'unsent_requests']
SLO = 'ttft_le_5s_and_tpot_le_100ms'


def load_optional(path, issues):
    if not path:
        return {}, None
    try:
        raw = Path(path).read_bytes()
        data = json.loads(raw)
        if not isinstance(data, dict):
            raise ValueError('JSON object required')
        return data, hashlib.sha256(raw).hexdigest()
    except (OSError, ValueError) as exc:
        issues.append(f'{path}: {type(exc).__name__}')
        return {}, None


def clean(value):
    if isinstance(value, float) and not math.isfinite(value):
        return None
    if isinstance(value, dict):
        return {k: clean(v) for k, v in value.items()}
    if isinstance(value, list):
        return [clean(v) for v in value]
    return value


def queue_report(path, output):
    path, output = Path(path), Path(output)
    data = read_json(path)
    rows = []
    for label, task in data.get('tasks', {}).items():
        issues = []
        summary, summary_sha = load_optional(task.get('request_summary_path'), issues)
        outcome, outcome_sha = load_optional(task.get('outcome_path'), issues)
        row = {'label': label, 'status': task.get('status'), 'outcome_status': outcome.get('status'),
               'request_summary_path': task.get('request_summary_path'), 'summary_sha256': summary_sha,
               'outcome_path': task.get('outcome_path'), 'outcome_sha256': outcome_sha,
               'trace_sha256': task.get('trace_sha256'), 'source_fingerprint': task.get('source_fingerprint'),
               'measurement_eligible': summary.get('measurement_eligible'), 'issues': issues,
               'main_slo_rate': None, 'completion_rate': None}
        for key in COUNTS:
            value = summary.get(key)
            if value is None and key in {'completed_requests','failed_requests','unfinished_requests'}:
                value = summary.get(key.removesuffix('_requests'))
            row[key] = value
        planned = row['planned_requests']
        valid_planned = type(planned) is int and planned > 0
        components = [row[k] for k in ('completed_requests','failed_requests','unfinished_requests','unsent_requests')]
        sent = row['sent_requests']
        conserved = (all(type(n) is int and n >= 0 for n in components) and valid_planned
                     and sum(components) == planned and type(sent) is int and sent >= 0
                     and sent + row['unsent_requests'] == planned)
        if valid_planned and conserved:
            row['completion_rate'] = row['completed_requests'] / planned
        else:
            issues.append('request conservation unknown or inconsistent')
        arrival = summary.get('arrival_metrics', {})
        slos = arrival.get('slo', summary.get('slo', {}))
        slo = slos.get(SLO, {})
        if valid_planned and slo.get('total') == planned and type(slo.get('passed')) is int and 0 <= slo['passed'] <= planned:
            row['main_slo_rate'] = slo['passed'] / planned
        elif slo:
            issues.append('SLO denominator mismatch')
        row['metric_origin'] = 'scheduled_arrival' if arrival else 'summary_top_level'
        row['slo'] = slos
        row['latencies_sec'] = arrival.get('latencies_sec', summary.get('latencies_sec'))
        row['tpot_sec'] = summary.get('latencies_sec', {}).get('tpot')
        row['throughput'] = summary.get('throughput')
        row['goodput'] = arrival.get('goodput', summary.get('goodput'))
        row['duration_sec'] = summary.get('duration_sec')
        row['cached_tokens'] = summary.get('cached_tokens')
        row['metric_definitions'] = summary.get('metric_definitions')
        row['arrival_definitions'] = arrival.get('definitions')
        if not summary:
            issues.append('summary unavailable; request quantities unknown')
        rows.append(clean(row))
    output.mkdir(parents=True, exist_ok=True)
    stem = digest(str(path))[:16]
    json_path, csv_path = output/(stem+'.json'), output/(stem+'.csv')
    atomic_json(json_path, {'schema_version': 1, 'source': str(path),
        'source_sha256': hashlib.sha256(path.read_bytes()).hexdigest(), 'queue_status': data.get('status'),
        'note': 'Every queue task retained. Unknown values are null; failed/unfinished/unsent included in planned SLO denominator. No formal eligibility inferred.',
        'rows': rows})
    # Atomic publication of CSV through the same durable primitive as JSON.
    columns = ['label','status','outcome_status',*COUNTS,'completion_rate','main_slo_rate','measurement_eligible',
               'duration_sec','metric_origin','request_summary_path','summary_sha256','issues']
    stream = io.StringIO()
    writer = csv.DictWriter(stream, fieldnames=columns, extrasaction='ignore')
    writer.writeheader()
    for row in rows:
        writer.writerow({**row, 'issues': '; '.join(row['issues'])})
    import os, tempfile
    fd, temp = tempfile.mkstemp(dir=output, prefix='.'+stem)
    try:
        with os.fdopen(fd, 'w') as f:
            f.write(stream.getvalue()); f.flush(); os.fsync(f.fileno())
        os.replace(temp, csv_path)
        fsync_dir(output)
    finally:
        if os.path.exists(temp):
            os.unlink(temp)
    return {'source': str(path), 'json': str(json_path), 'csv': str(csv_path), 'tasks': len(rows)}


def batch_reports(batch, folder):
    reports = []
    seen = set()
    for event in batch['events']:
        if event['kind'] != 'batch_terminal' or event['path'] in seen:
            continue
        seen.add(event['path'])
        try:
            data = read_json(event['path'])
            if isinstance(data.get('tasks'), dict):
                reports.append(queue_report(event['path'], folder/'reports'))
        except (OSError, ValueError, TypeError, AttributeError, KeyError, OverflowError) as exc:
            reports.append({'source': event['path'], 'error': f'{type(exc).__name__}: {exc}'})
    return reports
