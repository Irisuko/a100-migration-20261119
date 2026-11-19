import csv
import json
from pathlib import Path
import sys
import tempfile
import unittest

sys.path.insert(0, str(Path(__file__).resolve().parent))
from reports import queue_report, batch_reports


class ReportTests(unittest.TestCase):
    def test_includes_failed_unsent_and_unknown_rows_without_success_subset_denominator(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            summary = root/'summary.json'
            summary.write_text(json.dumps({'planned_requests': 10, 'sent_requests': 6,
                'completed_requests': 4, 'failed_requests': 1, 'unfinished_requests': 1,
                'unsent_requests': 4, 'duration_sec': 12,
                'slo': {'ttft_le_5s_and_tpot_le_100ms': {'passed': 3, 'total': 10, 'rate': .75}},
                'metric_definitions': {'slo': 'all planned'}, 'measurement_eligible': False}))
            queue = root/'queue.json'
            queue.write_text(json.dumps({'status': 'failed', 'tasks': {
                'partial': {'status': 'failed', 'request_summary_path': str(summary)},
                'no-summary': {'status': 'failed'}, 'never-started': {'status': 'pending'}}}))
            report = queue_report(queue, root/'reports')
            data = json.loads(Path(report['json']).read_text())
            self.assertEqual(len(data['rows']), 3)
            partial = next(r for r in data['rows'] if r['label'] == 'partial')
            self.assertEqual(partial['main_slo_rate'], .3)
            self.assertEqual(partial['completion_rate'], .4)
            self.assertFalse(partial['measurement_eligible'])
            missing = next(r for r in data['rows'] if r['label'] == 'no-summary')
            self.assertIsNone(missing['planned_requests'])
            self.assertIsNone(missing['main_slo_rate'])
            with open(report['csv']) as f:
                self.assertEqual(len(list(csv.DictReader(f))), 3)

    def test_bad_slo_denominator_never_reused(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            summary = root/'summary.json'
            summary.write_text(json.dumps({'planned_requests': 10, 'completed_requests': 4,
                'slo': {'ttft_le_5s_and_tpot_le_100ms': {'passed': 4, 'total': 4, 'rate': 1}}}))
            q = root/'q.json'
            q.write_text(json.dumps({'tasks': {'a': {'request_summary_path': str(summary)}}}))
            r = queue_report(q, root/'out')
            row = json.loads(Path(r['json']).read_text())['rows'][0]
            self.assertIsNone(row['main_slo_rate'])
            self.assertIn('SLO denominator mismatch', row['issues'])

    def test_malformed_optional_metrics_cannot_block_terminal_notification(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            summary = root/'summary.json'
            summary.write_text(json.dumps({'planned_requests': 10, 'arrival_metrics': None}))
            q = root/'q.json'
            q.write_text(json.dumps({'tasks': {'a': {'request_summary_path': str(summary)}}}))
            batch = {'events': [{'kind': 'batch_terminal', 'path': str(q)}]}
            result = batch_reports(batch, root/'reports')
            self.assertEqual(result[0]['source'], str(q))
            self.assertIn('error', result[0])


if __name__ == '__main__':
    unittest.main()
