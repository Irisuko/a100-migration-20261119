"""Parser regression checks using the scheduler's actual log formats."""

import csv
import importlib.util
import json
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest


SCRIPT = Path(__file__).with_name("analyze_logs.py")
SPEC = importlib.util.spec_from_file_location("analyze_logs", SCRIPT)
parser = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(parser)


def line(stage, message):
    return f"[2026-11-14 01:02:03 PP{stage}] {message}\n"


def summary(migrations=0, **changes):
    stage = {
        "partition": 1, "samples": [32, 20], "mape": [0.1, 0.2],
        "slice_choices": [4, 1, 2],
        "timing": {"wall": {"prefill": {"upstream": 0.2}},
                   "cuda": {"prefill": 0.5}, "partitions": {0: 0.2, 1: 1.5}},
        "admission": {"blocked": {"inflight_tokens": 3}, "stops": {"budget": 2}},
        "cuda_memory": {"allocated": 1024},
    }
    stage.update(changes)
    other = dict(stage, samples=[31, 19], admission=None)
    return line(0, f"MuxWise-PP summary stages={ {0: stage, 1: other}!r} "
                f"mbs=[(2, 512), (1, 128)] waiting=3/4096 required=inf "
                f"budget=2048 inflight=2 pending=1 migrations={migrations}")


def idle(value):
    return line(0, "MuxWise-PP idle self-check PASS stage=0 free_kv=100 "
                f"evictable=20 free_reqs=32 prefill_slices={value} decode_steps={value * 2} "
                f"cached_tokens=0 overlap_launches={value} free_receive_slots=0")


class LogParserTest(unittest.TestCase):
    def test_actual_envelopes_preserve_both_stages_and_counter_deltas(self):
        text = line(1, "PD-Multiplexing enabled with 3 stream groups, "
                    "sm_counts (prefill_sm, decode_sm): [(108, 0), (80, 28), (0, 108)]")
        text += line(1, "MuxWise-PP decode calibration partition=1 SMs=(80, 28) "
                     "samples=[(1, 128, 0.0012), (4, 128, 0.002)]")
        text += line(1, "MuxWise-PP graph memory partition=1 graphs=5 "
                     "added_reserved_MiB=22.00 total_reserved_MiB=1262.00")
        text += line(1, "MuxWise-PP decode calibration elapsed=0.123s")
        text += idle(5) + summary(2) + idle(8) + summary(3, partition=2)
        result = parser.analyze_lines(text.splitlines(), source="fixture.log")
        self.assertEqual(result["calibration_rows"][0]["stage"], 1)
        self.assertEqual(result["calibration_rows"][0]["prefill_sms"], 80)
        self.assertEqual(result["graph_memory"][0]["added_reserved_MiB"], 22)
        self.assertEqual(result["calibration_elapsed"][0]["seconds"], 0.123)
        self.assertEqual(result["stages"]["1"]["latest_telemetry"]["samples"], [31, 19])
        self.assertEqual(result["stages"]["0"]["idle_counter_max"]["overlap_launches"], 8)
        self.assertEqual(result["idle_checks"][1]["increments"]["overlap_launches"], 3)
        self.assertEqual(result["migrations"]["max"], 3)
        self.assertEqual(result["migrations"]["observed_increments"], 1)
        self.assertEqual(result["stages"]["0"]["partition_transitions_observed"], 1)
        self.assertEqual(result["metadata"]["parse_errors"], [])

    def test_nonfinite_literals_are_null_and_code_is_never_executed(self):
        text = summary(mape=[float("nan"), -float("inf")], note="inf nan are words")
        text += line(0, "MuxWise-PP summary stages=__import__('os').getpid() mbs=[]")
        result = parser.analyze_lines(text.splitlines())
        latest = result["stages"]["0"]["latest_telemetry"]
        self.assertEqual(latest["mape"], [None, None])
        self.assertEqual(latest["note"], "inf nan are words")
        self.assertTrue(result["metadata"]["nonfinite_values"])
        self.assertEqual(len(result["metadata"]["parse_errors"]), 1)
        json.dumps(result, allow_nan=False)

    def test_coverage_reports_missing_partitions_and_cells(self):
        text = line(0, "PD-Multiplexing enabled with 3 stream groups, "
                    "sm_counts (prefill_sm, decode_sm): [(108, 0), (80, 28), (0, 108)]")
        text += line(0, "MuxWise-PP decode calibration partition=1 SMs=(80, 28) "
                     "samples=[(1, 128, 0.001), (4, 512, 0.002)]")
        result = parser.analyze_lines(text.splitlines())
        coverage = result["calibration_coverage"]["0"]
        self.assertFalse(coverage["complete"])
        self.assertEqual(coverage["missing_partitions"], [2])
        self.assertIn({"partition": 1, "batch_size": 1, "context_tokens": 512},
                      coverage["missing_cells"])

    def test_invalid_calibration_rows_are_preserved_and_flagged(self):
        text = line(0, "MuxWise-PP decode calibration partition=1 SMs=(100, 8) "
                    "samples=[(1, 128, nan), (4, 128, -0.1), (8, 128, 0.0)]")
        result = parser.analyze_lines(text.splitlines())
        self.assertEqual(len(result["calibration_rows"]), 3)
        self.assertTrue(all(not row["valid"] for row in result["calibration_rows"]))
        self.assertFalse(result["calibration_coverage"]["0"]["all_values_valid"])
        json.dumps(result, allow_nan=False)

    def test_counter_reset_is_not_silently_added_to_previous_counter(self):
        result = parser.analyze_lines((idle(9) + idle(2) + idle(4)).splitlines())
        stage = result["stages"]["0"]
        self.assertEqual(stage["idle_counter_max"]["overlap_launches"], 9)
        self.assertIsNone(result["idle_checks"][1]["increments"]["overlap_launches"])
        self.assertEqual(stage["idle_observed_increments"]["overlap_launches"], 2)
        self.assertTrue(stage["idle_counter_resets"])

    def test_complete_grid_retains_repeated_calibration_samples(self):
        text = line(0, "PD-Multiplexing enabled with 3 stream groups, "
                    "sm_counts (prefill_sm, decode_sm): [(108, 0), (80, 28), (0, 108)]")
        samples = [(1, 128, 0.001), (1, 512, 0.002),
                   (4, 128, 0.003), (4, 512, 0.004)]
        for partition, sms in [(1, (80, 28)), (2, (0, 108)), (1, (80, 28))]:
            text += line(0, f"MuxWise-PP decode calibration partition={partition} "
                         f"SMs={sms!r} samples={samples!r}")
        result = parser.analyze_lines(text.splitlines())
        coverage = result["calibration_coverage"]["0"]
        self.assertTrue(coverage["complete"])
        self.assertEqual(coverage["row_count"], 12)
        self.assertEqual(len(coverage["duplicate_cells"]), 4)

    def test_malformed_summary_is_recorded_and_later_records_survive(self):
        result = parser.analyze_lines((summary(timing=None) + summary(2)).splitlines())
        self.assertEqual(len(result["metadata"]["parse_errors"]), 1)
        self.assertEqual(len(result["summaries"]), 1)
        self.assertEqual(result["stages"]["0"]["summary_count"], 1)

    def test_cli_writes_strict_json_and_calibration_csv(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            log = root / "server.log"
            log.write_text(line(0, "MuxWise-PP decode calibration partition=1 SMs=(100, 8) "
                                "samples=[(1, 128, 0.001)]"))
            subprocess.run([sys.executable, str(SCRIPT), "--log", str(log),
                            "--output", str(root / "report.json"),
                            "--csv", str(root / "calibration.csv")], check=True,
                           capture_output=True, text=True)
            report = json.loads((root / "report.json").read_text())
            self.assertEqual(report["source_log"], str(log.resolve()))
            with (root / "calibration.csv").open() as stream:
                rows = list(csv.DictReader(stream))
            self.assertEqual(rows[0]["batch_size"], "1")
            self.assertEqual(rows[0]["seconds"], "0.001")

    def test_empty_log_does_not_invent_success_or_activity(self):
        result = parser.analyze_lines(["ordinary baseline server"])
        self.assertEqual(result["stages"], {})
        self.assertEqual(result["calibration_rows"], [])
        self.assertIsNone(result["migrations"]["max"])
        self.assertTrue(result["metadata"]["issues"])


if __name__ == "__main__":
    unittest.main()
