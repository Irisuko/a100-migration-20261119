"""CPU-only queue contracts. No test may launch a real GPU controller."""
import hashlib
import importlib.util
import json
from pathlib import Path
import tempfile
import unittest
from unittest import mock


SCRIPT = Path(__file__).with_name("queue.py")
queue = None
if SCRIPT.exists():
    spec = importlib.util.spec_from_file_location("a100_test_queue", SCRIPT)
    queue = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(queue)


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


class QueueTests(unittest.TestCase):
    def setUp(self):
        self.assertIsNotNone(queue, "Resumable queue implementation is missing")
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        self.tasks_path = self.root / "tasks.json"
        self.calls = []
        self.process_block = mock.patch.object(queue.subprocess, "Popen", side_effect=AssertionError("Real subprocess forbidden in CPU queue tests"))
        self.process_block.start()
        self.addCleanup(self.process_block.stop)

    def task(self, name, dependencies=None, phase="development", trace=None):
        argv = ["--model", "8b", "--family", "plain", "--suite", "smoke",
                "--out", str(self.root / "measurements" / name)]
        if trace:
            argv += ["--trace", str(trace), "--time-scale", "0.5"]
        return dict(id=name, phase=phase, argv=argv, timeout=60,
                    dependencies=dependencies or [])

    def write_plan(self, tasks, **extras):
        self.tasks_path.write_text(json.dumps(dict(tasks=tasks, **extras)))

    def launcher(self, fail=None, after=None):
        def launch(command, log_path):
            self.assertEqual(command[2:4], ["--root", str(self.root)])
            self.assertTrue(command[1].endswith("/control.py"))
            self.assertEqual(command[4:6], ["run", "--label"])
            name = command[6]
            self.calls.append(name)
            server = command[command.index("--") + 1:]
            self.assertEqual(server[:3], ["taskset", "-c", "0-31,64-95"])
            self.assertTrue(server[4].endswith("/server_run.py"))
            run_dir = self.root / "runs" / name
            run_dir.mkdir(parents=True)
            (run_dir / "command.json").write_text(json.dumps(dict(argv=server, timeout=60)))
            out = Path(server[server.index("--out") + 1])
            out.mkdir(parents=True)
            (out / "trace_summary.json").write_text('{"planned_requests": 3, "failed": 1}\n')
            failed = name == fail
            (run_dir / "outcome.json").write_text(json.dumps(dict(status="failed" if failed else "completed", returncode=1 if failed else 0)))
            if after:
                after(name)
            return 1 if failed else 0
        return launch

    def run_plan(self, **kwargs):
        return queue.run_queue(self.tasks_path, self.root, "test", **kwargs)

    def test_dependencies_run_serially_and_completed_labels_resume_without_replay(self):
        self.write_plan([self.task("b", ["a"]), self.task("a")])
        result = self.run_plan(launcher=self.launcher())
        self.assertEqual(self.calls, ["a", "b"])
        self.assertEqual(result["status"], "completed")
        result = self.run_plan(launcher=self.launcher())
        self.assertEqual(self.calls, ["a", "b"])
        self.assertEqual(result["tasks"]["a"]["status"], "completed")

    def test_failed_label_never_retries_and_partial_summary_bytes_are_preserved(self):
        self.write_plan([self.task("a"), self.task("b", ["a"])])
        first = self.run_plan(launcher=self.launcher(fail="a"))
        summary = self.root / "measurements/a/trace_summary.json"
        original = summary.read_bytes()
        second = self.run_plan(launcher=self.launcher())
        self.assertEqual(self.calls, ["a"])
        self.assertEqual(first["status"], "failed")
        self.assertEqual(second["tasks"]["a"]["status"], "failed")
        self.assertNotEqual(second["tasks"]["b"]["status"], "completed")
        self.assertEqual(summary.read_bytes(), original)
        self.assertEqual(second["tasks"]["a"]["request_summary_sha256"], digest(summary))

    def test_controller_exit_failure_cannot_be_upgraded_by_completed_outcome_on_resume(self):
        self.write_plan([self.task("a")])
        successful_artifacts = self.launcher()
        def contradictory_exit(command, log_path):
            successful_artifacts(command, log_path)
            return 1
        first = self.run_plan(launcher=contradictory_exit)
        second = self.run_plan(launcher=self.launcher())
        self.assertEqual(first["tasks"]["a"]["status"], "failed")
        self.assertEqual(second["tasks"]["a"]["status"], "failed")
        self.assertEqual(self.calls, ["a"])

    def test_feasibility_smoke_is_nonformal(self):
        self.write_plan([self.task("a", phase="feasibility")])
        result = self.run_plan(launcher=self.launcher())
        self.assertEqual(result["status"], "completed")

    def test_preexisting_incomplete_run_directory_is_not_reused(self):
        self.write_plan([self.task("a")])
        (self.root / "runs/a").mkdir(parents=True)
        result = self.run_plan(launcher=self.launcher())
        self.assertEqual(self.calls, [])
        self.assertEqual(result["tasks"]["a"]["status"], "interrupted")

    def test_changed_task_plan_is_rejected_on_resume(self):
        self.write_plan([self.task("a")])
        self.run_plan(launcher=self.launcher())
        changed = self.task("a")
        changed["argv"] += ["--mem", "0.9"]
        self.write_plan([changed])
        with self.assertRaisesRegex(ValueError, "plan changed"):
            self.run_plan(launcher=self.launcher())
        self.assertEqual(self.calls, ["a"])

    def test_failed_dependency_blocks_only_dependents_when_continuation_is_explicit(self):
        self.write_plan([self.task("a"), self.task("b", ["a"]), self.task("c")])
        result = self.run_plan(launcher=self.launcher(fail="a"), continue_independent=True)
        self.assertEqual(self.calls, ["a", "c"])
        self.assertEqual(result["tasks"]["b"]["status"], "blocked")
        self.assertEqual(result["tasks"]["c"]["status"], "completed")
        self.assertEqual(result["status"], "failed")

    def test_formal_without_freeze_cannot_launch(self):
        trace = self.root / "trace.jsonl"
        trace.write_text("trace")
        self.write_plan([self.task("formal1", phase="formal_main", trace=trace)])
        result = self.run_plan(launcher=self.launcher())
        self.assertEqual(self.calls, [])
        self.assertEqual(result["tasks"]["formal1"]["status"], "blocked")
        self.assertIn("freeze", result["tasks"]["formal1"]["error"].lower())

    def frozen_plan(self):
        source, trace = self.root / "source.py", self.root / "trace.jsonl"
        source.write_text("model source")
        trace.write_text("trace")
        task = self.task("formal1", phase="formal_main", trace=trace)
        self.write_plan([task])
        freeze = self.root / "freeze.json"
        freeze.write_text(json.dumps(dict(status="frozen", source_sha256={str(source): digest(source)},
                                         tasks={"formal1": dict(phase=task["phase"], argv=task["argv"], timeout=60, trace_sha256=digest(trace))})))
        return source, trace, task, freeze

    def test_exact_frozen_parameters_rates_source_and_trace_allow_formal(self):
        _, _, _, freeze = self.frozen_plan()
        result = self.run_plan(launcher=self.launcher(), freeze_path=freeze)
        self.assertEqual(self.calls, ["formal1"])
        self.assertEqual(result["status"], "completed")
        self.assertEqual(result["tasks"]["formal1"]["freeze_sha256"], digest(freeze))

    def test_changed_frozen_trace_prevents_launch(self):
        _, trace, _, freeze = self.frozen_plan()
        trace.write_text("changed trace")
        result = self.run_plan(launcher=self.launcher(), freeze_path=freeze)
        self.assertEqual(self.calls, [])
        self.assertEqual(result["tasks"]["formal1"]["status"], "blocked")
        self.assertIn("hash", result["tasks"]["formal1"]["error"].lower())

    def test_changed_frozen_parameters_prevent_launch(self):
        _, _, task, freeze = self.frozen_plan()
        task["argv"] += ["--running", "64"]
        self.write_plan([task])
        result = self.run_plan(launcher=self.launcher(), freeze_path=freeze)
        self.assertEqual(self.calls, [])
        self.assertIn("argv", result["tasks"]["formal1"]["error"])

    def test_source_is_rechecked_between_development_tasks(self):
        source = self.root / "source.py"
        source.write_text("before")
        self.write_plan([self.task("a"), self.task("b")], source_sha256={str(source): digest(source)})
        result = self.run_plan(launcher=self.launcher(after=lambda _: source.write_text("after")))
        self.assertEqual(self.calls, ["a"])
        self.assertEqual(result["tasks"]["b"]["status"], "blocked")

    def test_duplicate_ids_outputs_and_dependency_cycles_rejected_before_launch(self):
        a, b = self.task("a"), self.task("b")
        b["argv"] = a["argv"].copy()
        invalid = [[a, a], [a, b], [self.task("a", ["b"]), self.task("b", ["a"])],
                   [self.task("a", ["absent"])]]
        for tasks in invalid:
            with self.subTest(tasks=tasks):
                self.write_plan(tasks)
                with self.assertRaises(ValueError):
                    self.run_plan(launcher=self.launcher())
        self.assertEqual(self.calls, [])

    def test_stdlib_queue_interface_survives_script_filename_shadowing(self):
        fifo = queue.Queue()
        fifo.put("safe")
        self.assertEqual(fifo.get_nowait(), "safe")
        with self.assertRaises(queue.Empty):
            fifo.get_nowait()


if __name__ == "__main__":
    unittest.main()
