"""Catch optimistic SLOs from late sends, success-only denominators and token inflation."""
import asyncio
import argparse
import json
import os
from pathlib import Path
import signal
import sys
import tempfile
import time
import unittest

import aiohttp
from aiohttp import web
import metrics
from bench_client import encode_payload, issue_request, run


class ArrivalMetricsTests(unittest.TestCase):
    def test_late_send_changes_arrival_slo_and_keeps_unsent_in_denominator(self):
        self.assertTrue(hasattr(metrics, "summarize_arrivals"), "arrival-aware summary missing")
        planned = [dict(idx=i, arrival=0, input_len=10, output_len=3) for i in range(4)]
        records = [dict(idx=0, arrival=0, sent_at=6, first_token_at=7, finished_at=7.1,
                        success=True, status="completed", output_tokens=3, tpot=.05,
                        chunks=[dict(at=7, completion_tokens=1), dict(at=7.1, completion_tokens=3)]),
                   dict(idx=1, arrival=0, sent_at=0, first_token_at=1, finished_at=1.1,
                        success=True, status="completed", output_tokens=3, tpot=.05,
                        chunks=[dict(at=1, completion_tokens=1), dict(at=1.1, completion_tokens=3)]),
                   dict(idx=2, arrival=0, sent_at=0, first_token_at=None, finished_at=8,
                        success=False, status="unfinished", output_tokens=0, tpot=None, chunks=[])]
        result = metrics.summarize_arrivals(records, planned, cutoff_time=8, arrival_window_sec=2)
        self.assertEqual(result["latencies_sec"]["ttft"]["p50"], ">cutoff")
        self.assertEqual(result["slo"]["ttft_le_5s_and_tpot_le_100ms"]["rate"], .25)
        self.assertEqual(result["slo"]["ttft_le_5s_and_tpot_le_100ms"]["total"], 4)
        self.assertEqual(result["unsent_requests"], 1)
        self.assertEqual(result["goodput"]["arrival_window"]["main_slo_requests_per_sec"], .5)
        self.assertEqual(result["goodput"]["full_drain"]["main_slo_requests_per_sec"], .125)
        self.assertEqual(result["goodput"]["arrival_window"]["main_slo_output_tokens_per_sec"], 1.5)

    def test_drain_tokens_do_not_inflate_arrival_window_throughput(self):
        self.assertTrue(hasattr(metrics, "summarize_arrivals"), "arrival-aware summary missing")
        planned = [dict(idx=0, arrival=0, input_len=10, output_len=4)]
        records = [dict(idx=0, arrival=0, sent_at=0, first_token_at=1, finished_at=3,
                        success=True, output_tokens=4, tpot=2/3,
                        chunks=[dict(at=1, completion_tokens=1), dict(at=2, completion_tokens=2),
                                dict(at=3, completion_tokens=4)])]
        result = metrics.summarize_arrivals(records, planned, cutoff_time=4, arrival_window_sec=2)
        self.assertEqual(result["goodput"]["arrival_window"]["observed_output_tokens"], 2)
        self.assertEqual(result["goodput"]["full_drain"]["observed_output_tokens"], 4)
        self.assertEqual(result["goodput"]["full_drain"]["duration_sec"], 3)


class StreamIdentityTests(unittest.IsolatedAsyncioTestCase):
    async def test_warmup_failure_persists_immutable_plan_and_unsent_summary(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp); observed = []
            trace = root / "trace.jsonl"
            trace.write_text("".join(json.dumps(dict(idx=i, arrival=i, input_ids=[128000,42],
                                                     input_len=2, output_len=3)) + "\n" for i in range(2)))
            async def handler(request):
                plan = root / "startup_planned.jsonl"
                observed.append(plan.read_bytes() if plan.exists() else None)
                return web.Response(status=503, text="startup unavailable")
            app = web.Application(); app.router.add_post("/generate", handler)
            runner = web.AppRunner(app); await runner.setup()
            site = web.TCPSite(runner, "127.0.0.1", 0); await site.start()
            try:
                args = argparse.Namespace(trace=str(trace), limit=None, time_scale=1, output_id_mode="incremental",
                    vocab_size=128256, arrival_window_sec=2, base_url=f"http://127.0.0.1:{runner.addresses[0][1]}",
                    out_dir=tmp, label="startup", overwrite=False, timeout=10, force_close=True,
                    drain_timeout=1, warmup=1, progress_interval=30)
                try:
                    result = await run(args)
                except Exception as exc:
                    result = exc
                self.assertIsNotNone(observed[0], "plan must exist before first warmup HTTP request")
                self.assertEqual(result,1)
                self.assertEqual((root/"startup_planned.jsonl").read_bytes(),observed[0])
                summary = json.loads((root/"startup_summary.json").read_text())
                self.assertEqual(summary["run_status"],"startup_failed")
                self.assertFalse(summary["measurement_started"])
                self.assertFalse(summary["measurement_eligible"])
                self.assertIsNone(summary["run_started_unix"])
                self.assertEqual((summary["sent_requests"],summary["unsent_requests"],summary["output_tokens"]),(0,2,0))
                self.assertEqual(summary["throughput"]["output_tokens_per_sec"],0)
                self.assertTrue(any("warmup" in error for error in summary["errors"]))
                outcomes = [json.loads(line) for line in (root/"startup_outcomes.jsonl").read_text().splitlines()]
                self.assertEqual([r["status"] for r in outcomes],["unsent","unsent"])
            finally:
                await runner.cleanup()

    async def test_cli_sigterm_preserves_partial_stream_and_unsent_requests(self):
        await self.check_cli_sigterm()

    async def test_cli_sigterm_during_warmup_is_not_a_measurement(self):
        await self.check_cli_sigterm(warmup=1)

    async def test_cli_sigterm_during_drain_preserves_cutoff(self):
        await self.check_cli_sigterm(arrivals=[0])

    async def check_cli_sigterm(self, warmup=0, arrivals=None):
        arrivals = [0,20,40] if arrivals is None else arrivals
        first_token_sent = asyncio.Event()
        async def handler(request):
            response = web.StreamResponse(headers={"Content-Type":"text/event-stream"})
            await response.prepare(request)
            await response.write(b'data: {"output_ids":[10],"meta_info":{"completion_tokens":1}}\n\n')
            first_token_sent.set()
            for _ in range(500):
                if request.transport is None or request.transport.is_closing():
                    break
                await asyncio.sleep(.01)
            return response
        app = web.Application(); app.router.add_post("/generate", handler)
        runner = web.AppRunner(app); await runner.setup()
        site = web.TCPSite(runner, "127.0.0.1", 0); await site.start()
        process = None
        try:
            with tempfile.TemporaryDirectory() as tmp:
                root=Path(tmp);trace=root/"trace.jsonl"
                trace.write_text("".join(json.dumps(dict(idx=i, arrival=t, input_ids=[128000,42],
                                                         input_len=2, output_len=3))+"\n"
                                         for i,t in enumerate(arrivals)))
                process = await asyncio.create_subprocess_exec(sys.executable,str(Path(__file__).with_name("bench_client.py")),
                    "--trace",str(trace),"--base-url",f"http://127.0.0.1:{runner.addresses[0][1]}",
                    "--label","term","--out-dir",tmp,"--drain-timeout","1","--warmup",str(warmup),
                    stdout=asyncio.subprocess.PIPE,stderr=asyncio.subprocess.PIPE,
                    env=dict(os.environ,CUDA_VISIBLE_DEVICES="",PYTHONDONTWRITEBYTECODE="1"))
                await asyncio.wait_for(first_token_sent.wait(),5)
                plan_bytes=(root/"term_planned.jsonl").read_bytes()
                await asyncio.sleep(.05)
                process.send_signal(signal.SIGTERM)
                stdout,stderr=await asyncio.wait_for(process.communicate(),5)
                self.assertEqual(process.returncode,143,(stdout.decode(),stderr.decode()))
                summary=json.loads((root/"term_summary.json").read_text())
                self.assertEqual(summary["run_status"],"interrupted")
                self.assertEqual(summary["termination_signal"],"SIGTERM")
                self.assertFalse(summary["measurement_eligible"])
                self.assertEqual((root/"term_planned.jsonl").read_bytes(),plan_bytes)
                if warmup:
                    self.assertFalse(summary["measurement_started"])
                    self.assertEqual((summary["sent_requests"],summary["output_tokens"],summary["unsent_requests"]),(0,0,len(arrivals)))
                    self.assertEqual((root/"term_requests.jsonl").read_text(),"")
                    self.assertEqual(summary["throughput"]["output_tokens_per_sec"],0)
                else:
                    self.assertTrue(summary["measurement_started"])
                    self.assertEqual((summary["sent_requests"],summary["unfinished_requests"],summary["unsent_requests"]),(1,1,len(arrivals)-1))
                    self.assertEqual(summary["output_tokens"],1)
                    details=json.loads((root/"term_requests.jsonl").read_text().strip())
                    self.assertEqual(details["output_ids"],[10])
                    if len(arrivals)==1:
                        self.assertAlmostEqual(summary["cutoff_time"],details["sent_at"]+1)
                self.assertEqual(summary["slo"]["ttft_le_5s"]["total"],len(arrivals))
        finally:
            if process is not None and process.returncode is None:
                process.kill();await process.wait()
            await runner.cleanup()

    async def test_interruption_persists_all_planned_and_unsent_requests(self):
        first_seen = asyncio.Event()
        async def handler(request):
            first_seen.set()
            return web.Response(text='data: {"output_ids":[10,11,12],"meta_info":{"completion_tokens":3}}\n\ndata: [DONE]\n\n', content_type="text/event-stream")
        app = web.Application(); app.router.add_post("/generate", handler)
        runner = web.AppRunner(app); await runner.setup()
        site = web.TCPSite(runner, "127.0.0.1", 0); await site.start()
        try:
            with tempfile.TemporaryDirectory() as tmp:
                trace = Path(tmp)/"trace.jsonl"
                trace.write_text("".join(json.dumps(dict(idx=i, arrival=t, input_ids=[128000, 42],
                                                         input_len=2, output_len=3)) + "\n"
                                         for i,t in enumerate([0,10,20])))
                args = argparse.Namespace(trace=str(trace), limit=None, time_scale=1, output_id_mode="incremental",
                    vocab_size=128256, arrival_window_sec=20, base_url=f"http://127.0.0.1:{runner.addresses[0][1]}",
                    out_dir=tmp, label="interrupt", overwrite=False, timeout=10, force_close=True,
                    drain_timeout=1, warmup=0, progress_interval=30)
                task = asyncio.create_task(run(args)); await asyncio.wait_for(first_seen.wait(),2)
                await asyncio.sleep(.02); task.cancel(); self.assertEqual(await task,130)
                planned = [json.loads(s) for s in (Path(tmp)/"interrupt_outcomes.jsonl").read_text().splitlines()]
                self.assertEqual([r["status"] for r in planned], ["completed","unsent","unsent"])
                summary = json.loads((Path(tmp)/"interrupt_summary.json").read_text())
                self.assertEqual(summary["planned_requests"],3)
                self.assertEqual(summary["arrival_metrics"]["unsent_requests"],2)
                self.assertEqual(summary["arrival_metrics"]["slo"]["ttft_le_5s_and_tpot_le_100ms"]["rate"],1/3)
                details = json.loads((Path(tmp)/"interrupt_requests.jsonl").read_text().strip())
                self.assertEqual(details["validated_output_id_count"],3)
        finally:
            await runner.cleanup()

    async def fetch(self, events, mode="incremental", requested=3):
        async def handler(request):
            body = "".join("data: " + json.dumps(e) + "\n\n" for e in events) + "data: [DONE]\n\n"
            return web.Response(text=body, content_type="text/event-stream")
        app = web.Application(); app.router.add_post("/generate", handler)
        runner = web.AppRunner(app); await runner.setup()
        site = web.TCPSite(runner, "127.0.0.1", 0); await site.start()
        try:
            async with aiohttp.ClientSession() as session:
                request = dict(idx=0, arrival=0, input_len=2, output_len=requested,
                               output_id_mode=mode, vocab_size=128256,
                               payload=encode_payload([128000, 42], requested))
                return await issue_request(session, f"http://127.0.0.1:{runner.addresses[0][1]}/generate",
                                           request, asyncio.get_running_loop().time(), time.time())
        finally:
            await runner.cleanup()

    async def test_incremental_ids_preserved_with_coalesced_events(self):
        result = await self.fetch([dict(output_ids=[10, 11], meta_info=dict(completion_tokens=2)),
                                   dict(output_ids=[12], meta_info=dict(completion_tokens=3))])
        self.assertTrue(result["success"], result["error"])
        self.assertEqual(result.get("output_ids"), [10, 11, 12])
        self.assertEqual(result.get("validated_output_id_count"), 3)

    async def test_missing_or_mismatched_ids_cannot_pass_on_metadata(self):
        for event in [dict(meta_info=dict(completion_tokens=3)),
                      dict(output_ids=[10], meta_info=dict(completion_tokens=3)),
                      dict(output_ids=[10, -1, 12], meta_info=dict(completion_tokens=3)),
                      dict(output_ids=[10, 128256, 12], meta_info=dict(completion_tokens=3))]:
            with self.subTest(event=event):
                result = await self.fetch([event]); self.assertFalse(result["success"])
                self.assertEqual(result["output_tokens"], 0)

    async def test_count_above_request_is_rejected_before_accounting(self):
        result = await self.fetch([dict(output_ids=[10,11,12,13], meta_info=dict(completion_tokens=4))])
        self.assertFalse(result["success"])
        self.assertEqual(result["output_tokens"], 0)

    async def test_cumulative_mode_checks_prior_prefix(self):
        result = await self.fetch([dict(output_ids=[10], meta_info=dict(completion_tokens=1)),
                                   dict(output_ids=[10,11,12], meta_info=dict(completion_tokens=3))], "cumulative")
        self.assertTrue(result["success"], result["error"])
        self.assertEqual(result.get("output_ids"), [10,11,12])
        invalid = await self.fetch([dict(output_ids=[10], meta_info=dict(completion_tokens=1)),
                                    dict(output_ids=[99,11,12], meta_info=dict(completion_tokens=3))], "cumulative")
        self.assertFalse(invalid["success"])


if __name__ == "__main__":
    unittest.main()
