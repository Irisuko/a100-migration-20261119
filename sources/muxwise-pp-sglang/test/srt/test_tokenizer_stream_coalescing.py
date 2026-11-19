"""Exercise exact tokenizer methods without importing CUDA/runtime dependencies.

Two producer callbacks execute before the async response consumer wakes. This is
the scheduling interleaving that lost IDs when only the latest delta was kept.
AST extraction changes no method bodies; only unrelated class bases/imports are
omitted. Dataclasses also come from the actual runtime source.
"""

import ast
import asyncio
import dataclasses
from http import HTTPStatus
import logging
from pathlib import Path
import time
from types import SimpleNamespace
import unittest


class HttpError(Exception):
    def __init__(self,status_code,detail):
        super().__init__(detail)
        self.status_code=status_code


ROOT = Path(__file__).resolve().parents[2]
MANAGER_SOURCE = ROOT / "python/sglang/srt/managers/tokenizer_manager.py"
IO_SOURCE = ROOT / "python/sglang/srt/managers/io_struct.py"
METHODS = {
    "_handle_batch_output", "_wait_one_response", "_handle_abort_req",
    "convert_logprob_style", "detokenize_logprob_tokens", "detokenize_top_logprobs_tokens",
}


def load_cpu_methods():
    tree = ast.parse(MANAGER_SOURCE.read_text())
    state = next(n for n in tree.body if isinstance(n, ast.ClassDef) and n.name == "ReqState")
    manager = next(n for n in tree.body if isinstance(n, ast.ClassDef) and n.name == "TokenizerManager")
    manager.bases = []
    manager.body = [n for n in manager.body if isinstance(n, (ast.FunctionDef, ast.AsyncFunctionDef)) and n.name in METHODS]
    io = ast.parse(IO_SOURCE.read_text())
    batches = [n for n in io.body if isinstance(n, ast.ClassDef) and n.name in (
        "BatchStrOut", "BatchTokenIDOut", "BatchEmbeddingOut", "BatchMultimodalOut")]
    module = ast.Module(body=[ast.ImportFrom(module="__future__", names=[ast.alias(name="annotations")], level=0),
                              *batches, state, manager], type_ignores=[])
    scope = dict(globals(), dataclass=dataclasses.dataclass,
                 trace_req_finish=lambda *args, **kwargs: None,
                 is_health_check_generate_req=lambda obj: False,
                 logger=logging.getLogger(__name__),
                 fastapi=SimpleNamespace(HTTPException=HttpError))
    exec(compile(ast.fix_missing_locations(module), str(MANAGER_SOURCE), "exec"), scope)
    return SimpleNamespace(**{name: scope[name] for name in (
        "TokenizerManager", "ReqState", "BatchStrOut", "BatchTokenIDOut")})


class StreamCoalescingTest(unittest.IsolatedAsyncioTestCase):
    @classmethod
    def setUpClass(cls):
        cls.runtime = load_cpu_methods()

    def setup_request(self, *, stream=True, stream_output=True, logprobs=False):
        manager = self.runtime.TokenizerManager()
        manager.server_args = SimpleNamespace(
            tokenizer_worker_num=1, weight_version="test", skip_tokenizer_init=False,
            stream_output=stream_output, speculative_algorithm=None, enable_lora=False,
        )
        manager.log_requests = manager.enable_metrics = False
        manager.dump_requests_folder = manager.crash_dump_folder = None
        obj = SimpleNamespace(rid="r", stream=stream, background=False,
                              return_logprob=logprobs, top_logprobs_num=0,
                              token_ids_logprob=None, return_text_in_logprobs=False)
        state = self.runtime.ReqState([], False, asyncio.Event(), obj, created_time=time.time())
        manager.rid_to_state = {"r": state}
        self.manager, self.state, self.obj = manager, state, obj
        self.consumer = manager._wait_one_response(obj, state)

    def produce(self, ids, completion, *, text="x", final=False, raw=False):
        kind = self.runtime.BatchTokenIDOut if raw else self.runtime.BatchStrOut
        values = {f.name: None for f in dataclasses.fields(kind)}
        values.update(rids=["r"], output_ids=[ids], prompt_tokens=[2048],
                      completion_tokens=[completion], cached_tokens=[0],
                      finished_reasons=[{"type": "length", "length": completion} if final else None],
                      input_token_logprobs_val=[], input_token_logprobs_idx=[],
                      output_token_logprobs_val=[[-.1]], output_token_logprobs_idx=[[ids[-1]]])
        if not raw:
            values["output_strs"] = [text]
        self.manager._handle_batch_output(kind(**values))

    async def take(self):
        return await asyncio.wait_for(self.consumer.__anext__(), timeout=1)

    async def asyncTearDown(self):
        if hasattr(self, "consumer"):
            await self.consumer.aclose()

    async def test_initial_prompt_lookback_and_first_token_survive_two_pending_frames(self):
        self.setup_request(logprobs=True)
        first = [11, 12, 13, 14, 15, 101]
        self.produce(first, 1, text="A")
        self.produce([19791], 2, text="B")
        first.append(999)  # Producer inputs must not alias queued response IDs.
        result = await self.take()
        self.assertEqual(result["output_ids"], [11, 12, 13, 14, 15, 101, 19791])
        self.assertEqual(result["meta_info"]["completion_tokens"], 2)
        self.assertEqual(result["text"], "AB")
        self.assertEqual([x[1] for x in result["meta_info"]["output_token_logprobs"]], [101, 19791])

    async def test_midstream_pending_deltas_are_delivered_once(self):
        self.setup_request()
        self.produce([11, 12, 13, 14, 15, 101], 1, text="A")
        first = await self.take()
        self.produce([102], 2, text="B")
        self.produce([103], 3, text="C")
        second = await self.take()
        self.assertEqual(first["output_ids"], [11, 12, 13, 14, 15, 101])
        self.assertEqual(second["output_ids"], [102, 103])
        self.assertEqual(second["text"], "ABC")

    async def test_final_pending_frame_keeps_all_ids_and_finish_reason(self):
        self.setup_request()
        self.produce([101], 1)
        self.produce([102], 2, final=True)
        result = await self.take()
        self.assertEqual(result["output_ids"], [101, 102])
        self.assertEqual(result["meta_info"]["finish_reason"], {"type": "length", "length": 2})
        with self.assertRaises(StopAsyncIteration):
            await self.consumer.__anext__()

    async def test_one_pending_frame_retains_original_object(self):
        self.setup_request()
        self.produce([101], 1)
        queued = self.state.out_list[0]
        self.assertIs(await self.take(), queued)

    async def test_raw_streaming_mode_merges_deltas(self):
        self.setup_request(stream_output=True)
        self.produce([101], 1, raw=True)
        self.produce([102], 2, raw=True)
        self.assertEqual((await self.take())["output_ids"], [101, 102])

    async def test_raw_cumulative_mode_does_not_duplicate_prior_ids(self):
        self.setup_request(stream_output=False)
        self.produce([101], 1, raw=True)
        self.produce([102], 2, raw=True)
        self.assertEqual((await self.take())["output_ids"], [101, 102])

    async def test_nonstream_final_output_remains_cumulative(self):
        self.setup_request(stream=False)
        self.produce([101], 1)
        self.produce([102], 2, final=True)
        self.assertEqual((await self.take())["output_ids"], [101, 102])

    async def test_abort_without_ids_keeps_existing_error_handling(self):
        self.setup_request()
        self.produce([101], 1)
        reason = {"type": "abort", "status_code": 400, "message": "rejected"}
        self.manager._handle_abort_req(SimpleNamespace(rid="r", finished_reason=reason))
        with self.assertRaisesRegex(ValueError, "rejected"):
            await self.take()

    async def test_bare_abort_removes_terminal_state_before_waking_retained_waiter(self):
        self.setup_request(stream=False)
        self.manager._handle_abort_req(SimpleNamespace(rid='r',finished_reason=None))
        self.assertNotIn('r',self.manager.rid_to_state)
        self.assertTrue(self.state.finished)
        self.assertGreaterEqual(self.state.finished_time,self.state.created_time)
        self.assertTrue(self.state.event.is_set())
        result=await self.take()
        self.assertEqual(result['meta_info']['finish_reason']['type'],'abort')
        with self.assertRaises(StopAsyncIteration):await self.consumer.__anext__()

    async def test_duplicate_or_late_abort_is_idempotent(self):
        self.setup_request()
        abort=SimpleNamespace(rid='r',finished_reason=None)
        self.manager._handle_abort_req(abort)
        pending=len(self.state.out_list)
        self.manager._handle_abort_req(abort)
        self.assertEqual(len(self.state.out_list),pending)
        self.manager._handle_abort_req(SimpleNamespace(rid='unknown',finished_reason=None))
        self.assertEqual(self.manager.rid_to_state,{})

    async def test_late_abort_after_normal_completion_does_not_replace_final_output(self):
        self.setup_request()
        self.produce([101],1,final=True)
        self.manager._handle_abort_req(SimpleNamespace(rid='r',finished_reason=None))
        result=await self.take()
        self.assertEqual(result['meta_info']['finish_reason']['type'],'length')

    async def test_terminal_abort_releases_lora_once_even_for_http_error(self):
        self.setup_request()
        self.manager.server_args.enable_lora=True
        self.obj.lora_path='adapter';self.obj.lora_id='adapter-id'
        released=[]
        async def release(adapter_id):released.append(adapter_id)
        self.manager.lora_registry=SimpleNamespace(release=release)
        abort=SimpleNamespace(rid='r',finished_reason={'type':'abort','status_code':503,'message':'unavailable'})
        self.manager._handle_abort_req(abort)
        self.manager._handle_abort_req(abort)
        with self.assertRaises(HttpError):await self.take()
        await asyncio.sleep(0)
        self.assertEqual(released,['adapter-id'])
        self.assertEqual(self.manager.rid_to_state,{})


if __name__ == "__main__":
    unittest.main()
