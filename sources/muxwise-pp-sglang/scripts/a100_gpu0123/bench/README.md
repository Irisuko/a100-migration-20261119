# Four-GPU trace replay client

Adapted from `Irisuko/muxwise-pp-experiments` commit
`d578ee53a9243b799348952303a1185d255d9cb7`, retaining its open-loop
`/generate` replay, `--time-scale`, request detail schema, historical send-relative
metrics, all-planned SLO denominator, and drain cutoff. Original file hashes are
in `source-provenance.json`. The historical experiment repository is unchanged.

Run from the runtime repository with the existing `.venv`:

```bash
.venv/bin/python scripts/a100_gpu0123/bench/bench_client.py \
  --trace run/a100-gpu0123-20261114/traces/generated/llama31_8b_dev_toolagent_1200_240_x2.5.jsonl \
  --base-url http://127.0.0.1:30000 --label unique-development-label \
  --out-dir run/a100-gpu0123-20261114/client-results \
  --time-scale 1 --warmup 2 --force-close --drain-timeout 300 \
  --output-id-mode incremental
```

`time-scale` multiplies already stretched arrivals. Smaller values send faster;
lengths and hash-block ordering remain unchanged. Full-trace arrival-window
duration comes from the adjacent `.stats.json`, multiplied by `time-scale`.
With `--limit`, it defaults to the last selected arrival. An explicit
`--arrival-window-sec` is in final replay seconds and cannot precede the last
planned request. A zero-duration window produces null rates.

The CLI validates incremental `output_ids` by default. When the server uses
`--skip-tokenizer-init`, also pass **server `--stream-output`** for incremental
IDs. Otherwise select client `--output-id-mode cumulative`. Every event's IDs
must match the cumulative completion count and the model vocabulary (default
128256), and counts cannot exceed the requested output length. Cumulative mode
also validates the previously seen prefix. Full output IDs and independently
counted IDs are retained. `--output-id-mode none` explicitly restores historical
metadata-only behavior. Programmatic callers omitting this new argument also
retain historical behavior for compatibility.

New fields supplement the original metrics:

- Request `scheduled_ttft` and `scheduled_e2e` start at scheduled arrival,
  including `send_lag`.
- `arrival_metrics.latencies_sec` and `.slo` use those origins and every planned
  request. Failed, unfinished, unsent and undefined observations miss SLOs.
  TPOT is unchanged because subtracting TTFT from E2E cancels the origin.
- `arrival_metrics.goodput.arrival_window` counts completions and observed
  tokens only through the planned arrival-window end. `full_drain` counts
  through actual final completion or the drain cutoff. Main-SLO goodput counts
  only completed requests satisfying scheduled TTFT <= 5 seconds and TPOT <=
  100 ms, reporting both requests/s and their output tokens/s.
- `<label>_planned.jsonl` records immutable request intent. It is flushed,
  fsynced and atomically installed before the first warmup HTTP request.
  `<label>_outcomes.jsonl` adds the final status and send time for every planned
  request, including unsent requests. Historical `<label>_requests.jsonl`
  retains only sent measurement requests. Labels refuse overwrite by default.

Warmup failure creates an empty request log, all-unsent outcomes and a summary
with `run_status=startup_failed`, errors, zero observed throughput,
`measurement_started=false`, and `measurement_eligible=false`; it exits 1.
Warmup tokens never enter measurement totals. **Tuning must require
`measurement_eligible=true`.** This flag identifies completed client execution,
not SLO success: request failures and requests unfinished at the normal drain
cutoff remain measured outcomes.

The CLI handles SIGTERM and SIGINT by cancelling once, closing connections and
saving partial streams plus all unsent requests. These runs have
`run_status=interrupted`, `measurement_eligible=false`, the recorded signal, and
exit status 143 or 130 respectively. An existing drain cutoff is retained.
SIGKILL cannot execute finalization; the prewritten immutable plan remains.
Programmatic `run()` does not install process signal handlers.

This verifies streamed delivery integrity, not agreement with a reference
model. The original SSE ITL limitation remains: coalesced events do not reveal
the arrival time of each individual token within the event.

CPU tests:

```bash
PYTHONDONTWRITEBYTECODE=1 .venv/bin/python -m unittest discover \
  -s scripts/a100_gpu0123/bench -p 'test_*.py' -v
```
