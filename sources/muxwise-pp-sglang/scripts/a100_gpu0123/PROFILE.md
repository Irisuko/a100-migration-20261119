`profile_run.py` creates a separate diagnostic run. It must run under the existing
GPU0–3 controller; it never starts a controller or changes GPU ownership itself.
It imports the unchanged server runner in a child and replaces only that child's
Python launcher with a generated shim. Worker arguments, semantic environment,
shim hash, wrapper hash, and serving-source hash are preserved in the diagnostic
directory.

Example for root to review and run, using a development trace and a fresh output:

```bash
.venv/bin/python scripts/a100_gpu0123/control.py \
  --root run/a100-gpu0123-20261114 run \
  --label profile-8b-muxpp-pp4-v1 --timeout 4200 -- \
  .venv/bin/python scripts/a100_gpu0123/profile_run.py \
  --out run/a100-gpu0123-20261114/diagnostics/profile-8b-muxpp-pp4-v1 \
  --seconds 30 --profile-delay-sec 60 --startup-timeout 2400 --export-timeout 180 \
  --runner-timeout 1200 -- \
  --model 8b --family muxpp --tp 1 --pp 4 --replicas 1 \
  --trace /ABS/DEVELOPMENT_TRACE.jsonl --mem 0.85 --graph 16
```

Replace serving parameters with the reviewed configuration. Do not pass the
server runner's `--out`, `--suite`, or `--pdmux-pp-request-trace`; the wrapper owns
these. Other runner options must precede its final `--extra`. The wrapper adds
`--suite trace`, retains the normal preflight, cache reset, and two client warmups,
and lets the trace finish/drain after the bounded profile window. It does not
change trace arrivals or use the profile duration as the arrival window.

With the default delay of zero, the client shim waits until all worker `/start_profile` calls acknowledge and all
PP/TP rank start entries appear in logs. Those entries precede initialization,
so they are entry evidence rather than a distributed initialization barrier.
The wrapper then releases the client, including its warmups, and stops profiling
after the requested time or early client completion. It holds client process
completion until all four rank exports are readable. Failed endpoints, missing
rank kernels, absent MuxWise-PP lane ranges, or serving failure fail the diagnostic.
Partial files and client error/unsent accounting remain available.

For a middle window, use `--profile-delay-sec 60` with a development trace whose
arrivals continue for at least 180 seconds. Positive delay releases the client
first, waits for its post-warmup `Loaded` log entry, and starts profiling after
the requested delay. Early client completion before that window fails the
diagnostic. Log polling adds uncertainty; the report records observation time
and later reconciles each profile request/acknowledgement offset against the
client's exact `run_started_unix`. Delayed profiles exclude the two warmups.

The primary artifact is `profile.json`. It records endpoint acknowledgements,
client boundaries, exported-rank hashes and kernel/range counts, and exact CPU
analysis commands. Worker files reside in `workerN/`; semantic filenames cannot
collide between replicas or PD roles. Every worker gets ordinary CPU/GPU profiling.
MuxWise-PP additionally gets request JSONL, CUDA-event timing, round traces and
launch ranges. Other systems do not acquire these MuxWise-PP semantic labels.
Semantic JSONL spans the full server lifecycle, including preflight; filter its
timestamps to the recorded client/profile boundaries before computing diagnostics.

Summary counters are sampled at the profile-start boundary (before client release
in zero-delay mode) and profile stop.
Only differences between distinct samples are reported, with each stage's sample
interval and the age of each boundary sample. Startup calibration totals are not
used as window totals. Samples are periodic, so these differences do not exactly
cover the Torch profile window. Prefill/decode host lanes overlap and must not be
summed into elapsed time. Host control/upstream/send waits are not CUDA
communication durations. GPU bubbles require interval union over actual scoped
GPU timelines, including unmapped work; `gpu_bubble_seconds` stays null here.

`scripts/a100/analyze_gpu_trace.py` can analyze each exported rank using explicit
CPU-to-GPU correlations. Its mapped lane overlap is a lower bound when kernels are
unmapped or ambiguous. Selected SM partition is present in launch labels and
debug logs; there is no measured SM occupancy counter or exact switch-event
timeline. The host switch region excludes the following TP barrier/backend update.
The older plotting script expects a different artifact layout.

All output is diagnostic-only. Client output first goes to `client-raw/`, which
has no serving metadata and is not an ordinary measurement directory. Publication
into `serving/` sets `measurement_eligible=false` and `diagnostic_only=true` before
the summary becomes visible there. Raw summary bytes are retained; final serving
metadata is also preserved and marked. Do not use the raw copies
for tuning, rate calibration, or formal performance aggregation. Fresh server
processes without tracing are required for those measurements. Profiling overhead
is not estimated or subtracted. `profile_by_stage` is always false because split
prefill is unsupported by that profiler mode.

CPU verification:

```bash
CUDA_VISIBLE_DEVICES='' PYTHONDONTWRITEBYTECODE=1 .venv/bin/python -m pytest \
  -q -p no:cacheprovider scripts/a100_gpu0123/test_profile_run.py
```
