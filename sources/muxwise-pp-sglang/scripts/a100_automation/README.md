# Campaign event observer

`eventd.py` is a stdlib CPU service that reads already running queues, controllers,
outer drivers, outcomes, and dock files. It cannot submit or restart experiments.
It only invokes the explicit notification adapter configured by the campaign
owner. The adapter's independent Codex decision turn can use the established
queue/controller interfaces under the authorization recorded in HANDOFF.

The daemon performs no model call on ordinary polling, individual successful
tasks, unchanged failures, or progress counters. Events are batch terminal,
new task failure/timeout/exception, unresolved dock/decision, persistent source
read failure, and optionally loss of an expected process identity. New events
are coalesced into one model invocation. A running invocation serializes all
further model notifications; observations keep accumulating durably.

## Configuration

All paths must be absolute. Use separate source entries for the two native
queue roots. Their `*` automatically discovers new queues. Register only known
outer drivers and their dock directories; do not watch arbitrary analysis/test
status files. An explicit inbox is the interface for future decision producers.

```json
{
  "schema_version": 1,
  "state_dir": "/CAMPAIGN/automation/event-state-v1",
  "poll_seconds": 5,
  "coalesce_seconds": 10,
  "read_error_grace_sec": 30,
  "process_grace_sec": 30,
  "max_events_per_batch": 128,
  "sources": [
    {"id": "main_queues", "kind": "queue", "glob": "/CAMPAIGN/queues/*/status.json"},
    {"id": "aux_queues", "kind": "queue", "glob": "/CAMPAIGN/pool-gpu4567/queues/*/status.json"},
    {"id": "main_controller", "kind": "controller", "path": "/CAMPAIGN/controller.json", "required": true, "monitor_process": true},
    {"id": "aux_controller", "kind": "controller", "path": "/CAMPAIGN/pool-gpu4567/controller.json", "required": true, "monitor_process": true},
    {"id": "tool_outer", "kind": "outer", "path": "/CAMPAIGN/proposals/KNOWN_TOOL_OUTER/status.json", "required": true, "monitor_process": true},
    {"id": "remaining_outer", "kind": "outer", "path": "/CAMPAIGN/proposals/KNOWN_REMAINING_OUTER/status.json", "required": true, "monitor_process": true},
    {"id": "tool_docks", "kind": "dock", "glob": "/CAMPAIGN/proposals/KNOWN_TOOL_OUTER/docks/*/boundary.json"},
    {"id": "registered_outers", "kind": "outer", "glob": "/CAMPAIGN/automation/registered-outers/*/status.json", "monitor_process": true},
    {"id": "registered_docks", "kind": "dock", "glob": "/CAMPAIGN/automation/registered-outers/*/docks/*/boundary.json"},
    {"id": "decisions", "kind": "decision", "glob": "/CAMPAIGN/automation/inbox/*.json"}
  ],
  "transport": {
    "argv": ["/REPO/.venv/bin/python", "/REPO/scripts/a100_automation/invoke_codex.py", "--event-file", "{event_file}", "--ack-file", "{ack_file}", "--batch-id", "{batch_id}", "--config", "/CAMPAIGN/automation/adapter.json"],
    "timeout_sec": 1815
  }
}
```

`kind=outcome` may observe an explicitly scoped `runs/*/outcome.json` glob if
unmanaged controller runs need independent failure notifications. It ignores
successful individual outcomes. Native queues already expose their failures.

`monitor_process` is optional and only reads `/proc`. Controllers use the stored
active controller PID/UID/starttime, or the idle holder's identity. Missing
identity while a controller/holder is explicitly expected produces a grace-gated
process-loss event; the transient state with both null does not. Outer drivers
with a bare PID pin the first observed
start identity, rejecting later PID reuse. Brief process/file transitions have
a grace interval. No observed process is signalled. An unchanged status file
still receives liveness checks; unchanged large manifests are not reloaded.

The first successful scan establishes a persistent historical baseline. Existing
completed/failed queues do not generate a historical flood. Unreleased dock
boundaries and unresolved inbox entries catch up once by default (`catch_up:
false` disables that per source). Dock release is detected by sibling
`release.json` existence; `release_file` may specify another basename. Decision
inbox objects use `{ "id": "stable-id", "status": "pending", "reason": "...",
"request": {...} }`; statuses handled/resolved/acked/dismissed and `resolved:
true` suppress catch-up. Publishing files by atomic rename is recommended.

Configuration is hashed into persistent state. Accidental configuration drift
is rejected. Use the same configuration for restart; source additions require
an explicitly reviewed state migration or a new state directory. Broad native
queue globs, registered-outers globs and one explicit inbox avoid registry changes.
There is no configuration hot reload. To register a future approved outer,
atomically create a uniquely named symlink to its whole directory under
`automation/registered-outers/` (temporary symlink followed by `os.replace`), then
fsync that directory. The fixed status/dock globs discover it next poll. Do not
replace a registered name with a different outer; use a new unique name. This
registers observation only, never submits an experiment. No daemon restart is
needed. Old already-released docks are suppressed both by release-file presence
and by the outer's nested `dock.released` flag; each new stage has its own identity.

Native `queue.py` status/plan/queue-task receipts do not store the queue process
PID/starttime. Existing controller identities cover an active task but cannot
prove the queue orchestrator is alive between tasks. Future event workers MUST
run finite queues under a registered outer whose status includes its PID. If a
new independent queue is necessary, prepare a finite supervising wrapper that
publishes `status.pid` and the queue's terminal result, then register its directory.
Do not leave a bare queue unmonitored. This package does not add a generic scheduler.

## Commands and deployment order

Replace absolute paths in these commands. Root owns real deployment; tests do
not start a service or invoke real Codex.

```bash
CUDA_VISIBLE_DEVICES='' taskset -c 96-127 nice -n 10 /REPO/.venv/bin/python /REPO/scripts/a100_automation/eventd.py check --config /CAMPAIGN/automation/eventd.json
CUDA_VISIBLE_DEVICES='' taskset -c 96-127 nice -n 10 /REPO/.venv/bin/python /REPO/scripts/a100_automation/eventd.py scan --config /CAMPAIGN/automation/eventd.json
CUDA_VISIBLE_DEVICES='' taskset -c 96-127 nice -n 10 /REPO/.venv/bin/python /REPO/scripts/a100_automation/eventd.py run --config /CAMPAIGN/automation/eventd.json
```

`scan` initializes durable state and `HOLD`, records the baseline and any catch-up
events, and **never dispatches**. `run` observes continuously. While held it
collects events without invoking the adapter. Once the owner has finalized the
handoff, source registry and adapter, arm from a separate process:

```bash
/REPO/.venv/bin/python /REPO/scripts/a100_automation/eventd.py arm --config /CAMPAIGN/automation/eventd.json
/REPO/.venv/bin/python /REPO/scripts/a100_automation/eventd.py status --config /CAMPAIGN/automation/eventd.json
/REPO/.venv/bin/python /REPO/scripts/a100_automation/eventd.py hold --reason 'operator maintenance' --config /CAMPAIGN/automation/eventd.json
```

HOLD does not cancel a model turn already dispatched. `status` reads a heartbeat
snapshot without acquiring the writer lock; its `observed_unix` shows freshness.
A second `run`/`scan` owner fails with `OwnerBusy`.

For a user systemd unit, use `WorkingDirectory=/REPO`, `Environment=
CUDA_VISIBLE_DEVICES=`, `CPUAffinity=96-127`, `Nice=10`, `Restart=on-failure`,
`RestartSec=10`, and **`KillMode=process`**. Stopping/restarting the observer must
not kill its independent event worker, Codex, or experiments started by a decision
turn. `start_new_session=True` alone does not isolate systemd cgroups. The owner
must verify login linger separately if service lifetime must survive logout.
The daemon handles SIGTERM/SIGINT by ending observation; it never stops children.

## Adapter and recovery contract

Transport is an argv list, never shell text. Placeholders are `{event_file}`,
`{ack_file}`, `{batch_id}`, and `{batch_dir}`. The outer transport timeout must
exceed the adapter's internal timeout by at least 15 seconds. The worker clears
CUDA visibility before invoking this adapter. The adapter must deduplicate its
own batch claims and preserve all CLI output and action evidence.

Event file:

```json
{"schema_version":1,"batch_id":"HASH","event_ids":["EVENT_HASH"],"events":[{"event_id":"EVENT_HASH","kind":"batch_terminal","source_id":"main_queues","path":"/absolute/status.json","key":"terminal","details":{"status":"completed"}}],"created_unix":0,"reports":[]}
```

Successful adapter exit alone is insufficient. It must atomically publish ACK:

```json
{"batch_id":"HASH","event_ids":["EVENT_HASH"],"status":"handled","summary":"Concrete disposition and any submitted finite work"}
```

`status=blocked` is also a valid acknowledgement of a decision. Event IDs must
match exactly in order with no duplicates. The model must exit after one finite
decision turn, allowing the observer to detect future experiment events. It
must not wait or poll experiments. A blocked ACK means the decision was delivered;
it does not make any experiment successful.

The database uses SQLite DELETE journal and FULL synchronization, plus flock
single ownership. State contains immutable batches, event IDs, source snapshots,
ACKs, local incidents, worker logs, direct transport process identity and durable
results. Durable JSON publication fsyncs file and containing directory. Deploy on
a filesystem with working POSIX locks/fsync; WAL is deliberately not assumed.

Workers acquire a per-batch flock and persist `started.json` before invoking the
adapter. On observer restart a live worker continues, completed results are
reconciled, and a helper that demonstrably never began invoking can be restarted.
Started delivery without a durable result becomes **unknown**. Nonzero adapter
exit becomes **failed**. Both retain all unacknowledged events and block further
dispatch; neither automatically retries. Observation still runs. Own delivery
incidents are not recursively fed back into model events, preventing storms.

After reconciling adapter claims/action evidence, an operator can stop only the
observer, write a resolution file containing `batch_id` and a specific `reason`,
then use `resolve --batch-id HASH --resolution-file ABS --config ABS`. This marks
the delivery resolved, leaves its events unacknowledged for audit, and allows
future independent event batches. It never resubmits the old batch. Do not resolve
while an uncertain model/adapter may still be running. There is no mathematical
exactly-once guarantee across an arbitrary external CLI crash boundary.

## Terminal reports and finite postprocessing

Each newly terminal queue gets JSON and CSV in its event batch's `reports/`.
Every queue task is retained, including failures, pending/unstarted tasks and
missing summaries. Unknown quantities stay null. Main SLO rate is computed only
when the saved numerator's denominator equals all planned requests; inconsistent
success-only denominators are flagged. Completion fractions additionally require
request-count conservation. Original source hashes, eligibility, latency metric
origins, cache metrics, arrival/full-drain goodput and timing definitions are
retained. These are convenient extracts, not a replacement for formal assessor
validation. No formal eligibility or successful outcome is inferred.

Optionally register at most 16 finite CPU postprocessors:

```json
{"postprocess":[{"id":"summary-check","argv":["/absolute/python","/absolute/read_only_report.py","--batch","{event_file}"],"timeout_sec":60}]}
```

They execute once after the durable invocation marker, before the notification
adapter, and store `postprocess-N.json`/`.log` in the same batch directory. Their
failure does not create recursive events or suppress the decision notification.
They must terminate themselves, avoid background descendants and must not submit
experiments. The owner audits these explicit commands before registration.

## Tests

```bash
CUDA_VISIBLE_DEVICES='' taskset -c 96-127 nice -n 10 /REPO/.venv/bin/python -m pytest -q scripts/a100_automation
```

Tests use temporary status trees and fake CPU transports. They exercise event
edges, historical baseline/catch-up, no-call normal progress, hold/arm, partial
JSON, PID reuse, immutable batches, SQLite settings, concurrent ownership,
helper races, daemon/worker death, timeout, invalid ACK, persistent unknown
delivery, finite postprocessing, and all-planned reporting. The separate adapter
tests fake Codex and never use the real account.
