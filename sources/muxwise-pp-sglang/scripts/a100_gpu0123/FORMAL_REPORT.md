The offline report inventories formal tasks from immutable queue plans and their
referenced, hashed freezes before discovering measurement files. Every task in
such a freeze remains an attempt, including tasks with no output directory.
Run `python scripts/a100_gpu0123/report.py --root CAMPAIGN_ROOT` after preserving
the queue/platform evidence. The command is CPU-only.

`report/formal-attempts.json` and `.csv` retain each frozen denominator, outcome,
request-accounting status, provenance, and validation error.
`formal-failures.csv` includes every attempt without an eligible measurement,
including pending or missing attempts; its status columns distinguish them.
Malformed immutable queue evidence fails the report closed.

A finalized measured summary must agree with metrics recomputed from timestamped
raw request records and all frozen planned requests. Failed, unfinished and
unsent requests remain in the SLO denominator. Normal cutoff with unfinished
requests can be a valid measurement. A failed controller, failed queue task,
cleanup error or missing provenance cannot qualify as a successful repeat.

Without a finalized summary, raw observed record states remain diagnostics;
missing records are not assumed unsent, and final request counts and throughput
remain unknown. Explicit replay-not-started evidence with an empty request log
can establish that all planned requests were unsent, but supplies no observed
throughput. Startup-failure zero rates from the client are masked in the report.

Every comparison group includes all matching frozen attempts. One failed or
missing repeat withholds means/min/max, even when three other repeats succeeded.
Different verified transports form different groups; an attempt with unknown
transport conservatively blocks matching groups. Model, runtime, configuration
or arrival-intent discrepancies cannot be hidden by averaging surviving rows.

The default independent repetition requirement is three. Main and critical
comparisons must retain at least three. Auxiliary single-run results require
these explicit fields on **each task in the hashed freeze**:

```json
{
  "phase": "mechanism_ablation",
  "role": "mechanism_ablation",
  "required_independent_repetitions": 1,
  "argv": ["..."],
  "timeout": 7000,
  "trace_sha256": "..."
}
```

Allowed role/phase pairs are:

| Frozen role | Allowed phases |
| --- | --- |
| `mechanism_ablation` | `mechanism_ablation`, `formal_mechanism_ablation` |
| `matched_kv` or `matched_kv_ablation` | `matched_kv_ablation`, `formal_matched_kv_ablation` |
| `burst_auxiliary` | `burst_toolagent`, `burst_auxiliary`, `formal_burst_auxiliary` |
| `communication_auxiliary` | `communication`, `communication_auxiliary`, `formal_communication_auxiliary` |

The group must have one consistent frozen role and repetition requirement.
Missing attempts never reduce it. Conflicting requirements invalidate the group,
even if enough rows would satisfy the smaller number. Values above three remain
binding. A one-run auxiliary group still cannot drop any other frozen attempt.
`formal_repeated_comparison_ready` means that the explicit frozen requirement
is met; inspect `required_independent_repetitions` to distinguish auxiliary results.
