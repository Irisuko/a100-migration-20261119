`build_selected.py` renders queues from explicit reviewed choices. It does not
rank configurations, select rates, read formal measurements, or launch processes.
No actual configuration/rate selection is included with this helper.

The selection JSON has `schema_version: 1`,
`status: "reviewed_selected_configurations"`, the exact current serving
`source_sha256` map, and `records`. Each record supplies:

- `id`: a unique short selection name across model/workload groups.
- `candidate`: `id`, `model`, `family`, `tp`, `pp`, `replicas`, plus
  `prefill_replicas` for PD.
- `workload`, `selected_task_dir`: the exact completed development trial under
  this campaign's `runs/`. The existing selector validates its raw measurements,
  provenance and eligibility. Architecture/finalist budget selection is an
  explicit prior review, not an inference performed here.
- `full_gate_label`, `full_gate_run`: completed exact-source correctness plus
  stress200, every replica/rank, resources, actual overlap, model/configuration
  bytes, and the same serving arguments/platform as the selected trial.
- `gate_review: {path, sha256}`: the explicit review described below. Pin the
  finalist profile/slot ledger and selection review in its evidence map.

The gate review has `status: "REVIEWED"`, the selector's exact
`configuration_fingerprint`, `source_fingerprint`, `model_fingerprint`, the
absolute `full_gate_run`, and
`checks: {communication: true, full_preflight: true}`. It includes nonempty
`evidence_sha256` and `evidence_phases`, restricted to development, correctness,
communication and numerical_audit. These are reviewed evidence records; this
helper does not fabricate them or substitute a boolean for a failed runtime gate.

Its `numerical` object carries `topology` matching the candidate fields above,
nonempty hashed evidence and limitations, and one explicit disposition:

- `strict_pass` requires `strict_token_equality: true`, scoped to the evidence.
- `performance_only_non_bitwise` requires `strict_token_equality: false`,
  `allowed_claims: ["fixed_input_fixed_length_performance"]` and
  `prohibited_claims: ["quality", "numerical_equivalence"]`.

The disposition remains in every emitted task and formal freeze. A PP4 result
cannot provide the declared scope for a TP2 numerical disposition.

The rate JSON has `schema_version: 1`, `selection_manifest_sha256`,
`evidence_phases`, `evidence_sha256`, and `records` containing short `model`,
`workload`, positive `absolute_requests_per_sec`, and `role` (`main` or
`critical`). `trace_sha256` may disambiguate a trace; otherwise precisely one
audited matching record is required. Every selected model/workload needs rate
records, and every selected configuration gets each of its cohort's streams.

For `--stage rate_calibration`, status is `reviewed_rates`; evidence phases are
development/rate_calibration. Only complete raw[1200,1440), base600s `dev` records
are accepted. Configurations stay fixed. Each point has one new deployment.

For `--stage formal`, rates must already have status `frozen`, evidence phases
exactly `["rate_calibration"]`, and nonempty hashed development evidence.
`build_formal_windows.py` can construct the corresponding audited
`formal_frozen_rate` manifest. Its unchanged rate-record digest must match.
Every stream needs nominal duration at least210s, actual first-to-last arrival
span at least180s, and at least3 distinct arrival times; exact rational checks
prevent rounding or idle padding from qualifying a shorter stream. All eligible
raw rows are retained. Main/critical points require at least3 fresh labels.

Example command shapes (placeholders are not adopted choices):

```sh
CUDA_VISIBLE_DEVICES='' .venv/bin/python scripts/a100_gpu0123/build_selected.py \
  --stage rate_calibration --selection REVIEWED_SELECTION.json \
  --rates REVIEWED_GRID.json --trace-manifest ROOT/traces/manifest.json \
  --root ROOT --tag NEW_TAG --out-dir ROOT/proposals/NEW_DIRECTORY

CUDA_VISIBLE_DEVICES='' .venv/bin/python scripts/a100_gpu0123/build_selected.py \
  --stage formal --selection REVIEWED_SELECTION.json \
  --rates FROZEN_RATES.json --trace-manifest AUDITED_FORMAL_MANIFEST.json \
  --root ROOT --tag NEW_TAG --out-dir ROOT/freezes/NEW_DIRECTORY
```

Outputs are exclusive-created: transport-specific `tasks-default.json` and/or
`tasks-nvlink.json`, `protocol.json`, a dereferenced content archive and verified
archive manifest, plus `freeze.json` for formal work. All pinned serving/helper,
config, input/evidence, trace/raw-source, model JSON and supplied platform package
and library bytes are archived. The archive is restored into a temporary
directory and every member hash checked. Archive/control hashes enter the freeze;
unpinned native binaries and full model weights are not claimed archived.

Use the unchanged `run_campaign.py` with the task file's required transport and
the formal freeze when appropriate. This renderer does not launch. The existing
queue/controller provides independent deployments; existing formal reporting
keeps failed, missing and unsent requests in frozen attempt accounting and
withholds survivor averages. The client drains for300s after its last actual
send; the shared controller cap is `ceil(max(7000, N/r + 2400))`. No deadline is
extended after observing results. Auxiliary single-run protocols are outside
this helper's main/critical scope.
