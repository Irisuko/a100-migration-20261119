# A100 four-GPU campaign

Current authoritative progress: [REPORT.md](../../run/a100-4gpu-20261114/REPORT.md).
The user has released physical GPUs4/5 to other users. The persisted global
`RESERVATION_DISABLED.json` blocks all holders and model runners, including
custom output namespaces. Do not remove it merely because GPUs look idle.

- `campaign.py status`: 27 requested candidates, prerequisites and execution state.
- `campaign.py preflight --candidate ID --attempt N`: checked serial full preflight,
  only after explicit resource restoration; attempts use new labels.
- `control.py run --label LABEL --timeout SECONDS -- COMMAND...`: UUID binding,
  exclusive queue, owned-process cleanup and idle reservation lifecycle.
- `server_run.py`: requires live controller ownership; ordinary/Chunked/MuxWise/
  MuxWise-PP/DP/PD startup and preflight/replay. Trace runs are not automatically formal.
- `compare_preflight.py`: ordinary same-model/topology reference comparisons,
  including PD pairing. Never turns numerical mismatches into passes by tolerance.
- `bench/bench_client.py`: faithful open-loop `--time-scale` replay, historical and
  arrival-aware metrics, complete planned/outcome lists and verified output IDs.
- `platform_probe.py`: isolated four-rank payload/green-context/graph diagnostic;
  existing probe did not record NCCL transport logs and is not model validation.
- `build_attention.py`: sm80 BF16 FlashInfer build with CUDA visibility empty.
- `report.py`: status CSV, measurement CSV, separate-source aggregates and plots
  when real measurements exist. Empty measurements produce no performance plots.

The parameter selection/freeze, formal queue and final ablation matrix are pending
actual development results. Preserve historical `run/a100/` and the restored
`../muxwise-pp-experiments` repository. CPU regressions must hide CUDA; launch-boundary
unit tests must forbid real GPU subprocesses.
