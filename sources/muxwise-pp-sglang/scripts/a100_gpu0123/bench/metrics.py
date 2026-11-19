"""One cutoff-aware metric implementation for live replay and offline reports."""
import math

from common import distribution


def summarize_arrivals(records, planned, cutoff_time=None, arrival_window_sec=None):
    """Supplement historical send-relative metrics without dropping unsent requests."""
    planned_by_id = {r["idx"]: r for r in planned}
    if len(planned_by_id) != len(planned):
        raise ValueError("duplicate planned request IDs")
    actual = max((r.get("finished_at") or r.get("sent_at", 0) for r in records), default=0.0)
    end = min(actual, cutoff_time) if cutoff_time is not None else actual
    arrival_end = (max((r["arrival"] for r in planned), default=0.0)
                   if arrival_window_sec is None else arrival_window_sec)
    if arrival_end < max((r["arrival"] for r in planned), default=0.0):
        raise ValueError("arrival window ends before a planned request")
    sent = {r["idx"]: r for r in records}
    if len(sent) != len(records) or not set(sent) <= set(planned_by_id):
        raise ValueError("duplicate or unplanned result request IDs")
    projected, ttfts, e2es = [], [], []
    for item in planned:
        r = sent.get(item["idx"])
        first = r.get("first_token_at") if r else None
        done = bool(r and r["success"] and r.get("finished_at") is not None and r["finished_at"] <= end)
        ttft = first - item["arrival"] if first is not None and first <= end else math.inf
        e2e = r["finished_at"] - item["arrival"] if done else math.inf
        ttfts.append(ttft); e2es.append(e2e)
        if done:
            projected.append(dict(r, scheduled_ttft=ttft, scheduled_e2e=e2e))
    slo = {}
    for ttft in (None, 2, 5, 10):
        for ms in (None, 50, 100, 200):
            if ttft is None and ms is None:
                continue
            key = (f"ttft_le_{ttft}s" if ttft is not None else "")
            key += ("_and_" if key and ms is not None else "") + (f"tpot_le_{ms}ms" if ms is not None else "")
            passed = sum((ttft is None or r["scheduled_ttft"] <= ttft)
                         and (ms is None or (r.get("tpot") is not None and r["tpot"] <= ms / 1000))
                         for r in projected)
            slo[key] = dict(passed=passed, total=len(planned), rate=passed / len(planned) if planned else None,
                            ttft_threshold_sec=ttft, tpot_threshold_sec=ms / 1000 if ms is not None else None)
    goodput = {}
    for name, boundary, duration in (("arrival_window", min(end, arrival_end), arrival_end),
                                     ("full_drain", end, end)):
        complete = [r for r in projected if r["finished_at"] <= boundary]
        good = [r for r in complete if r["scheduled_ttft"] <= 5 and r.get("tpot") is not None and r["tpot"] <= .1]
        observed = sum(max((c["completion_tokens"] for c in r.get("chunks", []) if c["at"] <= boundary), default=0)
                       for r in records)
        good_tokens = sum(r["output_tokens"] for r in good)
        goodput[name] = dict(duration_sec=duration, observation_end_sec=boundary,
                             completed_requests=len(complete), main_slo_requests=len(good),
                             main_slo_output_tokens=good_tokens, observed_output_tokens=observed,
                             main_slo_requests_per_sec=len(good) / duration if duration else None,
                             main_slo_output_tokens_per_sec=good_tokens / duration if duration else None,
                             observed_output_tokens_per_sec=observed / duration if duration else None)
    return dict(planned_requests=len(planned), sent_requests=len(records), unsent_requests=len(planned)-len(records),
                latencies_sec=dict(ttft=censored_distribution(ttfts), e2e=censored_distribution(e2es)),
                slo=slo, goodput=goodput,
                definitions={"latency_origin": "Scheduled arrival, including client send lag. Historical top-level metrics remain send-relative.",
                             "slo": "All planned requests; failed, unfinished, unsent and undefined metrics miss. TPOT unchanged.",
                             "goodput": "Successfully completed requests meeting scheduled TTFT<=5s and TPOT<=100ms. Arrival window includes only completions/tokens observed by its end; full drain uses all observations through actual completion or cutoff.",
                             "arrival_window": "Replay seconds from origin through planned window end. Duration is preserved even if interrupted; see observation_end_sec."})


def censored_distribution(values):
    """Linear (NumPy default) quantiles on extended reals, JSON-safe."""
    values = sorted(values)
    n = len(values)
    result = {"count": n, "finite_count": sum(math.isfinite(x) for x in values)}
    for name, q in (("p50", .50), ("p90", .90), ("p95", .95), ("p99", .99)):
        if not n:
            result[name] = None
            continue
        pos = (n - 1) * q
        lo, hi = math.floor(pos), math.ceil(pos)
        result[name] = (">cutoff" if math.isinf(values[hi]) else
                        values[lo] + (values[hi] - values[lo]) * (pos - lo))
    result["mean"] = (">cutoff" if n and math.isinf(values[-1]) else
                      sum(values) / n if n else None)
    result["max"] = (">cutoff" if n and math.isinf(values[-1]) else
                     values[-1] if n else None)
    return result


def cutoff_records(records, cutoff_time=None):
    """Return a projection; never mutate original records or approximate token times."""
    cutoff = math.inf if cutoff_time is None else cutoff_time
    projected = []
    for original in records:
        r = dict(original)
        finished = r.get("finished_at")
        terminal_before = finished is not None and finished <= cutoff
        completed = bool(r["success"] and terminal_before)
        unfinished = r.get("status") == "unfinished" or not terminal_before
        r["status"] = "completed" if completed else "unfinished" if unfinished else "failed"
        r["success"] = completed
        first = r.get("first_token_at")
        # Old synthetic tests may provide only TTFT; real logs always have *_at.
        if "first_token_at" not in r and r.get("ttft") is not None:
            first = r.get("sent_at", 0) + r["ttft"]
        if first is None or first > cutoff:
            r["first_token_at"] = r["ttft"] = None
        else:
            r["first_token_at"] = first
        if "chunks" in r:
            r["chunks"] = [c for c in r["chunks"] if c["at"] <= cutoff]
            r["output_tokens"] = max((c["completion_tokens"] for c in r["chunks"]), default=0)
        elif not terminal_before:
            raise ValueError("Cannot truncate output tokens without timestamped chunks")
        if not completed:
            r["e2e"] = r["tpot"] = None
        projected.append(r)
    return projected


def summarize_records(records, planned_requests=None, cutoff_time=None):
    total = len(records) if planned_requests is None else planned_requests
    if total < len(records):
        raise ValueError("planned_requests smaller than sent records")
    actual = max((r.get("finished_at") or r.get("stopped_at") or r.get("sent_at", 0)
                  for r in records), default=0.0)
    duration = min(actual, cutoff_time) if cutoff_time is not None else actual
    records = cutoff_records(records, cutoff_time)
    successful = [r for r in records if r["success"]]
    unfinished = sum(r["status"] == "unfinished" for r in records)
    failed = sum(r["status"] == "failed" for r in records)
    # Input processing is observable only once an output token is delivered.
    input_tokens = sum(r["input_len"] for r in records if r["first_token_at"] is not None)
    output_tokens = sum(r["output_tokens"] for r in records)
    cached = [r for r in successful if r.get("cached_tokens") is not None]
    latencies = {}
    for name in ("ttft", "e2e"):
        values = [r.get(name) if r.get(name) is not None else math.inf for r in records]
        latencies[name] = censored_distribution(values + [math.inf] * (total - len(records)))
    latencies["tpot"] = distribution(r.get("tpot") for r in successful)
    latencies["itl"] = distribution(x for r in successful for x in r.get("itl", []))
    latencies["send_lag"] = distribution(r.get("send_lag") for r in records)
    slo = {}

    def add_slo(key, ttft=None, tpot=None):
        passed = sum((ttft is None or (r.get("ttft") is not None and r["ttft"] <= ttft))
                     and (tpot is None or (r.get("tpot") is not None and r["tpot"] <= tpot))
                     for r in successful)
        slo[key] = {"passed": passed, "total": total, "rate": passed / total if total else None,
                    "ttft_threshold_sec": ttft, "tpot_threshold_sec": tpot}

    for ttft in (2, 5, 10):
        add_slo(f"ttft_le_{ttft}s", ttft=ttft)
    for ms in (50, 100, 200):
        add_slo(f"tpot_le_{ms}ms", tpot=ms / 1000)
    for ttft in (2, 5, 10):
        for ms in (50, 100, 200):
            add_slo(f"ttft_le_{ttft}s_and_tpot_le_{ms}ms", ttft, ms / 1000)
    return {
        "planned_requests": total, "sent_requests": len(records),
        "completed": len(successful), "unfinished": unfinished, "failed": failed,
        "completed_requests": len(successful), "unfinished_requests": unfinished,
        "failed_requests": failed, "unsent_requests": total - len(records),
        "cutoff_time": cutoff_time, "actual_duration_sec": actual, "duration_sec": duration,
        "input_tokens": input_tokens, "output_tokens": output_tokens,
        "observed_output_tokens_including_failures": output_tokens,
        "throughput": {"input_tokens_per_sec": input_tokens / duration if duration else None,
                       "output_tokens_per_sec": output_tokens / duration if duration else None,
                       "requests_per_sec": len(successful) / duration if duration else None},
        "latencies_sec": latencies, "slo": slo,
        "cached_tokens": {
            "available_requests": len(cached),
            "mean_ratio": sum(r["cached_tokens"] / r["input_len"] for r in cached) / len(cached) if cached else None,
            "token_weighted_ratio": sum(r["cached_tokens"] for r in cached) / sum(r["input_len"] for r in cached) if cached else None,
            "total": sum(r["cached_tokens"] for r in cached)},
        "metric_definitions": {
            "time": "Seconds relative to replay origin, after warmup. cutoff_time = last actual sent_at + drain_timeout.",
            "duration": "min(actual terminal duration, cutoff_time), including initial idle and final drain.",
            "latencies": "TTFT/E2E: ALL planned requests. Missing first token / unsuccessful completion = +infinity, serialized as >cutoff. Linear quantiles on extended reals.",
            "tpot": "(E2E-TTFT)/(output_tokens-1); only completed before cutoff with output_tokens>1; count reports sample size.",
            "slo": "ALL planned requests; unfinished/failed/unsent/undefined metrics are misses, including TTFT-only SLO.",
            "throughput": "All output tokens received in timestamped SSE chunks by cutoff, including partial/failed requests. Input counted once for each request delivering a first token by cutoff (includes cached input); neither queued input nor unobserved partial prefill is counted. Divide by duration_sec.",
            "cache": "Completed responses with cached_tokens only; missing is null. PD decode metadata does not measure prefill cache.",
            "itl": "Observed SSE intervals for completed requests; batched chunks do not reveal exact per-token times."}}
