"""Pure send-relative goodput supplement; requires finalized validated accounting."""
import math


def send_relative_goodput(records, account, arrival_window_sec):
    """No I/O. `account` is formal_attempts.request_accounting's verified result.

    Call only after finalized all-planned/outcome validation. Preserve failed,
    unfinished, and unsent requests in account['planned_requests']. This function
    uses timestamps/chunks, never cached per-request TTFT/TPOT fields.
    """
    if account.get('status') != 'complete' or account.get('metrics_observed') is not True:
        raise ValueError('Finalized validated request accounting is required')
    if not math.isfinite(arrival_window_sec) or arrival_window_sec <= 0:
        raise ValueError('A positive planned arrival window is required')
    historical = account['historical']
    total, end = account['planned_requests'], historical['duration_sec']
    if historical['sent_requests'] != len(records) or total < len(records):
        raise ValueError('Records do not match the validated accounting denominator')
    complete, good = [], []
    for row in records:
        finished, first = row.get('finished_at'), row.get('first_token_at')
        if not (row['success'] and row['status'] == 'completed' and finished is not None and finished <= end):
            continue
        count = max((c['completion_tokens'] for c in row['chunks'] if c['at'] <= end), default=0)
        complete.append((row, count))
        if (first is not None and count == row['output_len'] and count > 1
                and first-row['sent_at'] <= 5 and (finished-first)/(count-1) <= .1):
            good.append((row, count))
    result = {}
    for name, boundary, duration in (('arrival_window', min(end, arrival_window_sec), arrival_window_sec),
                                      ('full_drain', end, end)):
        finished = [(row, count) for row, count in complete if row['finished_at'] <= boundary]
        passed = [(row, count) for row, count in good if row['finished_at'] <= boundary]
        tokens = sum(count for _, count in passed)
        observed = sum(max((c['completion_tokens'] for c in row['chunks'] if c['at'] <= boundary), default=0)
                       for row in records)
        result[name] = dict(duration_sec=duration, observation_end_sec=boundary, completed_requests=len(finished),
                            main_slo_requests=len(passed), main_slo_output_tokens=tokens, observed_output_tokens=observed,
                            main_slo_total_planned=total, main_slo_fraction_of_planned=len(passed)/total if total else None,
                            main_slo_requests_per_sec=len(passed)/duration if duration else None,
                            main_slo_output_tokens_per_sec=tokens/duration if duration else None,
                            observed_output_tokens_per_sec=observed/duration if duration else None)
    result['definitions'] = dict(latency_origin='Actual send; primary TTFT<=5s and TPOT<=100ms.',
        ttft='first_token_at - sent_at', tpot='(finished_at - first_token_at) / (completed_output_tokens - 1)',
        denominator='All planned requests; failed, unfinished, unsent and undefined TPOT never pass.',
        windows='Completions at the exact window/cutoff boundary are included; arrival duration stays nominal and full drain ends at validated observation/cutoff.',
        goodput='Only output tokens of successfully completed SLO-passing requests; partial/failed tokens appear only in observed_output_tokens.')
    return result
