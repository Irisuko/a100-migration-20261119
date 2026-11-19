# Event driven campaign supervision

This package observes existing experiment files. It never submits, restarts, stops,
or signals an experiment. The only configured child is a notification adapter;
root owns its Codex command and the separate experiment authorization policy.

1. Read queues, outer status, controllers, outcomes, dock boundaries, and an
   explicit decision inbox. Compare semantic facts, not file timestamps. Initial
   historical outcomes are baseline; unresolved decisions/docks catch up once.
2. Persist observations and events in one SQLite transaction (DELETE journal,
   FULL synchronous). A single-owner flock protects the coordinator. Bounded
   polling performs no model call during normal progress. Parse errors must
   survive a grace interval before becoming one source exception per episode.
3. Coalesce new events into an immutable batch. Initial setup is held by HOLD;
   an explicit arm removes it. One independently surviving worker at a time
   invokes an explicit argv adapter with event_file, ack_file, and batch_id.
4. Worker flock, durable invocation marker, process start identity, durable
   result, and explicit ACK permit restart recovery. Delivery with an uncertain
   outcome is retained as unknown and never automatically repeated. Exactly-once
   execution across an arbitrary external CLI crash is not claimed. The adapter
   must also deduplicate batch_id and preserve its decision/action ledger.
5. Terminal queue reports preserve every task, unknown quantities, and the full
   planned-request denominator. Reports copy existing metrics and definitions;
   they do not recompute or silently reinterpret measurement eligibility.

Implementation order: semantic-edge/hold tests first; durable outbox and worker;
failure/race tests with fake CPU transport; terminal reporting; interface and
recovery documentation. No live daemon or real model is launched by these tests.

The daemon's own delivery failures are local incidents, not new model events.
This prevents recursive supervision storms. Unresolved unknown delivery blocks
subsequent dispatch until explicit operator resolution, but observation continues.
