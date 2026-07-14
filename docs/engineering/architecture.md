# Architecture

## System Components

- **API service** — stateless FastAPI-equivalent HTTP layer. Handles auth,
  rate limiting, and request validation, then enqueues work or reads
  directly from Postgres for simple lookups.
- **Worker fleet** — consumes jobs from Redis (workflow runs, webhook
  delivery retries, scheduled trigger evaluation). Horizontally scalable;
  each worker polls a shared queue rather than owning a partition.
- **Postgres** — system of record for workflows, run history, and workspace
  metadata. Not sharded; a single primary handles all workspaces up to the
  scaling limits described below.
- **Redis** — job queue (via a Redis Streams-based queue) and rate-limit
  counters (sliding window, implemented with sorted sets).
- **Scheduler** — a single leader-elected process that evaluates cron
  triggers every 30 seconds and enqueues due workflow runs. Leader election
  uses a Redis lock with a 45-second TTL and 15-second renewal.

## Data Flow

1. A trigger fires (schedule, webhook, or manual API call).
2. The API (or scheduler, for cron triggers) writes a `run` row in
   `queued` status and pushes a job onto the Redis stream.
3. A worker picks up the job, transitions the run to `running`, and executes
   steps sequentially. Each step's output becomes available as input to
   subsequent steps via a shared run context.
4. On completion, the worker writes the final status (`succeeded` or
   `failed`) and enqueues any registered webhooks for `run.succeeded` /
   `run.failed` events.
5. Webhook delivery is itself a queued job with its own retry policy,
   decoupled from the workflow run so a slow customer endpoint can't block
   worker capacity.

## Technology Choices

- **Postgres over a NoSQL store**: workflow definitions and run history are
  relational (a run belongs to a workflow, steps belong to a run) and
  benefit from transactional guarantees when a run's status and its step
  logs are written together.
- **Redis Streams over a dedicated message broker (e.g. Kafka)**: at current
  scale, the operational simplicity of one fewer stateful service outweighs
  Kafka's throughput headroom. Revisit if sustained job volume exceeds
  ~5k jobs/sec.
- **Leader-elected scheduler over distributed cron**: cron evaluation must
  not double-fire a schedule across replicas; a single leader with a lock
  is simpler to reason about than a distributed consensus scheme for a
  30-second-granularity job.

## Scaling Considerations

- The worker fleet scales horizontally with no coordination needed beyond
  the shared Redis queue — this is the primary scaling lever under load.
- Postgres is the eventual bottleneck. Current guidance is to vertically
  scale the primary until read replicas become necessary for run-history
  queries, which are read-heavy and don't need read-after-write consistency.
- The scheduler is intentionally a single point of coordination (not a
  single point of failure — a standby takes over the lock within 45 seconds
  of the leader dying), so it does not scale horizontally by design.

## Known Limitations

- Workflow steps run sequentially within a run; there's no parallel-step
  execution yet, so a workflow with independent branches still pays their
  combined latency.
- The 30-second scheduler tick means cron triggers have up to 30 seconds of
  jitter — not suitable for sub-minute precision scheduling.
- Webhook delivery retries cap at 3 attempts (configurable via
  `WEBHOOK_MAX_RETRIES`); there's no dead-letter queue yet, so events that
  exhaust retries are logged but not replayable without a manual API call.
