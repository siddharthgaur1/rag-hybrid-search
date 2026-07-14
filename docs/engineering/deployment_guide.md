# Deployment Guide

This guide covers deploying Nimbus's self-hosted edition via Docker.

## Docker Deployment

1. Pull the image: `docker pull nimbus/nimbus-server:2.3.0`
2. Copy `.env.production.example` to `.env.production` and fill in secrets.
3. Start the stack: `docker compose -f docker-compose.prod.yml up -d`
4. Run migrations (see below).
5. Verify health: `curl http://localhost:8000/health` should return `{"status": "ok"}`.

The production compose file starts three services: `api`, `worker` (executes
scheduled workflow runs), and `redis` (job queue + rate-limit counters).
Postgres is expected to be external — Nimbus does not bundle a database
container in production mode.

## Environment Variables

| Variable | Required | Default | Description |
|----------|----------|---------|-------------|
| `DATABASE_URL` | yes | — | Postgres connection string. |
| `REDIS_URL` | yes | — | Redis connection string for the job queue. |
| `SECRET_KEY` | yes | — | 32+ char random string, signs session cookies and webhook secrets. |
| `NIMBUS_ENV` | no | `production` | `production` or `staging`. Affects log verbosity. |
| `API_PORT` | no | `8000` | Port the API server binds to. |
| `WORKER_CONCURRENCY` | no | `4` | Max concurrent workflow runs per worker process. |
| `RATE_LIMIT_PER_MINUTE` | no | `100` | Global default; can be overridden per API key. |
| `RATE_LIMIT_PER_HOUR` | no | `1000` | Global default. |
| `WEBHOOK_TIMEOUT_SECONDS` | no | `10` | Timeout for outbound webhook delivery. |
| `WEBHOOK_MAX_RETRIES` | no | `3` | Retries with exponential backoff before marking a webhook failed. |
| `LOG_LEVEL` | no | `info` | `debug`, `info`, `warning`, `error`. |
| `SENTRY_DSN` | no | — | If set, errors are reported to Sentry. |
| `SMTP_HOST` | no | — | Required only if email notifications are enabled. |
| `SMTP_PORT` | no | `587` | |
| `S3_BUCKET` | no | — | Required for workflow attachment storage; falls back to local disk if unset. |
| `OAUTH_CLIENT_ID` / `OAUTH_CLIENT_SECRET` | no | — | Required only if third-party OAuth apps are enabled. |

## Database Migrations

Migrations run via the bundled CLI, not automatically on boot:

```bash
docker compose -f docker-compose.prod.yml exec api nimbus-cli migrate up
```

To check pending migrations without applying them: `nimbus-cli migrate status`.
Rolling back one step: `nimbus-cli migrate down --steps 1`. Migrations are
versioned and idempotent; running `migrate up` twice is a no-op if nothing's
pending.

## Health Check Endpoints

- `GET /health` — liveness probe. Returns 200 if the process is up, regardless
  of downstream dependency health.
- `GET /health/ready` — readiness probe. Checks Postgres and Redis
  connectivity; returns 503 if either is unreachable. Use this one for load
  balancer health checks, not `/health`.

## Rollback Procedures

1. Identify the last known-good image tag from your deployment history.
2. `docker compose -f docker-compose.prod.yml pull nimbus/nimbus-server:<previous-tag>`
3. If the failed deploy included a migration, run `nimbus-cli migrate down`
   for the number of migrations introduced in the bad release **before**
   rolling back the image — running an old image against a newer schema is
   unsupported and can corrupt workflow state.
4. Restart: `docker compose -f docker-compose.prod.yml up -d`
5. Confirm `/health/ready` returns 200 before routing traffic back.

## Monitoring Setup

Nimbus exposes Prometheus-compatible metrics at `/metrics` on the API and
worker services (port 9090 by default, separate from the API port). Key
metrics to alert on: `nimbus_run_failure_rate`, `nimbus_webhook_delivery_p99_ms`,
`nimbus_queue_depth`. If `SENTRY_DSN` is set, unhandled exceptions are
reported automatically with request context attached.
