# Troubleshooting

## `429 Too Many Requests` on every call

**Symptom**: All API calls fail with `RATE_LIMITED`, even after waiting.
**Diagnosis**: Check `X-RateLimit-Reset` on the response — if it's far in
the future, you may be sharing an API key across multiple services that
collectively exceed 100 req/min.
**Fix**: Issue separate API keys per service, or request a higher limit for
Enterprise plans via support.
**Escalate if**: The reset time itself looks wrong (e.g. resets to a time in
the past) — that's a server-side bug, not a usage issue.

## Workflow stuck in `queued` status

**Symptom**: A run never transitions to `running`.
**Diagnostic command**: `GET /v1/runs/{run_id}` and check `queued_at` age.
**Fix**: Usually indicates worker fleet capacity exhaustion. If self-hosted,
check `nimbus_queue_depth` in your Prometheus metrics — a persistently
growing queue means you need more worker replicas or higher
`WORKER_CONCURRENCY`.
**Escalate if**: Queue depth is near zero but runs still don't start — that
points to a scheduler leader-election issue.

## `WORKFLOW_LOCKED` won't clear

**Symptom**: Editing a workflow returns `409 WORKFLOW_LOCKED` even though no
run appears to be active in the dashboard.
**Diagnostic command**: `GET /v1/runs?workflow_id={id}&status=running`
**Fix**: A run can be stuck in `running` if a worker crashed mid-execution
before self-hosted deployments have a stale-run reaper (introduced
alongside worker health checks). Cancel the stuck run explicitly via
`POST /v1/runs/{run_id}/cancel`.
**Escalate if**: Cancelling doesn't clear the lock within a minute.

## Webhook signature verification always fails

**Symptom**: Your endpoint rejects every Nimbus webhook as unsigned/invalid.
**Diagnostic**: Confirm which signature scheme your code implements.
**Fix**: If you built the integration before v2.0.0, you're likely still
verifying against the v1 scheme (body-only HMAC). v2.0.0 changed this to
include a timestamp and sign the raw body plus timestamp — see the
Changelog's v2.0.0 breaking-change entry and the API reference's webhook
section for the current scheme.
**Escalate if**: You've confirmed you're on the v2 scheme and it still
fails — could indicate a body-encoding mismatch (e.g. your framework
re-serializing JSON before you compute the signature, which changes byte
ordering).

## Self-hosted deploy fails health check after upgrade

**Symptom**: `/health/ready` returns 503 after upgrading to a new version.
**Diagnostic command**: `docker compose logs api | grep -i migration`
**Fix**: Most commonly a pending migration wasn't applied. Run
`nimbus-cli migrate status` then `nimbus-cli migrate up`.
**Escalate if**: Migrations are up to date but readiness still fails — check
Postgres/Redis connectivity directly, since `/health/ready` checks both.

## Scheduled trigger fires twice for the same tick

**Symptom**: Duplicate runs at the same scheduled time.
**Diagnostic**: This was a known issue prior to v1.2.0 (race condition during
scheduler leader-election handoff), fixed in that release.
**Fix**: Upgrade to v1.2.0 or later.
**Escalate if**: You're already on v1.2.0+ and still see duplicates — this
would be a regression and should go straight to engineering.

## OAuth token refresh returns `AUTH_EXPIRED_TOKEN`

**Symptom**: Using a refresh token that should still be valid returns
`AUTH_EXPIRED_TOKEN` instead of a new access token.
**Diagnostic**: Refresh tokens last 30 days from issuance, not from last
use — they don't extend on each refresh.
**Fix**: Re-run the full Authorization Code flow to get a new refresh token.
**Escalate if**: The refresh token is well within its 30-day window and
still rejected.

## High webhook delivery latency

**Symptom**: `nimbus_webhook_delivery_p99_ms` is elevated.
**Diagnostic command**: Check whether the delay is on Nimbus's side (queue
depth for webhook jobs) or the receiving endpoint's response time.
**Fix**: If it's your endpoint, note that `WEBHOOK_TIMEOUT_SECONDS` defaults
to 10s — slow endpoints should either respond faster or process
asynchronously and return 200 immediately.
**Escalate if**: Nimbus-side queue depth for webhook jobs is elevated even
though endpoints respond quickly — infrastructure-side issue.

## Run history missing older than expected

**Symptom**: Runs older than N days are gone, N doesn't match your
configured `retention_days`.
**Diagnostic**: Confirm your actual plan tier — Free is hard-capped at 7
days *regardless* of any `retention_days` configuration attempt.
**Fix**: Upgrade to Team or Enterprise if you need longer retention.
**Escalate if**: You're on Team/Enterprise, `retention_days` is set
correctly, and history is still disappearing early.

## Database migration hangs

**Symptom**: `nimbus-cli migrate up` doesn't complete.
**Diagnostic command**: Check for long-running transactions on the
migration's target table (`pg_stat_activity` on Postgres).
**Fix**: A large `run_logs` table can make certain migrations slow, not
stuck — check for progress via row counts before assuming it's hung.
**Escalate if**: No progress after 30+ minutes on a table under 10M rows.
