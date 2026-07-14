# Feature Specifications

## 1. Scheduled Triggers

Run workflows on a cron schedule. Configuration: `cron` (standard 5-field
cron string, evaluated in the workspace's configured timezone), `timezone`
(IANA name, default `UTC`). Feature flag: `scheduled_triggers` — enabled by
default on all tiers.

**Availability**: Free, Team, Enterprise.

## 2. Webhook Triggers

Start a workflow run when an inbound webhook is received at a
workspace-scoped URL. Configuration: `secret` (auto-generated, used for
HMAC signature verification of *inbound* requests if the caller supports
it), `allowed_ips` (optional CIDR allowlist). Feature flag:
`webhook_triggers`.

**Availability**: Team, Enterprise. Not available on Free tier.

## 3. Conditional Branching

Add `if`/`else` steps that route execution based on a JMESPath expression
evaluated against the run context. Configuration: `condition` (JMESPath
string), `on_true` / `on_false` (step IDs to jump to). Feature flag:
`conditional_branching`.

**Availability**: Team, Enterprise.

## 4. Retry Policies

Per-step retry configuration overriding the workflow default. Configuration:
`max_retries` (0-10), `backoff` (`fixed` | `exponential`), `initial_delay_ms`.
Feature flag: `custom_retry_policies` — without it, all steps use the
workspace default (3 retries, exponential backoff, 1000ms initial delay).

**Availability**: Enterprise only.

## 5. Run History Retention

Controls how long completed run records (including step logs) are kept
before automatic deletion. Configuration: `retention_days`. Feature flag:
`extended_retention`.

**Availability**: Free tier is hard-capped at 7 days regardless of
configuration. Team can configure up to 30 days. Enterprise can configure up
to 365 days or set `retention_days: -1` for indefinite retention.

## 6. Approval Gates

Pause a workflow run and require a human approval (via the dashboard or the
`POST /v1/runs/{run_id}/approve` endpoint) before continuing to the next
step. Configuration: `approvers` (list of user IDs or a role name),
`timeout_hours` (run fails if no approval within this window). Feature flag:
`approval_gates`.

**Availability**: Enterprise only.

## Enabling Feature Flags

Feature flags are managed in **Settings → Feature Flags** for Enterprise
plans (self-serve) or by contacting support for Team-tier flags that are
plan-gated rather than opt-in. Flags can also be queried via
`GET /v1/workspace/flags`, which returns each flag's enabled state and the
plan tier that unlocked it.
