# Frequently Asked Questions

## Account & Billing

**Q: What plan tiers are available?**
Free, Team, and Enterprise. See `docs/product/feature_specs.md` for what's
gated per tier.

**Q: How do I upgrade from Free to Team?**
Go to **Settings → Billing → Upgrade Plan**. Upgrades take effect
immediately and are prorated for the current billing period.

**Q: Can I downgrade from Enterprise to Team mid-cycle?**
Yes, but any Enterprise-only features you're using (Approval Gates, custom
Retry Policies, retention beyond 30 days) will stop working immediately, not
at the end of the billing period. Export any run history beyond 30 days
before downgrading — it becomes inaccessible, though it isn't deleted for
90 days in case you upgrade back.

**Q: Do unused API requests roll over?**
No, rate limits reset every minute/hour and usage quotas reset every
billing period. There's no rollover.

## Authentication

**Q: My API key stopped working, what happened?**
Check `GET /v1/usage` — if it 401s with `AUTH_INVALID_KEY`, the key may
have been revoked (check **Settings → API Keys** for its status) or you may
be using a `nb_test_` key against the production API URL.

**Q: How long do OAuth access tokens last?**
1 hour. Refresh tokens last 30 days. See `docs/engineering/api_reference.md`
for the full auth flow.

**Q: Can one API key be used across multiple workspaces?**
No, API keys are scoped to a single workspace at creation time.

## Workflows

**Q: Why did my workflow edit fail with `WORKFLOW_LOCKED`?**
The workflow has an in-progress run. Wait for it to complete or cancel the
run via `POST /v1/runs/{run_id}/cancel`, then retry the edit.

**Q: Can workflow steps run in parallel?**
Not currently — see the "Known Limitations" section of
`docs/engineering/architecture.md`. Steps execute sequentially.

**Q: How do I pass data between steps?**
Each step's output is available to later steps via the shared run context.
Reference prior step output using JMESPath expressions in step config, the
same syntax used for Conditional Branching conditions.

**Q: What happens to a workflow's run history if I delete the workflow?**
Deleting a workflow archives it (soft delete); run history is retained per
your plan's retention policy and only purged after 90 days total, even past
the archive.

## Webhooks

**Q: Why isn't my webhook receiving events?**
1. Confirm the webhook is registered for the right event type
   (`run.succeeded` / `run.failed`).
2. Check that your endpoint responds within `WEBHOOK_TIMEOUT_SECONDS`
   (default 10s) — slow responses count as delivery failures.
3. Verify your endpoint isn't rejecting the request based on the
   `X-Nimbus-Signature` header using the outdated v1 scheme; v2.0.0 changed
   the signing method (see Changelog).

**Q: How many times will Nimbus retry a failed webhook delivery?**
`WEBHOOK_MAX_RETRIES` times (default 3), with exponential backoff. After
that, delivery is marked failed and is not automatically retried.

## Data & Retention

**Q: How long is my run history kept?**
Depends on plan: Free is hard-capped at 7 days, Team configurable up to 30
days, Enterprise up to 365 days or indefinite (`retention_days: -1`).

**Q: Is data encrypted at rest?**
Yes, both Postgres and any S3-backed workflow attachments are encrypted at
rest. See your workspace's security documentation for the specific
encryption standard in use (not covered in this corpus).

## Troubleshooting Cross-Reference

For error-code-level troubleshooting (e.g. specific `429` or `500`
responses), see `docs/support/troubleshooting.md` and the Error Codes table
in `docs/engineering/api_reference.md`.
