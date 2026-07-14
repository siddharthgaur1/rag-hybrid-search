# Changelog

## v2.3.0 — 2026-06-18

- Added Approval Gates (Enterprise).
- `POST /v1/workflows/{id}/run` now accepts an `input` object passed into
  the run context as `trigger.input`.
- Fixed a bug where `retention_days: -1` (indefinite) on Enterprise was
  silently capped at 365 days.

## v2.2.0 — 2026-05-02

- Added per-step Retry Policies (Enterprise).
- **Breaking change**: the default retry backoff changed from `fixed` to
  `exponential` for all new workflows. Existing workflows keep their
  previously configured behavior; this only affects workflows created after
  v2.2.0 that don't explicitly set a retry policy.

## v2.1.0 — 2026-03-14

- Conditional Branching graduated from beta to general availability on
  Team and Enterprise tiers.
- Increased default rate limit from 60 req/min to 100 req/min for all tiers.

## v2.0.0 — 2026-01-20

- **Breaking change**: `GET /v1/workflows` pagination changed from
  offset-based (`?offset=20&limit=20`) to cursor-based (`?cursor=...`).
  Offset-based pagination is no longer accepted; requests using `offset`
  now return `VALIDATION_ERROR`.
  **Migration**: replace `offset` param usage with the `next_cursor` value
  returned in the previous page's response. There is no direct offset ->
  cursor mapping, so jump-to-page navigation is no longer supported —
  paginate forward from the first page.
- **Breaking change**: webhook payloads now include a `signature_version: 2`
  field, and the HMAC signature covers the raw request body plus a
  timestamp (previously body only), to prevent replay attacks.
  **Migration**: update webhook receivers to verify against
  `X-Nimbus-Signature` using the v2 scheme documented in the API reference,
  or requests will fail verification (Nimbus does not fall back to v1
  signing).
- Extended Run History Retention became configurable per-workspace instead
  of a fixed 90-day window for all plans.

## v1.4.0 — 2025-12-01

- Added `GET /v1/usage` endpoint for programmatic quota checks.
- Webhook delivery timeout is now configurable via `WEBHOOK_TIMEOUT_SECONDS`
  (previously hardcoded to 10 seconds — the new default matches the old
  hardcoded value, so no behavior change unless explicitly configured).

## v1.3.0 — 2025-11-05

- Added OAuth 2.0 support alongside existing API key authentication.
- Scheduled Triggers now support IANA timezones instead of UTC-only.

## v1.2.0 — 2025-10-01

- Introduced Conditional Branching in beta (Team, Enterprise; opt-in via
  feature flag).
- Fixed a race condition where two scheduler replicas could both fire the
  same cron trigger during a leader election handoff.

## v1.1.0 — 2025-09-03

- Added webhook triggers.
- Rate limit responses now include `Retry-After` header.

## v1.0.0 — 2025-08-01

- Initial public release: scheduled triggers, HTTP request steps, run
  history (fixed 90-day retention), API key authentication.
