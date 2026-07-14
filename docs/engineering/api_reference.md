# API Reference

Nimbus is a fictional workflow-automation product. This document describes the
public REST API exposed at `https://api.nimbus.dev/v1`.

## Authentication

Nimbus supports two authentication methods.

### API Keys

Generate a key from **Settings → API Keys**. Send it in the `Authorization`
header as a Bearer token:

```bash
curl https://api.nimbus.dev/v1/workflows \
  -H "Authorization: Bearer nb_live_xxxxxxxxxxxx"
```

API keys are scoped to a workspace and inherit the permissions of the user
who created them. Keys prefixed `nb_test_` hit a sandboxed environment with
no billing impact.

### OAuth 2.0

For third-party integrations, use the Authorization Code flow against
`https://api.nimbus.dev/oauth/authorize`. Nimbus issues access tokens valid
for 1 hour and refresh tokens valid for 30 days. Requested scopes must be one
or more of: `workflows:read`, `workflows:write`, `runs:read`, `webhooks:manage`.

## Rate Limiting

All endpoints are rate limited per API key:

- **100 requests/minute** (burst)
- **1,000 requests/hour** (sustained)

Responses include `X-RateLimit-Limit`, `X-RateLimit-Remaining`, and
`X-RateLimit-Reset` headers. Exceeding either limit returns `429 Too Many
Requests` with a `Retry-After` header in seconds. Enterprise plans can
request higher limits via support.

## Error Codes

| Code | HTTP Status | Meaning |
|------|-------------|---------|
| `AUTH_INVALID_KEY` | 401 | The API key is malformed or revoked. |
| `AUTH_EXPIRED_TOKEN` | 401 | OAuth access token has expired; use the refresh token. |
| `AUTH_INSUFFICIENT_SCOPE` | 403 | Token doesn't have the required OAuth scope. |
| `RATE_LIMITED` | 429 | Per-minute or per-hour limit exceeded. |
| `VALIDATION_ERROR` | 422 | Request body failed schema validation; see `details`. |
| `WORKFLOW_NOT_FOUND` | 404 | No workflow exists with the given ID in this workspace. |
| `WORKFLOW_LOCKED` | 409 | Workflow is currently running and can't be edited. |
| `INTERNAL_ERROR` | 500 | Unexpected server error; safe to retry with backoff. |

## Endpoints

### `GET /v1/workflows`

List workflows in the current workspace.

Query params: `limit` (default 20, max 100), `cursor` (pagination token),
`status` (`active` | `paused` | `archived`).

```json
{
  "data": [
    {"id": "wf_1a2b3c", "name": "Invoice Sync", "status": "active", "created_at": "2026-01-04T10:00:00Z"}
  ],
  "next_cursor": "eyJvZmZzZXQiOjIwfQ=="
}
```

### `POST /v1/workflows`

Create a workflow.

```json
{
  "name": "Invoice Sync",
  "trigger": {"type": "schedule", "cron": "0 * * * *"},
  "steps": [{"type": "http_request", "config": {"url": "https://example.com/sync"}}]
}
```

### `GET /v1/workflows/{workflow_id}`

Fetch a single workflow, including its full step configuration and last run status.

### `PATCH /v1/workflows/{workflow_id}`

Partial update. Returns `WORKFLOW_LOCKED` if the workflow has a run in progress.

### `DELETE /v1/workflows/{workflow_id}`

Archives the workflow (soft delete). Archived workflows are purged after 90 days.

### `POST /v1/workflows/{workflow_id}/run`

Trigger an immediate run, bypassing the configured trigger.

```python
import requests

resp = requests.post(
    "https://api.nimbus.dev/v1/workflows/wf_1a2b3c/run",
    headers={"Authorization": "Bearer nb_live_xxxxxxxxxxxx"},
    json={"input": {"invoice_id": "INV-2044"}},
)
print(resp.json())
```

### `GET /v1/runs/{run_id}`

Fetch run status: `queued`, `running`, `succeeded`, `failed`, or `cancelled`.
Includes per-step logs and duration in `duration_ms`.

### `POST /v1/webhooks`

Register a webhook. Nimbus signs payloads with HMAC-SHA256 using your
webhook secret; verify via the `X-Nimbus-Signature` header.

```json
{
  "url": "https://example.com/hooks/nimbus",
  "events": ["run.succeeded", "run.failed"]
}
```

### `GET /v1/usage`

Returns current-period request counts, run counts, and remaining quota for
the workspace's plan tier.
