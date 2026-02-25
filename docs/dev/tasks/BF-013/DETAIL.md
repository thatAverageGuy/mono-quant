# BF-013: Fix CI silently swallowing test failures

## Status
TODO

## Audit Reference
M5 (Medium — masks regressions)

## Problem
In `.github/workflows/ci.yml`, the test step uses:

```yaml
run: pytest tests/ -v || echo "No tests found yet - skipping"
```

The `|| echo` fallback causes the CI step to **always succeed**, even when
pytest finds tests and they fail. A test regression is indistinguishable from
a clean run in CI output. The `echo` message implies no tests exist — this was
a temporary stub from early development that was never removed.

The CI also runs `mypy` with `|| true` (non-blocking), which is intentional
(see commit `b02a2b4`). That is acceptable. The pytest fallback is not.

## Requirements
1. Remove `|| echo "..."` from the pytest command.
2. CI must fail when any test fails.
3. If the tests directory is empty, CI should fail with a clear message
   (not silently succeed) — no tests is a problem, not an acceptable state.

## Decisions
- **Decision:** Remove the `|| echo` suffix. No replacement fallback needed.
  Reason: Tests exist (`tests/test_api.py`). The stub comment is stale.
  A CI that can't fail on test failure provides no safety net.

## Success Criteria
- [ ] `pytest tests/ -v` in CI fails the workflow step when tests fail
- [ ] CI passes cleanly when all tests pass
- [ ] The `|| echo` suffix is removed from the pytest invocation

## Dependencies
- None

## Implementation Guidance

In `.github/workflows/ci.yml`, find the test step:

```yaml
# BEFORE
- name: Run tests
  run: pytest tests/ -v || echo "No tests found yet - skipping"

# AFTER
- name: Run tests
  run: pytest tests/ -v
```

Also verify `--tb=short` or similar flags are present for readable CI output.
Consider adding `--strict-markers` to catch undefined pytest marks.

## Testing Requirements
- Manual: Create a failing test; push to dev branch; verify CI step turns red.
- Manual: With all tests passing, verify CI remains green.

## Open Questions
<!-- MUST be empty before implementation begins -->
