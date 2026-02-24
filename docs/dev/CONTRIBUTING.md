# Contributing: Mono Quant

**Last Updated:** 2026-02-24

---

## Branch Model

```
main    ← production releases only. Version tags here.
dev     ← all development work. PRs merge into main.
```

- All work happens on `dev`
- Never delete `dev` after merge
- Version tags: `vMAJOR.MINOR.PATCH` applied on `main` only

---

## Commit Procedure

Follow this sequence for every commit — no exceptions:

```
1. Verify implementation is complete
2. Run full test suite (must pass, no skips)
3. Update task's IMPL_LOG.md
4. Update docs/dev/tasks/TASKS.md status
5. Update CHANGELOG.md
6. Update CONTEXT.md
7. Update README.md (L2+ only, if user-facing)
8. Update ARCHITECTURE.md / STATE_MACHINES.md (L3 only)
9. Stage ALL changes (code + docs + tests)
10. Present full diff summary
11. Get explicit user approval
12. Single commit with task-prefixed message
13. Push to dev
```

---

## Commit Message Format

```
<TASK-ID>: <Brief summary> - <What was done>
```

Example:
```
T-018: GPTQ export - 4-bit packing matching AutoGPTQ reference format

- Implemented INT4 weight packing into 32-bit integers
- Group size 128 as per GPTQ standard
- Dual-validated against AutoGPTQ and vLLM loaders
- Docs updated: CHANGELOG, CONTEXT, TASKS, IMPL_LOG
```

Rules:
- Imperative mood, under 72 chars on subject line
- Task ID prefix is mandatory
- Body is optional for simple changes

---

## Task ID Prefixes

| Prefix | Meaning |
|--------|---------|
| T-XXX | Main development tasks |
| BF-XXX | Bug fixes |
| CL-XXX | Cleanup / refactoring |
| AX-XXX | Auxiliary (CI, tooling, infra) |

Adding a new prefix: update this table AND `TASKS.md`.

---

## Change Level System

Declare CHANGE LEVEL before all work:

| Level | Scope | Restrictions |
|-------|-------|-------------|
| 0 | Read only | No file modifications |
| 1 | Surgical | Edit existing lines only; max ~20 lines/file |
| 2 | Local | Create/modify within one logical area; stable public interfaces |
| 3 | Structural | Multi-file / architectural; must update ARCHITECTURE.md or create ADR |

Default if not stated: **LEVEL 0**.

---

## Code Conventions

| Convention | Value |
|-----------|-------|
| Language | Python 3.11+ |
| Style | snake_case for functions/variables, PascalCase for classes |
| Line length | 100 characters (ruff enforced) |
| Linter | `ruff` (`E`, `F`, `I`, `N`, `W` rules) |
| Type checker | `mypy` (non-blocking in CI, enforce locally) |
| String quotes | Double quotes preferred |
| Imports | Sorted (ruff `I`) |

### Ruff Ignores (from pyproject.toml)

```
E501  — line too long (handled by 100-char rule)
N803  — arg name lowercase (allows P, Q for probability distributions)
N806  — var lowercase in function (allows P, Q)
N812  — lowercase imported as non-lowercase (allows `F` for torch.nn.functional)
```

### Import Discipline

- Public API imports go in `src/mono_quant/__init__.py`
- Avoid circular imports — use **local imports** (inside functions) if unavoidable
  - Pattern established: `core.quantizers` imports `modules.linear` locally
- Optional dependencies (`onnx`, `onnxruntime`) must use lazy imports:
  ```python
  def export_to_onnx(...):
      try:
          import onnx
      except ImportError:
          raise ImportError("Install mono-quant[onnx]")
  ```

---

## Testing

### Principles

- Tests prove real behavior, not theory
- No mocking of internal code — mock only external deps (file I/O, optional imports)
- Integration tests are first-class
- Every new feature needs tests before the task is considered DONE

### Structure

```
tests/
├── test_api.py            # End-to-end API flows
├── test_onnx_export.py    # ONNX export (23 tests)
└── [test_gptq_export.py]  # Future: Phase 6
```

### Running Tests

```bash
pytest tests/                      # all tests
pytest tests/test_api.py -v        # specific file, verbose
pytest --cov=mono_quant tests/     # with coverage
```

### Naming Convention

```
test_<unit>_<scenario>_<expected_outcome>

Examples:
  test_quantize_dynamic_int8_returns_result
  test_onnx_export_int4_produces_qdq_nodes
  test_static_quantize_low_sqnr_raises_warning
```

### Coverage

Report coverage in IMPL_LOG.md for each task. No coverage-gaming with trivial tests.
Target: all critical paths and edge cases covered.

---

## Documentation Rules by Change Level

| Document | L1 | L2 | L3 |
|----------|----|----|-----|
| IMPL_LOG.md | Always | Always | Always |
| TASKS.md | Always | Always | Always |
| CHANGELOG.md | Always | Always | Always |
| CONTEXT.md | Always | Always | Always |
| README.md | No | If user-facing | Always |
| ARCHITECTURE.md | No | No | Always |
| STATE_MACHINES.md | No | If flow changed | Always |
| ADR | No | No | If new decision |

---

## ADR Process

Create an ADR when:
- A significant architectural decision is made
- A technology or pattern is chosen over alternatives
- A constraint is formalized

ADR lives at: `docs/dev/adr/ADR-XXX.md`

Format: see `docs/dev/adr/ADR-001.md` as template.

---

## CI/CD

GitHub Actions at `.github/workflows/ci.yml`.

```
On push/PR to dev or main:
  1. Install dependencies
  2. Run ruff (linting)
  3. Run mypy (non-blocking)
  4. Run pytest
```

CI must be green before merging to main.

---

## Release Process

1. All planned tasks for milestone are DONE in TASKS.md
2. Tests pass on `dev`
3. CHANGELOG.md updated with release version and date
4. PR from `dev` → `main`
5. Merge
6. Tag: `git tag vMAJOR.MINOR.PATCH` on main
7. Publish to PyPI: `python -m build && twine upload dist/*`
8. Update `pyproject.toml` version for next cycle

---

*Contributing guide for: Mono Quant*
*Created: 2026-02-24 during docs/dev/ bootstrap*
