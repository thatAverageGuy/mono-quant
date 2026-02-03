# CI/CD Pipeline

Mono Quant uses GitHub Actions for continuous integration and deployment.

## Workflow

The CI/CD pipeline runs on:
- **Push** to `main` or `dev` branches
- **Pull requests** to `main` or `dev` branches
- **Releases** (triggers PyPI publishing)

## Jobs

### 1. Test
Runs on multiple OS and Python versions:
- **OS:** Ubuntu, Windows, macOS
- **Python:** 3.11, 3.12

**Steps:**
- Check out code
- Set up Python
- Install dependencies
- Run tests with pytest
- Lint with ruff
- Type check with mypy

### 2. Build
Builds the distribution packages:
- Source distribution (.tar.gz)
- Wheel distribution (.whl)
- Checks package metadata with twine

### 3. Publish to PyPI
Automatically publishes to PyPI when a release is created:
- Uses trusted publishing (PyPI API token)
- Only runs on release events
- Uploads both wheel and source distribution

## Setup

### Required Secrets

For the CI/CD pipeline to work, you need to configure these secrets in your GitHub repository:

#### 1. PYPI_API_TOKEN (for publishing)

**To create:**
1. Visit: https://pypi.org/manage/account/token/
2. Create a new token
3. Scope: "Entire account"
4. Copy the token

**To add to GitHub:**
1. Go to: https://github.com/thatAverageGuy/mono-quant/settings/secrets/actions
2. Click "New repository secret"
3. Name: `PYPI_API_TOKEN`
4. Paste the token
5. Click "Add secret"

### Required GitHub Environments

**PyPI Environment:**
1. Go to: https://github.com/thatAverageGuy/mono-quant/settings/environments
2. Create environment named `pypi`
3. Add protection rule (recommended):
   - Required reviewers: (you or your team)
   - Wait timer: 0 (for automated publishing)
4. Add environment secret: `PYPI_API_TOKEN`

## Badge

Add this to your README.md:

```markdown
[![CI/CD](https://github.com/thatAverageGuy/mono-quant/actions/workflows/ci.yml/badge.svg)](https://github.com/thatAverageGuy/mono-quant/actions/workflows/ci.yml)
```

## Local Testing

To test the CI pipeline locally:

```bash
# Install test dependencies
pip install -e ".[dev]"

# Run tests
pytest tests/ -v

# Run linting
ruff check .

# Run type checking
mypy src/
```

## Triggering Workflows

### On Push/PR
Automatically runs tests, linting, and type checking.

### On Release
To trigger a PyPI publish:

1. Update version in `pyproject.toml` and `src/mono_quant/__init__.py`
2. Commit changes: `git commit -m "bump: version X.Y.Z"`
3. Push: `git push`
4. Create release on GitHub:
   ```bash
   gh release create X.Y.Z --title "vX.Y.Z" --notes "Release notes"
   ```
5. The CI pipeline will automatically publish to PyPI

## Troubleshooting

### Tests Failing
- Check test output in Actions tab
- Run tests locally with same Python version
- Ensure all dependencies are installed

### Type Checking Failures
- Run `mypy src/` locally to see issues
- Add type hints to fix issues
- Or adjust mypy configuration in `pyproject.toml`

### Linting Failures
- Run `ruff check .` to see issues
- Run `ruff check --fix .` to auto-fix
- See ruff docs: https://docs.astral.sh/ruff/

### PyPI Publishing Fails
- Check that `PYPI_API_TOKEN` secret is set correctly
- Verify the token has "Entire account" scope
- Check PyPI environment is configured
- Ensure version number hasn't been published already
