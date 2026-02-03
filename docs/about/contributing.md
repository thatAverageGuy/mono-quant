# Contributing

Contributions are welcome! Here's how to get started.

## Development Setup

1. **Fork and clone:**

```bash
git clone https://github.com/YOUR_USERNAME/mono-quant.git
cd mono-quant
```

2. **Install in editable mode:**

```bash
pip install -e ".[dev]"
```

3. **Run tests:**

```bash
pytest
```

4. **Build documentation:**

```bash
mkdocs serve
```

Visit `http://localhost:8000` to preview docs.

## Code Style

- Follow PEP 8 guidelines
- Use type hints where appropriate
- Add docstrings (Google style) to public functions
- Keep functions focused and small

## Running Tests

```bash
# Run all tests
pytest

# Run with coverage
pytest --cov=mono_quant --cov-report=html

# Run specific test file
pytest tests/test_quantizers.py

# Run specific test
pytest tests/test_quantizers.py::test_dynamic_quantize
```

## Building Documentation

```bash
# Serve locally (live reload)
mkdocs serve

# Build static site
mkdocs build

# Deploy to GitHub Pages
mkdocs gh-deploy
```

## Submitting Changes

1. Create a branch: `git checkout -b feature/your-feature`
2. Make changes and commit
3. Push to fork: `git push origin feature/your-feature`
4. Create pull request

### Pull Request Guidelines

- **Title:** Use conventional commits (`feat:`, `fix:`, `docs:`, etc.)
- **Description:** Explain what and why
- **Tests:** Include tests for new features
- **Docs:** Update documentation for API changes

## Areas for Contribution

### High Priority

- [ ] Additional observer types (Entropy, KL-divergence variants)
- [ ] ONNX export support
- [ ] More calibration strategies
- [ ] Benchmarks and performance comparisons

### Medium Priority

- [ ] TFLite export support
- [ ] Model zoo integration examples
- [ ] Jupyter notebook tutorials
- [ ] Video tutorials

### Low Priority

- [ ] Quantization-aware training (QAT)
- [ ] Automatic layer sensitivity analysis
- [ ] Mixed precision quantization
- [ ] LLM.int8() style outlier detection

## Questions?

Open an issue on GitHub for:
- Bug reports
- Feature requests
- Documentation improvements
- General questions

## Code of Conduct

Be respectful, constructive, and inclusive. We're all here to build something great together.
