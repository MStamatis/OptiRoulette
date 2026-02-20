# Release and Publish Guide (Maintainers)

This document is for package maintainers only.

## Recommended: GitHub Actions + PyPI Trusted Publisher (OIDC)

1. Create a GitHub repo for this package (if not already).
2. Add publishing workflow in `.github/workflows/publish.yml`.
3. In PyPI, add a Trusted Publisher for that repo/workflow.
4. Create a release tag (for example `v0.1.1`) and push it.
5. GitHub Actions builds and uploads to PyPI without API tokens.

## Manual Publish (Fallback)

From the package repository root (`OptiRoulette`):

```bash
python -m pip install --upgrade build twine
python -m build
python -m twine check dist/*
python -m twine upload dist/*
```

Optional test upload first:

```bash
python -m twine upload --repository testpypi dist/*
```

## Version Bump Checklist

1. Update `version` in `pyproject.toml`.
2. Build and check artifacts.
3. Publish (OIDC workflow or manual).
4. Verify install:

```bash
pip install -U OptiRoulette
```
