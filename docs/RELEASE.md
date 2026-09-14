# Release checklist (v0.1.6)

## Before tagging

1. Merge PR [#25](https://github.com/TimeCopilot/foundationforecast/pull/25) (`feat/tabpfn-ts-3`) into `main`.
2. Confirm CI is green on `main`.
3. Verify version in `pyproject.toml` is `0.1.6` and `docs/changelogs/v0.1.6.md` is listed in `mkdocs.yml`.

## Publish to PyPI

```bash
git checkout main && git pull
git tag v0.1.6
git push origin v0.1.6
```

The [Release workflow](.github/workflows/release.yaml) publishes automatically on tag push.

## After publish

1. In **timecopilot**, remove `[tool.uv.sources]` for `foundationforecast` from `pyproject.toml`.
2. Run `uv lock --upgrade-package foundationforecast` and commit the lock file.
3. Tag and publish timecopilot `v0.0.33`.
