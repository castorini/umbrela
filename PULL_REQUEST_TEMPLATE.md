# Summary

Describe the change in 2-5 sentences. Be explicit about whether it affects CLI behavior, prompt semantics, judgment parsing, qrel evaluation, or only internal implementation details.

## Related Issue

Reference the GitHub issue this pull request addresses. If none exists, write `N/A`.

## What Changed

- 

## Validation

- [ ] `bash scripts/quality_gate.sh commit`
- [ ] `bash scripts/quality_gate.sh push`
- [ ] `uv run pytest -q`
- [ ] Not run, with justification explained below

List any additional commands or manual checks used:

```text
paste commands and outcomes here
```

## Checklist

- [ ] I followed [CONTRIBUTING.md](CONTRIBUTING.md).
- [ ] I updated relevant documentation, examples, or help text for user-facing changes.
- [ ] I added or updated tests for non-trivial behavior changes, or explained why not.
- [ ] I called out any CLI, prompt, schema, artifact, dependency, or performance impact.

## Type of Change

- [ ] Bug fix
- [ ] Feature
- [ ] Refactor
- [ ] Documentation
- [ ] Maintenance
