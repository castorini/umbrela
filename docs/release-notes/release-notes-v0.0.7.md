# umbrela v0.0.7

Initial release-note scaffold for the packaged CLI and prompt-template workflow.

## Included In This Baseline

- Packaged `umbrela` CLI for direct judging, qrel-backed evaluation, schema inspection, validation, doctor, and view commands.
- FastAPI `umbrela serve` command exposing `GET /healthz` and `POST /v1/judge` on port `8086` by default.
- Direct `judge` input now also accepts Anserini REST candidates where `candidates[].doc` is a plain string, so Anserini search results can be piped directly into `POST /v1/judge` without a `jq` reshape step.
- YAML-backed prompt templates with the legacy flat-prompt rendering contract preserved.
- Offline-first contributor workflow built around `uv`.

## Migration Notes

This baseline establishes the release-note policy for future changes.

Document a migration note whenever a change affects:

- prompt semantics or rendered prompt structure
- parsing heuristics or judgment extraction
- backend defaults or provider selection behavior
- qrel, evaluation, or artifact output formats
