# umbrela v0.0.7

Initial release-note scaffold for the packaged CLI and prompt-template workflow.

## Included In This Baseline

- Packaged `umbrela` CLI for direct judging, qrel-backed evaluation, schema inspection, validation, doctor, and view commands.
- FastAPI `umbrela serve` command exposing `GET /healthz` and `POST /v1/judge` on port `8084` by default.
- Direct `judge` input now also accepts Anserini REST candidates where `candidates[].doc` is a plain string, so Anserini search results can be piped directly into `POST /v1/judge` without a `jq` reshape step.
- Direct `judge` input now also accepts single-record `castorini.cli.v1` envelopes from upstream tools such as `rank_llm`, so `search | rerank | judge` can be piped through `POST /v1/judge` without unwrapping `.artifacts[0].value[0]` first.
- Direct `umbrela judge --output json` and `POST /v1/judge` responses are now compact by default. The default judgment records include `query`, `passage`, and `judgment`, with `reasoning` remaining opt-in via `--include-reasoning`. The older verbose fields `prediction`, `result_status`, and `prompt` are now exposed only through the new `--include-trace` flag, with `--redact-prompts` available to suppress raw prompt text.
- YAML-backed prompt templates with the legacy flat-prompt rendering contract preserved.
- Offline-first contributor workflow built around `uv`.

## Migration Notes

This baseline establishes the release-note policy for future changes.

Document a migration note whenever a change affects:

- prompt semantics or rendered prompt structure
- parsing heuristics or judgment extraction
- backend defaults or provider selection behavior
- qrel, evaluation, or artifact output formats
