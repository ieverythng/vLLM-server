# Future Package Handoff

This pass keeps the repo simple: top-level `server.py`, `vllm_manager.py`, scripts, docs, and tests.

For a later full package conversion, implement:

- Move runtime code to `src/vllm_gateway/`.
- Split config/profile loading into a module such as `vllm_gateway.config`.
- Split FastAPI app creation into an app factory so tests can inject config and backend URLs.
- Add console entry points:
  - `vllm-gateway`
  - `vllm-manager`
  - `vllm-benchmark`
- Move scripts that remain operational wrappers into `scripts/`.
- Convert `config.yaml` and `models.yaml` into documented example files, then load project-local or environment-selected config paths.
- Add CI jobs for unit tests, linting, dry-run config validation, and optional GPU self-hosted validation.
- Publish a versioned package only after the Windows runtime and first Qwen3.6 profile are stable.
