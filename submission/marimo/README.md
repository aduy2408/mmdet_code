# Marimo connection

Use `docs/local_github_marimo_workflow.md` and `utils/marimo_ops.py` for preflight, launch, monitoring, recovery, artifact checks, and upload verification. The expected runtime is `/marimo/mmdet-venv/bin/python`.

`HF_TOKEN` comes only from the live Marimo global kernel. Never print or store it. Keep split seed `42` fixed while changing training seeds, and use explicit model-to-config mappings rather than checkpoint names.
