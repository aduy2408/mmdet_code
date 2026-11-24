# Marimo connection

Use `docs/local_github_marimo_workflow.md` and `utils/marimo_ops.py` for preflight, launch, monitoring, recovery, artifact checks, and upload verification. The expected runtime is `/marimo/mmdet-venv/bin/python`.

Keep the workflow CLI/client code in the submission branch:

- `utils/marimo_ops.py`: MMDetection preflight, launch, status, artifacts and upload gates.
- `utils/marimo_client.py`: authenticated live-kernel HTTP client.
- `utils/marimo_run.py`: compatibility CLI for `-c/--code`, `-f/--file`, and `--set-config`.
- `docs/local_github_marimo_workflow.md`: exact-SHA sync and launch workflow.

Typical live-kernel commands:

```bash
python -m utils.marimo_ops preflight --help
python -m utils.marimo_ops launch --help
python utils/marimo_run.py -c 'print("connected")'
```

`HF_TOKEN` comes only from the live Marimo global kernel. Never print or store it. Keep split seed `42` fixed while changing training seeds, and use explicit model-to-config mappings rather than checkpoint names.
