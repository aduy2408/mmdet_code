# MMDetection agent instructions

Read `/mnt/data/varroa/AGENTS.md` first. This file adds MMDetection-specific rules.

## Canonical execution

- Do not launch training from the local machine for Marimo requests.
- Use `/marimo/mmdet-venv/bin/python` on Marimo.
- Use the repository's `utils.marimo_ops` / `marimo-pair` workflow for preflight, launch, status, recovery, and artifact verification. Do not use browser automation or ad-hoc `nohup`/`Popen` training launches.
- `HF_TOKEN` comes from the live Marimo global kernel namespace. Never print, store, or pass it in a command-line argument. Pass it only through the detached child environment.
- If upload is required, fail before training when auth, the target repo, or the upload verifier is missing. Never add `--no-hf-upload` or silently skip upload.

- MMDetection work normally starts from `/mnt/data/varroa` so `yolo_related` is available as a reference corpus. Do not modify YOLO files during an MMDetection task unless the request explicitly names a cross-project change.
- If a YOLO implementation is used as a reference, record the source path/commit and the MMDetection-specific adaptation. Do not infer MMDetection behavior, pretrained settings, or configs from YOLO filenames.

## Matrix and provenance

- Current requested baseline matrix: `DETR` and `RTMDet` on both `LEVIR-Ship` and `TinyPerson`, training seeds `42, 43, 44`, fixed split seed `42`.
- Canonical result repositories are:
  - `duyle2408/levir_ship_mmdet_runs_seed42`
  - `duyle2408/levir_ship_mmdet_runs_seed43`
  - `duyle2408/levir_ship_mmdet_runs_seed44`
  - `duyle2408/tinyperson_mmdet_runs_seed42`
  - `duyle2408/tinyperson_mmdet_runs_seed43`
  - `duyle2408/tinyperson_mmdet_runs_seed44`
- Resolve model names through the explicit model-to-config mapping. Never infer a model, backbone, pretrained source, dataset root, or result location from a filename or checkpoint name.
- Keep baseline and modified configs separate. Every run manifest must record canonical config, patched config or explicit change, source commit, dataset root, split seed, training seed, model/backbone/pretrained source, training settings, and HF target.
- A run is complete only after training, evaluation, local artifact checks, HF upload, and remote-path verification. A PID, checkpoint, local marker, or upload API return alone is not completion evidence.

## Recovery after compaction

Before resuming a compacted session, reread this file, `/mnt/data/varroa/AGENTS.md`, the relevant workflow, and the current run manifest/state. State the active dataset, model, config, seeds, auth source, HF target, and run state from those files. If any value is unknown, report `unknown` instead of guessing.
