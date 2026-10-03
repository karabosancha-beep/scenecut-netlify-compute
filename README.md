# SceneCut Netlify Compute Lane

This repo is the **cloud-build compute kit** for [SceneCut](https://github.com/karabosancha-beep/scenecut) (private main repo).
It is connected to a Netlify free-tier site; every build-hook fire runs a **bench chunk on Netlify's build machine**
(0 credits — build minutes are not metered on free plans; draft/branch deploys are free).

This validates "Path B: Git-connected cloud build" from the netlify-free-tier-maxxing kit (DOCUMENTED, lane-untested → now tested).

## Layout
- `netlify.toml` — build command + plugin wiring
- `build.mjs` — entry: reads `INCOMING_HOOK_BODY` (job spec), dispatches to a runner
- `runners/` — bench chunk runners (self-reported logs + results to `/tmp/scenecut-out/`)
- `plugins/store-data/` — onPostBuild plugin: egress `/tmp/scenecut-out/` → Netlify Blobs
- `scenecut_py/` — snapshot of the Python package (synced from the main repo; no secrets needed)
- `PROBE.md` — the Path-B validation evidence

## Firing a chunk
```bash
curl -X POST "https://api.netlify.com/build_hooks/<HOOK_ID>" \
  -d 'job=probe'   # → INCOMING_HOOK_BODY in the build env
```
Results: Blobs store `site:scenecut-bench`, key `runs/latest` (pointer) + `runs/raw-<ts>.jsonl`.

## Sync
From the main repo: `scripts/sync_netlify_compute.sh` refreshes the `scenecut_py` snapshot and pushes.
