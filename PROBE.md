# PROBE.md — Path B (Git-connected cloud build) validation evidence

**Validated live**: 2026-10-03, account `dollartree1994's team` (credit-free cohort), site `scenecut-bench-lane`.

This closes the "lane-untested" status of Path B in netlify-build-as-compute-kit's `docs/trigger-paths.md` (Q2/Q3/Q4). Every step below was executed via APIs only — no dashboard, no browser.

## The connection recipe (the missing piece)

An API-created site with `repo: {provider, repo, branch, cmd, dir}` does NOT build out of the box:

1. `POST /api/v1/sites` with the repo object → deploys fire but fail: *"preparing repo: Host key verification failed"* (SSH clone, no GitHub App installation, no deploy key).
2. `POST /api/v1/deploy_keys` `{}` → returns `{id, public_key}` (Netlify-generated SSH keypair).
3. Add `public_key` to the GitHub repo as a **deploy key** (read-only): `POST /repos/{owner}/{repo}/keys` — any repo admin PAT works. Public repos too — the clone still goes over SSH.
4. `PATCH /api/v1/sites/{id}` with the same repo object + `"deploy_key_id": <key id>`.
5. `POST /api/v1/sites/{id}/build_hooks {"title": ..., "branch": "main"}`.
6. Fire: `POST https://api.netlify.com/build_hooks/{hook_id} -d "job=..."` (unauthenticated) — the body lands in the build env as `INCOMING_HOOK_BODY` (URL-encoded form data).

After step 4, clones succeed (`commit_ref` populated) and builds run **on Netlify's build machine**.

## Evidence: compute ran on Netlify, not the trigger machine

banner.json served from the deploy URL (`/out/banner.json`, static pickup path):

| Field | Netlify build machine | Trigger sandbox (for contrast) |
|---|---|---|
| hostname | `(none)` | `c-6ac093ea-14810412-1c1e97b12810` |
| cpus | **3** | 2 |
| mem_gb | **8** | 4 |
| deploy_time | 11-17 s per probe build | — |

## pyprobe — the full detection toolchain on the build machine

```
Python 3.12.15 | numpy 2.5.3 | cv2 5.0.0          (pip install: 7 s — faster than the sandbox's 13 s)
decode: 100/100 frames | 3054 fps decode | cut localized 49->50 (GT-exact, bundled ffmpeg)
cpu: 30x 1080p absdiff+mean = 43 ms (695 iter/s)   (sandbox: 811 iter/s — comparable class)
PYPROBE_OK, wall 8 s
```

`python3 -m venv` works; `opencv-python-headless` wheels install clean; the bundled FFmpeg
encodes and decodes; the synthetic hard-cut is localized exactly. **No ffmpeg apt needed.**

## Operational notes

- **Blobs plugin degraded gracefully**: `@netlify/blobs` did not resolve for a local plugin in the cloud-build runtime (dynamic import + fallback — see `plugins/store-data/index.js`). The **static pickup path is the primary**: results are copied to `dist/out/` and served from the deploy URL. On the credit-free cohort this costs only build minutes (300/mo cap), not credits.
- Hook-triggered builds of the default branch report `context=production`; on credit-based accounts use a non-default branch (`trigger_branch`) to stay in the free branch-deploy lane.
- Build logs are dashboard-only — runners self-report: everything the orchestrator needs goes to `/tmp/scenecut-out/{banner.json,result.json,log.txt}` → `dist/out/`.
- Job queueing: the hook body IS the job spec (`job=pyprobe`, `job=sbd&n=200`...). One build slot on free → serial chunks.
- Runner timeout discipline: `execSync` timeout 9 min < build cap; external_bench-style `--time-budget` still applies inside long jobs.
