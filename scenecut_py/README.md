# SceneCut

A CLI-first (with Web UI companion) tool for detecting shot/scene transitions in MP4 videos, marking them, optionally cutting into clips, and exporting cut lists to NLE interchange formats.

**Status:** v1.0 — implemented and tested (34 tests passing).

---

## Installation

**From PyPI** (recommended):

```bash
pip install scenecut

# With the optional TransNetV2 neural detection pass:
pip install "scenecut[neural]"
```

> **CPU-only PyTorch note (Linux):** the `[neural]` extra resolves `torch` to
> the CUDA build (~2–3 GB). For CPU-only machines, install torch from the CPU
> wheel index *first*, then the extra (a per-extra `--index-url` cannot be
> expressed in PEP 621 — see `docs/pypi.md`):
>
> ```bash
> pip install torch --index-url https://download.pytorch.org/whl/cpu
> pip install "scenecut[neural]"
> ```

**From source** (editable install, for development):

```bash
cd scenecut_py
pip install -e .
```

Both install paths provide the `scenecut` console command and
`python3 -m scenecut` (same entry point). ffmpeg/ffprobe must be on `PATH` for
cutting, export metadata, preprocessing and thumbnails.

---

## Quick start

### CLI

```bash
# Detect scenes on a video
scenecut detect video.mp4 --output scenes.json --thumbnails

# List detected scenes
scenecut list scenes.json

# Cut into per-scene clips
scenecut cut video.mp4 --scenes scenes.json --outdir clips/ --accurate

# Export to NLE format
scenecut export scenes.json --format fcpxml --output cut.fcpxml

# Bundle all clips as tar
scenecut tar video.mp4 --scenes scenes.json --output clips.tar --gz
```

### Web UI

The Next.js dev server is already running on port 3000. The Web UI provides:

- **Source** tab: upload MP4/MOV/MKV, or fetch from YouTube/Vimeo URL
- **Detect** tab: tune algorithm + thresholds, run multi-pass detection
- **Scenes** tab: timeline with cut markers, scene grid with thumbnails, inline video preview per scene, manual add/remove/move cuts
- **Export** tab: cut into clips, download as FCPXML/EDL/Premiere XML/CSV/SRT, or tar bundle

A demo project (5 scenes with hard cuts + a fade) is pre-loaded.

---

## Architecture

```
scenecut_py/         Python CLI package — canonical implementation
  scenecut/
    cli.py           typer-based CLI
    detect.py        multi-pass detection (adaptive + threshold + hash + dissolve)
    cut.py           ffmpeg-based cutting (stream copy or accurate re-encode)
    export/          FCPXML 1.9, EDL CMX 3600, Premiere xmeml, CSV, SRT
    project.py       scene JSON load/save + override application
    timecode.py      HH:MM:SS.mmm <-> HH:MM:SS:FF conversions
    download.py      yt-dlp wrapper (auto-detects JS runtime)
    preprocess.py    ffmpeg downscale for fast proxy detection
    thumbnails.py    JPEG extraction per scene
    compose.py       pick subset of scenes and concatenate
    audit.py         interactive REPL for manual cut editing
  tests/             34 tests (unit + integration + CLI E2E)
  samples/           test videos + demo.mp4
  scripts/make_demo.py

src/                 Next.js 16 web UI
  app/page.tsx       single-page tabbed UI
  app/api/scenecut/  API routes wrapping the Python CLI
  lib/scenecut.ts    subprocess wrapper + types

upload/projects/     per-project working dir (source video, clips, thumbs, project.json)
download/            user-facing deliverables (spec, README, screenshots)
```

The Web UI is a thin wrapper — every UI feature has a 1:1 CLI command.

---

## Detection strategy

Multi-pass by default (`--algo all`):

1. **AdaptiveDetector** (PySceneDetect) — hard cuts + fast dissolves. Uses a rolling-window content-delta so the threshold scales with local content variance. Tunable: `--threshold` (default 27.0).
2. **ThresholdDetector** (PySceneDetect) — fades to/from black. Catches slow fades that Adaptive misses because per-frame delta stays below threshold. Tunable: `--fade-threshold` (default 12 luma).
3. **HashDetector** (PySceneDetect) — perceptual-hash-based cut detection. Catches fast cuts Adaptive might miss on synthetic content.
4. **Dissolve sweep** (custom) — scans rolling HSV-histogram correlation. Flags windows where colorimetry drifts continuously without a hard cut. Tunable: `--dissolve-window` (12 frames), `--dissolve-threshold` (0.6).
5. **Dedup** — collides cuts within `--min-scene-len` (default 0.6s). When sources collide:
   - Source priority: implicit > adaptive > hash > threshold > dissolve
   - Type re-classification: if `threshold` agreed, type = `fade`; else if `dissolve` agreed, type = `dissolve`; else `cut`
   - All agreeing sources recorded in `cuts[].sources` for transparency.

Each cut carries: `frame_num`, `timecode` (HH:MM:SS.mmm), `seconds`, `type`, `confidence` (0..1), `source` (winning detector), `sources` (all that agreed), `manual` (if user-added).

### Fade handling

The `threshold` detector fires on luminance crossings — this is unambiguous for fades to/from black. When a fade passes through black, both ThresholdDetector and HashDetector fire on the same boundary; the dedup picks the higher-priority source (hash) but preserves `type="fade"` because threshold agreed.

### Dissolve handling

The custom dissolve pass requires:
- Full-window HSV-hist correlation ≥ `--dissolve-threshold`
- Max per-frame delta < 0.35 (no hard cut inside the window)
- Average per-frame delta between 0.01 and 0.10 (genuine change, not static)
- End-of-window correlation > start (continuous drift)

This rejects false positives on synthetic content with constantly-changing patterns (e.g. mandelbrot).

### Slow fades below sensitivity

A slow fade where each individual frame delta stays below the threshold is caught by the ThresholdDetector because it tracks the absolute luminance level (not deltas). As long as the fade passes through the threshold value (default 12, near-black), it will be detected.

---

## Scene file format (`scenes.json`)

```json
{
  "version": "1.0",
  "id": "a9401a90a00d",
  "source": {
    "path": "/abs/path/video.mp4",
    "duration": 20.0,
    "fps": 30.0,
    "width": 640, "height": 360,
    "codec": "h264", "audio_codec": null
  },
  "detector": {"algo": "all", "threshold": 27.0, ...},
  "cuts": [
    {"index": 0, "frame_num": 0, "seconds": 0.0, "timecode": "00:00:00.000",
     "type": "cut", "confidence": 1.0, "source": "implicit"},
    {"index": 1, "frame_num": 120, "seconds": 4.0, "timecode": "00:00:04.000",
     "type": "cut", "confidence": 0.9, "source": "adaptive",
     "sources": ["adaptive", "hash"]}
  ],
  "scenes": [
    {"index": 0, "start": 0.0, "end": 4.0, "duration": 4.0,
     "start_frame": 0, "end_frame": 120,
     "thumbnail": "thumbs/scene_000.jpg", "label": null, "tags": [],
     "type": "cut"}
  ],
  "overrides": {
    "added_cuts": [],
    "removed_cuts": [],
    "moved_cuts": {},
    "merged_scenes": [],
    "split_scenes": []
  },
  "labels": {}, "tags": {}
}
```

The `overrides` block records manual edits. The original `cuts` array is preserved; the effective cut list is computed by `apply_overrides()` (used by `cut`, `export`, `tar`, `compose`, and the Web UI).

---

## CLI reference

```bash
scenecut --help                    # show all commands
scenecut version                   # 1.0.0
scenecut info <input> [--json]     # video metadata via ffprobe

scenecut download <url>            # yt-dlp wrapper
  [--format "best[height<=720]"]
  [--cookies-from-browser chrome]  # for YouTube auth
  [--output PATH]

scenecut preprocess <input>        # downscale to proxy for fast detection
  [--height 360] [--crf 28] [--preset veryfast]
  [--fps keep] [--output PATH]

scenecut detect <input>            # multi-pass scene detection
  [--algo adaptive|content|threshold|hash|all]
  [--threshold 27.0] [--min-scene-len 0.6]
  [--fade-threshold 12] [--fade-min-len 0.3]
  [--dissolve-window 12] [--dissolve-threshold 0.6]
  [--group-scenes] [--group-threshold 0.6]
  [--thumbnails] [--thumb-size 320]
  [--proxy PATH] [--output scenes.json]

scenecut list <scenes.json>        # pretty-print scenes
  [--format table|csv|json]

scenecut cut <input> --scenes <scenes.json>
  [--outdir clips/] [--accurate] [--format mp4|mov|mkv]
  [--crf 20] [--preset veryfast] [--keep-audio/--no-audio]

scenecut thumbnails <input> --scenes <scenes.json>
  [--outdir thumbs/] [--positions start|mid|both]
  [--size 320] [--format jpg]

scenecut export <scenes.json>
  [--format fcpxml|edl|premiere|csv|srt]
  [--output PATH] [--source PATH] [--reel-name SCENE] [--fps N]

scenecut tar <input> --scenes <scenes.json>
  [--output clips.tar] [--gz] [--accurate] [--format mp4]

scenecut compose <input> --scenes <scenes.json>
  [--pick "1,3,5-7"] [--output composed.mp4] [--accurate]

scenecut audit <scenes.json> [--input VIDEO]
  # Interactive REPL: list / add / del / move / merge / split / label / tags / save

scenecut project create <input> [--name NAME]
scenecut project list
scenecut project show <id>
scenecut project delete <id>

scenecut web [--port 3000] [--open]
```

All commands accept `--json` for machine-readable output.

### Exit codes

| Code | Meaning |
|------|---------|
| 0 | Success |
| 2 | Input missing or invalid |
| 3 | External tool failure (ffmpeg / yt-dlp) |
| 4 | Detection error |
| 5 | Export error |

---

## NLE export formats

| Format | Use case |
|--------|----------|
| `fcpxml` | Final Cut Pro X 10.4+ / DaVinci Resolve 17+ import |
| `edl` | CMX 3600 EDL — universal, older NLEs |
| `premiere` | FCP7 XML (xmeml 5) — Adobe Premiere |
| `csv` | Spreadsheet / quick inspection |
| `srt` | Subtitle-style cut list, opens in any text editor |

---

## YouTube download notes

YouTube requires cookie authentication for most downloads now. SceneCut supports two ways to provide cookies:

### Option A: Cookies file (recommended for servers / sandboxes)

1. Install the free browser extension **"Get cookies.txt LOCALLY"**:
   - Chrome: <https://chromewebstore.google.com/detail/get-cookies-txt-locally/cclelndahbckbenkjhflpdbgdldlbecc>
   - Firefox: <https://addons.mozilla.org/firefox/addon/get-cookies-txt-locally/>
2. Visit the YouTube video page in your browser (so the cookies are fresh)
3. Click the extension icon → "Export" → saves a `cookies.txt` (Netscape format)
4. Pass it to SceneCut:

```bash
scenecut download "https://www.youtube.com/watch?v=..." --cookies cookies.txt
```

Or via the Web UI: pick "Cookies file (recommended for servers)" in the Authentication dropdown, then drag-drop the `cookies.txt` file.

The cookies file is uploaded, used for the download, then deleted from the server.

### Option B: Browser cookies (desktop only)

If you're running SceneCut on your own desktop machine where Chrome / Firefox / Edge is installed with a logged-in profile:

```bash
scenecut download "https://www.youtube.com/watch?v=..." --cookies-from-browser chrome
```

This does **not** work on a server / sandbox — the server has no browser profile to read from. If you see `could not find chrome cookies database`, switch to Option A.

The download module auto-detects an available JS runtime (node / deno / bun) and passes it to yt-dlp, which is required for YouTube extraction in recent versions.

---

## Testing

```bash
cd scenecut_py
python3 -m pytest tests/ -v
```

34 tests across:
- `test_timecode.py` (8) — TC conversions
- `test_project.py` (8) — project file + overrides
- `test_export.py` (5) — FCPXML / EDL / Premiere / CSV / SRT
- `test_detect.py` (4) — integration on synthetic test videos (hard cuts, fades, proxy)
- `test_cli.py` (9) — full CLI lifecycle (info / detect / list / cut / thumbnails / tar / compose / project management / export all formats)

---

## Tech stack

- **Python**: typer (CLI), PySceneDetect 0.7 (detection), opencv-python-headless 4.13 (dissolve sweep), yt-dlp (download), ffmpeg/ffprobe (cutting, metadata)
- **Web UI**: Next.js 16, React 19, TypeScript 5, Tailwind CSS 4, shadcn/ui, Lucide icons
- **Architecture**: Python CLI is canonical; Next.js API routes shell out to `python3 -m scenecut <cmd> --json` and pass JSON back to the React frontend.

---

## Future work (v2+)

- Audio-based scene detection (silence + beat tracking)
- ML-based semantic scene segmentation
- Face / object tracking across shots for character-based grouping
- Multi-camera sync detection
- Real-time preview in Web UI without pre-extracting thumbnails (MSE stream)
- Docker image
