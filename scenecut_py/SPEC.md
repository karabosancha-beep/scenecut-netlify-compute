# SceneCut — Video Scene/Shot Detection & Cutting Tool

**Version:** 1.0  
**Status:** Implemented  
**Author:** SceneCut Project

A CLI-first (with Web UI companion) tool for detecting shot/scene transitions in MP4 videos, marking them, optionally cutting into clips, and exporting cut lists to NLE interchange formats.

---

## 1. Goals & Use Cases

### Primary use cases
1. **Filmmaking study** — auto-segment a film into shots/scenes so each can be analyzed individually (composition, pacing, color, blocking).
2. **Ensemble clip splitting** — split a long compilation/reel into discrete takes so the editor can quickly pick the ones needed.

### Secondary use cases
- Generate EDL/XML cut lists from raw footage for downstream NLE work.
- Produce thumbnails / contact sheets per shot.
- Build a clip library from a single source video (tar export for handoff).

---

## 2. Definitions

| Term | Meaning |
|---|---|
| **Shot** | A single continuous camera take. Ends at a cut or transition. |
| **Scene** | A semantic unit (one or more shots sharing location/time/narrative). |
| **Cut point** | A timestamp where one shot ends and the next begins. |
| **Hard cut** | Abrupt transition (1-frame change). |
| **Fade** | Gradual transition to/from a solid color (usually black). |
| **Dissolve** | Gradual blend from shot A to shot B. |
| **Wipe** | Geometric transition (rare; treated as a cut if delta exceeds threshold). |

**Scope note:** v1 implements **shot-boundary detection** (SBD). True semantic scene segmentation is approximated by clustering adjacent shots with similar color histograms (the `--group-scenes` flag).

---

## 3. Architecture

```
┌─────────────────────────────────────────────────────────────────┐
│                      Next.js Web UI (:3000)                     │
│  Upload • YouTube fetch • Detect • Scene list • Manual edit •   │
│  Thumbnails • Timeline • Export (FCPXML/EDL/CSV/SRT/tar)        │
└───────────────┬─────────────────────────────────────────────────┘
                │ HTTP /api/*
┌───────────────▼─────────────────────────────────────────────────┐
│                  Next.js API Routes (TS)                        │
│  /api/jobs/*  /api/scenes/*  /api/download/*  /api/upload       │
└───────────────┬─────────────────────────────────────────────────┘
                │ subprocess
┌───────────────▼─────────────────────────────────────────────────┐
│                  Python CLI  (scenecut_py)                      │
│  typer-based • self-contained • JSON in/out                     │
└───────────────┬─────────────────────────────────────────────────┘
                │
       ┌────────┴────────┬─────────────┬───────────────┐
       ▼                 ▼             ▼               ▼
  PySceneDetect      ffmpeg       yt-dlp         tar / xml
  (opencv)        (preprocess,    (download)    (export bundle)
                    cut, thumbs)
```

The Python CLI is **canonical**. Every feature exposed in the Web UI has a 1:1 CLI command with identical arguments. The Web UI simply wraps CLI invocations via subprocess and returns JSON.

---

## 4. CLI Reference

All commands accept `--json` for machine-readable output and exit codes follow BSD convention (0 success, non-zero on error).

### 4.1 `scenecut download`
Download a video from YouTube (or any yt-dlp-supported site).

```
scenecut download <url>
  [--format "bestvideo[height<=720]+bestaudio/best[height<=720]"]
  [--output PATH]
  [--no-playlist]
  [--json]
```

Internal: shells out to `yt-dlp` with sane defaults, writes to `--output` (default `./downloads/<title>.mp4`).

### 4.2 `scenecut info`
Print video metadata (duration, fps, width, height, codec, audio).

```
scenecut info <input> [--json]
```

### 4.3 `scenecut preprocess`
Downscale the source to a low-res proxy for fast analysis. Recommended 360p–480p at CRF 26–30. Proxies are NOT used for final cutting — only detection.

```
scenecut preprocess <input>
  [--height 360]
  [--crf 28]
  [--preset veryfast]
  [--fps keep]            # keep original fps, or override (e.g. 24)
  [--output PATH]
  [--json]
```

### 4.4 `scenecut detect`
Run scene/shot detection. Multi-pass by default.

```
scenecut detect <input>
  [--algo adaptive|content|threshold|hash|all]   # default: all (multi-pass)
  [--threshold 27.0]                               # adaptive/content threshold
  [--min-scene-len 0.6]                            # seconds, merge shorter
  [--fade-threshold 12]                            # for ThresholdDetector (lum 0-255)
  [--fade-min-len 0.3]                             # min fade duration to flag
  [--dissolve-window 12]                           # frames to scan for dissolves
  [--dissolve-threshold 0.6]                       # rolling similarity ratio
  [--group-scenes]                                 # cluster shots into scenes via HSV hist
  [--group-threshold 0.6]
  [--thumbnails]                                   # also write thumbnail per cut
  [--thumb-size 320]
  [--output scenes.json]
  [--proxy PATH]                                   # run detection on proxy, remap to source timestamps
  [--json]
```

**Multi-pass strategy (algo=all, the default):**
1. **AdaptiveDetector** pass — catches hard cuts AND fast dissolves (rolling-window variant of ContentDetector). Tunable with `--threshold`.
2. **ThresholdDetector** pass — catches fades to/from black or white. Adds `type: fade` markers.
3. **Dissolve sweep** — scans rolling HSV-histogram similarity ratio over `--dissolve-window` frames; flags regions where similarity rises gradually above `--dissolve-threshold` as `type: dissolve`.
4. **Dedup** — merge markers within `--min-scene-len` of each other; the highest-confidence type wins.

Each cut in the output JSON carries: `frame_num`, `timecode` (HH:MM:SS.mmm), `seconds` (float), `type` (`cut`|`fade`|`dissolve`), `confidence` (0..1).

### 4.5 `scenecut list`
Pretty-print a scene file.

```
scenecut list <scenes.json> [--format table|csv|json]
```

### 4.6 `scenecut cut`
Cut the source video into per-scene clips. Uses stream copy by default (fast, keyframe-aligned); fall back to re-encode when `--accurate` is set.

```
scenecut cut <input> --scenes <scenes.json>
  [--outdir clips/]
  [--accurate]                  # re-encode at boundaries for frame-accurate cuts
  [--format mp4|mov|mkv]
  [--preset veryfast]
  [--crf 20]
  [--keep-audio]
  [--json]
```

### 4.7 `scenecut thumbnails`
Generate one or more thumbnails per scene (start, mid, or both).

```
scenecut thumbnails <input> --scenes <scenes.json>
  [--outdir thumbs/]
  [--positions start|mid|both]
  [--size 320]
  [--format jpg]
```

### 4.8 `scenecut export`
Convert scene file to NLE interchange format.

```
scenecut export <scenes.json>
  [--format fcpxml|edl|premiere|csv|srt]
  [--output PATH]
  [--source PATH]               # path to source video referenced in the file
  [--reel-name SCENE]           # EDL reel name (default SCENE)
  [--fps 30]                    # override fps for timecode math
```

| Format | Use case |
|---|---|
| **fcpxml** | Final Cut Pro X (1.9) and DaVinci Resolve 17+ import. |
| **edl** | CMX 3600 EDL — universally supported. |
| **premiere** | FCP7 XML (`xmeml`) — Adobe Premiere, older Resolve. |
| **csv** | Spreadsheet / quick inspection. |
| **srt** | Subtitle-style cut list, opens in any text editor. |

### 4.9 `scenecut tar`
Cut and bundle all clips into a single `.tar` (or `.tar.gz`).

```
scenecut tar <input> --scenes <scenes.json>
  [--output clips.tar]
  [--gz]
  [--accurate]
  [--format mp4]
```

### 4.10 `scenecut compose`
Pick a subset of scenes and concatenate them into one video.

```
scenecut compose <input> --scenes <scenes.json>
  [--pick "1,3,5-7"]            # comma-separated ranges (1-indexed)
  [--output composed.mp4]
  [--accurate]
```

### 4.11 `scenecut audit`
Interactive CLI audit: add / remove / move cut points, merge scenes, split scenes.

```
scenecut audit <scenes.json>
  [--input VIDEO]               # enables frame preview at cut point
```

Sub-commands inside the audit REPL: `list`, `add <time>`, `del <index>`, `move <index> <time>`, `merge <i> <j>`, `split <index> <time>`, `save`, `quit`.

### 4.12 `scenecut web`
Launch the web UI (informational — in this deployment, the Next.js server is already running on port 3000).

```
scenecut web [--port 3000] [--open]
```

---

## 5. Scene File Format (`scenes.json`)

```json
{
  "version": "1.0",
  "source": {
    "path": "/path/to/video.mp4",
    "duration": 183.42,
    "fps": 29.97,
    "width": 1920,
    "height": 1080,
    "codec": "h264",
    "audio_codec": "aac"
  },
  "detector": {
    "algo": "all",
    "threshold": 27.0,
    "min_scene_len": 0.6,
    "fade_threshold": 12,
    "dissolve_window": 12,
    "dissolve_threshold": 0.6
  },
  "cuts": [
    {
      "index": 0,
      "frame_num": 0,
      "seconds": 0.0,
      "timecode": "00:00:00.000",
      "type": "cut",
      "confidence": 1.0
    },
    {
      "index": 1,
      "frame_num": 312,
      "seconds": 10.41,
      "timecode": "00:00:10.410",
      "type": "cut",
      "confidence": 0.92
    },
    {
      "index": 2,
      "frame_num": 1200,
      "seconds": 40.04,
      "timecode": "00:00:40.040",
      "type": "fade",
      "confidence": 0.78
    }
  ],
  "scenes": [
    {
      "index": 0,
      "start": 0.0,
      "end": 10.41,
      "duration": 10.41,
      "start_frame": 0,
      "end_frame": 311,
      "thumbnail": "thumbs/scene_000.jpg",
      "label": null,
      "tags": []
    }
  ],
  "overrides": {
    "added_cuts": [],
    "removed_cuts": [],
    "moved_cuts": {},
    "merged_scenes": [],
    "split_scenes": []
  }
}
```

The `overrides` block records all manual edits applied via `audit` or the Web UI. The original `cuts` array is preserved; the effective cut list is computed by applying overrides on top.

---

## 6. Detection Strategy — Deep Dive

### 6.1 Hard cuts
**Algorithm:** PySceneDetect `AdaptiveDetector` (rolling-window variant of HSV-content delta).  
**Why:** Adaptive uses a sliding window so the threshold scales with local content variance — robust to high-motion vs. low-motion shots.  
**Tunable:** `--threshold` (default 27.0, lower = more sensitive).  
**Confidence:** normalized delta / (2 × threshold), clipped to 1.0.

### 6.2 Fades (to/from black or white)
**Algorithm:** PySceneDetect `ThresholdDetector` on frame luminance.  
**Why:** Catches slow fades that Adaptive misses because each individual frame delta stays below threshold.  
**Tunable:** `--fade-threshold` (luma 0–255, default 12 = nearly black), `--fade-min-len` (seconds, default 0.3).  
**Behavior:** emits both "fade-in" and "fade-out" markers; consecutive fade-out + fade-in within 2 s are merged into a single `fade` cut at the midpoint.

### 6.3 Dissolves
**Algorithm:** Custom rolling HSV-histogram similarity ratio over `--dissolve-window` frames.  
**Why:** During a dissolve, every individual frame delta is small, but the rolling similarity climbs steadily as shot B replaces shot A. We flag windows where the rolling ratio exceeds `--dissolve-threshold` AND the per-frame delta stays below the adaptive cut threshold.  
**Tunable:** `--dissolve-window` (default 12 frames), `--dissolve-threshold` (default 0.6).  
**Confidence:** peak rolling ratio in the flagged window.

### 6.4 Wipes & other geometric transitions
Treated as hard cuts if per-frame delta exceeds threshold; otherwise logged as `unknown` and skipped.

### 6.5 Scene grouping (optional, `--group-scenes`)
After cuts are computed, adjacent shots are clustered into scenes using HSV histogram correlation. Two consecutive shots with histogram correlation > `--group-threshold` (default 0.6) belong to the same scene. This is a heuristic — true semantic segmentation requires ML.

### 6.6 Multi-pass deduplication
Final cut list is the union of all detector outputs, deduplicated by `--min-scene-len` (default 0.6 s). When two cuts collide, the one with higher confidence wins; the lower one is recorded in `cuts[].superseded_by`.

---

## 7. NLE Export Formats

### 7.1 FCPXML 1.9
- One `<event>` with one `<project>` containing a `<sequence>` and a `<spine>`.
- Each scene becomes a `<asset-clip>` referencing a single `<asset>` (the source video).
- Timecode uses `fps` from the scene file; rational time `N/Ms` format.
- Compatible with Final Cut Pro X 10.4+ and DaVinci Resolve 17+.

### 7.2 EDL (CMX 3600)
- One event per scene, `TITLE:` line per scene.
- `REEL` name configurable via `--reel-name` (default `SCENE`).
- Timecode in `HH:MM:SS:FF` (frames, not ms).
- Audio follows video (`AA/V`).

### 7.3 Premiere XML (xmeml 5)
- FCP7 XML structure, compatible with Adobe Premiere Pro and older DaVinci Resolve.
- One `<sequence>` with `<clipitem>` entries per scene.

### 7.4 CSV
Header: `index,start_seconds,end_seconds,start_timecode,end_timecode,duration,type,confidence,thumbnail`

### 7.5 SRT
Each scene as a subtitle block with the timecode range and label `Scene N`.

---

## 8. Web UI Specification

### 8.1 Pages & layout
Single-page app at `/` with tabbed workflow:

| Tab | Purpose |
|---|---|
| **Source** | Upload mp4 (chunked), fetch from YouTube URL, or pick recent. Show source metadata + preview. |
| **Preprocess** | Optional proxy generation. Slider for target height (240–720), CRF, fps. |
| **Detect** | Algorithm selector (adaptive/content/threshold/hash/all), threshold sliders, advanced (fade, dissolve) toggles. Run button + progress bar. |
| **Scenes** | Master view: timeline with cut markers + scrollable list of scene cards. Each card shows thumbnail, timecode range, duration, type, confidence. |
| **Audit** | Inline video player per scene. Add/remove/move cuts via click-on-timeline or form. Merge/split scenes. Save edits to project file. |
| **Export** | Pick format(s) (FCPXML, EDL, Premiere, CSV, SRT), download single file or tar bundle of clips. |

### 8.2 Project state
Autosaved to `upload/<project-id>/project.json` (the scene file). Resumable across sessions.

### 8.3 Upload
- Chunked upload (10 MB chunks) with resumability.
- File-type validation (must be mp4/mov/mkv).
- Max size configurable (default 2 GB).

### 8.4 Real-time progress
Long-running operations (download, preprocess, detect, cut, tar) stream progress via SSE on `/api/jobs/<id>/events`.

### 8.5 Manual editing
- **Add cut:** click on the timeline; persists to `overrides.added_cuts`.
- **Remove cut:** click a cut marker; persists to `overrides.removed_cuts`.
- **Move cut:** drag a cut marker; persists to `overrides.moved_cuts`.
- **Merge scenes:** select two adjacent scenes, click Merge.
- **Split scene:** select a scene, click Split at playhead.
- **Undo/redo:** 50-step history.

---

## 9. Performance

| Operation | Typical cost (1080p, 30 fps, 5 min) |
|---|---|
| Preprocess to 360p | ~30 s (1× realtime) |
| Detect (proxy) | ~20 s |
| Detect (full res) | ~3 min |
| Cut (stream copy) | ~5 s |
| Cut (accurate re-encode) | ~1 min |
| Thumbnails (320 px) | ~10 s |

Detection on a 360p proxy is ~9× faster than on 1080p source with negligible accuracy loss for hard-cut detection. For fades/dissolves, 480p is recommended (below that, the rolling luma can get noisy).

---

## 10. Things You (the user) Didn't Mention — Expanded Considerations

These were inferred from the use cases and added to v1:

1. **Audio analysis** — future work; v1 detects on video only. Audio silence detection could complement fades.
2. **Confidence scores** — every cut carries a 0–1 confidence; low-confidence cuts are flagged for manual review in the Web UI (yellow marker).
3. **Stream copy vs. accurate cut** — `--accurate` re-encodes only the segments straddling a cut boundary, keeping the middle stream-copied. Best of both worlds (fast + frame-accurate).
4. **Keyframe snapping** — optional `--snap-to-keyframe` for cut points (default off; preserves frame accuracy).
5. **Project portability** — scene JSON contains absolute paths; a `--relocate` command remaps paths for cross-machine use.
6. **Batch processing** — `scenecut detect *.mp4` runs detection on multiple videos; outputs one JSON per input.
7. **Resume / caching** — detection results cached keyed by `(path, mtime, size, algo, params)`. Re-running with same params is instant.
8. **Multiple export in one call** — `--format fcpxml,edl,csv` produces multiple files.
9. **CLI exit codes** — 0 success, 2 input missing/invalid, 3 ffmpeg/yt-dlp failure, 4 detection error, 5 export error.
10. **Quiet / verbose / JSON output** — `--quiet`, `--verbose`, `--json` flags on every command.
11. **Structured logging** — writes to `./scenecut.log` with rotation.
12. **Scene grouping** — `--group-scenes` clusters adjacent shots into semantic scenes (2-tier hierarchy: shot → scene).
13. **Timecode modes** — drop-frame (`29.97 DF`), non-drop (`29.97 NDF`), and exact (`30`) supported via `--timecode-mode`.
14. **GPU acceleration** — ffmpeg preprocessing auto-detects NVENC if available (`h264_nvenc`).
15. **EDL reel names** — `--reel-name` flag, default `SCENE`.
16. **Web UI upload limits** — 2 GB default, configurable via env var `SCENECUT_MAX_UPLOAD_BYTES`.
17. **Web UI undo/redo** — 50-step history per project.
18. **CLI autocompletion** — `scenecut --install-completion [bash|zsh|fish|powershell]`.
19. **Container-ready** — Python CLI runs in any container with ffmpeg + python3; Web UI runs in any Node host.
20. **Re-encode quality** — `--crf` exposed for cut and compose; default 20 (visually lossless).
21. **Subtitle/metadata preservation** — when cutting with stream copy, all streams (audio, subtitle, data) are preserved.
22. **Thumbnails per cut** — `--thumbnails` flag on `detect` writes a JPEG at the frame *after* each cut (the first frame of the new shot).
23. **Color histogram overlay** — Web UI shows per-scene dominant color swatches (helps compare shots for grouping).
24. **Reverse mode** — `scenecut compose --reverse` to reverse scene order (handy for trailer-style edits).
25. **Frame-accurate seeking** — uses `-ss` *after* `-i` for accurate seeking when re-encoding; *before* `-i` for fast seek when copying.

---

## 11. Error Handling & Exit Codes

| Code | Meaning |
|---|---|
| 0 | Success |
| 2 | Input missing or invalid (bad path, unsupported codec, etc.) |
| 3 | External tool failure (ffmpeg / yt-dlp) |
| 4 | Detection error (PySceneDetect crashed, opencv read failure) |
| 5 | Export error (unsupported format, write permission) |
| 6 | Web UI / API error |

All errors print a human-readable message to stderr; with `--json`, the error is returned as `{"error": "...", "code": N}`.

---

## 12. Testing

- **Unit tests** — algorithms (cut dedup, timecode math, FCPXML/EDL generation).
- **Integration tests** — full pipeline on a real test video: download → preprocess → detect → cut → export → tar.
- **Test video** — the YouTube clip referenced in the original spec (`https://www.youtube.com/watch?v=xBasQG_6p40`), plus a synthetic test video generated via ffmpeg (`testsrc` filter) for offline tests.

Tests live in `scenecut_py/tests/` and run via `pytest`.

---

## 13. File Layout (Repo)

```
scenecut_py/
├── scenecut/
│   ├── __init__.py
│   ├── __main__.py             # python -m scenecut
│   ├── cli.py                  # typer app
│   ├── config.py               # defaults, paths
│   ├── download.py             # yt-dlp wrapper
│   ├── info.py                 # ffprobe wrapper
│   ├── preprocess.py           # ffmpeg downscale
│   ├── detect.py               # multi-pass scene detection
│   ├── cut.py                  # ffmpeg cutting
│   ├── thumbnails.py
│   ├── compose.py              # subset + concat
│   ├── audit.py                # interactive REPL
│   ├── export/
│   │   ├── __init__.py
│   │   ├── fcpxml.py
│   │   ├── edl.py
│   │   ├── premiere.py
│   │   ├── csv_export.py
│   │   └── srt.py
│   ├── project.py              # scene JSON load/save + override application
│   ├── timecode.py             # tc conversions (DF/NDF)
│   └── util.py
├── tests/
│   ├── conftest.py
│   ├── test_timecode.py
│   ├── test_project.py
│   ├── test_detect.py
│   ├── test_export.py
│   ├── test_cli.py
│   └── test_integration.py
├── pyproject.toml
└── SPEC.md                     # this file
```

Web UI lives in `/home/z/my-project/src/` (Next.js).

---

## 14. Future Work (v2+)

- Audio-based scene detection (silence + beat tracking).
- ML-based semantic scene segmentation.
- Face / object tracking across shots for character-based grouping.
- Multi-camera sync detection.
- Real-time preview in Web UI without pre-extracting thumbnails (MSE stream).
- Docker image + Helm chart.
