"""SceneCut CLI — typer-based entrypoint."""
from __future__ import annotations

import json
import os
import sys
import tarfile
from pathlib import Path
from typing import Optional

import typer

from . import __version__
from .config import DEFAULTS, PROJECTS_DIR, DOWNLOADS_DIR, project_dir, project_file
from .util import SceneCutError, probe_video

app = typer.Typer(
    name="scenecut",
    help="Video scene/shot detection & cutting tool.",
    no_args_is_help=True,
    add_completion=False,
)


# ------------------------------------------------------------------ helpers


def _emit(data, json_mode: bool):
    if json_mode:
        typer.echo(json.dumps(data, indent=2, default=str))
    else:
        if isinstance(data, dict):
            for k, v in data.items():
                if isinstance(v, (dict, list)):
                    typer.echo(f"{k}: {json.dumps(v, default=str)[:200]}")
                else:
                    typer.echo(f"{k}: {v}")
        elif isinstance(data, list):
            for x in data:
                typer.echo(json.dumps(x, default=str))
        else:
            typer.echo(str(data))


def _err(e: Exception, json_mode: bool):
    code = getattr(e, "code", 1) if isinstance(e, SceneCutError) else 1
    if json_mode:
        typer.echo(json.dumps({"error": str(e), "code": code}))
    else:
        typer.echo(f"ERROR: {e}", err=True)
    raise typer.Exit(code=code)


# ------------------------------------------------------------------ commands


@app.command()
def version():
    """Print version."""
    typer.echo(__version__)


@app.command()
def info(input: Path = typer.Argument(..., help="Input video file"),
         json: bool = typer.Option(False, "--json", help="JSON output")):
    """Print video metadata."""
    try:
        info = probe_video(input)
        _emit(info.to_dict(), json)
    except Exception as e:
        _err(e, json)


@app.command()
def download(url: str = typer.Argument(..., help="Video URL (YouTube etc.)"),
             output: Optional[Path] = typer.Option(None, "--output", "-o"),
             format: Optional[str] = typer.Option(None, "--format", "-f",
                                                  help="yt-dlp format spec"),
             no_playlist: bool = typer.Option(True, "--no-playlist/--playlist"),
             cookies_from_browser: Optional[str] = typer.Option(
                 None, "--cookies-from-browser",
                 help="Use cookies from chrome|firefox|edge|safari|brave "
                      "(browser must be installed on THIS machine with a profile)"),
             cookies: Optional[Path] = typer.Option(
                 None, "--cookies",
                 help="Path to Netscape-format cookies.txt file (export from "
                      "your browser via the 'Get cookies.txt LOCALLY' extension). "
                      "Recommended for server / sandbox environments."),
             json: bool = typer.Option(False, "--json")):
    """Download a video via yt-dlp."""
    from .download import download as dl
    try:
        result = dl(url, output=str(output) if output else None,
                    format_spec=format, no_playlist=no_playlist,
                    cookies_from_browser=cookies_from_browser,
                    cookies_file=str(cookies) if cookies else None)
        _emit(result, json)
    except Exception as e:
        _err(e, json)


@app.command()
def preprocess(input: Path = typer.Argument(...),
               height: int = typer.Option(DEFAULTS["proxy_height"], "--height"),
               crf: int = typer.Option(DEFAULTS["proxy_crf"], "--crf"),
               preset: str = typer.Option(DEFAULTS["proxy_preset"], "--preset"),
               fps: Optional[float] = typer.Option(None, "--fps"),
               denoise: bool = typer.Option(
                   False, "--denoise",
                   help="Denoise after upscale (hqdn3d) — for heavily compressed low-res sources."),
               output: Optional[Path] = typer.Option(None, "--output", "-o"),
               json: bool = typer.Option(False, "--json")):
    """Downscale a video to a low-res proxy for fast detection."""
    from .preprocess import preprocess as pp
    try:
        out = pp(input, height=height, crf=crf, preset=preset,
                 fps=fps, denoise=denoise,
                 output=str(output) if output else None)
        _emit({"input": str(input), "proxy": out, "height": height, "crf": crf,
               "denoise": denoise}, json)
    except Exception as e:
        _err(e, json)


@app.command()
def detect(input: Path = typer.Argument(...),
           algo: str = typer.Option(DEFAULTS["algo"], "--algo"),
           threshold: Optional[float] = typer.Option(None, "--threshold"),
           min_scene_len: Optional[float] = typer.Option(None, "--min-scene-len"),
           fade_threshold: Optional[int] = typer.Option(None, "--fade-threshold"),
           fade_ceiling: Optional[int] = typer.Option(
               None, "--fade-ceiling",
               help="Luma ceiling for fade-to-WHITE detection (default 243)."),
           fade_min_len: Optional[float] = typer.Option(None, "--fade-min-len"),
           dissolve_window: Optional[int] = typer.Option(
               None, "--dissolve-window",
               help="DEPRECATED: alias for a single-element --dissolve-windows list."),
           dissolve_windows: Optional[str] = typer.Option(
               None, "--dissolve-windows",
               help='Multi-scale dissolve windows, comma-separated (default "8,16,32").'),
           dissolve_threshold: Optional[float] = typer.Option(
               None, "--dissolve-threshold",
               help="DEPRECATED (Pearson->JS semantics change): ignored; use --dissolve-dissim."),
           dissolve_dissim: Optional[float] = typer.Option(
               None, "--dissolve-dissim",
               help="Normalized JS endpoint-dissimilarity floor for dissolves (D19 default 0.15)."),
           hard_cut_mad_min: Optional[float] = typer.Option(
               None, "--hard-cut-mad-min",
               help="Window-MAD ceiling for dissolve candidacy (D19 default 40; "
                    "hard cuts measure 54+)."),
           consec_js_min: Optional[float] = typer.Option(
               None, "--consec-js-min",
               help="Window-mean consec-JS floor for the dissolve plateau (D19 "
                    "default 0.025)."),
           consec_js_frame_min: Optional[float] = typer.Option(
               None, "--consec-js-frame-min",
               help="D19 sustain-gate frame-level elevation floor (default 0.03)."),
           consec_js_sustain_frac: Optional[float] = typer.Option(
               None, "--consec-js-sustain-frac",
               help="D19 fraction of window frames above the frame-level floor "
                    "(default 0.35)."),
           consec_js_ratio_min: Optional[float] = typer.Option(
               None, "--consec-js-ratio-min",
               help="D19 localization ratio vs clip-baseline median (default "
                    "1.8; rejects pans/global motion)."),
           motion_gate: bool = typer.Option(
               False, "--motion-gate",
               help="Enable motion gate: downgrades likely-motion false cuts (confidence<=0.3, kept in list)."),
           hash_threshold: Optional[float] = typer.Option(
               None, "--hash-threshold",
               help="HashDetector firing distance (default 0.4); higher = fewer hash fires."),
           hash_corroboration: Optional[bool] = typer.Option(
               None, "--hash-corroboration/--no-hash-corroboration",
               help="D16 v4.3: solo-hash feature-evidence corroboration gate (default on)."),
           hash_corroboration_spike: Optional[float] = typer.Option(
               None, "--hash-corroboration-spike",
               help="Solo-hash gate MAD spike-ratio floor (default 3.0)."),
           hash_corroboration_js: Optional[float] = typer.Option(
               None, "--hash-corroboration-js",
               help="Solo-hash gate consecutive-JS floor (default 0.15)."),
           hash_corroboration_luma: Optional[float] = typer.Option(
               None, "--hash-corroboration-luma",
               help="Solo-hash gate mean-luma step floor (default 8.0)."),
           spatial_dissolve: Optional[bool] = typer.Option(
               None, "--spatial-dissolve/--no-spatial-dissolve",
               help="D18 (Phase 1.5): tier-2 spatial match-dissolve detector (default on)."),
           neural: bool = typer.Option(
               False, "--neural",
               help="Enable the TransNetV2 pass + arbiter (D13; needs scenecut[neural])"),
           group_scenes: bool = typer.Option(False, "--group-scenes"),
           group_threshold: Optional[float] = typer.Option(None, "--group-threshold"),
           merge_short_scenes: Optional[float] = typer.Option(
               None, "--merge-short-scenes",
               help="Merge scenes shorter than this (seconds) with their most "
                    "similar neighbor. E.g. --merge-short-scenes 2.0 merges "
                    "all scenes < 2s. Useful for content with high intra-scene "
                    "motion that causes false cuts."),
           thumbnails: bool = typer.Option(False, "--thumbnails"),
           thumb_size: int = typer.Option(DEFAULTS["thumb_size"], "--thumb-size"),
           proxy: Optional[Path] = typer.Option(None, "--proxy", help="Proxy video for detection"),
           output: Optional[Path] = typer.Option(None, "--output", "-o"),
           json: bool = typer.Option(False, "--json")):
    """Run scene/shot detection on a video."""
    from .detect import detect_scenes
    try:
        project = detect_scenes(
            input, algo=algo, threshold=threshold, min_scene_len=min_scene_len,
            fade_threshold=fade_threshold, fade_ceiling=fade_ceiling,
            fade_min_len=fade_min_len,
            dissolve_window=dissolve_window, dissolve_windows=dissolve_windows,
            dissolve_threshold=dissolve_threshold, dissolve_dissim=dissolve_dissim,
            hard_cut_mad_min=hard_cut_mad_min, consec_js_min=consec_js_min,
            consec_js_frame_min=consec_js_frame_min,
            consec_js_sustain_frac=consec_js_sustain_frac,
            consec_js_ratio_min=consec_js_ratio_min,
            motion_gate=motion_gate, neural=neural,
            hash_threshold=hash_threshold,
            hash_corroboration=hash_corroboration,
            hash_corroboration_spike=hash_corroboration_spike,
            hash_corroboration_js=hash_corroboration_js,
            hash_corroboration_luma=hash_corroboration_luma,
            spatial_dissolve=spatial_dissolve,
            group_scenes=group_scenes, group_threshold=group_threshold,
            proxy=str(proxy) if proxy else None,
            merge_short_scenes=merge_short_scenes,
        )
        # Compute output path early so thumbnail generation can use it
        out_path = output or Path(str(input) + ".scenes.json")

        if thumbnails:
            from .thumbnails import make_thumbnails
            # Clear old thumbnails first to avoid stale files when scene count changes.
            # Thumbnails go next to the OUTPUT (project.json), not the input video,
            # so they always live in the project dir regardless of where source.mp4 is.
            import shutil
            from pathlib import Path as _P
            out_p = _P(str(out_path))
            thumb_dir = out_p.parent / "thumbs"
            if thumb_dir.exists():
                shutil.rmtree(thumb_dir)
            thumb_dir.mkdir(parents=True, exist_ok=True)
            thumbs = make_thumbnails(input, project, outdir=thumb_dir, size=thumb_size)
            for t in thumbs:
                if t["scene_index"] < len(project["scenes"]):
                    project["scenes"][t["scene_index"]]["thumbnail"] = t["thumbnail"]
        from .project import save_project
        save_project(project, out_path)
        _emit({"project_path": str(out_path),
               "n_cuts": len(project["cuts"]),
               "n_scenes": len(project["scenes"]),
               "cuts": [{"i": c["index"], "t": c["seconds"], "type": c["type"], "conf": c.get("confidence", 0)} for c in project["cuts"]]},
              json)
    except Exception as e:
        _err(e, json)


@app.command("list")
def list_scenes(scenes: Path = typer.Argument(...),
                format: str = typer.Option("table", "--format", "-f")):
    """Pretty-print a scene file."""
    from .project import load_project, apply_overrides
    project = apply_overrides(load_project(scenes))
    if format == "json":
        typer.echo(json.dumps(project, indent=2, default=str))
        return
    # Table
    typer.echo(f"Source: {project['source'].get('path', '?')}")
    typer.echo(f"Duration: {project['source'].get('duration', 0):.2f}s | FPS: {project['source'].get('fps', 30):.3f}")
    typer.echo(f"Cuts: {len(project['cuts'])} | Scenes: {len(project['scenes'])}")
    typer.echo("")
    typer.echo(f"{'#':>3} {'Start':>14} {'End':>14} {'Dur':>8} {'Type':>9} {'Conf':>5}")
    typer.echo("-" * 60)
    from .timecode import seconds_to_hmsms
    for i, s in enumerate(project["scenes"]):
        cut = next((c for c in project["cuts"] if c["index"] == i), {})
        typer.echo(f"{i + 1:>3} {seconds_to_hmsms(s['start']):>14} "
                   f"{seconds_to_hmsms(s['end']):>14} {s['duration']:>7.2f}s "
                   f"{cut.get('type', ''):>9} {cut.get('confidence', 0):>5.2f}")


@app.command()
def cut(input: Path = typer.Argument(...),
        scenes: Path = typer.Option(..., "--scenes", help="scenes.json file"),
        outdir: Optional[Path] = typer.Option(None, "--outdir", "-o"),
        accurate: bool = typer.Option(False, "--accurate", help="Re-encode at boundaries"),
        format: str = typer.Option("mp4", "--format"),
        crf: int = typer.Option(DEFAULTS["cut_crf"], "--crf"),
        preset: str = typer.Option(DEFAULTS["cut_preset"], "--preset"),
        keep_audio: bool = typer.Option(True, "--keep-audio/--no-audio"),
        json: bool = typer.Option(False, "--json")):
    """Cut input into per-scene clips."""
    from .cut import cut_video
    from .project import load_project
    try:
        project = load_project(scenes)
        results = cut_video(input, project, outdir=outdir, accurate=accurate,
                            format=format, crf=crf, preset=preset, keep_audio=keep_audio)
        _emit({"clips": results, "outdir": str(outdir or Path(str(input)).parent / (Path(str(input)).stem + "_clips"))}, json)
    except Exception as e:
        _err(e, json)


@app.command()
def thumbnails(input: Path = typer.Argument(...),
               scenes: Path = typer.Option(..., "--scenes"),
               outdir: Optional[Path] = typer.Option(None, "--outdir", "-o"),
               positions: str = typer.Option("start", "--positions"),
               size: int = typer.Option(DEFAULTS["thumb_size"], "--size"),
               format: str = typer.Option("jpg", "--format"),
               json: bool = typer.Option(False, "--json")):
    """Generate thumbnails for each scene."""
    from .thumbnails import make_thumbnails
    from .project import load_project, apply_overrides
    try:
        raw = load_project(scenes)
        # Apply overrides to get effective scenes for thumbnail generation
        applied = apply_overrides(raw)
        # Clear old thumbnails before regenerating
        import shutil
        from pathlib import Path as _P
        out_p = _P(str(scenes))
        thumb_dir = outdir or (out_p.parent / "thumbs")
        if not outdir:
            thumb_dir = out_p.parent / "thumbs"
        if thumb_dir.exists():
            shutil.rmtree(thumb_dir)
        thumb_dir.mkdir(parents=True, exist_ok=True)
        # Generate thumbnails using the APPLIED scenes (effective boundaries)
        # NOTE: we do NOT save the project file — the thumbnail API route
        # finds files by scene index on disk, so the project JSON stays clean.
        results = make_thumbnails(input, applied, outdir=thumb_dir, positions=positions,
                                  size=size, format=format)
        _emit(results, json)
    except Exception as e:
        _err(e, json)


@app.command("hover-clips")
def hover_clips(input: Path = typer.Argument(...),
                scenes: Path = typer.Option(..., "--scenes"),
                outdir: Optional[Path] = typer.Option(None, "--outdir", "-o"),
                height: int = typer.Option(144, "--height", help="Height in pixels (width auto)"),
                max_duration: float = typer.Option(10.0, "--max-duration",
                                                    help="Cap clip length (seconds). Scenes longer than this show only the beginning."),
                crf: int = typer.Option(35, "--crf", help="Quality (higher = smaller/lower quality)"),
                json: bool = typer.Option(False, "--json")):
    """Generate small video clips for hover-to-play preview in the UI.

    Each clip is a tiny MP4 (default: 144p, no audio, CRF 35) capped at
    --max-duration seconds. These are pre-cached by the browser and played
    on hover over the scene card.
    """
    from .project import load_project, apply_overrides
    from .util import ffmpeg_path, run
    import shutil
    from pathlib import Path as _P
    try:
        raw = load_project(scenes)
        project = apply_overrides(raw)
        out_p = _P(str(scenes))
        clip_dir = outdir or (out_p.parent / "hover_clips")
        if clip_dir.exists():
            shutil.rmtree(clip_dir)
        clip_dir.mkdir(parents=True, exist_ok=True)

        results = []
        for s in project["scenes"]:
            start = s["start"]
            dur = min(s["duration"], max_duration)
            if dur <= 0:
                continue
            clip_path = clip_dir / f"scene_{s['index']:03d}.mp4"
            cmd = [
                ffmpeg_path(), "-y", "-hide_banner", "-loglevel", "error",
                "-ss", f"{start:.3f}",
                "-i", str(input),
                "-t", f"{dur:.3f}",
                "-vf", f"scale=-2:{height}",
                "-c:v", "libx264", "-preset", "ultrafast", "-crf", str(crf),
                "-an",  # no audio
                "-movflags", "+faststart",
                str(clip_path),
            ]
            run(cmd)
            results.append({
                "scene_index": s["index"],
                "clip": str(clip_path),
                "duration": dur,
                "height": height,
            })
        _emit({"clips": results, "count": len(results), "outdir": str(clip_dir)}, json)
    except Exception as e:
        _err(e, json)


@app.command()
def export(scenes: Path = typer.Argument(...),
           format: str = typer.Option("fcpxml", "--format", "-f",
                                       help="fcpxml|edl|premiere|csv|srt"),
           output: Optional[Path] = typer.Option(None, "--output", "-o"),
           source: Optional[Path] = typer.Option(None, "--source"),
           reel_name: str = typer.Option("SCENE", "--reel-name"),
           fps: Optional[float] = typer.Option(None, "--fps"),
           json: bool = typer.Option(False, "--json")):
    """Export scene file to NLE interchange format."""
    from .export import export as do_export
    from .project import load_project
    try:
        project = load_project(scenes)
        ext = {"fcpxml": "fcpxml", "edl": "edl", "premiere": "xml",
               "csv": "csv", "srt": "srt"}[format]
        out_path = output or scenes.with_suffix(f".{ext}")
        result = do_export(project, format, out_path,
                           source=str(source) if source else None,
                           reel_name=reel_name, fps=fps)
        _emit({"path": result, "format": format}, json)
    except Exception as e:
        _err(e, json)


@app.command()
def tar(input: Path = typer.Argument(...),
        scenes: Path = typer.Option(..., "--scenes"),
        output: Optional[Path] = typer.Option(None, "--output", "-o"),
        gz: bool = typer.Option(False, "--gz"),
        accurate: bool = typer.Option(False, "--accurate"),
        format: str = typer.Option("mp4", "--format"),
        json: bool = typer.Option(False, "--json")):
    """Cut and bundle all clips into a single tar archive."""
    from .cut import cut_video
    from .project import load_project
    import tempfile, shutil
    try:
        project = load_project(scenes)
        with tempfile.TemporaryDirectory() as tmpdir:
            results = cut_video(input, project, outdir=tmpdir, accurate=accurate, format=format)
            out_path = output or scenes.with_suffix(".tar" + (".gz" if gz else ""))
            mode = "w:gz" if gz else "w"
            with tarfile.open(out_path, mode) as tf:
                for r in results:
                    tf.add(r["path"], arcname=Path(r["path"]).name)
                # Also add the scenes.json for reference
                tf.add(scenes, arcname="scenes.json")
        _emit({"path": str(out_path), "clips": len(results)}, json)
    except Exception as e:
        _err(e, json)


@app.command()
def compose(input: Path = typer.Argument(...),
            scenes: Path = typer.Option(..., "--scenes"),
            pick: str = typer.Option("all", "--pick", help="e.g. '1,3,5-7'"),
            output: Optional[Path] = typer.Option(None, "--output", "-o"),
            accurate: bool = typer.Option(True, "--accurate/--fast"),
            json: bool = typer.Option(False, "--json")):
    """Pick a subset of scenes and concatenate them."""
    from .compose import compose as do_compose
    from .project import load_project
    try:
        project = load_project(scenes)
        out = do_compose(input, project, pick=pick,
                         output=str(output) if output else None, accurate=accurate)
        _emit({"path": out, "pick": pick}, json)
    except Exception as e:
        _err(e, json)


@app.command("seams")
def seams_cmd(input: Path = typer.Argument(...),
              scenes: Path = typer.Option(..., "--scenes"),
              outdir: Optional[Path] = typer.Option(None, "--outdir", "-o"),
              radius: float = typer.Option(5.0, "--radius",
                                            help="Max seconds on each side of the cut"),
              height: int = typer.Option(360, "--height"),
              json_out: bool = typer.Option(False, "--json")):
    """Generate seam clips at each cut point for transition analysis.

    Each seam clip spans ±X seconds around a cut, where X = min(radius, half
    of each adjacent scene's duration). This gives the VLM both sides of the
    cut in one continuous video.

    Writes seams/manifest.json with metadata for each seam.
    """
    from .seam import generate_seam_clips
    from .project import load_project
    try:
        project = load_project(scenes)
        results = generate_seam_clips(input, project, outdir=outdir,
                                       max_seam_radius=radius, height=height)
        _emit({"seams": results, "count": len(results),
               "manifest": str(Path(outdir or Path(str(input)).parent / "seams") / "manifest.json")},
              json_out)
    except Exception as e:
        _err(e, json_out)


@app.command("analyze")
def analyze(input: Path = typer.Argument(...),
            scenes: Path = typer.Option(..., "--scenes"),
            outdir: Optional[Path] = typer.Option(None, "--outdir", "-o"),
            what: str = typer.Option("all", "--what",
                                      help="all|scenes|seams (what to analyze)"),
            vlm_script: Optional[Path] = typer.Option(
                None, "--vlm-script",
                help="Path to vlm_analyze.ts (default: auto-detect in scripts/)"),
            json_out: bool = typer.Option(False, "--json", help="JSON output")):
    """Run VLM analysis on scene clips and/or seam clips.

    Requires the z-ai-web-dev-sdk (Bun/TypeScript). The CLI calls a bridge
    script (vlm_analyze.ts) that sends each clip as base64 video to the VLM.

    What gets analyzed:
      scenes: Each scene clip gets a 14-point cinematography analysis
      seams:  Each seam clip gets a transition/cut analysis (type, false-positive check)
      all:    Both

    Prerequisites:
      1. Run `scenecut cut --for-vlm` first to generate scene clips
      2. Run `scenecut seams` first to generate seam clips (if analyzing seams)

    Output:
      <outdir>/scene_NNN_vlm.json  — raw VLM response for each scene
      <outdir>/seam_NNN_vlm.json   — raw VLM response for each seam
      <outdir>/report.json         — consolidated report with all analyses
    """
    import subprocess
    from .project import load_project, apply_overrides

    try:
        project = load_project(scenes)
        applied = apply_overrides(project)
        src_path = str(input)

        if outdir is None:
            outdir = Path(str(scenes)).parent / "analysis"
        outdir = Path(outdir)
        outdir.mkdir(parents=True, exist_ok=True)

        # Find the VLM bridge script
        if vlm_script is None:
            # Try multiple locations
            candidates = [
                Path(__file__).parent.parent / "scripts" / "vlm_analyze.ts",
                Path(__file__).resolve().parent.parent / "scripts" / "vlm_analyze.ts",
                Path.cwd() / "scenecut_py" / "scripts" / "vlm_analyze.ts",
                Path.cwd() / "scripts" / "vlm_analyze.ts",
            ]
            for c in candidates:
                if c.exists():
                    vlm_script = c
                    break
            if vlm_script is None:
                raise SceneCutError(
                    "vlm_analyze.ts not found. Pass --vlm-script <path> or "
                    "place it in scenecut_py/scripts/",
                    code=2,
                )

        results = {"scenes": [], "seams": []}

        # --- Analyze scene clips ---
        if what in ("all", "scenes"):
            clips_dir = Path(str(scenes)).parent / "clips"
            if not clips_dir.exists():
                print(f"  ⚠ No clips dir at {clips_dir}. Run `scenecut cut --for-vlm` first.")
            else:
                manifest_path = clips_dir / "manifest.json"
                if manifest_path.exists():
                    manifest = json.load(open(manifest_path))
                    clip_list = manifest.get("clips", [])
                else:
                    clip_list = [{"file": f, "index": i} for i, f in enumerate(sorted(os.listdir(clips_dir))) if f.endswith(".mp4")]

                total = len(clip_list)
                for i, clip_info in enumerate(clip_list):
                    clip_file = clip_info.get("file", f"scene_{i:03d}.mp4")
                    clip_path = clips_dir / clip_file
                    if not clip_path.exists():
                        continue

                    out_file = outdir / f"scene_{i:03d}_vlm.json"
                    if out_file.exists():
                        print(f"  [{i+1}/{total}] Scene {i} — cached")
                        results["scenes"].append({"index": i, "file": clip_file, "result": str(out_file), "cached": True})
                        continue

                    print(f"  [{i+1}/{total}] Analyzing scene {i}...", end=" ", flush=True)
                    proc = subprocess.run(
                        ["bun", "run", str(vlm_script), str(clip_path), "scene", str(out_file)],
                        capture_output=True, text=True, timeout=120,
                        cwd=str(Path(__file__).resolve().parent.parent),
                    )
                    if proc.returncode == 0:
                        try:
                            res = json.loads(proc.stdout.strip().split("\n")[-1])
                            if res.get("ok"):
                                print(f"✓ ({res.get('content_length', 0)} chars)")
                                results["scenes"].append({"index": i, "file": clip_file, "result": str(out_file)})
                            else:
                                print(f"✗ {res.get('error', 'unknown')}")
                        except json.JSONDecodeError:
                            print("✗ (parse error)")
                    else:
                        print(f"✗ (exit {proc.returncode})")

        # --- Analyze seam clips ---
        if what in ("all", "seams"):
            seams_dir = Path(str(scenes)).parent / "seams"
            if not seams_dir.exists():
                print(f"  ⚠ No seams dir at {seams_dir}. Run `scenecut seams` first.")
            else:
                manifest_path = seams_dir / "manifest.json"
                if manifest_path.exists():
                    seam_manifest = json.load(open(manifest_path))
                    seam_list = seam_manifest.get("seams", [])
                else:
                    seam_list = [{"file": f, "cut_index": i} for i, f in enumerate(sorted(os.listdir(seams_dir))) if f.endswith(".mp4")]

                total = len(seam_list)
                for i, seam_info in enumerate(seam_list):
                    seam_file = seam_info.get("file", f"seam_{i:03d}.mp4")
                    seam_path = seams_dir / seam_file
                    if not seam_path.exists():
                        continue

                    out_file = outdir / f"seam_{i:03d}_vlm.json"
                    if out_file.exists():
                        print(f"  [{i+1}/{total}] Seam {i} — cached")
                        results["seams"].append({"index": i, "file": seam_file, "result": str(out_file), "cached": True})
                        continue

                    print(f"  [{i+1}/{total}] Analyzing seam {i}...", end=" ", flush=True)
                    proc = subprocess.run(
                        ["bun", "run", str(vlm_script), str(seam_path), "seam", str(out_file)],
                        capture_output=True, text=True, timeout=120,
                        cwd=str(Path(__file__).resolve().parent.parent),
                    )
                    if proc.returncode == 0:
                        try:
                            res = json.loads(proc.stdout.strip().split("\n")[-1])
                            if res.get("ok"):
                                print(f"✓ ({res.get('content_length', 0)} chars)")
                                results["seams"].append({"index": i, "file": seam_file, "result": str(out_file)})
                            else:
                                print(f"✗ {res.get('error', 'unknown')}")
                        except json.JSONDecodeError:
                            print("✗ (parse error)")
                    else:
                        print(f"✗ (exit {proc.returncode})")

        # Write consolidated report
        report = {
            "project_id": applied.get("id", ""),
            "source": os.path.basename(src_path),
            "analyzed_at": __import__("time").time(),
            "scene_count": len(results["scenes"]),
            "seam_count": len(results["seams"]),
            "results": results,
        }
        report_path = outdir / "report.json"
        with open(report_path, "w") as f:
            json.dump(report, f, indent=2)

        _emit({
            "scenes_analyzed": len(results["scenes"]),
            "seams_analyzed": len(results["seams"]),
            "report": str(report_path),
            "outdir": str(outdir),
        }, json_out)

    except Exception as e:
        _err(e, json_out)


@app.command("report")
def report_cmd(project: Path = typer.Argument(
                   ..., help="Project directory (containing project.json) "
                             "or a project JSON file"),
               output: Optional[Path] = typer.Option(
                   None, "--output", "-o",
                   help="Write the Markdown report to this file (default: stdout)"),
               json_out: bool = typer.Option(
                   False, "--json", help="JSON status output")):
    """Generate a Markdown cinematography report from stored VLM scene analyses.

    Sources (merged, project JSON entries win on conflict):
      - <project dir>/analysis/scene_NNN_vlm.json — per-scene files written
        by `scenecut analyze` (raw 14-point cinematography answers, parsed)
      - structured project["analysis"] entries, when present

    The report contains summary stats (scenes, duration, analyzed count),
    a scene overview table with timecodes, frequency observations over
    categorical fields (shot size, movement, ...), and per-scene detail
    sections with ALL stored analysis fields. Unanalyzed scenes are noted.
    """
    from .project import load_project, apply_overrides
    from .report import generate_report, load_scene_analyses, collect_analysis
    try:
        p = Path(project)
        if p.is_dir():
            pjson = p / "project.json"
            if not pjson.exists():
                raise SceneCutError(f"No project.json in directory: {p}", code=2)
        else:
            pjson = p
            if not pjson.exists():
                raise SceneCutError(f"Project file not found: {p}", code=2)

        proj = load_project(pjson)
        disk = load_scene_analyses(pjson)
        md = generate_report(proj, analyses=disk)

        applied = apply_overrides(proj)
        entries = collect_analysis(applied, disk)
        n_scenes = len(applied.get("scenes", []))
        n_analyzed = sum(1 for i in range(n_scenes) if i in entries)

        if output is not None:
            out = Path(output)
            out.parent.mkdir(parents=True, exist_ok=True)
            out.write_text(md, encoding="utf-8")
            if json_out:
                typer.echo(json.dumps({"ok": True, "path": str(out),
                                       "n_scenes": n_scenes,
                                       "n_analyzed": n_analyzed}))
            else:
                typer.echo(f"Report written to {out} — {n_scenes} scenes, "
                           f"{n_analyzed} analyzed")
        else:
            if json_out:
                # stdout mode would otherwise lose the markdown — include it.
                typer.echo(json.dumps({"ok": True, "path": None,
                                       "n_scenes": n_scenes,
                                       "n_analyzed": n_analyzed,
                                       "markdown": md}))
            else:
                typer.echo(md)
    except Exception as e:
        _err(e, json_out)


@app.command()
def web(port: int = typer.Option(3000, "--port"),
        open_browser: bool = typer.Option(False, "--open")):
    """Launch the web UI (informational — Next.js runs separately)."""
    typer.echo(f"Web UI is served by the Next.js dev server on port {port}.")
    typer.echo(f"Open: http://localhost:{port}")
    if open_browser:
        import webbrowser
        webbrowser.open(f"http://localhost:{port}")


@app.command("apply-markers")
def apply_markers(
    scenes: Path = typer.Argument(..., help="Scene file (JSON) produced by `detect` (typically on a proxy)"),
    video: Path = typer.Argument(..., help="Target video to apply markers to (typically the full-resolution original)"),
    output: Optional[Path] = typer.Option(None, "--output", "-o", help="Output scene file (default: in-place)"),
    tolerance: float = typer.Option(0.5, "--tolerance",
                                     help="Allowed duration mismatch in seconds (proxy vs original)"),
    dry_run: bool = typer.Option(False, "--dry-run", help="Don't write, just print what would change"),
    json: bool = typer.Option(False, "--json"),
):
    """Apply proxy-derived scene markers to a different (typically higher-res) video.

    Useful workflow:
      1. scenecut preprocess big.mp4 --height 360 --output proxy.mp4
      2. scenecut detect proxy.mp4 --output scenes.json --thumbnails
      3. (optional) scenecut audit scenes.json   # fix any missed cuts
      4. scenecut apply-markers scenes.json big.mp4 --output big.scenes.json
      5. scenecut cut big.mp4 --scenes big.scenes.json --outdir big_clips/

    This command:
      - Probes the target video's metadata
      - Validates duration matches the scene file's source duration (within --tolerance)
      - Rewrites source metadata (path, fps, width, height, codec, duration, etc.)
      - Validates every cut's frame_num still makes sense at the new fps
        (if proxy and original have the same fps, this is a no-op; if fps
        differs, cut seconds are preserved and frame_num is recomputed)
      - Thumbnails are cleared (they were extracted from the proxy, not the original)
    """
    from .project import load_project, save_project
    try:
        project = load_project(scenes)
        target_info = probe_video(video)

        # Duration sanity check
        src = project.get("source", {})
        old_dur = float(src.get("duration", 0))
        new_dur = target_info.duration
        if old_dur <= 0:
            raise SceneCutError(f"Scene file has no source.duration — was it produced by `detect`?", code=2)
        mismatch = abs(new_dur - old_dur)
        if mismatch > tolerance:
            raise SceneCutError(
                f"Duration mismatch too large: scene file says {old_dur:.3f}s but "
                f"target video is {new_dur:.3f}s (diff {mismatch:.3f}s > tolerance {tolerance}s). "
                f"Pass --tolerance {mismatch + 0.1:.1f} to override, but this likely means "
                f"the scene file was produced from a DIFFERENT video.",
                code=2,
            )

        old_fps = float(src.get("fps", 0)) or 30.0
        new_fps = target_info.fps

        # Rewrite source metadata
        project["source"] = target_info.to_dict()

        # Validate cuts: keep seconds (always correct), recompute frame_num for new fps
        for c in project.get("cuts", []):
            c["frame_num"] = int(round(c["seconds"] * new_fps))

        # Validate scenes: recompute start_frame / end_frame for new fps
        for s in project.get("scenes", []):
            s["start_frame"] = int(round(s["start"] * new_fps))
            s["end_frame"] = int(round(s["end"] * new_fps))
            # Clear proxy-derived thumbnails — they're from the proxy, not the original
            s["thumbnail"] = None

        # Add a marker noting this was remapped
        project.setdefault("history", []).append({
            "action": "apply_markers",
            "from": {"path": str(scenes), "duration": old_dur, "fps": old_fps},
            "to": {"path": str(video), "duration": new_dur, "fps": new_fps},
            "duration_mismatch": mismatch,
            "tolerance": tolerance,
        })

        out_path = output or scenes
        if not dry_run:
            save_project(project, out_path)

        result = {
            "input_scenes": str(scenes),
            "target_video": str(video),
            "output": str(out_path),
            "duration_mismatch_seconds": round(mismatch, 4),
            "fps_change": f"{old_fps:.3f} -> {new_fps:.3f}" if abs(old_fps - new_fps) > 0.01 else "unchanged",
            "n_cuts": len(project.get("cuts", [])),
            "n_scenes": len(project.get("scenes", [])),
            "thumbnails_cleared": True,
            "dry_run": dry_run,
        }
        _emit(result, json)
    except Exception as e:
        _err(e, json)


# ------------------------------------------------------------------ project management


projects_app = typer.Typer(help="Project management")
app.add_typer(projects_app, name="project")


@projects_app.command("create")
def project_create(input: Path = typer.Argument(...),
                   name: Optional[str] = typer.Option(None, "--name"),
                   json_out: bool = typer.Option(False, "--json", help="JSON output")):
    """Create a new project from an input video."""
    import uuid as _uuid
    pid = _uuid.uuid4().hex[:12]
    pdir = project_dir(pid)
    info = probe_video(input)
    # Copy/symlink video into project
    src_link = pdir / "source.mp4"
    if not src_link.exists():
        try:
            src_link.symlink_to(Path(input).resolve())
        except OSError:
            import shutil
            shutil.copy2(input, src_link)
    from .project import empty_project, save_project
    project = empty_project(info.to_dict())
    project["id"] = pid
    project["name"] = name or Path(input).stem
    project["source"]["path"] = str(src_link)
    save_project(project, project_file(pid))
    _emit({"project_id": pid, "path": str(project_file(pid)), "project": project}, json_out)


@projects_app.command("list")
def project_list(json_out: bool = typer.Option(False, "--json", help="JSON output")):
    """List all projects."""
    out = []
    for p in PROJECTS_DIR.iterdir():
        if not p.is_dir() or p.name.startswith("."):
            continue
        pf = p / "project.json"
        if pf.exists():
            from .project import load_project
            try:
                proj = load_project(pf)
                out.append({
                    "id": proj.get("id", p.name),
                    "name": proj.get("name", p.name),
                    "source": proj.get("source", {}).get("path", "?"),
                    "duration": proj.get("source", {}).get("duration", 0),
                    "n_scenes": len(proj.get("scenes", [])),
                    "n_cuts": len(proj.get("cuts", [])),
                    "created_at": proj.get("created_at", 0),
                    "updated_at": proj.get("updated_at", 0),
                    "path": str(pf),
                })
            except Exception:
                continue
    out.sort(key=lambda p: p.get("updated_at", 0), reverse=True)
    _emit(out, json_out)


@projects_app.command("show")
def project_show(id: str = typer.Argument(...),
                 json_out: bool = typer.Option(False, "--json", help="JSON output")):
    """Show a project's full JSON."""
    from .project import load_project
    pf = project_file(id)
    if not pf.exists():
        _err(SceneCutError(f"Project not found: {id}", code=2), json_out)
        return
    typer.echo(json.dumps(load_project(pf), indent=2, default=str))


@projects_app.command("delete")
def project_delete(id: str = typer.Argument(...),
                   json_out: bool = typer.Option(False, "--json", help="JSON output")):
    """Delete a project."""
    import shutil
    pdir = PROJECTS_DIR / id
    if pdir.exists():
        shutil.rmtree(pdir)
        _emit({"deleted": id}, json_out)
    else:
        _err(SceneCutError(f"Project not found: {id}", code=2), json_out)



@app.command("autotune")
def autotune_cmd(input: Path = typer.Argument(..., help="Input video file"),
                 scenes: Optional[Path] = typer.Option(
                     None, "--scenes",
                     help="Existing project.json — starting params inherit its "
                          "detector block (A12); autotune always re-detects fresh"),
                 proxy: Optional[Path] = typer.Option(None, "--proxy",
                                                      help="Proxy video for detection"),
                 audit: str = typer.Option("vlm", "--audit",
                                           help="vlm | heuristic (offline fallback)"),
                 max_iterations: int = typer.Option(3, "--max-iterations"),
                 max_calls: int = typer.Option(200, "--max-calls",
                                               help="VLM call budget (incl. retries, "
                                                    "tie-breaks, recall guard)"),
                 seed: int = typer.Option(42, "--seed"),
                 apply: bool = typer.Option(False, "--apply",
                                            help="Write the tuned project (last-accepted "
                                                 "params only; see --force)"),
                 force: bool = typer.Option(False, "--force",
                                            help="With --apply: also write when nothing "
                                                 "was accepted (current params)"),
                 output: Optional[Path] = typer.Option(None, "--output", "-o",
                                                       help="Output project path "
                                                            "(default: <input>.scenes.json)"),
                 json_out: bool = typer.Option(False, "--json",
                                               help="NDJSON progress events + final "
                                                    "summary on stdout")):
    """Auto-tune detection params via VLM-audited false-split sampling (D9).

    Statistical protocol (v3.1): stratified sample (confidence quartile x
    source, census when population <= 24), 2-call decorrelated VLM vote per
    seam, HT estimate + Wilson-95%-upper acceptance gate (<= 0.20),
    source-routed monotone tightening, recall guard on scene interiors.
    Report: <output.parent>/autotune/report.json.
    """
    from .autotune import autotune as run_autotune
    try:
        start_params = None
        if scenes is not None:
            from .project import load_project
            proj = load_project(scenes)
            start_params = proj.get("detector", {})

        if json_out:
            def progress(ev: dict):
                typer.echo(json.dumps(ev, default=str))
        else:
            _HUMAN = {
                "iteration_start": lambda e: f"\n[iter {e['iteration']}] params: {e['params']}",
                "detected": lambda e: f"  detected {e['n_cuts']} cuts / {e['n_scenes']} scenes",
                "sampled": lambda e: f"  sampled {e['n']}/{e['target']} ({e['mode']}; population {e['realized_population']})",
                "seam_verdict": lambda e: f"  cut {e['cut']}: {e['verdict'] or 'abstain'} ({e['agreement']}, conf {e.get('confidence')})",
                "guard_verdict": lambda e: f"  interior s{e['scene']}: shot_change={e['shot_change']}",
                "estimate": lambda e: (f"  estimate p_hat={e['p_hat']} k={e['k_falses']}/{e['n_audited']} "
                                       f"wilson95 upper={e['wilson_upper_95']} -> {'ACCEPT' if e['accepted'] else 'tighten'}"),
                "adjust": lambda e: f"  adjust {e['klass']}: {e['knob']} {e['from_']} -> {e['to']}",
                "tripwire": lambda e: f"  ! recall tripwire: cut-count drop {e.get('drop_prev', e.get('drop_orig'))}",
                "accepted": lambda e: f"  ACCEPTED (params {e['params']})",
                "heuristic_converged": lambda e: "  heuristic converged (no suspects)",
                "no_cuts": lambda e: "  zero cuts — nothing to audit",
                "budget_exhausted": lambda e: f"  budget exhausted ({e.get('remaining')} left)",
                "vlm_unavailable": lambda e: f"  VLM unavailable (abstain rate {e.get('abstain_rate')})",
                "no_adjustment": lambda e: f"  no routable adjustment ({e.get('status')})",
                "apply": lambda e: f"  applied -> {e['path']}",
                "apply_skipped": lambda e: f"  apply skipped: {e['reason']}",
            }

            def progress(ev: dict):
                fn = _HUMAN.get(ev.get("event", ""))
                if fn:
                    typer.echo(fn(ev))
                elif ev.get("event") == "done":
                    typer.echo(f"\ndone: {ev.get('status')} ({ev.get('calls_used')} VLM calls)")

        report = run_autotune(
            str(input), proxy=str(proxy) if proxy else None, audit=audit,
            max_iterations=max_iterations, max_calls=max_calls, seed=seed,
            start_params=start_params, apply=apply, force=force,
            output=str(output) if output else None, progress_cb=progress)
        if json_out:
            typer.echo(json.dumps({"event": "summary", **report}, default=str))
        else:
            est = report.get("false_split_rate_estimate") or {}
            typer.echo(f"\nstatus: {report['status']}")
            if est:
                typer.echo(f"false-split estimate: p_hat={est.get('p_hat')} "
                           f"({est.get('k_falses')}/{est.get('n_audited')} audited) "
                           f"wilson95-upper={est.get('wilson_upper_95')}")
            typer.echo(f"recommended params: {report['recommended_params']}")
            typer.echo(f"report: {report['report_path']}")
    except Exception as e:
        _err(e, json_out)


@app.command("bench")
def bench_cmd(datasets: str = typer.Option("local", "--dataset",
                                           help="comma list: local,bbc,clipshots"),
              config: str = typer.Option("default", "--config",
                                         help="default|conservative|sensitive"),
              params: Optional[str] = typer.Option(None, "--params",
                                                   help="JSON param overrides"),
              neural: bool = typer.Option(False, "--neural",
                                           help="Enable the TransNetV2 pass (scenecut[neural])"),
              quick: bool = typer.Option(False, "--quick",
                                         help="BBC: first 3 episodes; ClipShots: first 100 sorted"),
              gate: bool = typer.Option(False, "--gate",
                                        help="Regression gate vs bench/golden.json (exit 1 floor-fail, 3 missing)"),
              download: Optional[str] = typer.Option(None, "--download",
                                                     help="Download a dataset (bbc) then exit"),
              output: Optional[Path] = typer.Option(None, "--output", "-o"),
              json_out: bool = typer.Option(False, "--json")):
    """Benchmark detection accuracy vs ground truth (D12).

    Local corpus: 10 fixtures + ground_truth.json. BBC Planet Earth:
    `--download bbc` first (Zenodo 14873790, ~4.7GB, resumable). ClipShots:
    place videos + annotations under datasets/clipshots/ manually.
    Metrics: cut-level greedy nearest matching (tau 0 and 1), micro-aggregated;
    frame-level secondary; gradual = point-in-interval with fade-first ordering.
    """
    from .bench import run_bench, run_gate, default_golden_path
    from .bench_datasets import DATASETS, download_bbc
    try:
        if download:
            dest = Path("datasets") / download
            if download == "bbc":
                res = download_bbc(dest)
                _emit(res, json_out)
            else:
                _err(SceneCutError(f"no downloader for {download!r}", code=2), json_out)
            return

        param_over = json.loads(params) if params else {}
        if neural:
            # Flows to detect_scenes as a kwarg when the D13 pass is enabled.
            param_over = {**param_over, "neural": True}
        ds_names = [d.strip() for d in datasets.split(",") if d.strip()]
        results = {}
        for name in ds_names:
            if name not in DATASETS:
                _err(SceneCutError(
                    f"unknown dataset {name!r} (have: {sorted(DATASETS)})", code=2), json_out)
                return
            rep = run_bench(name, config, param_over, quick=quick)
            results[name] = rep.get("datasets", {}).get(name, rep)
        out = {"datasets": results, "config": config, "params": param_over,
               "quick": quick}
        if gate:
            g = run_gate(out, default_golden_path(), quick=quick)
            out["gate"] = g
            if json_out:
                typer.echo(json.dumps(out, indent=2, default=str))
            else:
                _emit(out, True)
            raise typer.Exit(code={"pass": 0, "fail_floor": 1,
                                   "fail_missing": 3, "no_golden": 4}.get(g.get("gate", "no_golden"), 2))
        if output:
            Path(output).write_text(json.dumps(out, indent=2, default=str))
        _emit({"benchmarks": list(results.keys()),
               "report": str(output) if output else None,
               "summary": {k: ((v.get("aggregate") or {})
                               .get("hard_cuts@tol1", {}) or {}).get("f1")
                           for k, v in results.items()}},
              json_out)
    except typer.Exit:
        raise
    except Exception as e:
        _err(e, json_out)


if __name__ == "__main__":
    app()
