"""Generate seam clips — short clips spanning a cut point ±X seconds.

For each cut at time T between scene N and scene N+1:
  - seam_start = max(0, T - X)
  - seam_end = min(duration, T + X)
  - X = min(5, scene_N.duration / 2, scene_N+1.duration / 2)

This gives the VLM both sides of the cut in one continuous video, enabling:
  - Cut type identification (match cut, jump cut, dissolve, etc.)
  - False-positive detection (is this actually a continuous shot?)
"""
from __future__ import annotations

import json
import os
from pathlib import Path

from .project import apply_overrides
from .util import SceneCutError, ffmpeg_path, run, probe_video


def generate_seam_clips(input_path: str | Path,
                        project: dict,
                        outdir: str | Path | None = None,
                        max_seam_radius: float = 5.0,
                        height: int = 360,
                        crf: int = 23,
                        cut_indices: list[int] | None = None) -> list[dict]:
    """Generate seam clips at each cut point.

    Args:
        cut_indices: restrict generation to these cut indices (D9-impl — autotune
            audits a sampled subset; None = all cuts, backward compatible).

    Returns list of {cut_index, seam_start, seam_end, seam_duration, cut_time, path}.
    """
    input_path = str(input_path)
    project = apply_overrides(project)
    scenes = project["scenes"]
    src_info = probe_video(input_path)
    duration = src_info.duration

    if outdir is None:
        outdir = Path(input_path).parent / "seams"
    outdir = Path(outdir)
    outdir.mkdir(parents=True, exist_ok=True)

    results = []
    # For each adjacent scene pair, the cut is at the boundary
    for i in range(len(scenes) - 1):
        if cut_indices is not None and i not in cut_indices:
            continue
        scene_a = scenes[i]
        scene_b = scenes[i + 1]
        cut_time = scene_a["end"]  # = scene_b["start"]

        # X = min(max_radius, half of each adjacent scene's duration)
        x = min(max_seam_radius,
                scene_a["duration"] / 2,
                scene_b["duration"] / 2)
        # Ensure at least 0.5s on each side
        x = max(x, 0.5)

        seam_start = max(0, cut_time - x)
        seam_end = min(duration, cut_time + x)
        seam_duration = seam_end - seam_start

        if seam_duration < 0.2:
            continue

        out_path = outdir / f"seam_{i:03d}.mp4"
        cmd = [
            ffmpeg_path(), "-y", "-hide_banner", "-loglevel", "error",
            "-ss", f"{seam_start:.3f}",
            "-i", input_path,
            "-t", f"{seam_duration:.3f}",
            "-vf", f"scale=-2:{height}",
            "-c:v", "libx264", "-preset", "veryfast", "-crf", str(crf),
            "-pix_fmt", "yuv420p",
            "-an",  # no audio for VLM
            "-movflags", "+faststart",
            str(out_path),
        ]
        try:
            run(cmd)
            size_bytes = os.path.getsize(out_path)
            results.append({
                "cut_index": i,
                "cut_time": round(cut_time, 3),
                "seam_start": round(seam_start, 3),
                "seam_end": round(seam_end, 3),
                "seam_duration": round(seam_duration, 3),
                "seam_radius": round(x, 3),
                "scene_before": i,
                "scene_after": i + 1,
                "path": str(out_path),
                "size_bytes": size_bytes,
            })
        except SceneCutError:
            continue

    # Write manifest
    manifest = {
        "source": os.path.basename(input_path),
        "source_duration": duration,
        "seam_count": len(results),
        "max_seam_radius": max_seam_radius,
        "seams": [
            {
                "cut_index": r["cut_index"],
                "cut_time": r["cut_time"],
                "seam_start": r["seam_start"],
                "seam_end": r["seam_end"],
                "seam_duration": r["seam_duration"],
                "scene_before": r["scene_before"],
                "scene_after": r["scene_after"],
                "file": os.path.basename(r["path"]),
                "size_bytes": r["size_bytes"],
            }
            for r in results
        ],
    }
    manifest_path = outdir / "manifest.json"
    with open(manifest_path, "w") as f:
        json.dump(manifest, f, indent=2)

    return results


def generate_interior_clips(project: dict,
                            outdir: str | Path,
                            n_windows: int = 4,
                            window: float = 4.0,
                            cap: float = 6.0,
                            longest: int = 2,
                            rng: "random.Random | None" = None,
                            height: int = 360,
                            crf: int = 23,
                            source_path: str | None = None) -> list[dict]:
    """Generate scene-INTERIOR window clips for the autotune recall guard (D9-impl).

    Windows are strictly interior (no cut inside by construction): centered on
    scene midpoints. Scenes shorter than 2×window/2 = window seconds are
    ineligible for the random stratum; the `longest` stratum takes the N
    longest scenes with a capped window (A16/A18: cap bounds VLM input size).

    Returns list of {scene_index, start, end, duration, stratum, path}.
    """
    import random as _random

    input_path = source_path or project.get("source", {}).get("path", "")
    if not input_path or not os.path.exists(input_path):
        raise SceneCutError(
            f"interior clips need the source video path (got {input_path!r})", code=2)

    rng = rng or _random.Random()
    outdir = Path(outdir)
    outdir.mkdir(parents=True, exist_ok=True)
    src_info = probe_video(input_path)
    duration = src_info.duration

    scenes = project["scenes"]
    # Random stratum: scenes long enough for a full window at the midpoint.
    eligible = [s for s in scenes if s["duration"] >= window]
    picked = rng.sample(eligible, min(n_windows, len(eligible))) if eligible else []

    # Longest stratum: N longest scenes (dedup vs picked), window capped.
    by_len = sorted(scenes, key=lambda s: -s["duration"])
    longest_picked = []
    for s in by_len:
        if len(longest_picked) >= longest:
            break
        if s in picked:
            continue
        if s["duration"] >= 2.0:  # need >= 2s for a meaningful interior window
            longest_picked.append(s)

    windows: list[dict] = []
    for s in picked:
        w = window
        mid = s["start"] + s["duration"] / 2
        windows.append({"scene_index": s["index"], "start": mid - w / 2,
                        "end": mid + w / 2, "stratum": "random"})
    for s in longest_picked:
        w = min(cap, max(2.0, s["duration"] / 3))
        mid = s["start"] + s["duration"] / 2
        windows.append({"scene_index": s["index"], "start": mid - w / 2,
                        "end": mid + w / 2, "stratum": "longest"})

    results = []
    for k, win in enumerate(windows):
        start = max(0.0, win["start"])
        end = min(duration, win["end"])
        dur = end - start
        if dur < 1.0:
            continue
        out_path = outdir / f"interior_{k:03d}.mp4"
        cmd = [
            ffmpeg_path(), "-y", "-hide_banner", "-loglevel", "error",
            "-ss", f"{start:.3f}",
            "-i", input_path,
            "-t", f"{dur:.3f}",
            "-vf", f"scale=-2:{height}",
            "-c:v", "libx264", "-preset", "veryfast", "-crf", str(crf),
            "-pix_fmt", "yuv420p", "-an", "-movflags", "+faststart",
            str(out_path),
        ]
        try:
            run(cmd)
            results.append({
                "scene_index": win["scene_index"],
                "stratum": win["stratum"],
                "start": round(start, 3), "end": round(end, 3),
                "duration": round(dur, 3),
                "path": str(out_path),
            })
        except SceneCutError:
            continue

    return results
