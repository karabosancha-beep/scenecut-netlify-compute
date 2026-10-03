"""Cut a video into per-scene clips using ffmpeg."""
from __future__ import annotations

import re
from pathlib import Path

from .project import apply_overrides
from .util import SceneCutError, ffmpeg_path, run

# --format accepts both codec directives and container names. "copy" is a
# codec directive (stream copy, no re-encode): the Web UI's CUT_FORMATS
# ("copy" | "mp4" | "mkv" — src/lib/validation.ts) forwards it verbatim, so
# it must be mapped to a real container for the output EXTENSION — otherwise
# ffmpeg gets "scene_NNN.copy" and fails with "Unable to find a suitable
# output format" (surfaced as an API 500 on POST /cut and /tar).
_FORMAT_CONTAINERS = {"copy": "mp4", "mp4": "mp4", "mkv": "mkv"}


def cut_video(input_path: str | Path,
              project: dict,
              outdir: str | Path | None = None,
              accurate: bool = False,
              format: str = "mp4",
              crf: int = 20,
              preset: str = "veryfast",
              keep_audio: bool = True,
              progress_cb=None) -> list[dict]:
    """Cut input into per-scene clips. Returns list of {scene_index, path, start, end, duration}.

    ``format`` selects the output: "mp4"/"mkv" name the container (used as
    the clip extension), while "copy" requests stream-copy encoding into an
    mp4 container (honoured even with ``accurate=True`` — an explicit codec
    directive beats the re-encode-at-boundaries default).
    """
    input_path = str(input_path)
    fmt = str(format or "mp4").strip().lower()
    # Map the codec directive to a container; unknown values keep the legacy
    # behavior of being used verbatim as the extension (e.g. "mov").
    container = _FORMAT_CONTAINERS.get(fmt, fmt)
    reencode = accurate and fmt != "copy"
    project = apply_overrides(project)
    scenes = project["scenes"]
    fps = project["source"].get("fps", 30)

    if outdir is None:
        outdir = Path(input_path).parent / (Path(input_path).stem + "_clips")
    outdir = Path(outdir)
    outdir.mkdir(parents=True, exist_ok=True)

    results = []
    for i, scene in enumerate(scenes):
        start = scene["start"]
        end = scene["end"]
        duration = end - start
        if duration <= 0:
            continue
        out_path = outdir / f"scene_{i:03d}.{container}"

        # Build ffmpeg command
        # For accurate cuts, place -ss after -i (slow, accurate)
        # For fast cuts (stream copy), place -ss before -i (keyframe-aligned)
        if reencode:
            cmd = [
                ffmpeg_path(), "-y",
                "-i", input_path,
                "-ss", f"{start:.3f}",
                "-t", f"{duration:.3f}",
                "-c:v", "libx264", "-preset", preset, "-crf", str(crf),
                "-c:a", "aac" if keep_audio else "an",
                "-movflags", "+faststart",
                str(out_path),
            ]
            if not keep_audio:
                cmd[cmd.index("aac") if "aac" in cmd else -1] = "an"
        else:
            # Stream copy (fast, keyframe-aligned)
            cmd = [
                ffmpeg_path(), "-y",
                "-ss", f"{start:.3f}",
                "-i", input_path,
                "-t", f"{duration:.3f}",
                "-c", "copy",
                "-avoid_negative_ts", "make_zero",
                str(out_path),
            ]

        try:
            run(cmd)
            results.append({
                "scene_index": i,
                "path": str(out_path),
                "start": start,
                "end": end,
                "duration": duration,
                "accurate": reencode,
            })
        except SceneCutError as e:
            # Stream copy may fail on some videos; fall back to re-encode
            if not reencode:
                cmd = [
                    ffmpeg_path(), "-y",
                    "-ss", f"{start:.3f}",
                    "-i", input_path,
                    "-t", f"{duration:.3f}",
                    "-c:v", "libx264", "-preset", preset, "-crf", str(crf),
                    "-c:a", "aac" if keep_audio else "an",
                    "-movflags", "+faststart",
                    str(out_path),
                ]
                try:
                    run(cmd)
                    results.append({
                        "scene_index": i,
                        "path": str(out_path),
                        "start": start,
                        "end": end,
                        "duration": duration,
                        "accurate": True,
                        "fallback": True,
                    })
                except SceneCutError:
                    raise
            else:
                raise
        if progress_cb:
            progress_cb((i + 1) / len(scenes))
    return results
