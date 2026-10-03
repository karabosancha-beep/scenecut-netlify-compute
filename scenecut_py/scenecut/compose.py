"""Compose: pick a subset of scenes and concatenate."""
from __future__ import annotations

import re
from pathlib import Path

from .cut import cut_video
from .project import apply_overrides
from .util import SceneCutError, ffmpeg_path, run


def parse_pick(spec: str) -> list[int]:
    """Parse '1,3,5-7' -> [1, 3, 5, 6, 7] (1-indexed)."""
    out: list[int] = []
    for part in spec.split(","):
        part = part.strip()
        if not part:
            continue
        if "-" in part:
            a, b = part.split("-", 1)
            out.extend(range(int(a), int(b) + 1))
        else:
            out.append(int(part))
    return out


def compose(input_path: str | Path,
            project: dict,
            pick: str = "all",
            output: str | Path | None = None,
            accurate: bool = True) -> str:
    """Pick a subset of scenes and concatenate them."""
    input_path = str(input_path)
    project = apply_overrides(project)
    scenes = project["scenes"]

    if pick == "all":
        indices = list(range(len(scenes)))
    else:
        indices = [i - 1 for i in parse_pick(pick) if 1 <= i <= len(scenes)]

    if not indices:
        raise SceneCutError("No scenes selected", code=5)

    if output is None:
        output = Path(input_path).with_suffix(f".composed{Path(input_path).suffix or '.mp4'}")
    output = Path(output)
    output.parent.mkdir(parents=True, exist_ok=True)

    # Cut each picked scene to a temp dir, then concat
    tmp_dir = output.parent / f".compose_tmp_{output.stem}"
    tmp_dir.mkdir(exist_ok=True)

    # Cut all picked scenes (re-encode for safe concat)
    tmp_clips = []
    for idx in indices:
        scene = scenes[idx]
        out_path = tmp_dir / f"clip_{idx:03d}.mp4"
        cmd = [
            ffmpeg_path(), "-y",
            "-i", input_path,
            "-ss", f"{scene['start']:.3f}",
            "-t", f"{scene['duration']:.3f}",
            "-c:v", "libx264", "-preset", "veryfast", "-crf", "20",
            "-c:a", "aac", "-b:a", "128k",
            "-vf", "scale=trunc(iw/2)*2:trunc(ih/2)*2,setsar=1,fps=30",
            "-r", "30",
            "-movflags", "+faststart",
            str(out_path),
        ]
        run(cmd)
        tmp_clips.append(out_path)

    # Write concat list
    list_file = tmp_dir / "concat.txt"
    with list_file.open("w") as f:
        for c in tmp_clips:
            f.write(f"file '{c.absolute()}'\n")

    # Concat
    cmd = [
        ffmpeg_path(), "-y",
        "-f", "concat", "-safe", "0",
        "-i", str(list_file),
        "-c", "copy",
        str(output),
    ]
    run(cmd)

    # Cleanup
    for c in tmp_clips:
        try: c.unlink()
        except Exception: pass
    try: list_file.unlink()
    except Exception: pass
    try: tmp_dir.rmdir()
    except Exception: pass

    return str(output)
