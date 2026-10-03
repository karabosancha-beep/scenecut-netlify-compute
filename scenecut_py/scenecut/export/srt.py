"""SRT export — subtitle-style cut list."""
from __future__ import annotations

from pathlib import Path

from ..project import apply_overrides
from ..timecode import seconds_to_hmsms


def _srt_time(seconds: float) -> str:
    """SRT timestamp HH:MM:SS,mmm."""
    h = int(seconds // 3600)
    m = int((seconds % 3600) // 60)
    s = int(seconds % 60)
    ms = int(round((seconds - int(seconds)) * 1000))
    if ms == 1000:
        ms = 0
        s += 1
    return f"{h:02d}:{m:02d}:{s:02d},{ms:03d}"


def export_srt(project: dict, output) -> str:
    project = apply_overrides(project)
    scenes = project["scenes"]
    lines = []
    for i, scene in enumerate(scenes):
        lines.append(str(i + 1))
        lines.append(f"{_srt_time(scene['start'])} --> {_srt_time(scene['end'])}")
        label = scene.get("label") or f"Scene {i + 1}"
        lines.append(f"[{label}] {scene['duration']:.2f}s ({seconds_to_hmsms(scene['start'])} - {seconds_to_hmsms(scene['end'])})")
        lines.append("")
    output = Path(output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text("\n".join(lines), encoding="utf-8")
    return str(output)
