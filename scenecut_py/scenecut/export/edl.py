"""EDL (CMX 3600) export."""
from __future__ import annotations

from pathlib import Path

from ..project import apply_overrides
from ..timecode import seconds_to_tcff
from ..util import SceneCutError


def export_edl(project: dict, output, source=None, reel_name="SCENE", fps=None) -> str:
    """Generate CMX 3600 EDL."""
    project = apply_overrides(project)
    src = project["source"]
    fps = fps or src.get("fps", 30.0)
    src_path = source or src.get("path", "video.mp4")
    scenes = project["scenes"]
    title = Path(src_path).stem.upper().replace(" ", "_")[:7] or "VIDEO"

    lines = []
    lines.append(f"TITLE: {title}")
    lines.append(f"FCM: NON-DROP FRAME")
    lines.append("")

    for i, scene in enumerate(scenes):
        evt = i + 1
        src_in = seconds_to_tcff(scene["start"], fps)
        src_out = seconds_to_tcff(scene["end"], fps)
        rec_in = seconds_to_tcff(scene["start"], fps)
        rec_out = seconds_to_tcff(scene["end"], fps)
        lines.append(f"{evt:03d}  AX       AA/V  C        {src_in} {src_out} {rec_in} {rec_out}")
        lines.append(f"FROM CLIP NAME: SCENE_{scene['index'] + 1:03d}")
        lines.append("")

    output = Path(output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return str(output)
