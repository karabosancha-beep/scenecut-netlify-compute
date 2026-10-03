"""CSV export — simple table of scenes."""
from __future__ import annotations

import csv
from pathlib import Path

from ..project import apply_overrides
from ..timecode import seconds_to_hmsms


def export_csv(project: dict, output) -> str:
    project = apply_overrides(project)
    scenes = project["scenes"]
    cuts = project["cuts"]
    output = Path(output)
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["scene_index", "start_seconds", "end_seconds",
                    "start_timecode", "end_timecode", "duration",
                    "type", "confidence", "thumbnail", "label"])
        for i, scene in enumerate(scenes):
            cut = next((c for c in cuts if c["index"] == i), {})
            w.writerow([
                i,
                f"{scene['start']:.3f}",
                f"{scene['end']:.3f}",
                seconds_to_hmsms(scene["start"]),
                seconds_to_hmsms(scene["end"]),
                f"{scene['duration']:.3f}",
                cut.get("type", ""),
                cut.get("confidence", ""),
                scene.get("thumbnail") or "",
                scene.get("label") or "",
            ])
    return str(output)
