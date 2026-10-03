"""FCPXML 1.9 export — Final Cut Pro X / DaVinci Resolve 17+."""
from __future__ import annotations

import math
import uuid
from pathlib import Path
from xml.etree import ElementTree as ET

from ..project import apply_overrides
from ..timecode import seconds_to_tcff
from ..util import SceneCutError


def _rational(seconds: float, fps: float) -> str:
    """Convert seconds to a rational time string 'N/Ms'."""
    base = round(fps * 100)
    n = int(round(seconds * base))
    return f"{n}/{base}s"


def export_fcpxml(project: dict, output, source=None, fps=None) -> str:
    """Generate FCPXML 1.9 file."""
    project = apply_overrides(project)
    src = project["source"]
    fps = fps or src.get("fps", 30.0)
    duration = src.get("duration", 0)
    src_path = source or src.get("path", "video.mp4")
    src_name = Path(src_path).stem or "video"
    scenes = project["scenes"]

    fcpxml = ET.Element("fcpxml", version="1.9")
    resources = ET.SubElement(fcpxml, "resources")
    asset_id = f"asset-{uuid.uuid4().hex[:8]}"
    format_id = "format-0"
    ET.SubElement(resources, "format",
                  id=format_id,
                  frameRate=f"{fps:.4f}",
                  width=str(src.get("width", 1920)),
                  height=str(src.get("height", 1080)))
    asset = ET.SubElement(resources, "asset",
                          id=asset_id,
                          name=src_name,
                          start=_rational(0, fps),
                          duration=_rational(duration, fps),
                          hasVideo="1",
                          hasAudio="1" if src.get("audio_codec") else "0",
                          format=format_id,
                          uid=str(abs(hash(src_path)) % (10**12)))
    media_rep = ET.SubElement(asset, "media-rep",
                              kind="original-media",
                              src=f"file://{Path(src_path).resolve()}")

    library = ET.SubElement(fcpxml, "library")
    event = ET.SubElement(library, "event",
                          name="SceneCut Export",
                          uid=str(uuid.uuid4()))
    project_el = ET.SubElement(event, "project",
                                name=src_name,
                                uid=str(uuid.uuid4()),
                                modDate=str(int(60 * 60 * 24 * 365.25 * 55)))
    sequence = ET.SubElement(project_el, "sequence",
                              format=format_id,
                              duration=_rational(duration, fps),
                              tcStart="0s",
                              tcFormat="NDF")
    spine = ET.SubElement(sequence, "spine")

    offset = 0.0
    for scene in scenes:
        clip_dur = scene["duration"]
        if clip_dur <= 0:
            continue
        ET.SubElement(spine, "asset-clip",
                      name=f"Scene {scene['index'] + 1}",
                      ref=asset_id,
                      offset=_rational(offset, fps),
                      duration=_rational(clip_dur, fps),
                      start=_rational(scene["start"], fps),
                      tcFormat="NDF")
        offset += clip_dur

    ET.indent(fcpxml, space="  ")
    xml_str = '<?xml version="1.0" encoding="UTF-8"?>\n' + ET.tostring(fcpxml, encoding="unicode")
    output = Path(output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(xml_str, encoding="utf-8")
    return str(output)
