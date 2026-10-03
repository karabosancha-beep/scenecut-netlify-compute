"""Premiere XML (xmeml 5) export — Adobe Premiere / older Resolve."""
from __future__ import annotations

import uuid
from pathlib import Path
from xml.etree import ElementTree as ET

from ..project import apply_overrides


def export_premiere(project: dict, output, source=None, fps=None) -> str:
    """Generate FCP7 xmeml XML for Adobe Premiere."""
    project = apply_overrides(project)
    src = project["source"]
    fps = fps or src.get("fps", 30.0)
    duration = src.get("duration", 0)
    src_path = source or src.get("path", "video.mp4")
    src_name = Path(src_path).stem or "video"
    scenes = project["scenes"]
    width = src.get("width", 1920)
    height = src.get("height", 1080)

    xmeml = ET.Element("xmeml", version="5")
    sequence = ET.SubElement(xmeml, "sequence")
    ET.SubElement(sequence, "uuid").text = str(uuid.uuid4())
    ET.SubElement(sequence, "name").text = f"{src_name}_scenecut"
    ET.SubElement(sequence, "duration").text = str(int(duration * fps))
    ET.SubElement(sequence, "rate")
    rate = sequence.find("rate")
    ET.SubElement(rate, "timebase").text = str(int(round(fps)))
    ET.SubElement(rate, "ntsc").text = "TRUE" if abs(fps - round(fps)) > 0.01 else "FALSE"
    media = ET.SubElement(sequence, "media")
    video = ET.SubElement(media, "video")
    ET.SubElement(video, "format")
    fmt = video.find("format")
    ET.SubElement(fmt, "samplecharacteristics")
    sc = fmt.find("samplecharacteristics")
    ET.SubElement(sc, "width").text = str(width)
    ET.SubElement(sc, "height").text = str(height)
    ET.SubElement(sc, "pixelaspectratio").text = "square"
    ET.SubElement(video, "track")

    track = video.find("track")
    offset = 0
    for scene in scenes:
        clipitem = ET.SubElement(track, "clipitem")
        ET.SubElement(clipitem, "name").text = f"Scene {scene['index'] + 1}"
        ET.SubElement(clipitem, "enabled").text = "TRUE"
        ET.SubElement(clipitem, "duration").text = str(int(scene["duration"] * fps))
        ET.SubElement(clipitem, "rate")
        crate = clipitem.find("rate")
        ET.SubElement(crate, "timebase").text = str(int(round(fps)))
        ET.SubElement(clipitem, "start").text = str(int(offset * fps))
        ET.SubElement(clipitem, "end").text = str(int((offset + scene["duration"]) * fps))
        ET.SubElement(clipitem, "in").text = str(int(scene["start"] * fps))
        ET.SubElement(clipitem, "out").text = str(int(scene["end"] * fps))
        file_el = ET.SubElement(clipitem, "file")
        ET.SubElement(file_el, "name").text = src_name
        pathurl = ET.SubElement(file_el, "pathurl")
        pathurl.text = f"file://{Path(src_path).resolve()}"
        offset += scene["duration"]

    ET.indent(xmeml, space="  ")
    xml_str = '<?xml version="1.0" encoding="UTF-8"?>\n<!DOCTYPE xmeml>\n' + ET.tostring(xmeml, encoding="unicode")
    output = Path(output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(xml_str, encoding="utf-8")
    return str(output)
