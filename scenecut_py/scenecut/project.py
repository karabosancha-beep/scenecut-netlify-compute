"""Project file (scenes.json) load/save and override application."""
from __future__ import annotations

import json
import time
import uuid
from pathlib import Path
from typing import Any

from .config import SCENE_FILE_VERSION
from .util import SceneCutError


def empty_project(source: dict | None = None, detector: dict | None = None) -> dict:
    return {
        "version": SCENE_FILE_VERSION,
        "id": uuid.uuid4().hex[:12],
        "created_at": time.time(),
        "updated_at": time.time(),
        "source": source or {},
        "detector": detector or {},
        "cuts": [],
        "scenes": [],
        "overrides": {
            "added_cuts": [],
            "removed_cuts": [],
            "moved_cuts": {},
            "merged_scenes": [],
            "split_scenes": [],
        },
        "labels": {},
        "tags": {},
    }


def load_project(path: str | Path) -> dict:
    path = Path(path)
    if not path.exists():
        raise SceneCutError(f"Project file not found: {path}", code=2)
    with path.open("r", encoding="utf-8") as f:
        data = json.load(f)
    # Backfill missing overrides block (older files)
    data.setdefault("overrides", {
        "added_cuts": [],
        "removed_cuts": [],
        "moved_cuts": {},
        "merged_scenes": [],
        "split_scenes": [],
    })
    for k in ("added_cuts", "removed_cuts", "moved_cuts", "merged_scenes", "split_scenes"):
        data["overrides"].setdefault(k, [] if k in ("added_cuts", "removed_cuts", "merged_scenes", "split_scenes") else {})
    data.setdefault("labels", {})
    data.setdefault("tags", {})
    return data


def save_project(project: dict, path: str | Path) -> Path:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    project["updated_at"] = time.time()
    # Atomic write (tmp + rename): the GET project route can read project.json
    # while autotune --apply / detect writes it — truncate-in-place yields
    # torn JSON under concurrency (CR2-10).
    tmp = path.with_suffix(path.suffix + ".tmp")
    with tmp.open("w", encoding="utf-8") as f:
        json.dump(project, f, indent=2, ensure_ascii=False)
    import os as _os
    _os.replace(tmp, path)
    return path


def apply_overrides(project: dict) -> dict:
    """Return a new project with overrides applied to cuts and scenes.

    - removed_cuts: drop cuts whose index is in the list
    - moved_cuts: {index: new_seconds} replace the cut time
    - added_cuts: append new cut entries
    - merged_scenes: list of [i, j] pairs of scene indices to merge
    - split_scenes: list of {scene: i, at: seconds} splits
    """
    p = json.loads(json.dumps(project))  # deep copy
    cuts = p.get("cuts", [])
    ov = p.get("overrides", {})

    # Apply removals and moves
    removed = set(ov.get("removed_cuts", []))
    moved = ov.get("moved_cuts", {})
    new_cuts = []
    for c in cuts:
        idx = c["index"]
        if idx in removed:
            continue
        if str(idx) in moved:
            c = dict(c)
            new_sec = float(moved[str(idx)])
            from .timecode import seconds_to_hmsms
            c["seconds"] = new_sec
            c["timecode"] = seconds_to_hmsms(new_sec)
            c["frame_num"] = int(round(new_sec * (c.get("fps") or p.get("source", {}).get("fps") or 30)))
            c["type"] = c.get("type", "cut")
            c["confidence"] = 1.0
            c["manual"] = True
            # D14/CR3 F4: an editor-defined position invalidates any
            # detector-derived transition interval — a stale [s,e) would
            # mis-score under interval semantics.
            c.pop("interval", None)
            c.pop("interval_fallback", None)
        new_cuts.append(c)

    # Add new cuts
    for add in ov.get("added_cuts", []):
        new_cuts.append({
            "index": -len(new_cuts) - 1,  # temporary; will be re-indexed
            "frame_num": int(round(add["seconds"] * (p.get("source", {}).get("fps") or 30))),
            "seconds": float(add["seconds"]),
            "timecode": add.get("timecode") or _hmsms(float(add["seconds"])),
            "type": add.get("type", "cut"),
            "confidence": 1.0,
            "manual": True,
        })

    # Sort by seconds and reindex
    new_cuts.sort(key=lambda c: c["seconds"])
    for i, c in enumerate(new_cuts):
        c["index"] = i

    p["cuts"] = new_cuts

    # Rebuild scenes from cuts (skip empty scenes where start == end)
    src = p.get("source", {})
    duration = src.get("duration", 0)
    fps = src.get("fps", 30)
    scenes = []
    starts = [c["seconds"] for c in new_cuts] + [duration]
    scene_idx = 0
    for i, start in enumerate(starts[:-1]):
        end = starts[i + 1]
        if end - start <= 1e-3:
            continue
        scenes.append({
            "index": scene_idx,
            "start": start,
            "end": end,
            "duration": end - start,
            "start_frame": int(round(start * fps)),
            "end_frame": int(round(end * fps)),
            "thumbnail": p.get("scenes", [{}])[i].get("thumbnail") if i < len(p.get("scenes", [])) else None,
            "label": p.get("labels", {}).get(str(i)),
            "tags": p.get("tags", {}).get(str(i), []),
            "type": new_cuts[i].get("type") if i < len(new_cuts) else "cut",
        })
        scene_idx += 1
    p["scenes"] = scenes
    return p


def _hmsms(seconds: float) -> str:
    from .timecode import seconds_to_hmsms
    return seconds_to_hmsms(seconds)


def add_override(project: dict, kind: str, value: Any) -> dict:
    """Add an override entry in-place and return project."""
    ov = project.setdefault("overrides", {})
    if kind in ("added_cuts", "removed_cuts", "merged_scenes", "split_scenes"):
        ov.setdefault(kind, []).append(value)
    elif kind == "moved_cuts":
        idx, new_sec = value
        ov.setdefault("moved_cuts", {})[str(idx)] = new_sec
    return project


def set_label(project: dict, scene_index: int, label: str | None) -> dict:
    if label is None:
        project.setdefault("labels", {}).pop(str(scene_index), None)
    else:
        project.setdefault("labels", {})[str(scene_index)] = label
    return project


def set_tags(project: dict, scene_index: int, tags: list[str]) -> dict:
    project.setdefault("tags", {})[str(scene_index)] = tags
    return project
