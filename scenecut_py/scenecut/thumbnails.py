"""Generate thumbnails for scenes."""
from __future__ import annotations

from pathlib import Path

import cv2

from .util import SceneCutError


def make_thumbnails(input_path: str | Path,
                    project: dict,
                    outdir: str | Path | None = None,
                    positions: str = "start",  # "start" | "mid" | "both"
                    size: int = 320,
                    format: str = "jpg") -> list[dict]:
    """Generate thumbnails for each scene.

    NOTE: `project` should already have overrides applied (effective scenes).
    This function does NOT call apply_overrides — the caller is responsible.
    """
    input_path = str(input_path)
    scenes = project["scenes"]

    if outdir is None:
        outdir = Path(input_path).parent / (Path(input_path).stem + "_thumbs")
    outdir = Path(outdir)
    outdir.mkdir(parents=True, exist_ok=True)

    cap = cv2.VideoCapture(input_path)
    if not cap.isOpened():
        raise SceneCutError(f"OpenCV failed to open {input_path}", code=4)

    results = []
    for i, scene in enumerate(scenes):
        # Pick frame at start (or 1 frame in to avoid black at cut)
        ts = scene["start"] + 0.05 if positions in ("start", "both") else (scene["start"] + scene["end"]) / 2
        cap.set(cv2.CAP_PROP_POS_MSEC, ts * 1000)
        ok, frame = cap.read()
        if not ok:
            results.append({"scene_index": i, "thumbnail": None})
            continue
        h, w = frame.shape[:2]
        scale = size / max(h, w)
        new_w, new_h = int(w * scale), int(h * scale)
        frame = cv2.resize(frame, (new_w, new_h), interpolation=cv2.INTER_AREA)
        out_path = outdir / f"scene_{i:03d}.{format}"
        cv2.imwrite(str(out_path), frame, [cv2.IMWRITE_JPEG_QUALITY, 85])
        results.append({"scene_index": i, "thumbnail": str(out_path)})
    cap.release()
    return results
