"""Preprocess: downscale video to a low-res proxy for fast detection."""
from __future__ import annotations

import shutil
from pathlib import Path

from .util import SceneCutError, ffmpeg_path, run


def preprocess(input_path: str | Path,
               height: int = 360,
               crf: int = 28,
               preset: str = "veryfast",
               fps: float | None = None,
               denoise: bool = False,
               output: str | Path | None = None) -> str:
    """Downscale a video to a proxy. Returns output path.

    denoise=True adds hqdn3d AFTER the upscale (scale-then-denoise smooths the
    upscaled block artifacts of low-res sources — DECISIONS.md D10) — for
    heavily compressed (<=360p) sources whose block-edge noise inflates
    detection deltas.
    """
    input_path = Path(input_path)
    if not input_path.exists():
        raise SceneCutError(f"Input not found: {input_path}", code=2)
    if output is None:
        output = input_path.with_suffix(f".proxy{input_path.suffix or '.mp4'}")
    output = Path(output)

    # Use scale with -2 to keep aspect ratio (width auto-derived, even number)
    vf = f"scale=-2:{height}:flags=lanczos"
    if denoise:
        vf += ",hqdn3d=4:3:6:4"
    if fps is not None:
        vf += f",fps={fps}"

    # Detect NVENC availability
    enc = "libx264"
    # Test once (cached by ffmpeg)
    test = shutil.which("ffmpeg")
    if test:
        probe = run([ffmpeg_path(), "-hide_banner", "-encoders"])
        if "h264_nvenc" in (probe.stdout or ""):
            # Try a tiny nvenc encode to confirm GPU available
            try:
                run([ffmpeg_path(), "-f", "lavfi", "-i", "color=black:s=64x64:d=0.1",
                     "-c:v", "h264_nvenc", "-f", "null", "-"])
                enc = "h264_nvenc"
            except SceneCutError:
                enc = "libx264"

    cmd = [
        ffmpeg_path(), "-y", "-i", str(input_path),
        "-vf", vf,
        "-c:v", enc,
        "-preset", "p4" if enc == "h264_nvenc" else preset,
        "-crf", str(crf) if enc == "libx264" else "23",
        "-c:a", "aac", "-b:a", "96k",
        "-movflags", "+faststart",
        str(output),
    ]
    run(cmd)
    return str(output)
