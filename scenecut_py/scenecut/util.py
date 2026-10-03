"""Shared utilities: subprocess helpers, ffprobe, path helpers."""
from __future__ import annotations

import json
import os
import shutil
import subprocess
from dataclasses import dataclass
from pathlib import Path
from typing import Any


class SceneCutError(Exception):
    """Base error. Carries an exit code."""

    def __init__(self, message: str, code: int = 1):
        super().__init__(message)
        self.code = code


@dataclass
class VideoInfo:
    path: str
    duration: float
    fps: float
    width: int
    height: int
    codec: str
    audio_codec: str | None
    bitrate: int | None
    n_frames: int | None

    def to_dict(self) -> dict[str, Any]:
        return {
            "path": self.path,
            "duration": self.duration,
            "fps": self.fps,
            "width": self.width,
            "height": self.height,
            "codec": self.codec,
            "audio_codec": self.audio_codec,
            "bitrate": self.bitrate,
            "n_frames": self.n_frames,
        }


def ffmpeg_path() -> str:
    p = shutil.which("ffmpeg")
    if not p:
        raise SceneCutError("ffmpeg not found on PATH", code=3)
    return p


def ffprobe_path() -> str:
    p = shutil.which("ffprobe") or shutil.which("ffmpeg")
    if not p:
        raise SceneCutError("ffprobe not found", code=3)
    # ffprobe usually lives next to ffmpeg
    ffmpeg = shutil.which("ffmpeg")
    if ffmpeg:
        candidate = Path(ffmpeg).parent / "ffprobe"
        if candidate.exists():
            return str(candidate)
    return p


def ytdlp_path() -> str:
    p = shutil.which("yt-dlp")
    if not p:
        raise SceneCutError("yt-dlp not found on PATH", code=3)
    return p


def run(cmd: list[str], **kw) -> subprocess.CompletedProcess:
    """Run a subprocess, raise on failure with stderr included."""
    proc = subprocess.run(cmd, capture_output=True, text=True, **kw)
    if proc.returncode != 0:
        stderr = (proc.stderr or "").strip()
        raise SceneCutError(
            f"Command failed ({' '.join(cmd[:3])}...): {stderr[:500]}",
            code=3,
        )
    return proc


def probe_video(path: str | Path) -> VideoInfo:
    """Use ffprobe to extract video metadata."""
    path = str(path)
    if not Path(path).exists():
        raise SceneCutError(f"Input not found: {path}", code=2)
    cmd = [
        ffprobe_path(),
        "-v", "quiet",
        "-print_format", "json",
        "-show_format",
        "-show_streams",
        path,
    ]
    proc = run(cmd)
    data = json.loads(proc.stdout)

    fmt = data.get("format", {})
    streams = data.get("streams", [])
    vstream = next((s for s in streams if s.get("codec_type") == "video"), None)
    astream = next((s for s in streams if s.get("codec_type") == "audio"), None)

    if not vstream:
        raise SceneCutError(f"No video stream in {path}", code=2)

    # Parse fps (r_frame_rate is "N/M")
    fps = 30.0
    rfr = vstream.get("r_frame_rate") or vstream.get("avg_frame_rate")
    if rfr and "/" in rfr:
        num, den = rfr.split("/")
        try:
            den_f = float(den)
            fps = float(num) / den_f if den_f else 30.0
        except ValueError:
            pass

    duration = float(fmt.get("duration") or vstream.get("duration") or 0)

    return VideoInfo(
        path=path,
        duration=duration,
        fps=fps,
        width=int(vstream.get("width", 0)),
        height=int(vstream.get("height", 0)),
        codec=vstream.get("codec_name", "unknown"),
        audio_codec=astream.get("codec_name") if astream else None,
        bitrate=int(fmt.get("bit_rate", 0)) or None,
        n_frames=int(vstream.get("nb_frames", 0)) or None,
    )


def ensure_parent(path: Path) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    return path
