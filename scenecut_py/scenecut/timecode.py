"""Timecode conversions (seconds <-> HH:MM:SS.mmm <-> HH:MM:SS:FF)."""
from __future__ import annotations

from dataclasses import dataclass


def seconds_to_hmsms(seconds: float) -> str:
    """Convert seconds to HH:MM:SS.mmm string."""
    if seconds < 0:
        seconds = 0
    h = int(seconds // 3600)
    m = int((seconds % 3600) // 60)
    s = int(seconds % 60)
    ms = int(round((seconds - int(seconds)) * 1000))
    if ms == 1000:
        ms = 0
        s += 1
        if s == 60:
            s = 0
            m += 1
            if m == 60:
                m = 0
                h += 1
    return f"{h:02d}:{m:02d}:{s:02d}.{ms:03d}"


def hmsms_to_seconds(tc: str) -> float:
    """Convert HH:MM:SS.mmm or HH:MM:SS:FF to seconds."""
    tc = tc.strip()
    if "." in tc:
        # HH:MM:SS.mmm
        h, m, rest = tc.split(":")
        s, _, ms = rest.partition(".")
        return int(h) * 3600 + int(m) * 60 + int(s) + (int(ms.ljust(3, "0")[:3]) / 1000.0)
    elif tc.count(":") == 3:
        # HH:MM:SS:FF — need fps
        # default to 30 if not provided
        return tcff_to_seconds(tc, 30.0)
    elif tc.count(":") == 2:
        h, m, s = tc.split(":")
        return int(h) * 3600 + int(m) * 60 + int(s)
    raise ValueError(f"Unrecognized timecode: {tc}")


def tcff_to_seconds(tc: str, fps: float) -> float:
    """HH:MM:SS:FF -> seconds given fps."""
    h, m, s, f = tc.split(":")
    return (int(h) * 3600 + int(m) * 60 + int(s)) + int(f) / fps


def seconds_to_tcff(seconds: float, fps: float, drop_frame: bool = False) -> str:
    """HH:MM:SS:FF for given fps. Optional drop-frame for 29.97/59.94."""
    if fps <= 0:
        fps = 30.0
    nominal_fps = round(fps)
    h = int(seconds // 3600)
    rem = seconds - h * 3600
    m = int(rem // 60)
    rem -= m * 60
    s = int(rem)
    f = int(round((rem - s) * nominal_fps))
    if f >= nominal_fps:
        f -= nominal_fps
        s += 1
        if s >= 60:
            s -= 60
            m += 1
            if m >= 60:
                m -= 60
                h += 1
    return f"{h:02d}:{m:02d}:{s:02d}:{f:02d}"


def frame_to_seconds(frame: int, fps: float) -> float:
    return frame / fps if fps > 0 else 0.0
