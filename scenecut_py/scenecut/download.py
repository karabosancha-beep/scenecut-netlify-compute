"""yt-dlp wrapper."""
from __future__ import annotations

import json
import shutil
import subprocess
from pathlib import Path

from .config import DOWNLOADS_DIR
from .util import SceneCutError, ytdlp_path


def _detect_js_runtime() -> list[str]:
    """Find a JS runtime yt-dlp can use. Returns extra args."""
    for cmd in ("node", "deno", "bun"):
        if shutil.which(cmd):
            return ["--js-runtimes", cmd]
    return []


def download(url: str, output: str | Path | None = None,
             format_spec: str | None = None, no_playlist: bool = True,
             extra_args: list[str] | None = None,
             cookies_from_browser: str | None = None,
             cookies_file: str | Path | None = None) -> dict:
    """Download a video via yt-dlp. Returns metadata dict.

    Authentication options (use one):
      cookies_from_browser: 'chrome' | 'firefox' | 'edge' | 'safari' | 'brave'
                            (browser must be installed on this machine with a
                            logged-in profile — typically only works on the
                            user's own desktop, not on a server)
      cookies_file:         path to a Netscape-format cookies.txt file exported
                            from a browser extension like "Get cookies.txt
                            LOCALLY" (works on any machine — recommended for
                            server / sandbox environments)
    """
    if cookies_from_browser and cookies_file:
        raise SceneCutError(
            "Pass either --cookies-from-browser OR --cookies, not both.",
            code=2,
        )

    output = str(output or (DOWNLOADS_DIR / "%(title).80s.%(ext)s"))

    fmt = format_spec or "bestvideo[height<=720]+bestaudio/best[height<=720]/best"
    cmd = [
        ytdlp_path(),
        "-f", fmt,
        "--merge-output-format", "mp4",
        "-o", output,
        "--print", "%(filename)s|%(duration)s|%(title)s|%(uploader)s|%(id)s",
        "--no-progress",
    ]
    cmd.extend(_detect_js_runtime())
    if no_playlist:
        cmd.append("--no-playlist")
    if cookies_from_browser:
        cmd.extend(["--cookies-from-browser", cookies_from_browser])
    if cookies_file:
        cookies_path = Path(cookies_file)
        if not cookies_path.exists():
            raise SceneCutError(f"Cookies file not found: {cookies_path}", code=2)
        cmd.extend(["--cookies", str(cookies_path)])
    if extra_args:
        cmd.extend(extra_args)
    cmd.append(url)

    proc = subprocess.run(cmd, capture_output=True, text=True)
    if proc.returncode != 0:
        stderr = (proc.stderr or "")
        # Detect the YouTube bot-detection error and surface a helpful message
        if "Sign in to confirm" in stderr or "not a bot" in stderr:
            raise SceneCutError(
                "YouTube requires cookie authentication. Either:\n"
                "  (a) export a cookies.txt from your browser (extension: "
                "'Get cookies.txt LOCALLY') and pass --cookies <path>, or\n"
                "  (b) if running on your own desktop, pass "
                "--cookies-from-browser chrome|firefox|edge.\n"
                "See https://github.com/yt-dlp/yt-dlp/wiki/FAQ for details.",
                code=3,
            )
        # Friendly error for missing browser cookies database
        if "could not find" in stderr and "cookies database" in stderr:
            raise SceneCutError(
                f"Browser cookie database not found for '{cookies_from_browser}'. "
                f"This usually means the browser isn't installed on this machine. "
                f"On a server/sandbox, use --cookies <cookies.txt> instead "
                f"(export from your local browser via the 'Get cookies.txt LOCALLY' extension).",
                code=3,
            )
        raise SceneCutError(f"yt-dlp failed: {stderr[:500]}", code=3)

    out_line = (proc.stdout or "").strip().splitlines()
    info = {}
    if out_line:
        parts = out_line[-1].split("|", 4)
        if len(parts) >= 1:
            info["path"] = parts[0]
        if len(parts) >= 2:
            try: info["duration"] = float(parts[1]) if parts[1] != "NA" else None
            except ValueError: info["duration"] = None
        if len(parts) >= 3:
            info["title"] = parts[2]
        if len(parts) >= 4:
            info["uploader"] = parts[3]
        if len(parts) >= 5:
            info["video_id"] = parts[4]

    if "path" not in info or not Path(info["path"]).exists():
        mp4s = sorted(Path(DOWNLOADS_DIR).glob("*.mp4"), key=lambda p: p.stat().st_mtime, reverse=True)
        if mp4s:
            info["path"] = str(mp4s[0])

    if "path" not in info or not Path(info["path"]).exists():
        raise SceneCutError("yt-dlp reported success but output file not found", code=3)

    return info
