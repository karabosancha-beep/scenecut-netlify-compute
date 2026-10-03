"""Generate a rich demo video: 5 scenes with hard cuts + a fade transition."""
import subprocess
from pathlib import Path

OUT = Path(__file__).resolve().parent.parent / "samples" / "demo.mp4"
TMP = Path(__file__).resolve().parent.parent / "samples" / ".demo_tmp"
TMP.mkdir(exist_ok=True)

SEGMENTS = [
    ("seg1_testsrc", "testsrc2=size=640x360:rate=30", 4, None),
    ("seg2_mandel", "mandelbrot=size=640x360:rate=30", 4, None),
    ("seg3_bars", "smptebars=size=640x360:rate=30", 4, "fade=t=out:st=3:d=1"),
    ("seg4_color", "color=color=blue:size=640x360:rate=30", 4, "fade=t=in:st=0:d=1"),
    ("seg5_mandeldot", "mandelbrot=size=640x360:rate=30", 4, None),
]

seg_files = []
for name, src, dur, fade in SEGMENTS:
    out = TMP / f"{name}.mp4"
    cmd = ["ffmpeg", "-y", "-hide_banner", "-loglevel", "error",
           "-f", "lavfi", "-i", src, "-t", str(dur),
           "-c:v", "libx264", "-preset", "ultrafast", "-crf", "28",
           "-pix_fmt", "yuv420p"]
    if fade:
        cmd = ["ffmpeg", "-y", "-hide_banner", "-loglevel", "error",
               "-f", "lavfi", "-i", src, "-t", str(dur),
               "-vf", fade,
               "-c:v", "libx264", "-preset", "ultrafast", "-crf", "28",
               "-pix_fmt", "yuv420p"]
    cmd.append(str(out))
    subprocess.run(cmd, check=True)
    seg_files.append(out)

list_file = TMP / "concat.txt"
with list_file.open("w") as f:
    for s in seg_files:
        f.write(f"file '{s.absolute()}'\n")

cmd = ["ffmpeg", "-y", "-hide_banner", "-loglevel", "error",
       "-f", "concat", "-safe", "0", "-i", str(list_file),
       "-c", "copy", str(OUT)]
subprocess.run(cmd, check=True)

for s in seg_files:
    s.unlink()
list_file.unlink()
TMP.rmdir()

print(f"Generated: {OUT} ({OUT.stat().st_size} bytes)")
