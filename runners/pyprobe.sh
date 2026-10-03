#!/usr/bin/env bash
# pyprobe — validates the full Python detection toolchain on a Netlify build machine:
#   python3 + pip + venv (or --user fallback) + numpy + opencv-python-headless
#   + bundled-ffmpeg video encode/decode + a synthetic hard-cut sanity check.
# Everything prints to stdout; build.mjs captures it into log.txt.
set -o pipefail
log() { echo "[$(date -u +%H:%M:%S)] $*"; }

log "pyprobe start"
python3 --version || { echo "PYPROBE_FAIL: no python3"; exit 1; }

log "creating venv"
if python3 -m venv /tmp/venv 2>/dev/null; then
  PY=/tmp/venv/bin/python
  PIP="/tmp/venv/bin/python -m pip"
else
  log "venv unavailable — falling back to --user installs"
  PY=python3
  PIP="python3 -m pip --user"
fi
$PIP install --quiet --disable-pip-version-check numpy opencv-python-headless 2>&1 | tail -3
log "pip install done"

$PY - <<'EOF'
import sys, time
import numpy as np
import cv2
print(f"python {sys.version.split()[0]} | numpy {np.__version__} | cv2 {cv2.__version__}")

# 1) bundled-ffmpeg encode/decode round-trip with a hard cut at frame 50
w, h, fps = 320, 240, 25
a = np.tile(np.linspace(40, 80, h).reshape(h, 1, 1), (1, w, 3)).astype(np.uint8)
b = np.tile(np.linspace(160, 220, h).reshape(h, 1, 1), (1, w, 3)).astype(np.uint8)
rng = np.random.default_rng(0)
vw = cv2.VideoWriter("/tmp/test.mp4", cv2.VideoWriter_fourcc(*"mp4v"), fps, (w, h))
for i in range(100):
    base = a if i < 50 else b
    frame = (base.astype(np.int16) + rng.integers(-8, 9, (h, w, 3))).clip(0, 255).astype(np.uint8)
    vw.write(frame)
vw.release()

t0 = time.time()
cap = cv2.VideoCapture("/tmp/test.mp4")
prev, diffs, n = None, [], 0
while True:
    ok, f = cap.read()
    if not ok:
        break
    n += 1
    g = cv2.cvtColor(f, cv2.COLOR_BGR2GRAY)
    if prev is not None:
        diffs.append(float(np.abs(g.astype(np.int16) - prev).mean()))
    prev = g
cap.release()
dt = time.time() - t0
cut = int(np.argmax(diffs))
print(f"decode: {n}/100 frames | {n/dt:.0f} fps decode | max frame-diff at boundary frame {cut}->{cut+1} (expect 49->50) value {max(diffs):.1f}")
assert n == 100, f"decode lost frames: {n}"
assert abs(cut - 49) <= 1, f"cut localized at {cut}, expected 49"

# 2) CPU sanity: time a 2-frame-family absdiff at 1080p (the detection inner loop shape)
f1 = rng.integers(0, 255, (1080, 1920), dtype=np.uint8)
f2 = rng.integers(0, 255, (1080, 1920), dtype=np.uint8)
t0 = time.time()
for _ in range(30):
    d = cv2.absdiff(f1, f2)
    _ = float(d.mean())
dt = time.time() - t0
print(f"cpu: 30x 1080p absdiff+mean = {dt*1000:.0f} ms ({30/dt:.0f} iter/s)")
print("PYPROBE_OK")
EOF
rc=$?
log "pyprobe exit=$rc"
exit $rc
