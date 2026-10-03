#!/usr/bin/env bash
# sbd — the REAL bench job for the compute lane.
#
#   venv + numpy/cv2  ->  (ffprobe if missing: static BtbN build)
#   ->  SBD corpus fetch (scripts/sbd_fetch.py, streaming + resumable)
#   ->  external bench (scripts/external_bench.py sbd, time-budgeted)
#   ->  copy report + state (+ corpus manifest) to $OUT_DIR
#
# Everything logs to stdout — build.mjs captures it into log.txt.
# The whole job is time-disciplined for the 9-min execSync cap in build.mjs:
#   pip ~10-60s, ffprobe fetch ~10-60s (skipped when ffprobe is on PATH),
#   corpus fetch <= SBD_FETCH_TMO (240s), bench stops at SBD_BENCH_CAP (500s
#   runner-elapsed) — always under build.mjs's 9-min execSync kill.
# Every stage is resumable, so a re-run of this script continues where the
# previous one stopped (fetch: per-tar manifest; bench: state cursor).
set -o pipefail

log() { echo "[$(date -u +%H:%M:%S)] $*"; }
T0=$(date +%s)
elapsed() { echo $(( $(date +%s) - T0 )); }
fail() { echo "SBD_FAIL: $*"; exit 1; }

OUT_DIR="${OUT_DIR:-/tmp/scenecut-out}"
SBD_REAL="${SBD_REAL:-60}"
SBD_SYN="${SBD_SYN:-40}"
SBD_BUDGET="${SBD_BUDGET:-420}"      # external_bench --time-budget per invocation
SBD_FETCH_TMO="${SBD_FETCH_TMO:-240}" # wall-clock cap on the corpus fetch
SBD_BENCH_CAP="${SBD_BENCH_CAP:-500}" # wall-clock cap on the whole bench phase

mkdir -p "$OUT_DIR"
cd "$(dirname "$0")/.." || fail "cannot cd to repo root"
# keep the package's config.py from writing upload/ dirs into the checkout
export SCENECUT_WORKDIR="${SCENECUT_WORKDIR:-/tmp/scenecut-workdir}"
# external_bench.py inserts <repo>/scenecut_py on sys.path itself; this is
# belt-and-braces so `import scenecut` works for any helper too.
export PYTHONPATH="${PWD}/scenecut_py${PYTHONPATH:+:$PYTHONPATH}"

log "sbd job start: cwd=$(pwd) out=$OUT_DIR quotas real=$SBD_REAL syn=$SBD_SYN"

# ---- 1) python venv (reuse if present) ---------------------------------
if [ ! -x /tmp/venv/bin/python ]; then
  log "creating venv at /tmp/venv"
  python3 -m venv /tmp/venv || fail "venv creation failed"
fi
PY=/tmp/venv/bin/python
log "pip install numpy scenedetect + opencv-python-headless (headless LAST)"
# scenedetect (runtime dep of the orchestrated detector pass) pulls the GUI
# opencv-python build, which cannot import on a display-less build machine.
# Both wheels own the same cv2/ package dir — last unpack wins, so the
# headless variant is force-reinstalled AFTER it, deterministically.
"$PY" -m pip install --quiet --disable-pip-version-check \
      numpy scenedetect 2>&1 | tail -3
"$PY" -m pip install --quiet --disable-pip-version-check \
      --force-reinstall --no-deps opencv-python-headless 2>&1 | tail -2
"$PY" - <<'PYEOF' || fail "python toolchain import failed"
import numpy, cv2, scenedetect
from scenedetect import open_video
from scenedetect.detectors import AdaptiveDetector
print(f"toolchain: numpy {numpy.__version__} cv2 {cv2.__version__} "
      f"scenedetect {scenedetect.__version__}")
PYEOF
log "python toolchain ready ($(elapsed)s)"

# ---- 2) ffprobe — detect_scenes probes every clip via ffprobe -----------
# (opencv's bundled ffmpeg handles decode; probe_video() shells out to the
#  ffprobe BINARY, which is not guaranteed on the build machine)
if command -v ffprobe >/dev/null 2>&1; then
  log "ffprobe on PATH: $(command -v ffprobe) | $(ffprobe -version 2>&1 | head -1)"
else
  log "ffprobe NOT on PATH — fetching static ffmpeg+ffprobe (BtbN GitHub build, ~154MB)"
  FF_DIR=/tmp/ffstatic
  FF_TGZ="$FF_DIR/ffmpeg-static.tar.xz"
  URL="https://github.com/BtbN/ffmpeg-builds/releases/download/latest/ffmpeg-master-latest-linux64-gpl.tar.xz"
  mkdir -p "$FF_DIR"
  ok=0
  if command -v curl >/dev/null 2>&1; then
    curl -fsSL --retry 2 --max-time 240 -o "$FF_TGZ" "$URL" && ok=1
  fi
  if [ "$ok" -ne 1 ]; then
    "$PY" - "$URL" "$FF_TGZ" <<'PYEOF' && ok=1 || ok=0
import sys, urllib.request
url, dst = sys.argv[1], sys.argv[2]
req = urllib.request.Request(url, headers={"User-Agent": "scenecut-lane/1.0"})
with urllib.request.urlopen(req, timeout=240) as r, open(dst, "wb") as f:
    while True:
        b = r.read(1 << 20)
        if not b:
            break
        f.write(b)
PYEOF
  fi
  [ "$ok" = 1 ] || fail "static ffmpeg download failed ($URL)"
  "$PY" - <<'PYEOF' || fail "static ffmpeg extraction failed"
import tarfile
with tarfile.open("/tmp/ffstatic/ffmpeg-static.tar.xz") as tf:
    keep = [m for m in tf.getmembers()
            if m.name.rstrip("/").endswith(("bin/ffmpeg", "bin/ffprobe"))]
    tf.extractall("/tmp/ffstatic", members=keep)
    print("extracted:", sorted(m.name for m in keep))
    assert any(m.name.endswith("bin/ffprobe") for m in keep), "no ffprobe in tarball"
PYEOF
  FF_BIN=$(dirname "$(echo /tmp/ffstatic/*/bin/ffprobe)")
  export PATH="$FF_BIN:$PATH"
  command -v ffprobe >/dev/null 2>&1 \
    || fail "static ffprobe not usable after extraction"
  log "static ffprobe ready: $(ffprobe -version 2>&1 | head -1) ($(elapsed)s)"
fi

# ---- 3) fetch the SBD corpus (streaming, resumable, time-capped) --------
log "fetching SBD corpus (per-label real=$SBD_REAL syn=$SBD_SYN, tmo ${SBD_FETCH_TMO}s)"
timeout "$SBD_FETCH_TMO" "$PY" scenecut_py/scripts/sbd_fetch.py \
    --per-label-real "$SBD_REAL" --per-label-syn "$SBD_SYN"
rc=$?
log "sbd_fetch rc=$rc ($(elapsed)s elapsed)"
[ -f datasets/sbd/gt.json ] || fail "datasets/sbd/gt.json missing after fetch"
"$PY" - <<'PYEOF'
import json, sys
from collections import Counter
gt = json.load(open("datasets/sbd/gt.json"))
n = len(gt.get("samples", []))
c = Counter((s["label"], bool(s.get("real"))) for s in gt["samples"])
per_label = Counter(s["label"] for s in gt["samples"])
print(f"corpus: {n} clips {dict(c)} (tars_scanned={gt.get('tars_scanned')})")
sys.exit(0 if n >= 60 and min(per_label.get(x, 0) for x in "CTE") >= 10 else 1)
PYEOF
[ $? -eq 0 ] || fail "corpus too small/unbalanced after fetch (see counts above)"

# ---- 4) external bench (resumable, time-budgeted) -----------------------
mkdir -p bench
log "running external bench sbd (budget ${SBD_BUDGET}s, cap ${SBD_BENCH_CAP}s)"
BENCH_DONE=0
for attempt in 1 2 3; do
  left=$(( SBD_BENCH_CAP - $(elapsed) ))
  if [ "$left" -le 45 ]; then
    log "bench window exhausted (left=${left}s) — stopping"
    break
  fi
  budget=$(( left < SBD_BUDGET ? left : SBD_BUDGET ))
  "$PY" scenecut_py/scripts/external_bench.py sbd --time-budget "$budget"
  rc=$?
  log "external_bench rc=$rc (attempt $attempt, $(elapsed)s elapsed)"
  if [ "$rc" -eq 0 ]; then BENCH_DONE=1; break; fi
done
[ "$BENCH_DONE" = 1 ] || fail "bench did not reach the corpus end (partial report on disk, but job incomplete)"

# ---- 5) outputs ---------------------------------------------------------
[ -f bench/external_sbd_default.json ] || fail "bench report missing"
cp bench/external_sbd_default.json "$OUT_DIR/"
[ -f bench/.external_sbd_default.state.json ] \
  && cp bench/.external_sbd_default.state.json "$OUT_DIR/"
cp datasets/sbd/gt.json "$OUT_DIR/sbd_gt.json"
"$PY" - <<'PYEOF'
import json, sys
rep = json.load(open("bench/external_sbd_default.json"))
print(f"SBD metrics: n_videos={rep['n_videos']} n_skipped={rep['n_skipped']}")
for k in sorted(rep["aggregate"]):
    c = rep["aggregate"][k]
    print(f"  {k}: P={c['precision']} R={c['recall']} F1={c['f1']} "
          f"(tp {c['tp']}/fp {c['fp']}/fn {c['fn']})")
sys.exit(0 if rep["n_videos"] > 0 else 1)
PYEOF
[ $? -eq 0 ] || fail "bench produced 0 rows (all videos skipped?)"

log "SBD_OK ($(elapsed)s total)"
echo "SBD_OK"
