#!/usr/bin/env python3
"""Deterministic fault-injection fixture corpus for scene-detection regression
testing (DECISIONS.md D11 "Test fixtures", Task IMPL-A1).

Generates the fixture set into scenecut_py/samples/ plus a machine-checkable
ground_truth.json (expected cuts as frame numbers + measured structure).
Every fixture is self-verified at generation time -- frame count, duration,
and content sanity probes (hue/luma drift, motion, blend frames, blur
episodes, hard-cut spikes) -- and the script fails loudly if a fixture does
not match its intent.

Base recipes: .agents/research/design-review-r2.md §1.7 (verified feasible
on ffmpeg 7.1.5). Two measured deviations from the literal recipes, both
documented in the ground-truth notes:

* itest_pan -- the review's crop x='mod(t*300,320)' WRAPS: at t*300 = 320k
  the window jumps 320px back to x=0 (measured consecutive mean-abs-diff
  ~127 at frames 32/64/96 -- a hard cut, defeating the "NO cuts" intent).
  Replaced with a triangle-wave pan x='320-abs(mod(t*320,640)-320)'
  (constant 320 px/s, position mathematically continuous on [0,4s], zero
  content jumps, max consecutive diff ~13) which preserves the stated
  intent ("fast pan over busy pattern, NO cuts").

* itest_sunset -- eq=brightness='-0.18*t/8' is a SILENT NO-OP for per-frame
  expressions on ffmpeg 7.1.5 (measured: flat output; the option is
  evaluated once at init with t=0). The hue filter's own b= option uses the
  same per-frame expression engine as h= (verified ramping), so the intended
  brightness ramp is applied as hue=...:b='-1.8*t/8' (b units are ~10x
  weaker than eq brightness; -1.8 reproduces the intended -0.18*255 ~= -46
  luma drop, measured -51).

Usage (from repo root):
    PYTHONPATH=scenecut_py python3 scenecut_py/scripts/make_fixtures.py [--force]

--force regenerates the 8 new fixtures + base images. The pinned pre-existing
fixtures (itest_hard_cuts.mp4, itest_fade.mp4) are only generated when
missing (never force-regenerated) so their bytes stay stable.
"""
from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path

import cv2
import numpy as np

SAMPLES = Path(__file__).resolve().parent.parent / "samples"
W, H, FPS = 640, 360, 30
BINS = 50  # HSV histogram bins per channel (matches detect.py's helper)
ENCODER = ["-c:v", "libx264", "-crf", "18", "-pix_fmt", "yuv420p", "-an"]
EPS = 1e-7


class FixtureError(RuntimeError):
    """Raised when a fixture does not match its intent."""


# ------------------------------------------------------------------ helpers


def run_ffmpeg(args: list[str], desc: str) -> None:
    cmd = ["ffmpeg", "-y", "-hide_banner", "-loglevel", "error", *args]
    proc = subprocess.run(cmd, capture_output=True, text=True)
    if proc.returncode != 0:
        raise FixtureError(f"[{desc}] ffmpeg failed: {(proc.stderr or '').strip()[:400]}")


def probe_duration(path: Path) -> float:
    proc = subprocess.run(
        ["ffprobe", "-v", "quiet", "-print_format", "json", "-show_format", str(path)],
        capture_output=True, text=True, check=True)
    return float(json.loads(proc.stdout)["format"]["duration"])


def decode(path: Path) -> list[np.ndarray]:
    cap = cv2.VideoCapture(str(path))
    if not cap.isOpened():
        raise FixtureError(f"OpenCV cannot open {path}")
    frames = []
    while True:
        ok, frame = cap.read()
        if not ok:
            break
        frames.append(frame)
    cap.release()
    if not frames:
        raise FixtureError(f"{path} decoded to zero frames")
    return frames


def gray(frame: np.ndarray) -> np.ndarray:
    return cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY).astype(np.float32)


def mad_series(frames: list[np.ndarray]) -> np.ndarray:
    """Consecutive mean-abs gray diffs: d[i] = MAD(frame[i], frame[i+1])."""
    g = [gray(f) for f in frames]
    return np.array([float(np.mean(np.abs(g[i + 1] - g[i]))) for i in range(len(g) - 1)])


def hsv_hist(frame: np.ndarray) -> np.ndarray:
    """L1-normalized (hue,sat) histogram on a 160x90 downscale (D3 geometry)."""
    small = cv2.resize(frame, (160, 90), interpolation=cv2.INTER_AREA)
    hsv = cv2.cvtColor(small, cv2.COLOR_BGR2HSV)
    hist = cv2.calcHist([hsv], [0, 1], None, [BINS, BINS], [0, 180, 0, 256])
    hist = hist.flatten().astype(np.float64)
    return (hist + EPS) / (hist + EPS).sum()


def js_norm(a: np.ndarray, b: np.ndarray) -> float:
    """Normalized Jensen-Shannon divergence D_JS/ln2 in [0,1]."""
    m = 0.5 * (a + b)

    def kl(p: np.ndarray, q: np.ndarray) -> float:
        mask = p > 0
        return float(np.sum(p[mask] * np.log(p[mask] / q[mask])))

    return (0.5 * kl(a, m) + 0.5 * kl(b, m)) / np.log(2.0)


def mean_hue_v(frame: np.ndarray) -> tuple[float, float]:
    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
    return float(np.mean(hsv[:, :, 0])), float(np.mean(hsv[:, :, 2]))


def laplacian_var_series(frames: list[np.ndarray]) -> np.ndarray:
    """Per-frame sharpness (Laplacian variance on 8-bit gray)."""
    return np.array([
        float(cv2.Laplacian(cv2.cvtColor(f, cv2.COLOR_BGR2GRAY), cv2.CV_64F).var())
        for f in frames])


def check(cond: bool, fixture: str, msg: str) -> None:
    if not cond:
        raise FixtureError(f"[{fixture}] self-verification FAILED: {msg}")


def common_checks(name: str, frames: list[np.ndarray], n_expected: int,
                  dur_expected: float) -> float:
    """Frame count / geometry / duration checks shared by all fixtures."""
    h, w = frames[0].shape[:2]
    check(w == W and h == H, name, f"geometry {w}x{h}, expected {W}x{H}")
    check(len(frames) == n_expected, name,
          f"decoded {len(frames)} frames, expected {n_expected}")
    dur = probe_duration(SAMPLES / f"{name}.mp4")
    check(abs(dur - dur_expected) <= 0.15, name,
          f"duration {dur:.3f}s, expected ~{dur_expected}s")
    return dur


def entry(name: str, purpose: str, frames: list[np.ndarray], dur: float,
          cuts: list[dict], notes: str, measured: dict | None = None) -> dict:
    e = {
        "fixture": f"{name}.mp4",
        "purpose": purpose,
        "width": W,
        "height": H,
        "fps": FPS,
        "frames": len(frames),
        "duration": round(dur, 3),
        "expected_cuts": cuts,
        "notes": notes,
    }
    if measured:
        e["measured"] = measured
    return e


# ------------------------------------------------- deterministic base images
# All synthesized with numpy + cv2, no RNG: a vertical sky gradient, an
# opaque sun disc (row-independent palette) and soft elliptical clouds.


def _sky_gradient(w: int, h: int) -> np.ndarray:
    top = np.array([190, 105, 62], dtype=np.float32)     # BGR sky blue
    bottom = np.array([105, 170, 238], dtype=np.float32)  # BGR warm horizon
    rows = np.linspace(0, 1, h, dtype=np.float32)[:, None, None]
    return np.repeat(top[None, None, :] * (1 - rows) + bottom[None, None, :] * rows, w, axis=1)


def _add_sun(img: np.ndarray, cx: int, cy: int, r: int = 42) -> None:
    h, w = img.shape[:2]
    yy, xx = np.mgrid[0:h, 0:w].astype(np.float32)
    dist = np.sqrt((xx - cx) ** 2 + (yy - cy) ** 2)
    m = np.clip((float(r) - dist) / 10.0, 0, 1)
    img[:] = img * (1 - m[..., None]) + np.array([210, 243, 255], np.float32) * m[..., None]


def _add_cloud(img: np.ndarray, cx: int, cy: int, ax: int, ay: int,
               alpha: float = 0.72) -> None:
    h, w = img.shape[:2]
    m = np.zeros((h, w), np.float32)
    cv2.ellipse(m, (cx, cy), (ax, ay), 0, 0, 360, 1.0, -1)
    m = cv2.GaussianBlur(m, (0, 0), 6) * alpha
    img[:] = img * (1 - m[..., None]) + np.array([230, 240, 252], np.float32) * m[..., None]


def _finish(img: np.ndarray) -> np.ndarray:
    return np.clip(img, 0, 255).astype(np.uint8)


def write_base_images(force: bool) -> dict[str, Path]:
    """Write the deterministic source PNGs (idempotent unless force)."""

    def put(path: Path, img: np.ndarray) -> tuple[Path, np.ndarray]:
        if force or not path.exists():
            cv2.imwrite(str(path), img)
        return path, img

    # Standalone sky: base for itest_sunset / itest_static.
    sky = _sky_gradient(W, H)
    _add_sun(sky, 170, 120)
    _add_cloud(sky, 110, 250, 90, 15)
    _add_cloud(sky, 480, 90, 110, 16, 0.68)
    _add_cloud(sky, 320, 300, 120, 13, 0.60)
    _, sky_img = put(SAMPLES / "src_sky.png", _finish(sky))

    # Dialog pair (itest_reverse_dialog): identical component multiset; clouds
    # sit at the SAME rows in both images (palette contribution cancels), the
    # opaque sun moves far away (row-independent palette). Same palette, very
    # different composition -> same-palette TRUE hard cut.
    dlg_l = _sky_gradient(W, H)
    _add_sun(dlg_l, 170, 100)
    _add_cloud(dlg_l, 110, 250, 90, 15)
    _add_cloud(dlg_l, 420, 200, 110, 17, 0.70)
    _, dlg_l_img = put(SAMPLES / "src_dialog_L.png", _finish(dlg_l))

    dlg_r = _sky_gradient(W, H)
    _add_sun(dlg_r, 470, 260)
    _add_cloud(dlg_r, 530, 250, 90, 15)
    _add_cloud(dlg_r, 250, 200, 110, 17, 0.70)
    _, dlg_r_img = put(SAMPLES / "src_dialog_R.png", _finish(dlg_r))

    # Match pair (itest_match_dissolve): sun-left vs cloud-heavy-right --
    # same palette family but a measurably wider (still sub-tier-1) histogram
    # gap, i.e. the tier-2 "same-palette dissolve" corner.
    mat_l = _sky_gradient(W, H)
    _add_sun(mat_l, 180, 120, 46)
    _add_cloud(mat_l, 130, 250, 95, 15)
    _add_cloud(mat_l, 330, 300, 110, 13, 0.60)
    _, mat_l_img = put(SAMPLES / "src_match_L.png", _finish(mat_l))

    mat_r = _sky_gradient(W, H)
    _add_cloud(mat_r, 480, 120, 150, 22, 0.80)
    _add_cloud(mat_r, 560, 245, 100, 15, 0.72)
    _add_cloud(mat_r, 320, 300, 110, 13, 0.60)
    _, mat_r_img = put(SAMPLES / "src_match_R.png", _finish(mat_r))

    # Source-image sanity (raw, pre-encoding): dialog pair must be near
    # palette-identical; match pair must sit in the tier-2 band.
    js_dialog = js_norm(hsv_hist(dlg_l_img), hsv_hist(dlg_r_img))
    js_match = js_norm(hsv_hist(mat_l_img), hsv_hist(mat_r_img))
    check(js_dialog <= 0.05, "src_dialog", f"raw palette JS {js_dialog:.4f} > 0.05")
    check(0.03 <= js_match <= 0.11, "src_match",
          f"raw palette JS {js_match:.4f} outside tier-2 band [0.03, 0.11]")
    return {"sky": SAMPLES / "src_sky.png",
            "dialog_L": SAMPLES / "src_dialog_L.png",
            "dialog_R": SAMPLES / "src_dialog_R.png",
            "match_L": SAMPLES / "src_match_L.png",
            "match_R": SAMPLES / "src_match_R.png"}


# ------------------------------------------------------------ fixture builders


def build_pan(force: bool) -> dict:
    name = "itest_pan"
    out = SAMPLES / f"{name}.mp4"
    if force or not out.exists():
        # Triangle-wave pan: constant 320 px/s, position continuous (see
        # module docstring for why the review's mod() recipe was replaced).
        run_ffmpeg(
            ["-f", "lavfi", "-i", "testsrc2=s=960x360:r=30:d=4",
             "-vf", "crop=w=640:h=360:x='320-abs(mod(t*320,640)-320)':y=0",
             *ENCODER, str(out)], name)
    frames = decode(out)
    dur = common_checks(name, frames, 120, 4.0)
    mads = mad_series(frames)
    med, mx = float(np.median(mads)), float(mads.max())
    check(med >= 4.0, name, f"no motion: median consecutive diff {med:.2f} < 4")
    check(mx <= 20.0, name,
          f"content jump: max consecutive diff {mx:.2f} > 20 (hard-cut-like)")
    return entry(name,
                 "negative: fast pan over busy pattern -- motion-FP stress (D4)",
                 frames, dur, [],
                 "Fast pan (constant 320 px/s, triangle wave, position continuous) "
                 "over testsrc2 at 960x360 with a moving 640x360 crop. NO content "
                 "discontinuities: max consecutive mean-abs-diff "
                 f"{mx:.1f} vs ~127 for the reviewed mod(t*300,320) recipe, whose "
                 "wrap-around at frames 32/64/96 is a hard cut. Ground truth: zero "
                 "cuts; any detector cut on this clip is a motion false positive.",
                 {"median_consec_mad": round(med, 2), "max_consec_mad": round(mx, 2)})


def build_sunset(force: bool) -> dict:
    name = "itest_sunset"
    out = SAMPLES / f"{name}.mp4"
    if force or not out.exists():
        run_ffmpeg(
            ["-loop", "1", "-framerate", "30", "-i", str(SAMPLES / "src_sky.png"),
             "-t", "8",
             "-vf", "hue=h='20*t/8':s=1.0:b='-1.8*t/8'",
             *ENCODER, str(out)], name)
    frames = decode(out)
    dur = common_checks(name, frames, 240, 8.0)
    v_first = float(np.mean([mean_hue_v(f)[1] for f in frames[:15]]))
    v_last = float(np.mean([mean_hue_v(f)[1] for f in frames[-15:]]))
    drop = v_first - v_last
    hists = [hsv_hist(f) for f in frames]
    cjs = np.array([js_norm(hists[i], hists[i + 1]) for i in range(len(hists) - 1)])
    ep32 = js_norm(hists[0], hists[32])
    mads = mad_series(frames)
    check(drop >= 25.0, name, f"no brightness drift: V drops only {drop:.1f}")
    check(float(cjs.max()) <= 0.20, name,
          f"hard-cut-like palette jump: max consecutive JS {cjs.max():.3f} > 0.20")
    check(ep32 >= 0.12, name,
          f"missing tier-1 stress: w=32 endpoint JS {ep32:.3f} < 0.12")
    check(float(mads.max()) <= 3.0, name,
          f"content jump: max consecutive diff {mads.max():.2f} > 3")
    return entry(name,
                 "negative: gradual color+luma drift -- dissolve-FP stress (D3)",
                 frames, dur, [],
                 "Static synthesized sky (vertical gradient + opaque sun + soft "
                 "clouds, no source motion) with hue=h='20*t/8' (+20 deg) and a "
                 "gradual brightness ramp. Gradual drift only: measured max "
                 f"consecutive JS {cjs.max():.3f} (no hard cut) while w=32 endpoint "
                 f"JS {ep32:.3f} >= dissim_min 0.12 -- stresses tier-1's dissim "
                 "conjunct; only the gradual/no-jump character blocks a dissolve. "
                 "DEVIATION from §1.7: eq=brightness is a silent no-op for "
                 "per-frame expressions on ffmpeg 7.1.5 (measured), so the "
                 "-0.18*t/8 luma ramp is applied via hue's b= option at -1.8*t/8 "
                 f"(same engine as h=; measured V drop {drop:.1f} ~= -0.18*255).",
                 {"v_drop": round(drop, 1), "max_consec_js": round(float(cjs.max()), 4),
                  "endpoint_w32_js": round(ep32, 4), "max_consec_mad": round(float(mads.max()), 2)})


def build_static(force: bool) -> dict:
    name = "itest_static"
    out = SAMPLES / f"{name}.mp4"
    if force or not out.exists():
        run_ffmpeg(
            ["-loop", "1", "-framerate", "30", "-i", str(SAMPLES / "src_sky.png"),
             "-t", "6", *ENCODER, str(out)], name)
    frames = decode(out)
    dur = common_checks(name, frames, 180, 6.0)
    mads = mad_series(frames)
    h0, v0 = mean_hue_v(frames[0])
    h1, v1 = mean_hue_v(frames[-1])
    check(float(mads.max()) <= 0.5, name,
          f"not static: max consecutive diff {mads.max():.3f} > 0.5")
    check(abs(h1 - h0) <= 2.0 and abs(v1 - v0) <= 2.0, name,
          f"drifted: hue {h1 - h0:+.2f}, V {v1 - v0:+.2f}")
    return entry(name,
                 "negative: undrifted static sky -- FP floor (JS/delta noise floor)",
                 frames, dur, [],
                 "The undrifted src_sky.png looped for 6 s, no filters. Codec "
                 f"noise only: max consecutive mean-abs-diff {mads.max():.3f}. "
                 "Zero cuts expected; any detector cut here is a pure FP.",
                 {"max_consec_mad": round(float(mads.max()), 3)})


def build_whippan(force: bool) -> dict:
    name = "itest_whippan"
    out = SAMPLES / f"{name}.mp4"
    if force or not out.exists():
        run_ffmpeg(
            ["-f", "lavfi", "-i", "testsrc2=s=640x360:r=30:d=1.5",
             "-f", "lavfi", "-i",
             "testsrc2=s=1280x720:r=30:d=0.2,"
             "crop=w=640:h=360:x='min(640,max(0,(t/0.2)*640))':y=180,"
             "boxblur=8:2,scale=640:360",
             "-f", "lavfi", "-i", "testsrc2=s=640x360:r=30:d=1.5,hue=h=120",
             "-filter_complex", "[0:v][1:v][2:v]concat=n=3:v=1:a=0[out]",
             "-map", "[out]", *ENCODER, str(out)], name)
    frames = decode(out)
    dur = common_checks(name, frames, 96, 3.2)
    # Boundaries measured via blur state: boxblur(8:2) collapses the Laplacian
    # variance to ~1 (sharp frames are ~470-670).
    lap = laplacian_var_series(frames)
    blurred = [i for i, v in enumerate(lap) if v < 10.0]
    check(len(blurred) == 6, name,
          f"blur episode is {len(blurred)} frames (expected 6): {blurred}")
    check(blurred[0] == 45, name, f"blur episode starts at frame {blurred[0]}, expected 45")
    first_b, last_b = blurred[0], blurred[-1]
    b2 = last_b + 1  # first sharp frame after the blur episode
    check(b2 == 51, name, f"sharp resumption at frame {b2}, expected 51")
    mads = mad_series(frames)
    check(float(mads[first_b - 1]) >= 40.0, name,
          f"boundary A not a hard change: diff into {first_b} is {mads[first_b - 1]:.1f}")
    check(float(mads[b2 - 1]) >= 40.0, name,
          f"boundary B not a hard change: diff into {b2} is {mads[b2 - 1]:.1f}")
    outside = [mads[i] for i in range(len(mads)) if not (43 <= i <= 50)]
    check(max(outside) <= 25.0, name,
          f"unexpected jump outside the episode: {max(outside):.1f}")
    return entry(name,
                 "motion-gate ROC: whip-pan blur episode between palette-distinct shots (D4)",
                 frames, dur,
                 [{"frame": first_b, "type_hint": "cut"}, {"frame": b2, "type_hint": "cut"}],
                 "testsrc2 (1.5 s) | concat | blurred fast pan (0.2 s, boxblur 8:2) | "
                 "concat | hue+120 testsrc2 (1.5 s). MEASURED REALITY: the two concat "
                 f"boundaries are genuine hard cuts (frames {first_b} and {b2}; "
                 f"consec diffs {mads[first_b - 1]:.0f}/{mads[b2 - 1]:.0f}); the blur "
                 f"episode spans frames {first_b}-{last_b} (Laplacian variance ~1 vs "
                 "~500-670 sharp). DECISIONS.md D11 lists this as a negative, but the "
                 "bitstream contains the two boundary cuts -- the ideal detector "
                 "reports exactly these 2 and does NOT fragment the blur episode "
                 f"(intra-episode consec diffs ~56-58) into extra cuts. Cuts recorded "
                 "as the measured blur-state transitions.",
                 {"blur_episode": [first_b, last_b],
                  "boundary_mads": [round(float(mads[first_b - 1]), 1), round(float(mads[b2 - 1]), 1)]})


def build_dissolve(force: bool) -> dict:
    name = "itest_dissolve"
    out = SAMPLES / f"{name}.mp4"
    if force or not out.exists():
        run_ffmpeg(
            ["-f", "lavfi", "-i", "testsrc2=s=640x360:r=30:d=3",
             "-f", "lavfi", "-i", "testsrc2=s=640x360:r=30:d=3,hue=h=120",
             "-filter_complex",
             "[0:v][1:v]xfade=transition=fade:duration=0.7:offset=2.3[out]",
             "-map", "[out]", *ENCODER, str(out)], name)
    frames = decode(out)
    dur = common_checks(name, frames, 159, 5.3)
    hists = [hsv_hist(f) for f in frames]
    ref_a, ref_b = hists[5], hists[154]
    blend = [i for i in range(len(frames))
             if js_norm(hists[i], ref_a) > 0.1 and js_norm(hists[i], ref_b) > 0.1]
    check(len(blend) >= 15, name,
          f"blend too short: {len(blend)} frames clearly between the endpoints")
    check(65 <= blend[0] and blend[-1] <= 92, name,
          f"blend span {blend[0]}..{blend[-1]} outside the xfade window (~69..90)")
    mid = (blend[0] + blend[-1] + 1) // 2
    mads = mad_series(frames)
    ep = js_norm(hists[20], hists[100])
    cjs = np.array([js_norm(hists[i], hists[i + 1]) for i in range(len(hists) - 1)])
    check(float(mads.max()) <= 10.0, name,
          f"hard cut inside: max consecutive diff {mads.max():.2f} > 10")
    check(ep >= 0.12, name, f"endpoint JS {ep:.3f} < 0.12 (not a tier-1 dissolve)")
    return entry(name,
                 "positive: palette-distinct xfade -- tier-1 dissolve recall (D3)",
                 frames, dur,
                 [{"frame": mid, "type_hint": "dissolve"}],
                 "xfade transition=fade duration=0.7 offset=2.3 between testsrc2 and "
                 f"hue+120 testsrc2. ONE dissolve: measured blend frames {blend[0]}-"
                 f"{blend[-1]} (alpha 0.05-0.95), geometric midpoint ~{mid} "
                 "(2.65 s). Endpoint JS "
                 f"{ep:.3f} >> dissim_min 0.12 (tier-1 dissim satisfied); no hard cut "
                 f"by pixel diff (max consecutive {mads.max():.1f}). HAZARD for "
                 "JS-based no-hard-cut guards: max consecutive-frame JS "
                 f"{cjs.max():.2f} mid-blend -- flat-color regions make tall "
                 "histogram peaks walk bins during the crossfade (a sustained "
                 "plateau over ~20 frames, unlike a single-frame hard-cut spike).",
                 {"blend_span": [blend[0], blend[-1]], "midpoint": mid,
                  "endpoint_js": round(ep, 4), "max_consec_js": round(float(cjs.max()), 4),
                  "max_consec_mad": round(float(mads.max()), 2)})


def build_match_dissolve(force: bool) -> dict:
    name = "itest_match_dissolve"
    out = SAMPLES / f"{name}.mp4"
    ref_l = SAMPLES / "src_match_L.png"
    ref_r = SAMPLES / "src_match_R.png"
    if force or not out.exists():
        run_ffmpeg(
            ["-loop", "1", "-framerate", "30", "-t", "3", "-i", str(ref_l),
             "-loop", "1", "-framerate", "30", "-t", "3", "-i", str(ref_r),
             "-filter_complex",
             "[0:v][1:v]xfade=transition=fade:duration=0.7:offset=2.3[out]",
             "-map", "[out]", "-r", "30", *ENCODER, str(out)], name)
    frames = decode(out)
    dur = common_checks(name, frames, 159, 5.3)
    g_l, g_r = gray(cv2.imread(str(ref_l))), gray(cv2.imread(str(ref_r)))
    gs = [gray(f) for f in frames]
    mse = lambda a, b: float(np.mean((a - b) ** 2))
    mse_l = [mse(x, g_l) for x in gs]
    mse_r = [mse(x, g_r) for x in gs]
    check(mse_l[10] <= 15.0, name, f"frame 10 != source A (MSE {mse_l[10]:.1f})")
    check(mse_r[148] <= 15.0, name, f"frame 148 != source B (MSE {mse_r[148]:.1f})")
    blend = [i for i in range(len(frames)) if mse_l[i] > 25.0 and mse_r[i] > 25.0]
    check(len(blend) >= 8, name,
          f"no intermediate blend frames: only {len(blend)} clearly-between frames")
    check(65 <= blend[0] and blend[-1] <= 92, name,
          f"blend span {blend[0]}..{blend[-1]} outside the xfade window")
    mid = (blend[0] + blend[-1] + 1) // 2
    mads = mad_series(frames)
    hists = [hsv_hist(f) for f in frames]
    ep = js_norm(hists[10], hists[148])
    check(float(mads.max()) <= 2.0, name,
          f"hard cut inside: max consecutive diff {mads.max():.2f} > 2")
    check(0.035 <= ep < 0.12, name,
          f"endpoint JS {ep:.4f} outside the tier-2 band [0.035, 0.12)")
    return entry(name,
                 "positive: same-palette xfade -- tier-2 match-dissolve recall (D3)",
                 frames, dur,
                 [{"frame": mid, "type_hint": "dissolve"}],
                 "xfade fade 0.7 s @ 2.3 s between same-palette static crops of one "
                 "synthesized sky (src_match_L/R.png: sun-left vs cloud-heavy-right). "
                 f"ONE dissolve: measured blend frames {blend[0]}-{blend[-1]}, "
                 f"midpoint ~{mid}. Endpoint JS {ep:.4f} < dissim_min 0.12 but >= "
                 "dissim_min/3 0.04 -- the exact tier-2 corner the two-tier rule "
                 "exists for (edge-dip + accumulation must carry the decision). "
                 "Both endpoints are bit-stable statics (MSE to source ~2).",
                 {"blend_span": [blend[0], blend[-1]], "midpoint": mid,
                  "endpoint_js": round(ep, 4), "max_consec_mad": round(float(mads.max()), 3)})


def build_reverse_dialog(force: bool) -> dict:
    name = "itest_reverse_dialog"
    out = SAMPLES / f"{name}.mp4"
    ref_l = SAMPLES / "src_dialog_L.png"
    ref_r = SAMPLES / "src_dialog_R.png"
    if force or not out.exists():
        run_ffmpeg(
            ["-loop", "1", "-framerate", "30", "-t", "2", "-i", str(ref_l),
             "-loop", "1", "-framerate", "30", "-t", "2", "-i", str(ref_r),
             "-filter_complex", "[0:v][1:v]concat=n=2:v=1:a=0[out]",
             "-map", "[out]", "-r", "30", *ENCODER, str(out)], name)
    frames = decode(out)
    dur = common_checks(name, frames, 120, 4.0)
    mads = mad_series(frames)
    peak = int(np.argmax(mads)) + 1
    check(59 <= peak <= 61, name, f"hard cut at frame {peak}, expected 60")
    check(float(mads.max()) >= 5.0, name,
          f"cut too weak to be a true shot change: {mads.max():.2f}")
    others = [float(mads[i]) for i in range(len(mads)) if not (58 <= i <= 62)]
    check(max(others) <= 1.0, name,
          f"not static around the cut: max other diff {max(others):.2f} > 1")
    hists = [hsv_hist(f) for f in frames]
    js_pal = js_norm(hists[15], hists[45])
    check(js_pal <= 0.05, name,
          f"not same-palette: endpoint JS {js_pal:.4f} > 0.05")
    g_l, g_r = gray(cv2.imread(str(ref_l))), gray(cv2.imread(str(ref_r)))
    mse = lambda a, b: float(np.mean((a - b) ** 2))
    check(mse(gray(frames[15]), g_l) <= 15.0, name, "first shot != source L")
    check(mse(gray(frames[90]), g_r) <= 15.0, name, "second shot != source R")
    return entry(name,
                 "positive: same-palette TRUE hard cut -- motion-gate ROC negative (D4)",
                 frames, dur,
                 [{"frame": peak, "type_hint": "cut"}],
                 "Hard concat (filter concat, single encode) of two static crops of "
                 "one synthesized sky: identical component multiset (same sun, same "
                 "clouds at the same rows) rearranged into different compositions "
                 "(src_dialog_L/R.png). ONE true hard cut at the 2 s midpoint "
                 f"(frame {peak}, consec diff {mads.max():.1f}; every other frame "
                 f"pair <= {max(others):.2f}). Same palette: endpoint JS "
                 f"{js_pal:.4f} << dissim_min 0.12 -- a motion gate that suppresses "
                 "same-palette cuts would delete this true positive (D4's "
                 "downgrade-not-delete requirement).",
                 {"cut_mad": round(float(mads.max()), 2), "palette_js": round(js_pal, 4)})


def build_fade_white(force: bool) -> dict:
    name = "itest_fade_white"
    out = SAMPLES / f"{name}.mp4"
    if force or not out.exists():
        run_ffmpeg(
            ["-f", "lavfi", "-i", "testsrc2=s=640x360:r=30:d=4",
             "-vf", "fade=t=out:st=2:d=1:color=white",
             *ENCODER, str(out)], name)
    frames = decode(out)
    dur = common_checks(name, frames, 120, 4.0)
    luma = [float(np.mean(gray(f))) for f in frames]
    base = float(np.mean(luma[:10]))
    white = float(np.mean(luma[-10:]))
    onset = next((i for i, l in enumerate(luma) if l > base + 8.0), None)
    check(white >= base + 100.0, name,
          f"does not reach white: final luma {white:.1f} vs base {base:.1f}")
    check(onset is not None and 58 <= onset <= 65, name,
          f"fade onset at frame {onset}, expected ~60 (st=2)")
    return entry(name,
                 "positive: fade to white with white plateau -- CEILING pass recall (D5)",
                 frames, dur,
                 [{"frame": 60, "type_hint": "fade"}],
                 "testsrc2 with fade=t=out:st=2:d=1:color=white. ONE fade-to-white: "
                 "onset at frame 60 (t=2.0), luma ramps to 255 by frame 90 and holds "
                 f"a pure-white plateau for the final 30 frames (measured: base "
                 f"{base:.0f} -> {white:.0f}, detectable rise from frame ~{onset}). "
                 "D5 target: ThresholdDetector(CEILING) with the transit/edge-"
                 "collapse gate; single cut at onset typed fade.",
                 {"base_luma": round(base, 1), "white_luma": round(white, 1),
                  "measured_rise_onset": onset})


# ------------------------------------------------- pinned pre-existing fixtures

def _make_concat_video(name: str, segments: list[tuple[str, int]],
                       fades: dict[int, str] | None = None) -> None:
    """Regenerate a pinned fixture with the exact tests/conftest.py recipe
    (crf 28 / ultrafast) so the corpus is complete on a fresh checkout."""
    fades = fades or {}
    tmp = SAMPLES / f".tmp_{name}"
    tmp.mkdir(exist_ok=True)
    parts = []
    for i, (src, d) in enumerate(segments):
        part = tmp / f"part_{i:02d}.mp4"
        cmd = ["-f", "lavfi", "-i", src, "-t", str(d),
               "-c:v", "libx264", "-preset", "ultrafast", "-crf", "28",
               "-pix_fmt", "yuv420p"]
        if i in fades:
            st = "0" if fades[i] == "in" else str(max(0, d - 1))
            cmd.extend(["-vf", f"fade=t={fades[i]}:st={st}:d=1"])
        cmd.append(str(part))
        run_ffmpeg(cmd, f"{name}/part{i}")
        parts.append(part)
    lst = tmp / "concat.txt"
    with lst.open("w") as fh:
        for p in parts:
            fh.write(f"file '{p.absolute()}'\n")
    run_ffmpeg(["-f", "concat", "-safe", "0", "-i", str(lst), "-c", "copy",
                str(SAMPLES / f"{name}.mp4")], name)
    for p in parts:
        p.unlink()
    lst.unlink()
    tmp.rmdir()


def build_hard_cuts(force: bool) -> dict:
    name = "itest_hard_cuts"
    out = SAMPLES / f"{name}.mp4"
    if not out.exists():  # pinned: never force-regenerated
        _make_concat_video(name, [
            ("testsrc2=size=640x360:rate=30", 4),
            ("mandelbrot=size=640x360:rate=30", 4),
            ("smptebars=size=640x360:rate=30", 4),
        ])
    frames = decode(out)
    dur = common_checks(name, frames, 360, 12.0)
    mads = mad_series(frames)
    # boundaries measured: largest spike within +/-10 of the segment math
    def peak_near(target: int) -> int:
        lo, hi = max(0, target - 10), min(len(mads), target + 10)
        return int(np.argmax(mads[lo:hi])) + lo + 1
    b1, b2 = peak_near(120), peak_near(240)
    check(b1 == 120, name, f"first boundary at frame {b1}, expected 120")
    check(b2 == 240, name, f"second boundary at frame {b2}, expected 240")
    check(float(mads[b1 - 1]) >= 40.0 and float(mads[b2 - 1]) >= 40.0, name,
          "segment boundaries are not hard changes")
    return entry(name,
                 "positive: 3 palette-distinct segments -- hard-cut baseline / "
                 "orchestration regression net",
                 frames, dur,
                 [{"frame": b1, "type_hint": "cut"}, {"frame": b2, "type_hint": "cut"}],
                 "PINNED pre-existing fixture (tests/conftest.py recipe: testsrc2 | "
                 "mandelbrot | smptebars, 4 s each, crf 28 ultrafast -- bytes "
                 "intentionally preserved). Two true hard cuts at the segment "
                 f"boundaries (measured frames {b1}, {b2}; consec diffs "
                 f"{mads[b1 - 1]:.0f}/{mads[b2 - 1]:.0f}).",
                 {"boundary_mads": [round(float(mads[b1 - 1]), 1), round(float(mads[b2 - 1]), 1)]})


def build_fade(force: bool) -> dict:
    name = "itest_fade"
    out = SAMPLES / f"{name}.mp4"
    if not out.exists():  # pinned: never force-regenerated
        _make_concat_video(name, [
            ("testsrc2=size=640x360:rate=30", 5),
            ("mandelbrot=size=640x360:rate=30", 5),
        ], fades={0: "out", 1: "in"})
    frames = decode(out)
    dur = common_checks(name, frames, 300, 10.0)
    luma = [float(np.mean(gray(f))) for f in frames]
    lo, hi = 145, 156
    black_frame = int(np.argmin(luma[lo:hi])) + lo
    check(abs(black_frame - 150) <= 1, name,
          f"black midpoint at frame {black_frame}, expected ~150")
    check(luma[black_frame] <= 8.0, name,
          f"boundary frame is not black: luma {luma[black_frame]:.1f}")
    check(luma[0] >= 100.0, name, "opening frame is not a normal shot")
    return entry(name,
                 "positive: fade-out-to-black + hidden shot change + fade-in",
                 frames, dur,
                 [{"frame": black_frame, "type_hint": "fade"}],
                 "PINNED pre-existing fixture (tests/conftest.py recipe: testsrc2 "
                 "5 s fading out to black over the last 1 s, then mandelbrot 5 s "
                 "fading in from black; crf 28 ultrafast -- bytes preserved). ONE "
                 f"transition: fade-out frames ~120-{black_frame - 1}, the shot "
                 f"change hidden at the black frame {black_frame}, fade-in "
                 f"{black_frame}-~180. Ground truth records the hidden shot-change "
                 "frame (the black midpoint).",
                 {"black_frame_luma": round(luma[black_frame], 1)})


# ------------------------------------------------------------------ main


BUILDERS = [
    build_pan,
    build_sunset,
    build_static,
    build_whippan,
    build_dissolve,
    build_match_dissolve,
    build_reverse_dialog,
    build_fade_white,
    build_hard_cuts,
    build_fade,
]


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--force", action="store_true",
                    help="regenerate the 8 new fixtures + base PNGs even if they "
                         "exist (pinned fixtures are only generated when missing)")
    args = ap.parse_args()

    SAMPLES.mkdir(parents=True, exist_ok=True)
    write_base_images(args.force)

    entries = []
    failures = []
    for builder in BUILDERS:
        name = builder.__name__.replace("build_", "")
        try:
            entries.append(builder(args.force))
            print(f"  ok   {name}")
        except FixtureError as exc:
            failures.append(str(exc))
            print(f"  FAIL {name}: {exc}", file=sys.stderr)
    if failures:
        print(f"\n{len(failures)} fixture(s) failed self-verification", file=sys.stderr)
        return 1

    gt = {
        "schema": "1.0",
        "description": (
            "Ground truth for the SceneCut scene-detection fixture corpus "
            "(DECISIONS.md D11). Machine-checkable: expected_cuts is a list of "
            "{frame, type_hint}; an EMPTY list means the fixture must produce "
            "ZERO cuts (type_hint 'none'). Frame numbering is 0-based; a cut's "
            "frame is the first frame of the new shot (matches detect.py "
            "scene-start emission). Dissolve/fade positions are the measured "
            "midpoint/onset of the transition."
        ),
        "generated_by": "scenecut_py/scripts/make_fixtures.py",
        "type_hint_values": ["cut", "fade", "dissolve", "none"],
        "conventions": {
            "frame_numbering": "0-based; cut frame = first frame of the new shot",
            "no_cut_fixtures": "expected_cuts == [] encodes type_hint 'none'",
            "dissolve_position": "measured midpoint of the blend span",
            "fade_position": "onset frame of the fade (D5: single cut at onset)",
        },
        "fixtures": entries,
    }
    gt_path = SAMPLES / "ground_truth.json"
    gt_path.write_text(json.dumps(gt, indent=2) + "\n")

    total_cuts = sum(len(e["expected_cuts"]) for e in entries)
    print(f"\n{len(entries)} fixtures verified, {total_cuts} expected cuts total")
    print(f"ground truth written: {gt_path}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
