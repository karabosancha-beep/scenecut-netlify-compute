#!/usr/bin/env python3
"""Gradual-transition stratification diagnostic (D14 §7, deferred item).

Classify every GT gradual span in the ClipShots ``only_gradual`` corpus by its
per-span feature signature, so future tuning can target gradual sub-families
(fade-like vs dissolve-like vs other) instead of treating "gradual" as one
undifferentiated mass.  DIAGNOSTIC ONLY — D14 §7 prohibitions apply: never
feeds F1/gate, never tunes thresholds, never scores `--neural` runs.

Feature source
--------------
Reuses the detector's own feature pass (``scenecut.detect._feature_pass`` —
the exact per-frame arrays detection consumes): 160x90 downscale per frame,
``mean_v`` = mean GRAY (luma), ``mad`` = mean-abs pixel diff vs prev frame,
``consec_js`` = normalized JS divergence between consecutive H-S histograms
(``consec_js[i] = JS(H[i], H[i+1])``), ``mean_s`` = mean HSV saturation,
``edge`` = Canny edge count, plus lagged JS at windows {8, 16}
(``lagged_js[w][t] = JS(H[t-w], H[t])``).  No detection runs; this is the
feature pass only (~1 decode per video, ~168 fps measured on this box).

Per-span signals (GT span = half-open [s, e), frames s..e-1)
------------------------------------------------------------
* luma trajectory: start/end/min/max/range/slope, directional consistency
  (``luma_mono``), Pearson corr vs time, black/white dwell fractions, and
  pre/post-span luma context (15-frame means before/after) — the fade-family
  signature.
* consec_js profile: mean/max, plateau fractions at the detector's 0.15 level
  (``plateau_frac``, per D14§7 spec) and at the empirically measured 0.05
  level (``plateau_frac05`` — real cross-dissolves drift at ~0.02-0.09, not
  0.15), spike share (top-pair share of total in-span JS mass).
* windowed JS ``js_win_max`` = max(lagged_js[8], lagged_js[16]) inside the
  span — "do the two sides of this span look like different shots" (the
  detector's own dissolve endpoint signal, floor ``dissolve_dissim`` 0.12).
* MAD level: mean/max plus motion-normalized ratios vs the video's median MAD
  (``mad_ratio_*``) — separates "hard-cut-like jump" from "busy content".
* saturation mean/min/max, edge collapse (``edge_min_ratio``,
  ``edge_min_pos``) — flat-color-transit signature.
* span length.

Class rules (first match wins; RULES dict is the single knob surface)
----------------------------------------------------------------------
1. ``ambiguous``             — length <= 1 (degenerate GT span).
2. ``fade_to_black``         — black dwell (fraction of span frames at
                               mean-gray <= 24) >= 0.15, or >= 0.05 at reduced
                               confidence, UNLESS both luma contexts are dark
                               (dark-scene dissolve guard).  Confident fades
                               also show edge collapse mid-span.
3. ``fade_to_white``         — symmetric at mean-gray >= 232.
4. ``colored_or_other_fade`` — edge collapse (``edge_min_ratio`` <= 0.12) at
                               a mid-span position (not the destination) with
                               low js_mean: a transit through a FLAT frame
                               (color card / brief flash) that never reaches
                               the black/white luma extremes.  Confidence
                               boosted when the flat frame is saturated
                               (``sat_max`` >= 150 — a color card, not gray).
5. ``dissolve_plateau``      — ``js_win_max`` >= 0.12 (sides differ like
                               different shots) AND the change is SPREAD
                               (spike_share < 0.45) AND no hard-cut-like jump
                               (not mad_max >= 20 with mad_ratio_max >= 4):
                               sustained gradual blend, the classic dissolve.
6. ``spiky_or_wipe``         — the change is CONCENTRATED: spike_share >= 0.45
                               with js_max >= 0.10, or a hard-cut-like MAD
                               jump inside the span (mad_max >= 20 AND
                               mad_ratio_max >= 4): fast wipe / flash cut /
                               near-hard-cut annotated as gradual.
7. ``ambiguous``             — weak global signature (js_win_max < 0.12 and
                               nothing else fires) — match-dissolve territory
                               (global H-S is blind there by design; that is
                               the Phase-1.5 spatial-detection backlog).

Confidence is a heuristic strength in [0, 1], never a probability.

Resume / chunked execution
--------------------------
The output JSON is its own durable state: it carries ``videos_done`` +
per-span rows with RAW features; classification is re-derived from raw
features on every invocation, so rule tweaks never force a re-decode
(``--reclassify-only`` re-classifies instantly).  The sandbox caps single
commands at 10 min, so run repeatedly (``--time-budget`` / ``--limit``) until
``meta.complete`` is true; each invocation rewrites the file atomically.

Usage
-----
  python3 gradual_stratify.py                       # process what fits in the
                                                     # time budget, then stop
  python3 gradual_stratify.py --limit 20            # at most 20 new videos
  python3 gradual_stratify.py --reclassify-only     # rules changed; no decode
  python3 gradual_stratify.py --reset               # start over
  python3 gradual_stratify.py --bench ../../bench/external_clipshots_only_gradual_default.json
                                                     # add per-class recall
                                                     # estimate (video-level
                                                     # proportional join)

Methodology epilogue — what the signals can and cannot say (measured)
----------------------------------------------------------------------
* Fades are a GLOBAL luminance phenomenon: the whole frame ramps toward an
  extreme while pixel structure stays (the detector's own fade logic scans
  exactly the ``mean_v`` array).  Dwell fraction + pre/post context is the
  most robust global signature available and survives motion during the ramp
  (unlike MAD/JS, which motion inflates).  Fades also collapse the edge count
  mid-span (flat black/white frames have no Canny edges) — recorded as
  ``edge_min_ratio``.
* MEASURED (this corpus, first 2-video calibration run): real cross-dissolves
  do NOT elevate consecutive-JS to the detector's 0.15 plateau level —
  consecutive blends of two static histograms differ by ~1/L of mixture mass,
  giving js_mean ~0.02-0.09.  The discriminative scale is the WINDOWED
  divergence (lagged_js at 8/16 frames — the detector's own dissolve endpoint
  signal): a real A->B content change gives js_win_max >= ~0.12, a no-op /
  match-dissolve stays below.  The cascade therefore keys "dissolve" on
  js_win_max + spread (spike_share), not on a 0.15 plateau.  Both plateau
  fractions are still recorded (0.15 per the D14§7 spec, 0.05 empirical).
* Wipes/flash transitions change the frame LOCALLY and FAST: consecutive-JS
  mass concentrates in one-two pairs (high spike_share), and/or MAD jumps
  hard-cut-like relative to the video's own motion level (mad_ratio).
* ``colored_or_other_fade`` keys on EDGE COLLAPSE at a mid-span position —
  the only global signature of a transit through a flat color card that never
  touches the luma extremes.  Mean saturation alone is useless as a
  discriminator here (ordinary YouTube content already sits at mean_s 40-155;
  measured).  The saturated-card boost uses sat_max >= 150, which ordinary
  per-frame means do not reach.
* Thresholds are pinned to the detector's own constants where they exist
  (windowed JS 0.12 = ``dissolve_dissim``; hard-cut MAD 20 =
  ``hard_cut_mad_min``; consec 0.15 = ``consec_js_min``; fade extremes from
  ``fade_threshold`` 12 / ``fade_ceiling`` 243 with compression tolerance) so
  the strata speak the detector's language.
* The ``ambiguous`` stratum is itself a deliverable: it lower-bounds the
  global-feature-blind gradual mass (match-dissolves & friends) that only
  Phase-1.5 spatial detection can address.
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT / "scenecut_py"))

import cv2  # noqa: E402
import numpy as np  # noqa: E402

from scenecut.detect import _feature_pass  # noqa: E402
from scenecut.util import probe_video  # noqa: E402

# ------------------------------------------------------------------ knobs

RULES = {
    # luma extremes (mean-gray scale, 0-255)
    "black_level": 24.0,      # at/below -> "black" (detector fade floor 12 + compression lift)
    "white_level": 232.0,     # at/above -> "white" (detector fade_ceiling 243 - tolerance)
    "dwell_strong": 0.15,     # fraction of span frames at the extreme -> confident fade
    "dwell_weak": 0.05,       # brief dip/flash still fades, reduced confidence
    "ctx_dark": 34.0,         # both luma contexts below -> dark-scene dissolve, not fade
    "ctx_bright": 200.0,      # both above -> bright-scene guard for white
    # consec-JS profile
    "plateau_js": 0.15,       # = detector consec_js_min (pinned; reported per D14§7 spec)
    "plateau_js_emp": 0.05,   # empirical drift level (measured on this corpus)
    # windowed JS (dissolve sides-differ signal; detector dissolve_dissim = 0.12)
    "win_js_min": 0.12,
    # concentration / spikiness
    "spike_share": 0.45,      # top-pair share of total in-span JS mass
    "spike_js_min": 0.10,     # js_max floor for the concentrated case
    # motion
    "mad_hardcut": 20.0,      # = detector hard_cut_mad_min (pinned)
    "mad_ratio_spike": 4.0,   # jump vs video's own median MAD
    # colored fade (flat-frame transit)
    "edge_flat": 0.12,        # edge_min_ratio <= -> a span frame is (nearly) edgeless
    "edge_min_pos_max": 0.85, # flat frame must be mid-span (not the destination)
    "colored_js_max": 0.10,
    "sat_card": 150.0,        # sat_max >= -> the flat frame was a saturated color card
}

CLASS_ORDER = ["fade_to_black", "fade_to_white", "colored_or_other_fade",
               "dissolve_plateau", "spiky_or_wipe", "ambiguous"]

CLASS_RULE_DOC = {
    "fade_to_black": "black_dwell >= dwell_strong (or >= dwell_weak at reduced confidence) at "
                     "mean-gray <= black_level, unless both luma contexts are dark "
                     "(dark-scene dissolve guard)",
    "fade_to_white": "white_dwell >= dwell_strong (or >= dwell_weak) at mean-gray >= white_level, "
                     "unless both contexts are bright",
    "colored_or_other_fade": "edge_min_ratio <= edge_flat at a mid-span position AND js_mean <= "
                             "colored_js_max (transit through a flat color card / brief flash "
                             "that never reaches the luma extremes; conf boosted by sat_max >= sat_card)",
    "dissolve_plateau": "js_win_max >= win_js_min (sides differ like different shots) AND "
                        "spike_share < spike_share (change is spread, not concentrated) AND "
                        "no hard-cut-like MAD jump",
    "spiky_or_wipe": "concentrated change: (spike_share >= spike_share AND js_max >= spike_js_min) "
                     "OR (mad_max >= mad_hardcut AND mad_ratio_max >= mad_ratio_spike)",
    "ambiguous": "length <= 1 (degenerate) or weak global signature (js_win_max < win_js_min "
                 "and nothing else fires) — match-dissolve territory",
}

DEFAULT_DATASET_DIR = REPO_ROOT / "datasets" / "clipshots"
DEFAULT_OUT = REPO_ROOT / "bench" / "gradual_stratification.json"
CTX_RADIUS = 15  # frames of pre/post-span luma context
TRAJ_SAMPLES = 48  # max downsampled trajectory points stored per span


# ------------------------------------------------------------------ GT loading

def load_gt_spans(dataset_dir: Path) -> dict[str, list[tuple[int, int]]]:
    """{video_name: [(s, e), ...]} for on-disk videos bearing gradual spans.

    ClipShots annotation format: {"<video>.mp4": {"transitions": [[s, e], ...],
    "frame_num": N}}.  Hard cut iff e == s+1 (loader semantics, A17); wider
    spans are gradual intervals, half-open [s, e).  A degenerate e == s span
    still counts as gradual (the bench loader does the same — 640 total).
    Videos whose transitions are all hard cuts carry no gradual spans and are
    skipped (they are GT-bearing for the bench, but not for stratification).
    """
    ann = dataset_dir / "annotations" / "only_gradual.json"
    data = json.loads(ann.read_text())
    out: dict[str, list[tuple[int, int]]] = {}
    for key, info in sorted(data.items()):
        name = Path(key).name
        transitions = (info or {}).get("transitions") or []
        if not transitions:
            continue
        video = dataset_dir / "videos" / name
        if not video.is_file():
            continue
        spans = []
        for tr in transitions:
            try:
                s, e = int(tr[0]), int(tr[1])
            except (ValueError, IndexError, TypeError):
                continue
            if e != s + 1:  # wider-than-1 => gradual (half-open [s, e))
                spans.append((s, e))
        if spans:
            out[name] = spans
    return out


# ------------------------------------------------------------------ extraction

def _safe_corr(y: np.ndarray) -> float:
    """Pearson corr of y vs its index; 0.0 when degenerate."""
    n = len(y)
    if n < 2:
        return 0.0
    t = np.arange(n, dtype=np.float64)
    if float(np.std(y)) < 1e-9 or float(np.std(t)) < 1e-9:
        return 0.0
    return float(np.corrcoef(y, t)[0, 1])


def _downsample(arr: np.ndarray, k: int) -> list[float]:
    """Evenly-spaced <= k samples of arr, rounded for JSON."""
    if len(arr) == 0:
        return []
    if len(arr) <= k:
        return [round(float(x), 3) for x in arr]
    idx = np.linspace(0, len(arr) - 1, k).round().astype(int)
    return [round(float(arr[i]), 3) for i in idx]


def _span_features(feat, s: int, e: int) -> dict:
    """Raw per-span feature dict from a filled _Features object (span [s, e))."""
    n = feat.n
    s_c = max(0, min(s, n - 1))
    e_c = max(s_c, min(e, n))
    L = e_c - s_c

    y = feat.mean_v[s_c:e_c].astype(np.float64)
    js = feat.consec_js[s_c:max(s_c, e_c - 1)].astype(np.float64)  # pairs inside span
    mad = feat.mad[s_c:e_c].astype(np.float64)
    sat = feat.mean_s[s_c:e_c].astype(np.float64)
    edge = feat.edge[s_c:e_c].astype(np.float64)

    # windowed JS (8 / 16) inside span — valid only where t >= w
    js8 = js16 = None
    for w in (8, 16):
        if w in feat.lagged_js:
            seg = feat.lagged_js[w][s_c:e_c]
            valid = seg[max(0, w - s_c):] if s_c < w else seg
            m = float(np.max(valid)) if len(valid) else None
            if w == 8:
                js8 = m
            else:
                js16 = m
    js_win_max = max((v for v in (js8, js16) if v is not None), default=None)

    # luma context around the span
    ctx_b = feat.mean_v[max(0, s_c - CTX_RADIUS):s_c]
    ctx_a = feat.mean_v[e_c:min(n, e_c + CTX_RADIUS)]
    ctx_before = round(float(np.mean(ctx_b)), 2) if len(ctx_b) else None
    ctx_after = round(float(np.mean(ctx_a)), 2) if len(ctx_a) else None

    # motion normalization: the video's own median MAD
    mad_med = float(np.median(feat.mad)) if n else 1.0
    mad_med = max(mad_med, 0.5)

    steps = np.diff(y) if L > 1 else np.zeros(0)
    net = float(np.sum(steps)) if L > 1 else 0.0
    if L > 1 and net != 0.0:
        direction = 1.0 if net > 0 else -1.0
        mono = float(np.mean((steps == 0) | (np.sign(steps) == direction)))
    else:
        mono = 0.5 if L > 1 else 0.0

    js_sum = float(np.sum(js)) if len(js) else 0.0
    js_max = float(np.max(js)) if len(js) else 0.0
    plateau15 = float(np.mean(js >= RULES["plateau_js"])) if len(js) else 0.0
    plateau05 = float(np.mean(js >= RULES["plateau_js_emp"])) if len(js) else 0.0
    spike_share = (js_max / js_sum) if js_sum > 1e-9 else 0.0

    typical_edge = float(np.mean(feat.edge)) if n else 1.0
    edge_min = float(np.min(edge)) if len(edge) else 0.0
    edge_min_pos = (float(np.argmin(edge)) / (L - 1)) if L > 1 else 0.0
    mad_mean = float(np.mean(mad)) if len(mad) else 0.0
    mad_max = float(np.max(mad)) if len(mad) else 0.0

    return {
        "length": L,
        "clipped": (s, e) != (s_c, e_c),
        "luma_start": round(float(y[0]), 2) if L else None,
        "luma_end": round(float(y[-1]), 2) if L else None,
        "luma_min": round(float(np.min(y)), 2) if L else None,
        "luma_max": round(float(np.max(y)), 2) if L else None,
        "luma_range": round(float(np.max(y) - np.min(y)), 2) if L else None,
        "luma_slope": round(float(y[-1] - y[0]) / (L - 1), 3) if L > 1 else None,
        "luma_mono": round(mono, 3),
        "luma_corr": round(_safe_corr(y), 3) if L > 1 else None,
        "black_dwell": round(float(np.mean(y <= RULES["black_level"])), 3) if L else None,
        "white_dwell": round(float(np.mean(y >= RULES["white_level"])), 3) if L else None,
        "ctx_luma_before": ctx_before,
        "ctx_luma_after": ctx_after,
        "js_mean": round(float(np.mean(js)), 4) if len(js) else None,
        "js_max": round(js_max, 4),
        "plateau_frac": round(plateau15, 3),       # at the detector's 0.15 (D14§7 spec)
        "plateau_frac05": round(plateau05, 3),     # at the empirical 0.05 level
        "spike_share": round(spike_share, 3),
        "n_js_pairs": int(len(js)),
        "js8_max": round(js8, 4) if js8 is not None else None,
        "js16_max": round(js16, 4) if js16 is not None else None,
        "js_win_max": round(js_win_max, 4) if js_win_max is not None else None,
        "mad_mean": round(mad_mean, 2),
        "mad_max": round(mad_max, 2),
        "mad_ratio_mean": round(mad_mean / mad_med, 2),
        "mad_ratio_max": round(mad_max / mad_med, 2),
        "sat_mean": round(float(np.mean(sat)), 1) if len(sat) else None,
        "sat_min": round(float(np.min(sat)), 1) if len(sat) else None,
        "sat_max": round(float(np.max(sat)), 1) if len(sat) else None,
        "edge_mean": round(float(np.mean(edge)), 1) if len(edge) else None,
        "edge_min_ratio": round(edge_min / typical_edge, 3) if typical_edge > 0 else None,
        "edge_min_pos": round(edge_min_pos, 3),
        "luma_traj": _downsample(y, TRAJ_SAMPLES),
        "js_traj": _downsample(js, TRAJ_SAMPLES),
        "edge_traj": _downsample(edge, TRAJ_SAMPLES),
    }


def process_video(job: tuple[str, str, list[tuple[int, int]]]) -> dict:
    """Decode one video's feature pass and extract raw per-span features.

    job = (video_name, video_path, gt_spans).  Returns a per-video record
    {video, ok, meta, spans: [{span, features}]}; failures are recorded as
    ok=False rows so a single bad video never kills the run.
    """
    name, path, spans = job
    try:
        cv2.setNumThreads(1)
        try:
            fps = probe_video(path).fps or 30.0
        except Exception:
            fps = 30.0
        feat = _feature_pass(path, [8, 16], fps)
        rows = []
        for (s, e) in spans:
            if s >= feat.n:
                rows.append({"span": [s, e], "features": None,
                             "error": f"span start {s} beyond decoded frames ({feat.n})"})
                continue
            rows.append({"span": [s, e], "features": _span_features(feat, s, e)})
        return {"video": name, "ok": True,
                "meta": {"n_frames": int(feat.n), "fps": round(float(fps), 3),
                         "n_gt_spans": len(spans)},
                "spans": rows}
    except Exception as exc:  # decode failure — record, continue
        return {"video": name, "ok": False, "meta": {"error": str(exc)[:200]},
                "spans": [{"span": list(sp), "features": None, "error": "video decode failed"}
                          for sp in spans]}


# ------------------------------------------------------------------ classification

def _classify(f: dict | None) -> tuple[str, float, str]:
    """(class, confidence, note) from raw span features — pure function."""
    if not f:
        return "ambiguous", 0.0, "no features (decode/span error)"
    L = f["length"]
    if L <= 1:
        return "ambiguous", 0.2, "degenerate GT span (<=1 frame: nothing to profile)"

    bd = f.get("black_dwell") or 0.0
    wd = f.get("white_dwell") or 0.0
    cb = f.get("ctx_luma_before")
    ca = f.get("ctx_luma_after")
    dark_ctx = ((cb is not None and cb < RULES["ctx_dark"]) and
                (ca is not None and ca < RULES["ctx_dark"]))
    bright_ctx = ((cb is not None and cb > RULES["ctx_bright"]) and
                  (ca is not None and ca > RULES["ctx_bright"]))

    # ---- fade family (dwell at an extreme, context not permanently there)
    for dwell, level, cls, ctx_bad in ((bd, "black", "fade_to_black", dark_ctx),
                                       (wd, "white", "fade_to_white", bright_ctx)):
        if dwell >= RULES["dwell_strong"] and not ctx_bad:
            conf = 0.55 + 0.40 * min(1.0, (dwell - RULES["dwell_strong"]) / 0.35)
            note = f"{level} dwell {dwell:.2f}"
            if (f.get("edge_min_ratio") or 1.0) <= 0.25:
                conf = min(0.99, conf + 0.04)  # edge collapse corroborates
                note += ", edge collapse"
            if cb is not None and ca is not None and \
                    (cb > RULES["ctx_dark"] + 20 and ca > RULES["ctx_dark"] + 20):
                conf = min(0.99, conf + 0.05)  # true transit: enters and leaves
                note += " (transit)"
            return cls, round(min(0.99, conf), 2), note
        if dwell >= RULES["dwell_strong"] and ctx_bad:
            break  # extreme dwell but permanently dark/bright scene -> NOT a fade
    for dwell, level, cls, ctx_bad in ((bd, "black", "fade_to_black", dark_ctx),
                                       (wd, "white", "fade_to_white", bright_ctx)):
        if RULES["dwell_weak"] <= dwell < RULES["dwell_strong"] and not ctx_bad:
            conf = 0.30 + 0.25 * (dwell - RULES["dwell_weak"]) / RULES["dwell_strong"]
            return cls, round(conf, 2), f"brief {level} dip/flash (dwell {dwell:.2f})"

    # ---- colored / other fade: transit through a FLAT mid-span frame
    if ((f.get("edge_min_ratio") is not None and f["edge_min_ratio"] <= RULES["edge_flat"])
            and (f.get("edge_min_pos") or 0.0) <= RULES["edge_min_pos_max"]
            and (f.get("js_mean") if f.get("js_mean") is not None else 1.0) <= RULES["colored_js_max"]):
        conf = 0.45 + 0.15 * min(1.0, (RULES["edge_flat"] - f["edge_min_ratio"]) / RULES["edge_flat"])
        note = f"flat-frame transit (edge_min_ratio {f['edge_min_ratio']} at pos {f.get('edge_min_pos')})"
        if (f.get("sat_max") or 0.0) >= RULES["sat_card"]:
            conf = min(0.85, conf + 0.15)
            note += f", saturated card (sat_max {f['sat_max']})"
        return "colored_or_other_fade", round(min(0.8, conf), 2), note

    # ---- dissolve plateau: sides differ, change is spread, no hard jump
    js_win = f.get("js_win_max") or 0.0
    hard_jump = ((f.get("mad_max") or 0.0) >= RULES["mad_hardcut"]
                 and (f.get("mad_ratio_max") or 0.0) >= RULES["mad_ratio_spike"])
    if js_win >= RULES["win_js_min"] and (f.get("spike_share") or 0.0) < RULES["spike_share"] \
            and not hard_jump:
        conf = 0.50 + 0.25 * min(1.0, (js_win - RULES["win_js_min"]) / 0.40)
        if (f.get("plateau_frac05") or 0.0) >= 0.30:
            conf = min(0.95, conf + 0.10)
        note = f"js_win_max {js_win}, spread (spike_share {f.get('spike_share')})"
        if dark_ctx:
            note += " (dark scene both sides)"
        return "dissolve_plateau", round(min(0.95, conf), 2), note

    # ---- spiky / wipe: concentrated change
    if ((f.get("spike_share") or 0.0) >= RULES["spike_share"]
            and (f.get("js_max") or 0.0) >= RULES["spike_js_min"]) or hard_jump:
        conf = 0.45 + 0.25 * min(1.0, (f.get("spike_share") or 0.0))
        if hard_jump:
            conf = min(0.9, conf + 0.10)
        return "spiky_or_wipe", round(min(0.9, conf), 2), \
            f"concentrated (spike_share {f.get('spike_share')}, js_max {f.get('js_max')}, " \
            f"mad_max {f.get('mad_max')}, mad_ratio_max {f.get('mad_ratio_max')})"

    if js_win < RULES["win_js_min"]:
        return "ambiguous", 0.3, \
            f"weak global signature (js_win_max {js_win} < {RULES['win_js_min']}: " \
            "match-dissolve territory)"
    return "ambiguous", 0.3, "mixed signature: no rule fired cleanly"


# ------------------------------------------------------------------ aggregation

def _agg(vals: list[float]) -> dict:
    if not vals:
        return {"n": 0}
    sv = sorted(vals)
    return {"n": len(vals),
            "mean": round(float(np.mean(vals)), 3),
            "median": round(float(sv[len(sv) // 2]), 3),
            "min": round(float(sv[0]), 3), "max": round(float(sv[-1]), 3)}


STAT_FIELDS = ["length", "luma_range", "luma_slope", "js_mean", "js_max",
               "plateau_frac", "plateau_frac05", "spike_share", "js_win_max",
               "mad_mean", "mad_max", "mad_ratio_mean", "black_dwell",
               "white_dwell", "sat_mean", "sat_max", "edge_min_ratio",
               "confidence"]


def build_report(videos: dict[str, dict], rules: dict, complete: bool,
                 totals: dict) -> dict:
    """Full output JSON: classes re-derived from raw features + aggregates."""
    spans = []
    for name in sorted(videos):
        rec = videos[name]
        for row in rec["spans"]:
            f = row.get("features")
            cls, conf, note = _classify(f)
            spans.append({
                "video": name,
                "span": row["span"],
                "length": (f or {}).get("length", row["span"][1] - row["span"][0]),
                "class": cls,
                "confidence": conf,
                **({"note": note} if note else {}),
                **({"features": f} if f else {"error": row.get("error", "missing")}),
            })

    counts = {c: 0 for c in CLASS_ORDER}
    for sp in spans:
        counts[sp["class"]] = counts.get(sp["class"], 0) + 1
    n_total = len(spans)

    per_class = {}
    for c in CLASS_ORDER:
        rows = [sp for sp in spans if sp["class"] == c]
        entry = {
            "n": len(rows),
            "share": round(len(rows) / n_total, 4) if n_total else 0.0,
            "rule": CLASS_RULE_DOC[c],
        }
        for field in STAT_FIELDS:
            if field == "confidence":
                vals = [r["confidence"] for r in rows]
            elif field == "length":
                vals = [r["length"] for r in rows]
            else:
                vals = [r["features"][field] for r in rows
                        if r.get("features")
                        and isinstance(r["features"].get(field), (int, float))]
            entry[field] = _agg(vals)
        per_class[c] = entry

    return {
        "meta": {
            "script": "scenecut_py/scripts/gradual_stratify.py",
            "purpose": "D14 §7 model-assisted gradual stratification — DIAGNOSTIC ONLY "
                       "(never feeds F1/gate, never tunes thresholds)",
            "generated": datetime.now(timezone.utc).isoformat(timespec="seconds"),
            "dataset": "clipshots:only_gradual",
            "feature_source": "scenecut.detect._feature_pass (160x90: mean gray luma, MAD, "
                              "consec JS on 50x50 H-S hists, HSV-S, Canny edge, lagged JS w={8,16})",
            "gt_convention": "half-open [s, e); hard cut iff e == s+1 (excluded); "
                             "degenerate e == s kept as gradual (bench-loader parity)",
            "videos_total": totals["videos_total"],
            "videos_done": len(videos),
            "spans_total_expected": totals["spans_total"],
            "spans_scored": n_total,
            "complete": complete,
            "rules": rules,
            "class_rules": CLASS_RULE_DOC,
        },
        "class_counts": counts,
        "per_class": per_class,
        "videos_done": sorted(videos),
        "video_meta": {name: rec["meta"] for name, rec in sorted(videos.items())},
        "spans": spans,
    }


# ------------------------------------------------------------------ bench join

def bench_join(report: dict, bench_path: Path) -> dict | None:
    """Per-class gradual-recall ESTIMATE from existing bench rows.

    The bench rows carry per-video tp/fn counts but NOT detected interval
    positions, so span-level matching is impossible; this is a video-level
    PROPORTIONAL attribution: each video's gradual recall is attributed to
    classes in proportion to that video's span counts (expected-TP math),
    plus an EXACT subset restricted to videos whose spans are all one class.
    Coverage caveat: the bench run covers only its own video subset.
    """
    if not bench_path.is_file():
        return None
    bench = json.loads(bench_path.read_text())
    rows = {r["video"]: r for r in bench.get("per_video", [])}
    covered = [v for v in report["videos_done"] if v in rows]
    if not covered:
        return None

    def video_recall(r: dict) -> tuple[float | None, float | None]:
        off = (r.get("clipshots_official") or {}).get("graduals") or {}
        native = r.get("transitions_gradual") or {}
        r_off = (off["tp"] / off["gts"]) if off.get("gts") else None
        r_nat = (native["tp"] / (native["tp"] + native["fn"])) \
            if (native.get("tp", 0) + native.get("fn", 0)) else None
        return r_off, r_nat

    out = {
        "bench_file": str(bench_path),
        "bench_videos_covered": len(covered),
        "method": "video-level proportional attribution (bench rows lack detected "
                  "interval positions — span-level matching impossible); exact subset "
                  "= videos whose spans are all one class",
        "metrics": {},
    }
    for label, idx in (("clipshots_official_graduals", 0), ("native_transitions_gradual", 1)):
        est_tp = {c: 0.0 for c in CLASS_ORDER}
        est_n = {c: 0 for c in CLASS_ORDER}
        exact_tp = {c: 0 for c in CLASS_ORDER}
        exact_n = {c: 0 for c in CLASS_ORDER}
        for v in covered:
            rec = video_recall(rows[v])
            recall = rec[idx]
            if recall is None:
                continue
            vrows = [sp for sp in report["spans"] if sp["video"] == v]
            vcounts = {c: 0 for c in CLASS_ORDER}
            for sp in vrows:
                vcounts[sp["class"]] += 1
            for c in CLASS_ORDER:
                est_tp[c] += recall * vcounts[c]
                est_n[c] += vcounts[c]
            if sum(1 for c in vcounts.values() if c > 0) == 1:
                c = next(c for c, k in vcounts.items() if k > 0)
                exact_tp[c] += round(recall * vcounts[c])
                exact_n[c] += vcounts[c]
        est = {c: {"est_recall": round(est_tp[c] / est_n[c], 4) if est_n[c] else None,
                   "n_spans_attributed": est_n[c]}
               for c in CLASS_ORDER}
        exact = {c: {"recall": round(exact_tp[c] / exact_n[c], 4) if exact_n[c] else None,
                     "n_spans": exact_n[c]}
                 for c in CLASS_ORDER if exact_n[c]}
        out["metrics"][label] = {"estimated": est, "exact_single_class_videos": exact}
    return out


# ------------------------------------------------------------------ printing

def print_summary(report: dict) -> None:
    m = report["meta"]
    print(f"\n=== gradual stratification — {m['dataset']} "
          f"({m['videos_done']}/{m['videos_total']} videos, "
          f"{m['spans_scored']}/{m['spans_total_expected']} spans, "
          f"complete={m['complete']}) ===")
    hdr = (f"{'class':<22}{'n':>5}{'share':>7}{'len_med':>9}{'js_mean':>9}"
           f"{'js_win':>8}{'plat05':>8}{'mad_mean':>9}{'black':>7}{'white':>7}{'conf':>7}")
    print(hdr)
    print("-" * len(hdr))

    def _cell(entry: dict, field: str, width: int, prec: int) -> str:
        v = entry.get(field) or {}
        val = v.get("median", v.get("mean"))
        return f"{val:>{width}.{prec}f}" if isinstance(val, (int, float)) else f"{'-':>{width}}"

    for c in CLASS_ORDER:
        e = report["per_class"][c]
        print(f"{c:<22}{e['n']:>5}{e['share'] * 100:>6.1f}%"
              f"{_cell(e, 'length', 9, 1)}{_cell(e, 'js_mean', 9, 3)}"
              f"{_cell(e, 'js_win_max', 8, 3)}{_cell(e, 'plateau_frac05', 8, 2)}"
              f"{_cell(e, 'mad_mean', 9, 2)}{_cell(e, 'black_dwell', 7, 2)}"
              f"{_cell(e, 'white_dwell', 7, 2)}{_cell(e, 'confidence', 7, 2)}")
    fade_n = report["class_counts"]["fade_to_black"] + report["class_counts"]["fade_to_white"] \
        + report["class_counts"]["colored_or_other_fade"]
    n = report["meta"]["spans_scored"]
    if n:
        print(f"\nfade-like family (black+white+colored): {fade_n}/{n} = {fade_n / n:.1%}")


# ------------------------------------------------------------------ main

def main() -> int:
    ap = argparse.ArgumentParser(
        description="Gradual-transition stratification diagnostic (D14 §7) — "
                    "classify GT gradual spans by feature signature. See module "
                    "docstring for methodology.")
    ap.add_argument("--dataset-dir", default=str(DEFAULT_DATASET_DIR))
    ap.add_argument("--out", default=str(DEFAULT_OUT),
                    help="output JSON (also the durable resume state)")
    ap.add_argument("--limit", type=int, default=None,
                    help="process at most N new videos this invocation")
    ap.add_argument("--time-budget", type=float, default=480.0,
                    help="stop pulling new videos after S seconds (default 480)")
    ap.add_argument("--workers", type=int, default=2,
                    help="parallel decode workers (default 2 = CPU count)")
    ap.add_argument("--reset", action="store_true", help="discard prior state, start over")
    ap.add_argument("--reclassify-only", action="store_true",
                    help="no decode; re-derive classes from cached raw features")
    ap.add_argument("--bench", default=None,
                    help="optional bench JSON for the per-class recall join")
    args = ap.parse_args()

    out_path = Path(args.out).resolve()
    dataset_dir = Path(args.dataset_dir).resolve()
    gt = load_gt_spans(dataset_dir)
    totals = {"videos_total": len(gt),
              "spans_total": sum(len(v) for v in gt.values())}
    print(f"[0] corpus: {totals['videos_total']} videos bearing gradual spans, "
          f"{totals['spans_total']} gradual spans "
          f"(of 150 GT-bearing only_gradual videos on disk)")

    # ---- resume state
    videos: dict[str, dict] = {}
    if out_path.is_file() and not args.reset:
        try:
            prior = json.loads(out_path.read_text())
            for name in prior.get("videos_done", []):
                if name in gt:  # stale names dropped; rows re-validated
                    videos[name] = {"meta": (prior.get("video_meta") or {}).get(name, {}),
                                    "spans": [sp_row_to_state(sp)
                                              for sp in prior.get("spans", [])
                                              if sp.get("video") == name]}
            if videos:
                print(f"[0] resumed: {len(videos)} videos already scored")
        except (json.JSONDecodeError, KeyError):
            print("[0] prior output unreadable — starting fresh")
    out_path.parent.mkdir(parents=True, exist_ok=True)

    todo = [v for v in sorted(gt) if v not in videos]
    if args.limit is not None:
        todo = todo[:args.limit]

    # ---- decode loop (chunked, budget-bounded, resumable)
    if not args.reclassify_only and todo:
        import multiprocessing as mp
        t0 = time.time()
        batch = max(args.workers, args.workers * 2)
        with mp.Pool(args.workers) as pool:
            for i in range(0, len(todo), batch):
                if time.time() - t0 > args.time_budget:
                    print(f"[1] time budget ({args.time_budget:.0f}s) reached — "
                          f"stopping cleanly, re-run to continue")
                    break
                chunk = todo[i:i + batch]
                jobs = [(v, str(dataset_dir / "videos" / v), gt[v]) for v in chunk]
                for rec in pool.imap_unordered(_worker_entry, jobs):
                    videos[rec["video"]] = {"meta": rec["meta"], "spans": rec["spans"]}
                    err = "" if rec["ok"] else f" [DECODE ERROR: {rec['meta'].get('error', '')}]"
                    print(f"[1] {rec['video']}: {len(rec['spans'])} spans, "
                          f"{rec['meta'].get('n_frames', '?')} frames{err}", flush=True)
        print(f"[1] decode loop done: {len(videos)}/{totals['videos_total']} videos "
              f"in {time.time() - t0:.0f}s")

    # ---- report (classes always re-derived from raw features)
    complete = len(videos) == totals["videos_total"]
    report = build_report(videos, RULES, complete, totals)
    if args.bench:
        join = bench_join(report, Path(args.bench))
        if join:
            report["bench_join"] = join

    tmp = out_path.with_suffix(".json.tmp")
    tmp.write_text(json.dumps(report, indent=1))
    tmp.replace(out_path)
    print(f"[2] wrote {out_path} ({report['meta']['spans_scored']} span rows)")
    print_summary(report)

    if not complete:
        print("\nINCOMPLETE — re-run this script (state is durable) until complete=true")
        return 1
    return 0


def sp_row_to_state(sp: dict) -> dict:
    """Output span row -> worker-state row (features may be None on error)."""
    if "features" in sp:
        return {"span": sp["span"], "features": sp["features"]}
    return {"span": sp["span"], "features": None, "error": sp.get("error", "missing")}


def _worker_entry(job):
    """Pool entry point (module-level for picklability)."""
    return process_video(job)


if __name__ == "__main__":
    sys.exit(main())
