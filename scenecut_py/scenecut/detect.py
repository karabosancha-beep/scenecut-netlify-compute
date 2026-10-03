"""Multi-pass scene/shot detection — v2.1 architecture (DECISIONS.md D3-D7).

Pipeline (2 decodes total, was 4):
  1. Orchestrated scenedetect pass: Adaptive + Threshold(FLOOR) + Threshold(CEILING)
     + Hash detectors driven manually over ONE VideoStream, with SceneManager's
     auto-downscale replicated (max_dim/256, INTER_LINEAR). Per-detector cut lists
     preserve source identity; per-detector StatsManagers capture metrics for
     score-derived confidences (D7).
  2. OpenCV feature-extraction pass (single decode): per-frame L1-normalized H-S
     histograms (50x50) kept in a bounded rolling window, plus full-length scalar
     arrays: MAD (mean abs diff, pixel level), Canny edge energy, mean S, mean V,
     consecutive-frame JS divergence, and lagged JS per dissolve scale. Also a
     subsampled mid-shot histogram store for merge/grouping (zero random seeks).
  3. Dissolve detection (D3): two-tier endpoint JS divergence + edge-energy dip
     (mixture signature) + MAD hard-cut guard, multi-scale windows {8,16,32} with
     NMS across scales. Replaces the inverted Pearson-correlation sweep.
  4. Source-priority dedup with capped noisy-OR confidence merge + conf_by_source.
  5. Optional motion gate (D4): downgrade-not-delete, family-based veto, default OFF.
  6. Optional merge-short-scenes: greedy boundary removal over precomputed
     shot-middle histograms (no seeks, no iteration cap, first/last scenes fixed).
  7. Optional scene grouping (JS scale).
"""
from __future__ import annotations

import math
import sys
from pathlib import Path
from typing import Any

import cv2
import numpy as np

from .config import DEFAULTS
from .timecode import seconds_to_hmsms, frame_to_seconds
from .util import SceneCutError, probe_video

# ------------------------------------------------------------------ histogram math

_HIST_BINS = 50  # H x S bins (50x50 = 2500-dim)
_JS_LN2 = math.log(2.0)
_HUE_WRAP_SHIFT = 25  # bins (of 50) = 90 degrees of the 180-degree hue circle
_FEATURE_W, _FEATURE_H = 160, 90
# D18: spatial block grid (8x8 blocks of ~20x11 px at feature scale)
_SPATIAL_GRID = 8


def _to_hs_hist(frame_bgr: np.ndarray, bins: int = _HIST_BINS) -> np.ndarray:
    """L1-normalized H-S histogram (probability distribution) of a frame.

    Zero-bin smoothing (epsilon) so JS divergence log terms are defined.
    """
    hsv = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2HSV)
    hist = cv2.calcHist([hsv], [0, 1], None, [bins, bins], [0, 180, 0, 256])
    hist = hist.astype(np.float64).flatten()
    hist += 1e-7
    return hist / hist.sum()


def _js_divergence(a: np.ndarray, b: np.ndarray, wrap: bool = True) -> float:
    """Normalized Jensen-Shannon divergence in [0, 1] between two L1 hists.

    `wrap=True` also evaluates the hue-rotated variant (bin roll by 25 = 90 deg)
    and returns the min — mitigates the hue-circle wraparound artifact where
    reddish mass splits across the 0/180 boundary (RV1 N15).
    """
    def _js(p: np.ndarray, q: np.ndarray) -> float:
        m = 0.5 * (p + q)
        kl_pm = float(np.sum(p * np.log(p / m)))
        kl_qm = float(np.sum(q * np.log(q / m)))
        return (0.5 * kl_pm + 0.5 * kl_qm) / _JS_LN2

    val = _js(a, b)
    if wrap and _HUE_WRAP_SHIFT:
        bins = _HIST_BINS
        ar = np.roll(a.reshape(bins, bins), _HUE_WRAP_SHIFT, axis=0).flatten()
        br = np.roll(b.reshape(bins, bins), _HUE_WRAP_SHIFT, axis=0).flatten()
        val = min(val, _js(ar, br))
    return float(np.clip(val, 0.0, 1.0))


# ------------------------------------------------------------------ feature pass


class _Features:
    """Full-length scalar arrays + bounded rolling histogram state (D6 v2.1).

    Memory for a 2h@30fps source: ~5 scalar arrays x 864 KB + O(w_max) hists
    + subsampled mid-hist store (~72 MB at 1 sample/sec) — bounded, never the
    2.2 GB full-matrix hazard (RV1 N5). mean_v is mean GRAY (luma).
    """

    def __init__(self, n_frames: int, windows: list[int]):
        n = max(n_frames, 1)
        self.n = n_frames
        self.mad = np.zeros(n, dtype=np.float32)          # pixel mean-abs-diff vs prev frame
        self.edge = np.zeros(n, dtype=np.float32)         # Canny edge pixel count (160x90)
        self.mean_s = np.zeros(n, dtype=np.float32)       # mean saturation
        self.mean_v = np.zeros(n, dtype=np.float32)       # mean luma (V channel)
        self.consec_js = np.ones(n, dtype=np.float32)     # JS(H[i], H[i+1])
        self.lagged_js: dict[int, np.ndarray] = {w: np.ones(n, dtype=np.float32) for w in windows}
        # D18 (Phase 1.5): 8x8 block-max MAD — global H-S histograms are
        # spatially blind to same-palette blends (match-dissolves: endpoint JS
        # 0.04, consec 0.008), but the spatial ARRANGEMENT differs (block-max
        # MAD at the blend endpoints: 17-84 vs sunset drift <= 10). 0 = no
        # spatial change (NOT the 1.0 JS-sentinel convention).
        self.spatial_consec = np.zeros(n, dtype=np.float32)   # block-max MAD, t vs t-1
        self.spatial_lag: dict[int, np.ndarray] = {w: np.zeros(n, dtype=np.float32) for w in windows}
        # Subsampled histogram store for merge/group (mid-shot similarity)
        self.mid_stride = 1
        self.mid_hists: list[np.ndarray] = []


def _feature_pass(video_path: str, windows: list[int], fps: float,
                  progress_cb=None) -> _Features:
    """ONE OpenCV decode building all detection features (replaces the old
    dissolve-pass decode + all random seeks in merge/group)."""
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise SceneCutError(f"OpenCV failed to open {video_path}", code=4)

    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0) or 1
    n_est = total
    feat = _Features(n_est, windows)

    def _grow(arr: np.ndarray) -> np.ndarray:
        # CR1 M3: the container frame count is an ESTIMATE — decodes can yield
        # more frames (VFR/metadata drift); grow scalar arrays in chunks.
        pad = np.zeros(4096, dtype=arr.dtype)
        return np.concatenate([arr, pad])

    def _ensure(t: int) -> None:
        need = t + 2
        while feat.mad.shape[0] < need:
            feat.mad = _grow(feat.mad)
            feat.edge = _grow(feat.edge)
            feat.mean_s = _grow(feat.mean_s)
            feat.mean_v = _grow(feat.mean_v)
            feat.consec_js = _grow(feat.consec_js)
            feat.spatial_consec = _grow(feat.spatial_consec)
            for w in windows:
                feat.lagged_js[w] = _grow(feat.lagged_js[w])
                feat.spatial_lag[w] = _grow(feat.spatial_lag[w])
    # Subsample stride: ~1 sample/sec, capped at 20k samples.
    stride = max(1, int(round(fps)) if fps and fps > 0 else 1)
    if n_est / stride > 20000:
        stride = int(math.ceil(n_est / 20000))
    feat.mid_stride = stride

    w_max = max(windows) if windows else 1
    ring: list[np.ndarray] = []  # hists for frames [t - w_max, t]
    gray_ring: list[np.ndarray] = []  # D18: small grays for lagged spatial MAD
    prev_gray: np.ndarray | None = None
    t = 0
    while True:
        ok, frame = cap.read()
        if not ok:
            break
        small = cv2.resize(frame, (_FEATURE_W, _FEATURE_H), interpolation=cv2.INTER_AREA)
        gray = cv2.cvtColor(small, cv2.COLOR_BGR2GRAY)
        hsv = cv2.cvtColor(small, cv2.COLOR_BGR2HSV)
        _ensure(t)

        # scalar features (mean GRAY, not HSV V — V=max(R,G,B) stays ~255 on
        # saturated content, blinding fade detection; gray matches luma semantics)
        if prev_gray is not None:
            ad = cv2.absdiff(gray, prev_gray)
            feat.mad[t] = float(np.mean(ad))
            # D18: 8x8 block-max of the SAME absdiff (area-mean per block via
            # INTER_AREA downsize; blocks are ~20x11 px). Match-dissolve blend
            # frames rearrange spatially while the global histogram walks
            # only ~0.008 — this array sees what consec_js cannot.
            feat.spatial_consec[t] = float(
                cv2.resize(ad, (_SPATIAL_GRID, _SPATIAL_GRID),
                           interpolation=cv2.INTER_AREA).max())
        feat.edge[t] = float(cv2.countNonZero(cv2.Canny(gray, 50, 150)))
        feat.mean_s[t] = float(np.mean(hsv[:, :, 1]))
        feat.mean_v[t] = float(np.mean(gray))

        hist = _to_hs_hist(small)
        if prev_gray is not None and t > 0:
            feat.consec_js[t - 1] = _js_divergence(ring[-1], hist)
        ring.append(hist)
        if len(ring) > w_max + 1:
            ring.pop(0)
        # lagged JS per scale: lagged_js[w][t - w] = JS(H[t-w], H[t])
        for w in windows:
            if t - w >= 0 and len(ring) >= w + 1:
                feat.lagged_js[w][t - w] = _js_divergence(ring[-(w + 1)], ring[-1])
        # D18: lagged spatial MAD per scale — spatial_lag[w][t-w] = block-max
        # MAD(gray[t-w], gray[t]). Blend endpoints separate from drift here
        # (MD 17-84 vs sunset <= 10); pans ALSO score high at lag, so the
        # per-frame band (spatial_consec) is the pan discriminator, not this.
        gray_ring.append(gray)
        if len(gray_ring) > w_max + 1:
            gray_ring.pop(0)
        for w in windows:
            if t - w >= 0 and len(gray_ring) >= w + 1:
                ad_w = cv2.absdiff(gray_ring[-(w + 1)], gray)
                feat.spatial_lag[w][t - w] = float(
                    cv2.resize(ad_w, (_SPATIAL_GRID, _SPATIAL_GRID),
                               interpolation=cv2.INTER_AREA).max())
        # subsampled hist store
        if t % stride == 0:
            feat.mid_hists.append(hist)

        prev_gray = gray
        t += 1
        if progress_cb and t % 200 == 0:
            _safe_progress(progress_cb, 0.4 + 0.35 * (t / total), "feature pass")
    cap.release()

    feat.n = t
    if t == 0:
        raise SceneCutError(f"No frames decoded from {video_path}", code=4)
    # Trim arrays to actual length
    for arr_name in ("mad", "edge", "mean_s", "mean_v", "consec_js",
                     "spatial_consec"):
        setattr(feat, arr_name, getattr(feat, arr_name)[:t])
    for w in windows:
        feat.lagged_js[w] = feat.lagged_js[w][:t]
        feat.spatial_lag[w] = feat.spatial_lag[w][:t]
    return feat


def _mid_hist_at(feat: _Features, frame: int) -> np.ndarray | None:
    """Nearest subsampled histogram to a frame position."""
    if not feat.mid_hists:
        return None
    k = int(round(frame / feat.mid_stride))
    k = max(0, min(k, len(feat.mid_hists) - 1))
    return feat.mid_hists[k]


def _safe_progress(cb, frac: float, label: str) -> None:
    try:
        cb(frac, label)
    except TypeError:
        try:
            cb(frac)
        except Exception:
            pass
    except Exception:
        pass


# ------------------------------------------------------------------ orchestrated scenedetect pass


def _pass_orchestrated(video_path: str, requested: set[str], threshold: float,
                       min_scene_len_s: float, fps: float, fade_threshold: int,
                       fade_ceiling: int, hash_threshold: float = 0.4,
                       progress_cb=None):
    """Manually drive all scenedetect detectors over ONE VideoStream decode.

    Replicates SceneManager's auto-downscale (compute_downscale_factor on the
    max dimension -> INTER_LINEAR resize) so detector behavior matches the old
    SceneManager runs (RV2 N20). Returns {source: [frame_num, ...]} plus the
    per-detector StatsManagers for confidence metrics (D7).
    """
    from scenedetect import open_video
    from scenedetect.detectors import AdaptiveDetector, ThresholdDetector, HashDetector
    from scenedetect.stats_manager import StatsManager
    from scenedetect.scene_manager import compute_downscale_factor

    video = open_video(video_path)
    fw, fh = video.frame_size
    factor = compute_downscale_factor(max(fw, fh))
    dw, dh = max(1, int(fw / factor)), max(1, int(fh / factor))

    min_len_frames = max(1, int(round(min_scene_len_s * fps)))

    detectors: dict[str, Any] = {}
    if "adaptive" in requested:
        detectors["adaptive"] = AdaptiveDetector(
            adaptive_threshold=threshold, min_scene_len=min_len_frames)
    if "threshold" in requested:
        detectors["threshold"] = ThresholdDetector(
            threshold=fade_threshold, min_scene_len=min_len_frames,
            method=ThresholdDetector.Method.FLOOR)
    if "threshold_ceiling" in requested:
        detectors["threshold_ceiling"] = ThresholdDetector(
            threshold=fade_ceiling, min_scene_len=min_len_frames,
            method=ThresholdDetector.Method.CEILING)
    if "hash" in requested:
        detectors["hash"] = HashDetector(
            min_scene_len=min_len_frames,
            threshold=float(hash_threshold))

    stats: dict[str, StatsManager] = {}
    metric_keys: dict[str, list[str]] = {}
    primary_metric: dict[str, str] = {}
    for name, det in detectors.items():
        sm = StatsManager()
        det.stats_manager = sm
        sm.register_metrics(det.get_metrics())
        stats[name] = sm
        metric_keys[name] = list(det.get_metrics())
        # Primary metric per source (CR1 H1: metric_keys lists ALL keys e.g.
        # content_val/delta_hue/.../adaptive_ratio — vals[0] would be content_val,
        # not the adaptive ratio the confidence formula needs)
        if name == "adaptive":
            primary_metric[name] = getattr(
                det, "_adaptive_ratio_key",
                AdaptiveDetector.ADAPTIVE_RATIO_KEY_TEMPLATE.format(
                    window_width=getattr(det, "window_width", 2), luma_only=""))
        elif name == "hash":
            primary_metric[name] = getattr(det, "_metric_key", metric_keys[name][0])
        elif name in ("threshold", "threshold_ceiling"):
            primary_metric[name] = ThresholdDetector.THRESHOLD_VALUE_KEY
        else:
            primary_metric[name] = metric_keys[name][0]

    total = int(video.duration.frame_num) if video.duration else 0
    cuts_by_source: dict[str, list[int]] = {name: [] for name in detectors}

    while True:
        frame = video.read()
        if frame is False or frame is None:
            break
        pos = video.position
        if factor > 1.0:
            small = cv2.resize(frame, (dw, dh), interpolation=cv2.INTER_LINEAR)
        else:
            small = frame
        for name, det in detectors.items():
            for cut in det.process_frame(pos, small):
                cuts_by_source[name].append(cut.frame_num)
        if progress_cb and total and pos.frame_num % 200 == 0:
            _safe_progress(progress_cb, 0.1 + 0.3 * (pos.frame_num / total), "detector pass")

    last_pos = video.position
    for name, det in detectors.items():
        for cut in det.post_process(last_pos):
            cuts_by_source[name].append(cut.frame_num)

    return cuts_by_source, stats, metric_keys, primary_metric


def _metric_value(stats, primary_metric: dict[str, str], source: str, frame_num: int) -> float | None:
    """Read the PRIMARY metric for a source at a cut frame (defensive).

    CR1 C1: must pass the single primary metric KEY, never the whole
    metric-keys dict — get_metrics would iterate dict keys (source names)
    and return None for everything, silently disabling calibration.
    """
    key = primary_metric.get(source)
    if not key:
        return None
    try:
        vals = stats[source].get_metrics(frame_num, [key])
        v = vals[0] if vals else None
        return float(v) if v is not None else None
    except Exception:
        return None


# ------------------------------------------------------------------ confidences (D7 v2.1)


def _sigmoid(x: float) -> float:
    return 1.0 / (1.0 + math.exp(-x))


def _conf_adaptive(ratio: float | None, threshold: float) -> float:
    if ratio is None or ratio <= 0 or threshold <= 0:
        return 0.7  # metric unavailable: neutral-default, agreement can still boost
    return float(np.clip(_sigmoid(2.0 * (ratio / threshold - 1.0)), 0.05, 0.99))


def _conf_hash(dist_norm: float | None, threshold: float = 0.4) -> float:
    # HashDetector fires when dist_norm >= threshold, so confidence must grow
    # WITH distance above the firing boundary (RV2 N21 — the old inverted
    # formula scored every firing cut <= 0).
    if dist_norm is None:
        return 0.7
    return float(np.clip((dist_norm - threshold) / max(1e-6, 1.0 - threshold), 0.05, 0.99))


def _conf_floor(trough: float | None, fade_threshold: int) -> float:
    if trough is None:
        return 0.7
    return float(np.clip((fade_threshold - trough) / max(1e-6, fade_threshold), 0.05, 0.99))


def _conf_ceiling(peak: float | None, fade_ceiling: int) -> float:
    if peak is None:
        return 0.7
    return float(np.clip((peak - fade_ceiling) / max(1e-6, 255 - fade_ceiling), 0.05, 0.99))


def _luma_window(feat: _Features, frame: int, radius: int) -> np.ndarray:
    lo = max(0, frame - radius)
    hi = min(feat.n, frame + radius + 1)
    return feat.mean_v[lo:hi]


# ------------------------------------------------------------------ dissolve detection (D3 v2.1)


def _refine_plateau(feat: _Features, mid: int, w_max: int, eps: float,
                    min_run: int = 3, gap_tol: int = 2,
                    arr: np.ndarray | None = None) -> dict | None:
    """Plateau refinement (D16 §3 v4.3 + CR3 F2/F3): expand the consec-JS
    plateau around a dissolve candidate's window mid, with hysteresis.

    D18: ``arr`` selects the walked array — the spatial tier passes
    ``feat.spatial_consec`` with ``eps = spatial_consec_lo`` (match-dissolve
    plateaus live in the spatial signal; the global JS plateau is flat there).

    Returns {"start", "end", "emission", "interval"} where the plateau is
    the elevated run [start, end] INCLUSIVE (consec_js[t] ≥ eps), BRIDGING
    sub-threshold dips ≤ gap_tol frames (2-frame dips are bridged; the run
    ends after gap_tol+1 = 3 consecutive clean frames); emission = blend
    CENTER = round((start + end + 1) / 2) — the local-corpus GT convention
    (measured: itest_dissolve plateau [69..89], GT 80); interval =
    [start, end + 1) (first clean frame is end + 1). Returns None when the
    mid isn't elevated or the run is shorter than min_run (caller falls
    back to window semantics).

    CR3 F3: the search bound is ±2·w_max (128 frames at default windows) —
    blends up to ~2·w_max refine whole instead of fragmenting into
    bound-truncated pieces (a 100-frame blend previously split into two
    adjacent non-overlapping intervals that escaped overlap suppression).
    F2 pin: never read consec_js[n−1] (stale 1.0 init) — hi ≤ n−2.
    """
    n = feat.n
    bound = 2 * w_max
    hi_cap = min(mid + bound, n - 2)   # F2: n-1 holds the init sentinel
    lo_cap = max(0, mid - bound)
    if mid < 0 or mid > hi_cap:
        return None
    js = arr if arr is not None else feat.consec_js
    if js[mid] < eps:
        return None
    # expand right: bridge sub-threshold dips <= gap_tol; stop after
    # gap_tol+1 consecutive clean frames
    end = mid
    t = mid + 1
    gap = 0
    while t <= hi_cap:
        if js[t] >= eps:
            end = t
            gap = 0
        else:
            gap += 1
            if gap > gap_tol:
                break
        t += 1
    # expand left (same hysteresis; independent gap counter)
    start = mid
    t = mid - 1
    gap = 0
    while t >= lo_cap:
        if js[t] >= eps:
            start = t
            gap = 0
        else:
            gap += 1
            if gap > gap_tol:
                break
        t -= 1
    if end - start + 1 < min_run:
        return None
    emission = int(round((start + end + 1) / 2))
    return {"start": int(start), "end": int(end),
            "emission": emission, "interval": [int(start), int(end) + 1]}


def _detect_dissolves(feat: _Features, windows: list[int], dissim_min: float,
                      hard_cut_mad_min: float, consec_js_min: float,
                      min_scene_len_s: float, fps: float,
                      consec_js_frame_min: float = 0.03,
                      consec_js_sustain_frac: float = 0.35,
                      consec_js_ratio_min: float = 1.8,
                      progress_cb=None) -> list[dict]:
    """JS-divergence dissolve detector over the feature arrays (D3, calibrated
    on the fixture corpus — see samples/ground_truth.json; RECALIBRATED on the
    external SBD snippet corpus by D19).

    Measured discriminators — two calibration generations:
      LOCAL fixtures (D3 era): endpoint JS 0.5-0.95 (distinct-shot dissolves);
      consec-JS plateau ~0.73; MAD blend 3-5 / pan 12.5 / whip 56.
      EXTERNAL SBD 61-frame snippets, n=520 (D19): real/synthetic blends show
      endpoint JS p5 0.177 (w=16), window-mean consec-JS p50 0.046 (NOT 0.73
      — natural-content histograms walk far more weakly than fixtures),
      window MAD p75 20 / p90 ~30 (visible blends exceed the old 20 ceiling),
      hard cuts p25 53.8. Negatives (E): consec-JS p95 0.074, endpoint JS
      p75 0.108; the endJS notch sits at 0.15.

    D19 gate structure (each conjunct's discriminator):
      1. endpoint JS >= dissim_min (0.15) — blend vs same-shot/static;
      2. window MAD max < hard_cut_mad_min (40) — cut/whip-blur guard
         (cuts 54+, blends p90 ~30 — the 20 fixture-era ceiling rejected
         half the external blends);
      3. SUSTAINED elevation: mean(consec_js) >= consec_js_min (0.03) AND
         frac(consec_js >= consec_js_frame_min=0.03) >= 0.35 — the plateau is
         many-frame moderate elevation, NOT one big spike (a hard cut's
         single-frame JS jump dilutes to ~0.06 mean but fails the frac);
         the 0.03 mean floor also rejects sunset-drift impostors (measured
         0.026-0.028 vs real-blend p25 0.030);
      4. LOCALIZATION ratio: mean(consec_js[window]) >= consec_js_ratio_min
         (1.8) x median(consec_js[whole clip], floored at 0.004) — blends are
         LOCALIZED elevation (quiet sides), sustained motion (pans) is GLOBAL
         (center ~= base -> rejected). This is the gate that lets 3's floor
         drop from 0.15 to 0.025 without opening motion FPs on natural video.
      5. CONTEXT-QUIET bands (fixture regression net): bands at mid±[w, 2w)
         must stay under 0.67 x consec_js_frame_min when at least w/2 context
         frames exist. Real transitions are LOCALIZED excursions — the shot
         before/after may move, but drift-class fixtures (sunset: continuous
         histogram walk; itest_hard_cuts' slow-motion segment: cjs 0.09,
         endJS 0.86, MAD 2.4 — a perfect blend impostor) keep the context
         bands elevated too, and ARE rejected here. Bands at ±[w, 2w) (not
         ±[2w, 3w)): the hard_cuts motion segment is 110 frames wide — its
         w=32 windows' ±[2w,3w) bands escape past the segment edges and read
         quiet, while ±[w,2w) bands stay inside and trip; conversely a real
         64-frame blend under a w=32 window has its ±[32,64) bands start
         exactly at the blend edge (outside) and passes. Multi-scale
         redundancy covers narrower-window band-overlap cases.

    NOTE (design deviation from DECISIONS.md v2.1, measured): the edge-energy
    DIP conjunct was dropped — busy-content dissolves show no dip (superposed
    edges ADD: measured dips 0.0-0.2, sometimes negative). Tier-2 (match
    dissolves between same-palette shots): D18 block-wise spatial detector.

    NMS across scales — REFINED-BEFORE-NMS (D16 §3 v4.3, RV4b F1): every
    candidate is plateau-refined FIRST (emission = blend center, interval
    attached, fallback = raw window semantics when refinement fails), and
    NMS distances use the REFINED emissions. Refinement happens before NMS
    so fallback zombies cannot survive past a nearby refined plateau (the
    RV4-B1 two-cut failure mode), and multi-scale candidates converge on
    the same plateau → identical emissions → collapse.
    Emission at plateau CENTER (measured convention: blend [70,90) has GT
    at 80 = center; also minimizes worst-case distance to unknown
    annotator conventions: ±w/2 vs +w for end-emission).
    """
    n = feat.n
    candidates: list[dict] = []
    w_max = max(windows) if windows else 8
    min_len_frames = max(1, int(round(min_scene_len_s * fps)))

    # D19 gate 4: clip-level consec-JS baseline (localization ratio).
    # Floored so near-static clips (median ~0.002) don't make the ratio test
    # the binding constraint — the absolute floor (gate 3) dominates there.
    cjs_all = feat.consec_js[:max(0, n - 1)]  # F2: drop the stale init sentinel
    cjs_baseline = max(float(np.median(cjs_all)) if cjs_all.size else 0.0, 0.004)

    for w in sorted(windows):
        if w < 8 or w >= n:
            continue
        lag = feat.lagged_js[w]
        for i in range(0, n - w):
            end_js = float(lag[i])
            if end_js < dissim_min:
                continue
            # No hard cut / whip-pan blur inside the window (pixel-level MAD)
            wmad = feat.mad[i:i + w]
            if len(wmad) == 0 or float(wmad.max()) >= hard_cut_mad_min:
                continue
            # Blend signature: SUSTAINED histogram walking (D19 gate 3) —
            # absolute floor + frame-level elevation fraction (spike-immune)
            cjs = feat.consec_js[i:i + w]
            if len(cjs) == 0 or float(np.mean(cjs)) < consec_js_min:
                continue
            if float(np.mean(cjs >= consec_js_frame_min)) < consec_js_sustain_frac:
                continue
            # D19 gate 4: localized elevation — reject global motion (pans)
            cjs_mean = float(np.mean(cjs))
            if cjs_mean < consec_js_ratio_min * cjs_baseline:
                continue
            # D19 gate 5: context-quiet bands at mid±[w, 2w) — transitions
            # are localized excursions; drift keeps the context elevated.
            mid = i + w // 2
            ctx: list[float] = []
            for lo, hi in ((mid - 2 * w, mid - w), (mid + w, mid + 2 * w)):
                if hi <= 0 or lo >= n:
                    continue
                ctx.extend(feat.consec_js[max(0, lo):min(n, hi)])
            if (len(ctx) >= max(4, w // 2)
                    and float(np.mean(ctx)) >= 0.67 * consec_js_frame_min):
                continue

            cut_frame = i + w // 2
            candidates.append({
                "frame_num": cut_frame,
                "confidence": float(min(end_js, 0.95)),
                # provenance for refinement (RV4b F1)
                "window_i": i,
                "window_w": w,
            })

    if progress_cb:
        _safe_progress(progress_cb, 0.78, "dissolve NMS")

    # ---- refine EVERY candidate before NMS (F1)
    # D19: the plateau walk eps drops to the FRAME-level floor (0.03) — the
    # old 0.15 walked only fixture-scale plateaus and truncated external ones.
    for c in candidates:
        ref = _refine_plateau(feat, c["frame_num"], w_max, consec_js_frame_min)
        if ref is not None:
            c["frame_num"] = ref["emission"]
            c["interval"] = ref["interval"]
            c["interval_fallback"] = False
        else:
            # fallback: raw window semantics, mid emission (pre-v4 behavior)
            c["interval"] = [c["window_i"], c["window_i"] + c["window_w"]]
            c["interval_fallback"] = True

    # D19 gate 6: minimum blend extent — real transitions blend for >= 10
    # frames (SBD GT is center±10; ClipShots gradual spans run ~15-40). Shorter
    # refined plateaus are drift BURSTS: the sunset impostors pass every
    # per-window gate marginally (cjs 0.032-0.038, endJS 0.151-0.166) but
    # refine to exactly their 8-frame windows. Applies to fallback candidates
    # too (a fallback's interval == its window length).
    candidates = [c for c in candidates
                  if (c.get("interval")[1] - c.get("interval")[0]) >= 10]

    # NMS across scales on REFINED emissions (F1: interval-overlap aware —
    # a fallback candidate whose raw window overlaps a kept candidate's
    # plateau is part of the SAME gradual → suppressed; two separate
    # graduals have disjoint plateaus and survive unless frame-close)
    radius = max(1, min(w_max // 2, min_len_frames - 1))
    candidates.sort(key=lambda c: (-c["confidence"], c["frame_num"]))
    kept: list[dict] = []
    for c in candidates:
        clash = False
        for k in kept:
            if abs(c["frame_num"] - k["frame_num"]) <= radius:
                clash = True
                break
            cs, ce = c["interval"]
            ks, ke = k["interval"]
            if cs < ke and ks < ce:   # half-open overlap
                clash = True
                break
        if not clash:
            kept.append(c)

    cuts = []
    for c in kept:
        sec = frame_to_seconds(c["frame_num"], fps)
        cut = {
            "frame_num": c["frame_num"],
            "seconds": sec,
            "timecode": seconds_to_hmsms(sec),
            "type": "dissolve",
            "confidence": round(c["confidence"], 3),
            "source": "dissolve",
            "interval": [int(c["interval"][0]), int(c["interval"][1])],
            "interval_fallback": bool(c["interval_fallback"]),
        }
        cuts.append(cut)
    return cuts


# ------------------------------------------------------------------ spatial dissolves (D18, Phase 1.5)

def _detect_spatial_dissolves(feat: _Features, windows: list[int],
                              hard_cut_mad_min: float, consec_js_min: float,
                              min_scene_len_s: float, fps: float,
                              spatial_dissim_min: float,
                              spatial_consec_lo: float,
                              spatial_motion_max: float,
                              progress_cb=None) -> list[dict]:
    """Tier-2 spatial dissolve detector (D18 / Phase 1.5) — the match-dissolve
    family: gradual transitions between DIFFERENT shots with near-identical
    color palettes. The global H-S histogram is blind there (measured on
    itest_match_dissolve: endpoint JS 0.042, consec 0.008 vs tier-1 floor
    0.12/0.15) — but the spatial ARRANGEMENT cross-fades (8x8 block-max MAD
    at endpoints 17-84 vs sunset drift <= 10, static 0.05).

    Gates (R8b measured margins, .agents/research/r8b-spatial-design.md):
      G0 tier-2 regime: mean(consec_js[i:i+w]) < consec_js_min — the global
         detector's regime is EXCLUDED (tier-1 dissolve 0.64, fade-white 0.80
         fail G0; MD 0.008 passes). Tier-1/tier-2 candidate sets are disjoint
         by construction, so dedup never arbitrates between them.
      G1 spatial endpoint: spatial_lag[w][i] >= spatial_dissim_min (14.0;
         MD spanning med 40 / sunset <= 10 — 1.4x margin at the floor).
      G2 cut/blur guard (reused): max(mad[i:i+w]) < hard_cut_mad_min —
         hard cuts (MAD 55-76) and whip-pan blur (56) stay out.
      G3 blend band + sustained run: median(spatial_consec[i:i+w]) in
         [spatial_consec_lo, spatial_motion_max] (pan 36 is 4.5x ABOVE the
         band — coherent slides displace whole blocks every frame, blends
         change each block by <= full_diff/span; fades 18.8-19.9 also above)
         AND longest run of spatial_consec >= lo within the window >= 6
         (MD run 20; sunset/reverse-dialog runs <= 1).

    Emission + refinement mirror the tier-1 D16 machinery (refine-before-NMS,
    plateau center, interval attached, fallback = raw window) — but the
    plateau is walked on spatial_consec (the signal that is actually elevated).
    Confidence: clip(0.5 + 0.45 * (lag - dissim_min) / (84 - dissim_min)).
    """
    n = feat.n
    candidates: list[dict] = []
    w_max = max(windows) if windows else 8
    min_len_frames = max(1, int(round(min_scene_len_s * fps)))

    for w in sorted(windows):
        if w < 8 or w >= n:
            continue
        lag = feat.spatial_lag[w]
        for i in range(0, n - w):
            # G0: only the tier-2 regime (global histogram quiet)
            cjs = feat.consec_js[i:i + w]
            if len(cjs) == 0 or float(np.mean(cjs)) >= consec_js_min:
                continue
            # G1: spatial endpoint separation
            slag = float(lag[i])
            if slag < spatial_dissim_min:
                continue
            # G2: no hard cut / whip-pan blur inside the window. CR4-M1:
            # the slice must include the TRAILING edge transition — mad[t] is
            # MAD(gray[t-1], gray[t]), so a hard cut between i+w-1 and i+w
            # lives at mad[i+w], INSIDE the G1 lag span but OUTSIDE the old
            # [i, i+w) slice. A motion segment bounded by a hard cut then
            # passed every gate and emitted a mid-motion "dissolve" ~w/2
            # early (CR4 probe case C: fired @86 for a cut @102).
            wmad = feat.mad[i:min(i + w + 1, n)]
            if len(wmad) == 0 or float(wmad.max()) >= hard_cut_mad_min:
                continue
            # G3: per-frame blend band (pan/fade reject) + sustained run
            scon = feat.spatial_consec[i:i + w]
            if len(scon) == 0:
                continue
            med = float(np.median(scon))
            if med < spatial_consec_lo or med > spatial_motion_max:
                continue
            # longest run of scon >= lo (>= 6 frames)
            run = best = 0
            for v in scon:
                run = run + 1 if float(v) >= spatial_consec_lo else 0
                best = max(best, run)
            if best < 6:
                continue
            # G4: near-static CONTEXT on both sides — the defining signature
            # of a match-dissolve between two composed/stable shots. Slow
            # coherent motion (itest_hard_cuts segments: sustained
            # spatial_consec ~5-8 with global JS ~0.04) sits INSIDE the blend
            # band; only the surrounding quiet separates it from a true blend.
            # Measured: MD within-shot spatial_consec ~0.04 both sides.
            pre = feat.spatial_consec[max(0, i - w):i]
            post = feat.spatial_consec[i + w:i + 2 * w]
            if len(pre) == 0 or len(post) == 0:
                continue
            if float(np.median(pre)) >= spatial_consec_lo \
                    or float(np.median(post)) >= spatial_consec_lo:
                continue
            # G5: luma guards (R8b defense-in-depth, validated live: the BBC
            # ep01 title-card fires — black->logo at f88, logo fade-out at
            # f200 — rearrange spatially with quiet context both sides).
            # (a) DARK-REGION floor: below mean_v ~8, block-MAD of 2-3 is
            #     proportionally quantization noise (measured fires at
            #     mean_v 1.7-3.5; MD blends ~120) — spatial rearrangement is
            #     not reliable evidence there.
            # (b) luma MOVEMENT across the window (range >= 8): a fade ramp
            #     or auto-exposure wobble — same-palette blends are luma-flat
            #     by construction (measured MD range ~1). Window-RANGE (not
            #     endpoint-diff) catches gradual ramps that hide from
            #     small-window endpoint checks.
            # (c) RELATIVE endpoint drift (dv >= 0.5 x local level): dark
            #     fades invisible to absolute floors.
            mv = feat.mean_v[i:i + w]
            if float(mv.max() - mv.min()) >= 8.0:
                continue
            if float(np.median(mv)) < 8.0:
                continue
            v_i = float(feat.mean_v[i])
            v_e = float(feat.mean_v[min(i + w, n - 1)])
            if abs(v_e - v_i) >= 0.5 * max(min(v_i, v_e), 1.0):
                continue

            candidates.append({
                "frame_num": i + w // 2,
                "confidence": float(min(0.95, max(0.05,
                    0.5 + 0.45 * (slag - spatial_dissim_min) / (84.0 - spatial_dissim_min)))),
                "spatial_lag": slag,
                "window_i": i,
                "window_w": w,
            })

    if progress_cb:
        _safe_progress(progress_cb, 0.78, "spatial dissolve NMS")

    # ---- refine EVERY candidate before NMS (D16 F1 pattern, spatial array)
    for c in candidates:
        ref = _refine_plateau(feat, c["frame_num"], w_max, spatial_consec_lo,
                              arr=feat.spatial_consec)
        if ref is not None:
            c["frame_num"] = ref["emission"]
            c["interval"] = ref["interval"]
            c["interval_fallback"] = False
        else:
            c["interval"] = [c["window_i"], c["window_i"] + c["window_w"]]
            c["interval_fallback"] = True

    # NMS on refined emissions (same interval-overlap-aware rule as tier-1)
    radius = max(1, min(w_max // 2, min_len_frames - 1))
    candidates.sort(key=lambda c: (-c["confidence"], c["frame_num"]))
    kept: list[dict] = []
    for c in candidates:
        clash = False
        for k in kept:
            if abs(c["frame_num"] - k["frame_num"]) <= radius:
                clash = True
                break
            cs, ce = c["interval"]
            ks, ke = k["interval"]
            if cs < ke and ks < ce:
                clash = True
                break
        if not clash:
            kept.append(c)

    cuts = []
    for c in kept:
        sec = frame_to_seconds(c["frame_num"], fps)
        cut = {
            "frame_num": c["frame_num"],
            "seconds": sec,
            "timecode": seconds_to_hmsms(sec),
            "type": "dissolve",
            "confidence": round(c["confidence"], 3),
            "source": "dissolve",
            "tier": 2,
            "spatial_lag": round(float(c.get("spatial_lag", 0.0)), 2),
            "interval": [int(c["interval"][0]), int(c["interval"][1])],
            "interval_fallback": bool(c["interval_fallback"]),
        }
        cuts.append(cut)
    return cuts


# ------------------------------------------------------------------ fade-to-white (D5, feature-scan implementation)


def _detect_fade_white(feat: _Features, fps: float, fade_ceiling: int,
                       min_scene_len_s: float, max_run_s: float = 1.5) -> list[tuple[int, int, int]]:
    """Detect fade-to-WHITE transitions by scanning the mean-gray array.

    Design deviation from DECISIONS.md v2.1 (measured): scenedetect's
    ThresholdDetector(Method.CEILING) emits its cut when LEAVING the white
    region (midpoint of the run) and NEVER fires when the video ENDS in white
    (fade-white fixtures end in white) — so we scan the luma array directly.

    A fade-to-white transits THROUGH white:
      - bounded above-ceiling run (<= max_run_s) — a sustained white SHOT is
        overexposure, not a transit
      - gray below the ceiling on both sides (unless end-of-video)
      - edge collapse on the plateau (white frames have no edges; an
        overexposed shot keeps horizon/cloud edges)
    Emission: the RAMP ONSET (walk back from the run start while gray keeps
    rising) — the editorially-correct scene boundary where the outgoing shot
    ends. Returns (onset, run_start, run_end) tuples.
    """
    n = feat.n
    gray = feat.mean_v
    max_run = int(max_run_s * fps)
    typical_edge = float(np.mean(feat.edge)) if n else 0.0
    regions: list[tuple[int, int, int]] = []

    i = 0
    while i < n:
        if gray[i] >= fade_ceiling:
            a = i
            while i < n and gray[i] >= fade_ceiling:
                i += 1
            b = i  # run [a, b)
            run = b - a
            if run <= 0 or run > max_run:
                continue  # too long = white shot; too short handled below
            # gray below ceiling before the run (else we're inside a longer
            # bright episode)
            if a > 0 and gray[max(0, a - 3)] >= fade_ceiling:
                continue
            # gray below ceiling after the run, unless the video ends here
            if b < n and gray[min(n - 1, b + 2)] >= fade_ceiling:
                continue
            # edge collapse on the plateau
            plateau_edge = float(np.mean(feat.edge[a:b]))
            if typical_edge > 0 and plateau_edge > 0.25 * typical_edge:
                continue
            # ramp onset: walk back while gray is rising toward the run
            k = a
            while k > 0 and gray[k - 1] < gray[k] - 0.5:
                k -= 1
            regions.append((k, a, b))
        else:
            i += 1
    return regions


# ------------------------------------------------------------------ dedup (v2.1)

SOURCE_PRIORITY = {
    # A23: integers only (float 1.5 broke priority-margin consumers).
    # Neural outranks heuristics for the source label on agreement (D13) but
    # TYPE_PRIORITY preservation keeps typed fades/dissolves informative.
    "implicit": 0,
    "neural": 1,
    "adaptive": 2,
    "hash": 3,
    "threshold": 4,
    "threshold_ceiling": 4,
    "dissolve": 5,
}
TYPE_PRIORITY = {"cut": 0, "fade": 1, "dissolve": 2}  # higher = more informative

_SOURCE_FAMILY = {
    "adaptive": "pixel",
    "hash": "pixel",
    "threshold": "luminance",
    "threshold_ceiling": "luminance",
    "dissolve": "histogram",
}


def _capped_noisy_or(c1: float, c2: float) -> float:
    """Capped noisy-OR (D7 v2.1): monotone, bounded, capped near the max so
    correlated pixel-family agreement (adaptive+hash failing together on whip
    pans) cannot manufacture near-certainty (RV1 N13 / RV2 N13). The dedup
    applies the same shape over ALL per-source confidences at once (see
    _dedup_cuts)."""
    noisy = 1.0 - (1.0 - c1) * (1.0 - c2)
    return float(min(noisy, max(c1, c2) + 0.15))


def _fade_run_interval(feat: _Features, f: int, fade_threshold: int,
                       max_run: int) -> list[int] | None:
    """D14: luma-run interval for a threshold-floor fade emission.

    Walks back through the falling ramp (luma rising backwards) and forward
    through the below-threshold run. Bounded by max_run frames; returns None
    when degenerate (no walk possible / emission at video head).
    """
    gray = feat.mean_v
    n = feat.n
    if not (0 < f < n - 1):
        return None
    k = f
    while k > 0 and gray[k - 1] > gray[k] + 0.5 and (f - k) < max_run:
        k -= 1
    start = k
    t = f
    while t < n and gray[t] <= fade_threshold and (t - start) < max_run:
        t += 1
    # CR3: t < n (not n-1) — an end-of-video fade keeps its last
    # below-threshold frame in the interval
    if t - start < 2:
        return None
    return [int(start), int(t)]


def _corroboration_gate(cuts: list[dict], feat: _Features,
                        spike_min: float, js_min: float, luma_min: float,
                        ) -> tuple[list[dict], int, int]:
    """D16 §1 v4.3: solo-hash corroboration gate + gradual-region absorption.

    Runs post-dedup (merged `sources` available), BEFORE the neural arbiter.

    1. ABSORPTION (first — RV4b F10): a solo-hash fire whose frame lies inside
       a REFINED (non-fallback) dissolve interval [s, e) is dropped — it is a
       mid-transition artifact of the gradual the dissolve cut already
       represents (same rationale as TransNetV2 transition-zone suppression).
       Fallback/raw-window intervals never absorb (the S4 overshoot).
    2. GATE: a solo-hash cut (hash fired with no other detector within the
       dedup radius — `len(sources) == 1 and sources[0] == 'hash'`) must show
       feature-evidence corroboration, else it is DROPPED (not downgraded:
       the measured FP population is 0.05-conf motion re-fire noise):
         spike_ratio = mad[p] / max(median(mad[p-30 .. p-5]), 0.5)  >= spike_min
         OR consec_js[p-1] >= js_min   (0.15 = consec_js_min — pinned)
         OR |mean_v[p] - mean_v[p-1]| >= luma_min
       Evidence (ep01, spec formula): solo-hash 377 TP / 73 FP; rule drops
       0 TP / 56 FP (56:1). p < 6 → gate skipped (video head: no baseline).

    Returns (kept_cuts, n_absorbed, n_gated).
    """
    refined_intervals: list[list[int]] = [
        c["interval"] for c in cuts
        if c.get("interval") and not c.get("interval_fallback")
        and "dissolve" in (c.get("sources") or [c.get("source")])
        # CR3 F1: membership by SOURCES, not the label — a hash-priority
        # dedup win inherits the dissolve interval; filtering on
        # source == "dissolve" made absorption blind to those intervals
        # (reproduced: solo-hash inside an inherited plateau survived).
    ]
    kept: list[dict] = []
    n_absorbed = 0
    n_gated = 0
    for cut in cuts:
        srcs = cut.get("sources") or [cut.get("source")]
        solo_hash = len(srcs) == 1 and srcs[0] == "hash"
        if not solo_hash:
            kept.append(cut)
            continue
        p = int(cut["frame_num"])
        # --- absorption into refined gradual intervals
        if any(s <= p < e for (s, e) in refined_intervals):
            n_absorbed += 1
            continue
        # --- corroboration rule
        if p < 6 or p >= feat.n:
            kept.append(cut)   # video head / out of range: skip the gate
            continue
        lo = max(0, p - 30)
        base = feat.mad[lo:max(lo + 1, p - 4)]   # frames p-30 .. p-5
        med = float(np.median(base)) if len(base) else 0.0
        spike = float(feat.mad[p]) / max(med, 0.5)
        js = float(feat.consec_js[p - 1])
        luma = abs(float(feat.mean_v[p]) - float(feat.mean_v[p - 1]))
        if spike >= spike_min or js >= js_min or luma >= luma_min:
            kept.append(cut)
        else:
            n_gated += 1
    return kept, n_absorbed, n_gated


def _dedup_cuts(cuts: list[dict], min_scene_len: float) -> list[dict]:
    """Sort + dedup by min_scene_len with source priority, capped noisy-OR
    confidence merge, conf_by_source bookkeeping, and type re-classification."""
    cuts.sort(key=lambda c: c["seconds"])
    if not cuts:
        return cuts
    deduped = [dict(cuts[0])]
    for c in cuts[1:]:
        gap = c["seconds"] - deduped[-1]["seconds"]
        if gap >= min_scene_len:
            deduped.append(dict(c))
            continue
        cur = deduped[-1]
        cur_pri = SOURCE_PRIORITY.get(cur.get("source", ""), 9)
        new_pri = SOURCE_PRIORITY.get(c.get("source", ""), 9)
        if new_pri < cur_pri or (new_pri == cur_pri and c["confidence"] > cur["confidence"]):
            winner, loser = c, cur
        else:
            winner, loser = cur, c
        winner = dict(winner)
        # Preserve the more informative type from either
        if TYPE_PRIORITY.get(loser.get("type", ""), 0) > TYPE_PRIORITY.get(winner.get("type", ""), 0):
            winner["type"] = loser["type"]
        # D14: interval survives the merge — winner keeps its own interval if
        # present, else inherits the loser's (a hash-priority win must not
        # drop the dissolve/fade interval it collided with — RV4b SHOULD-7)
        if not winner.get("interval") and loser.get("interval"):
            winner["interval"] = list(loser["interval"])
            winner["interval_fallback"] = loser.get("interval_fallback", False)
        # CR4-M2: D18 provenance inherits like intervals — a tier-2 dissolve
        # loses essentially every priority race (dissolve = priority 5), so
        # without this its tier/spatial_lag vanish exactly on merged cuts
        # (the CR3-F1 class; the UI TransitionBadge consumes tier).
        for k in ("tier", "spatial_lag"):
            if loser.get(k) is not None and winner.get(k) is None:
                winner[k] = loser[k]
        # Track all agreeing sources
        srcs = list(winner.get("sources", [winner.get("source")]))
        if loser.get("source") and loser["source"] not in srcs:
            srcs = srcs + [loser["source"]]
        winner["sources"] = srcs
        # Re-classify: luminance detectors present => fade; else dissolve; else cut
        if "threshold" in srcs or "threshold_ceiling" in srcs:
            winner["type"] = "fade"
        elif "dissolve" in srcs:
            winner["type"] = "dissolve"
        else:
            winner["type"] = winner.get("type", "cut")
        # Per-source confidences: merge BOTH cuts' existing maps, then add each
        # cut's own source at its OWN (pre-merge) confidence (CR1 M1: the old
        # code read cur's post-merge confidence and let chains inflate)
        cbs = dict(cur.get("conf_by_source") or {})
        cbs.update(dict(loser.get("conf_by_source") or {}))
        for src_cut, own_src in ((cur, cur.get("source")), (loser, loser.get("source"))):
            if own_src and own_src not in cbs:
                cbs[own_src] = src_cut.get("confidence")
        winner["conf_by_source"] = cbs
        # Collision confidence over ALL distinct sources (noisy-OR once, capped
        # near the max — chain-collision safe: recomputed from per-source values)
        per_src = [float(v) for v in cbs.values() if v is not None]
        if per_src:
            noisy = 1.0
            for pc in per_src:
                noisy *= (1.0 - pc)
            winner["confidence"] = round(min(1.0 - noisy, max(per_src) + 0.15), 3)
        deduped[-1] = winner
    return deduped


# ------------------------------------------------------------------ motion gate (D4 v2.1)


def _motion_gate(cuts: list[dict], feat: _Features, motion_gate_js: float) -> int:
    """Downgrade-not-delete motion gate on adaptive-source cuts.

    Histogram stability across the cut (low consecutive JS) + single-family
    corroboration => likely camera motion, not a content change. Gated cuts
    STAY in the list with motion_gated=true and confidence <= 0.3. A >=2
    distinct-family veto (pixel/luminance/histogram) keeps same-palette true
    cuts safe when luminance or histogram detectors corroborate (RV2 N22).
    """
    downgraded = 0
    for cut in cuts:
        if cut.get("source") != "adaptive":
            continue
        families = {_SOURCE_FAMILY.get(s) for s in cut.get("sources", [])}
        families.discard(None)
        if len(families) >= 2:
            continue  # family veto — multi-family corroboration wins
        f = cut["frame_num"]
        if 0 < f <= feat.n:
            js_at_cut = float(feat.consec_js[f - 1])
            if js_at_cut < motion_gate_js:
                cut["motion_gated"] = True
                cut["confidence"] = min(cut.get("confidence", 1.0), 0.3)
                downgraded += 1
    return downgraded


# ------------------------------------------------------------------ merge short scenes (D6 v2.1)


def _merge_short_scenes(cuts: list[dict], feat: _Features, fps: float,
                        total_duration: float, min_duration: float,
                        progress_cb=None) -> list[dict]:
    """Greedy boundary removal over precomputed shot-middle histograms.

    Zero random seeks (was <=150 per merge), no 50-iteration cap, first and
    last scenes are full candidates, and the both-histograms-missing case
    resolves by neighbor duration (was arbitrary-left).
    """
    if len(cuts) < 2 or min_duration <= 0:
        return cuts

    # Shot k spans [cuts[k].seconds, cuts[k+1].seconds); the LAST shot ends at
    # total_duration. cuts[0] is the IMPLICIT cut at 0.0 (starts shot 0) — it is
    # NOT a removable boundary. The boundary between shot k and shot k+1 is
    # cut index k+1 (CR1 H2: the previous off-by-one mapped boundary (k,k+1) to
    # cut k, letting merges delete the implicit cut and drop head-of-video
    # coverage — scenes[0].start became 1.5 instead of 0.0).
    starts = [c["seconds"] for c in cuts]
    ends = starts[1:] + [total_duration]
    n_shots = len(starts)

    def shot_mid_frame(k: int) -> int:
        return int(round(((starts[k] + ends[k]) / 2.0) * fps))

    mids = [_mid_hist_at(feat, shot_mid_frame(k)) for k in range(n_shots)]
    # JS dissimilarity of boundary k (between shot k and shot k+1), None if a
    # histogram is unavailable.
    sims: list[float | None] = [
        _js_divergence(mids[k], mids[k + 1]) if (mids[k] is not None and mids[k + 1] is not None) else None
        for k in range(n_shots - 1)
    ]

    alive = [True] * n_shots
    removed_boundary: set[int] = set()

    # Loop scanning live shot groups (cut counts are small; the old code's
    # real cost was per-iteration random SEEKS, which are gone).
    iterations = 0
    max_iterations = len(cuts) + 10
    while iterations < max_iterations:
        iterations += 1
        # live shot groups: shots j and j+1 are merged iff their boundary cut
        # (index j+1) was removed
        groups: list[tuple[int, int]] = []  # (start_shot, end_shot) inclusive
        k = 0
        while k < n_shots:
            j = k
            while j + 1 < n_shots and (j + 1) in removed_boundary:
                j += 1
            groups.append((k, j))
            k = j + 1
        # durations
        shortest_g, shortest_dur = None, float("inf")
        for gi, (a, b) in enumerate(groups):
            dur = ends[b] - starts[a]
            if dur < min_duration and dur < shortest_dur:
                shortest_dur, shortest_g = dur, gi
        if shortest_g is None:
            break
        a, b = groups[shortest_g]
        # Boundary CUT indices: between the previous group and this group is
        # cut index a (>= 1; index 0 is the implicit head cut, never removable);
        # between this group and the next is cut index b+1 (only if a next
        # group exists).
        left_b = a if a >= 1 else None
        right_b = b + 1 if b + 1 <= len(cuts) - 1 else None
        sim_left = sims[left_b - 1] if left_b is not None else None
        sim_right = sims[right_b - 1] if right_b is not None else None
        if sim_left is not None and sim_right is not None:
            remove = left_b if sim_left <= sim_right else right_b
        elif sim_left is not None:
            remove = left_b
        elif sim_right is not None:
            remove = right_b
        else:
            # both missing: merge into the LONGER neighbor for stability
            # (was arbitrary-left in the old code)
            left_dur = (starts[a] - starts[groups[shortest_g - 1][0]]) if shortest_g > 0 else -1.0
            right_dur = (ends[groups[shortest_g + 1][1]] - ends[b]) if shortest_g + 1 < len(groups) else -1.0
            if left_dur >= right_dur and left_b is not None:
                remove = left_b
            elif right_b is not None:
                remove = right_b
            else:
                remove = left_b
        if remove is None or remove in removed_boundary or remove <= 0:
            break  # nothing removable (implicit head cut or single shot) — done
        removed_boundary.add(remove)

    if progress_cb:
        _safe_progress(progress_cb, 0.88, "merging short scenes")

    merged = [c for i, c in enumerate(cuts) if i not in removed_boundary]
    for i, c in enumerate(merged):
        c["index"] = i
    return merged


# ------------------------------------------------------------------ grouping


def _group_scenes(cuts: list[dict], feat: _Features, fps: float,
                  total_duration: float, threshold: float) -> list[dict]:
    """Cluster adjacent shots with similar colorimetry (normalized JS scale).

    Uses the subsampled hist store (zero seeks). The first cut IS compared now
    (the old i==0 branch skipped it — RV1 2.5)."""
    if not cuts:
        return cuts
    starts = [0.0] + [c["seconds"] for c in cuts]
    ends = [c["seconds"] for c in cuts] + [total_duration]
    mids = [_mid_hist_at(feat, int(round(((starts[k] + ends[k]) / 2.0) * fps)))
            for k in range(len(starts))]
    group = 0
    for i, c in enumerate(cuts):
        if i > 0:
            h_prev, h_cur = mids[i], mids[i + 1]
            if h_prev is not None and h_cur is not None:
                if _js_divergence(h_prev, h_cur) >= threshold:
                    group += 1
        c["scene_group"] = group
    return cuts


# ------------------------------------------------------------------ main API


def detect_scenes(video_path: str | Path,
                  algo: str = "all",
                  threshold: float | None = None,
                  min_scene_len: float | None = None,
                  fade_threshold: int | None = None,
                  fade_min_len: float | None = None,
                  dissolve_window: int | None = None,
                  dissolve_threshold: float | None = None,
                  group_scenes: bool = False,
                  group_threshold: float | None = None,
                  proxy: str | None = None,
                  merge_short_scenes: float | None = None,
                  fade_ceiling: int | None = None,
                  motion_gate: bool = False,
                  hash_threshold: float | None = None,
                  hash_corroboration: bool | None = None,
                  hash_corroboration_spike: float | None = None,
                  hash_corroboration_js: float | None = None,
                  hash_corroboration_luma: float | None = None,
                  dissolve_dissim: float | None = None,
                  dissolve_windows: list[int] | str | None = None,
                  hard_cut_mad_min: float | None = None,
                  consec_js_min: float | None = None,
                  consec_js_frame_min: float | None = None,
                  consec_js_sustain_frac: float | None = None,
                  consec_js_ratio_min: float | None = None,
                  spatial_dissolve: bool | None = None,
                  neural: bool = False,
                  progress_cb=None) -> dict:
    """Run multi-pass scene detection (v2.1 architecture — see module docstring).

    New parameters:
      fade_ceiling: luma threshold for the fade-to-WHITE pass (default 243).
      motion_gate: enable the downgrade-not-delete motion gate (default False).
      dissolve_dissim: normalized JS endpoint-dissimilarity floor (D19: 0.15).
      dissolve_windows: multi-scale window list (default [8, 16, 32]) or a
        comma-string "8,16,32".
      hard_cut_mad_min: window-MAD ceiling for dissolve candidacy (D19: 40 —
        cuts measure 54+; the fixture-era 20 rejected half of external blends).
      consec_js_min: window-mean consec-JS floor (D19: 0.025 — external
        blends walk histograms far more weakly than fixture dissolves).
      consec_js_frame_min: frame-level elevation floor for the D19 sustain
        gate (default 0.03; spike-immune vs single-frame cut jumps).
      consec_js_sustain_frac: fraction of window frames that must clear
        consec_js_frame_min (default 0.35).
      consec_js_ratio_min: D19 localization ratio — window mean must exceed
        the clip-baseline median by this factor (default 1.8; rejects pans).
    Deprecated: dissolve_window (alias -> [w]), dissolve_threshold (ignored) —
    the old Pearson-correlation semantics cannot be mapped to JS (RV2 N25).
    """
    video_path = str(video_path)

    # ---- parameters (dual probe: D10 v2.1 — original for source metadata,
    # detect path for derivation fps/duration)
    info_orig = probe_video(video_path)
    detect_path = proxy or video_path
    info_det = probe_video(detect_path)
    fps = info_det.fps

    threshold = threshold if threshold is not None else DEFAULTS["threshold"]
    min_scene_len = min_scene_len if min_scene_len is not None else DEFAULTS["min_scene_len"]
    fade_threshold = fade_threshold if fade_threshold is not None else DEFAULTS["fade_threshold"]
    fade_min_len = fade_min_len if fade_min_len is not None else DEFAULTS["fade_min_len"]
    fade_ceiling = fade_ceiling if fade_ceiling is not None else DEFAULTS.get("fade_ceiling", 243)
    dissolve_dissim = dissolve_dissim if dissolve_dissim is not None else DEFAULTS.get("dissolve_dissim", 0.15)
    hard_cut_mad_min = hard_cut_mad_min if hard_cut_mad_min is not None else DEFAULTS.get("hard_cut_mad_min", 40.0)
    consec_js_min = consec_js_min if consec_js_min is not None else DEFAULTS.get("consec_js_min", 0.03)
    consec_js_frame_min = consec_js_frame_min if consec_js_frame_min is not None else DEFAULTS.get("consec_js_frame_min", 0.03)
    consec_js_sustain_frac = consec_js_sustain_frac if consec_js_sustain_frac is not None else DEFAULTS.get("consec_js_sustain_frac", 0.35)
    consec_js_ratio_min = consec_js_ratio_min if consec_js_ratio_min is not None else DEFAULTS.get("consec_js_ratio_min", 1.8)
    motion_gate_js = DEFAULTS.get("motion_gate_js", 0.05)
    group_threshold = group_threshold if group_threshold is not None else DEFAULTS.get("group_threshold", 0.15)
    spatial_dissolve_en = bool(spatial_dissolve
                               if spatial_dissolve is not None
                               else DEFAULTS.get("spatial_dissolve", True))
    spatial_dissim_min = DEFAULTS.get("spatial_dissim", 14.0)
    spatial_consec_lo = DEFAULTS.get("spatial_consec_lo", 1.0)
    spatial_motion_max = DEFAULTS.get("spatial_motion_max", 8.0)

    # windows: new list/str param, deprecated singular alias, default
    if dissolve_windows is not None:
        if isinstance(dissolve_windows, str):
            windows = sorted({int(x) for x in dissolve_windows.split(",") if x.strip()})
        else:
            windows = sorted({int(x) for x in dissolve_windows})
    elif dissolve_window is not None:
        print("[scenecut] --dissolve-window is deprecated; use --dissolve-windows "
              f'"8,16,32" (aliased to [{dissolve_window}])', file=sys.stderr)
        windows = [int(dissolve_window)]
    else:
        windows = sorted({int(x) for x in DEFAULTS.get("dissolve_windows", [8, 16, 32])})
    if not windows:
        windows = [8, 16, 32]
    if dissolve_threshold is not None:
        print("[scenecut] --dissolve-threshold is deprecated (Pearson->JS semantics "
              "change; no valid mapping) and is IGNORED — use --dissolve-dissim",
              file=sys.stderr)

    requested: set[str]
    if algo == "all":
        requested = {"adaptive", "threshold", "hash"}
    elif algo == "threshold":
        requested = {"threshold"}
    elif algo in ("adaptive", "hash"):
        requested = {algo}
    elif algo == "dissolve":
        requested = set()
    else:
        requested = {"adaptive", "threshold", "hash"}

    # VFR sanity (D10): nominal fps vs n_frames/duration
    if info_det.duration > 0 and info_det.n_frames:
        implied = info_det.n_frames / info_det.duration
        if fps > 0 and abs(implied - fps) / fps > 0.01:
            print(f"[scenecut] WARNING: nominal fps {fps:.3f} disagrees with "
                  f"n_frames/duration {implied:.3f} (>1%) — VFR source? Cut "
                  "positions may drift.", file=sys.stderr)

    # Drop end-of-video spurious cuts (any cut within 1/fps of the end)
    end_eps = 1.0 / fps if fps > 0 else 0.05
    duration = info_det.duration

    def _is_end_spurious(seconds: float) -> bool:
        return seconds >= duration - end_eps and seconds > 0

    def _mk_cut(frame_num: int, ctype: str, source: str, conf: float) -> dict:
        sec = frame_to_seconds(frame_num, fps)
        return {
            "frame_num": frame_num,
            "seconds": sec,
            "timecode": seconds_to_hmsms(sec),
            "type": ctype,
            "confidence": round(float(conf), 3),
            "source": source,
        }

    cuts: list[dict] = [{
        "index": 0, "frame_num": 0, "seconds": 0.0, "timecode": "00:00:00.000",
        "type": "cut", "confidence": 1.0, "source": "implicit",
    }]

    # ---- pass 1: orchestrated scenedetect detectors (one decode)
    cuts_by_source: dict[str, list[int]] = {}
    stats = {}
    metric_keys = {}
    primary_metric = {}
    if requested:
        if progress_cb:
            _safe_progress(progress_cb, 0.05, "detector pass")
        cuts_by_source, stats, metric_keys, primary_metric = _pass_orchestrated(
            detect_path, requested, threshold, min_scene_len, fps,
            fade_threshold, fade_ceiling,
            hash_threshold=hash_threshold if hash_threshold is not None
            else DEFAULTS.get("hash_threshold", 0.4),
            progress_cb=progress_cb)

    # ---- pass 2: feature extraction (one decode)
    if progress_cb:
        _safe_progress(progress_cb, 0.40, "feature pass")
    feat = _feature_pass(detect_path, windows, fps, progress_cb=progress_cb)

    # ---- confidences (D7) from detector metrics + luma troughs
    fade_radius = max(3, int(round(fade_min_len * fps)))
    for source, frames in cuts_by_source.items():
        for f in frames:
            iv: list[int] | None = None   # D14 fade interval (threshold path)
            if source == "adaptive":
                ratio = _metric_value(stats, primary_metric, source, f)
                conf = _conf_adaptive(ratio, threshold)
                ctype = "cut"
            elif source == "hash":
                dist = _metric_value(stats, primary_metric, source, f)
                conf = _conf_hash(dist)
                ctype = "cut"
            elif source == "threshold":
                trough = float(np.min(_luma_window(feat, f, fade_radius))) if f < feat.n else None
                conf = _conf_floor(trough, fade_threshold)
                ctype = "fade"
                iv = _fade_run_interval(feat, f, fade_threshold,
                                        int(1.5 * fps) + fade_radius)
            else:
                conf = 0.7
                ctype = "cut"
            c = _mk_cut(f, ctype, source, conf)
            if iv is not None:
                c["interval"] = iv
            if not _is_end_spurious(c["seconds"]):
                cuts.append(c)
            iv = None

    # ---- fade-to-white (D5): feature-scan regions + ramp-onset emission.
    # Computed ONCE and reused for the suppression pass (CR1 L4: double call).
    fade_regions = _detect_fade_white(feat, fps, fade_ceiling, min_scene_len) \
        if algo in ("all", "threshold") else []
    if fade_regions:
        if progress_cb:
            _safe_progress(progress_cb, 0.72, "fade-white pass")
        for onset, a, b in fade_regions:
            peak = float(np.max(feat.mean_v[a:b])) if b > a else None
            conf = _conf_ceiling(peak, fade_ceiling)
            c = _mk_cut(onset, "fade", "threshold_ceiling", conf)
            # D14: fade-white interval = [ramp onset, run end) — the full
            # fade span (contains the emission by construction)
            c["interval"] = [int(onset), int(b)]
            if not _is_end_spurious(c["seconds"]):
                cuts.append(c)

    # ---- dissolves (from features)
    if algo in ("all", "dissolve"):
        if progress_cb:
            _safe_progress(progress_cb, 0.75, "dissolve pass")
        for c in _detect_dissolves(feat, windows, dissolve_dissim, hard_cut_mad_min,
                                   consec_js_min, min_scene_len, fps,
                                   consec_js_frame_min=consec_js_frame_min,
                                   consec_js_sustain_frac=consec_js_sustain_frac,
                                   consec_js_ratio_min=consec_js_ratio_min,
                                   progress_cb=progress_cb):
            if not _is_end_spurious(c["seconds"]):
                cuts.append(c)

        # ---- tier-2 spatial dissolves (D18 / Phase 1.5): match-dissolves
        # between same-palette shots — global H-S JS is blind; the 8x8
        # block-max MAD arrays see the rearrangement. G0 excludes tier-1's
        # regime so the two candidate sets are disjoint.
        if spatial_dissolve_en:
            if progress_cb:
                _safe_progress(progress_cb, 0.77, "spatial dissolve pass")
            for c in _detect_spatial_dissolves(
                    feat, windows, hard_cut_mad_min, consec_js_min,
                    min_scene_len, fps, spatial_dissim_min,
                    spatial_consec_lo, spatial_motion_max,
                    progress_cb=progress_cb):
                if not _is_end_spurious(c["seconds"]):
                    cuts.append(c)

    # ---- optional neural pass (D13): TransNetV2 on the same detect-path
    # timeline. Emits cuts BEFORE dedup (priority merge applies); the arbiter
    # runs post-dedup below. Fail-closed on missing extra / bad weights.
    neural_report = {"enabled": False}
    neural_cuts_n = 0
    _neural_probs: tuple[np.ndarray, np.ndarray] = (np.zeros(0), np.zeros(0))
    if neural:
        if progress_cb:
            _safe_progress(progress_cb, 0.42, "neural pass")
        n_frames_cut, _neural_probs_ab, _neural_probs_gr = _pass_neural(
            detect_path, progress_cb=progress_cb)
        _neural_probs = (_neural_probs_ab, _neural_probs_gr)
        neural_cuts_n = len(n_frames_cut)
        neural_report = {
            "enabled": True,
            "cuts": neural_cuts_n,
            "suppressed": 0,
            "corroborated": 0,
            "model": _NEURAL_MODEL_NAME,
            "threshold": _NEURAL_THRESHOLD,
        }
        for f in n_frames_cut:
            # f = neural frame + 1 (our convention); confidence reads the
            # firing frame's prob (f-1 is always in-bounds: f >= 1).
            c = _mk_cut(f, "cut", "neural", float(_neural_probs_ab[f - 1]))
            if not _is_end_spurious(c["seconds"]):
                cuts.append(c)

    # ---- fade-region suppression: a fade-to-white transition owns its span —
    # drop non-fade cuts inside [onset, run_end) (measured: hash fires at the
    # full-white frame ~90 while the editorial boundary is the ramp onset ~63)
    if fade_regions:
        cuts = [c for c in cuts if not (
            c["source"] not in ("implicit", "threshold_ceiling")
            and any(k <= c["frame_num"] < b for (k, a, b) in fade_regions)
        )]

    # ---- dedup (priority + capped noisy-OR + typing)
    cuts = _dedup_cuts(cuts, min_scene_len)
    for i, c in enumerate(cuts):
        c["index"] = i

    # ---- D16 v4.3: gradual-region absorption + solo-hash corroboration
    # gate (post-dedup — needs merged `sources`; BEFORE the neural arbiter
    # whose WEAK set excludes hash — RV4b F10 insertion point). Default ON
    # (config hash_corroboration); disable via hash_corroboration=False.
    gate_report = {"enabled": bool(hash_corroboration
                                    if hash_corroboration is not None
                                    else DEFAULTS.get("hash_corroboration", True)),
                   "absorbed": 0, "gated": 0}
    if gate_report["enabled"]:
        cuts, n_absorbed, n_gated = _corroboration_gate(
            cuts, feat,
            spike_min=hash_corroboration_spike
            if hash_corroboration_spike is not None
            else DEFAULTS.get("hash_corroboration_spike", 3.0),
            js_min=hash_corroboration_js
            if hash_corroboration_js is not None
            else DEFAULTS.get("hash_corroboration_js", 0.15),
            luma_min=hash_corroboration_luma
            if hash_corroboration_luma is not None
            else DEFAULTS.get("hash_corroboration_luma", 8.0),
        )
        gate_report["absorbed"] = int(n_absorbed)
        gate_report["gated"] = int(n_gated)
        for i, c in enumerate(cuts):
            c["index"] = i

    # ---- optional neural arbiter (D13 A20): gradual corroboration FIRST,
    # then abrupt-based suppression of single-weak-source cuts. Post-dedup
    # (needs the merged `sources` list). Suppressed cuts are REMOVED (opt-in
    # mode — distinct from the motion gate's downgrade-not-delete).
    if neural:
        cuts, n_supp, n_corr = _neural_arbiter(
            cuts, _neural_probs[0], _neural_probs[1], fps)
        neural_report["suppressed"] = n_supp
        neural_report["corroborated"] = n_corr
        for i, c in enumerate(cuts):
            c["index"] = i

    # ---- optional motion gate (post-dedup, downgrade-not-delete)
    motion_gated = 0
    if motion_gate:
        motion_gated = _motion_gate(cuts, feat, motion_gate_js)

    # ---- optional merge short scenes
    if merge_short_scenes and merge_short_scenes > 0:
        if progress_cb:
            _safe_progress(progress_cb, 0.85, "merging short scenes")
        cuts = _merge_short_scenes(cuts, feat, fps, duration, merge_short_scenes,
                                   progress_cb=progress_cb)

    # ---- optional scene grouping
    if group_scenes:
        cuts = _group_scenes(cuts, feat, fps, duration, group_threshold)

    # ---- build scenes (skip empty scenes where start == end)
    scenes = []
    starts = [c["seconds"] for c in cuts] + [info_orig.duration]
    scene_idx = 0
    for i, start in enumerate(starts[:-1]):
        end = starts[i + 1]
        if end - start <= 1e-3:
            continue
        scenes.append({
            "index": scene_idx,
            "start": start,
            "end": end,
            "duration": end - start,
            "start_frame": int(round(start * fps)),
            "end_frame": int(round(end * fps)),
            "thumbnail": None,
            "label": None,
            "tags": [],
            "type": cuts[i].get("type", "cut") if i < len(cuts) else "cut",
        })
        scene_idx += 1

    return {
        "version": "1.0",
        "source": info_orig.to_dict(),
        "detector": {
            "algo": algo,
            "schema": "1.1",
            "threshold": threshold,
            "min_scene_len": min_scene_len,
            "fade_threshold": fade_threshold,
            "fade_ceiling_threshold": fade_ceiling,
            "fade_min_len": fade_min_len,
            "dissolve_dissim": dissolve_dissim,
            "dissolve_windows": windows,
            "hard_cut_mad_min": hard_cut_mad_min,
            "consec_js_min": consec_js_min,
            "consec_js_frame_min": consec_js_frame_min,
            "consec_js_sustain_frac": consec_js_sustain_frac,
            "consec_js_ratio_min": consec_js_ratio_min,
            "dissolve_window": dissolve_window,   # deprecated, kept one release
            "dissolve_threshold": dissolve_threshold,  # deprecated, kept one release
            "group_scenes": group_scenes,
            "group_threshold": group_threshold,
            "proxy": proxy,
            "merge_short_scenes": merge_short_scenes,
            "motion_gate": {"enabled": bool(motion_gate), "downgraded": int(motion_gated)},
            "hash_threshold": hash_threshold if hash_threshold is not None
                              else DEFAULTS.get("hash_threshold", 0.4),
            "hash_corroboration": gate_report,
            "neural": neural_report,
        },
        "cuts": cuts,
        "scenes": scenes,
        "overrides": {
            "added_cuts": [],
            "removed_cuts": [],
            "moved_cuts": {},
            "merged_scenes": [],
            "split_scenes": [],
        },
        "labels": {},
        "tags": {},
    }


# ------------------------------------------------------------------ neural pass (D13, v3.1)

# Pinned weights digest (R4 research: bundled in the transnetv2-pytorch 1.0.5
# wheel). FAIL-CLOSED per A24 — corrupted weights emit garbage into dedup +
# the arbiter, so a mismatch aborts the pass with a readable error.
_NEURAL_WEIGHTS_SHA256 = "a313d0b3bebfa9a71914b375bfdf918a30b5c3b1e6be51972d35dd8078b442de"
_NEURAL_MODEL_NAME = "transnetv2-pytorch-1.0.5"
_NEURAL_INPUT_W, _NEURAL_INPUT_H = 48, 27        # HWC (W, H) — model contract
_NEURAL_THRESHOLD = 0.5                          # author default (R4)
_NEURAL_CHUNK = 4000                             # ~15.5 MB of uint8 frames
_NEURAL_CHUNK_OVERLAP = 100                      # >= receptive field (A21)


def _neural_check_weights() -> None:
    """Fail-closed sha256 verification of the bundled weights (A24)."""
    import hashlib
    import importlib.resources
    try:
        pkg = importlib.resources.files("transnetv2_pytorch")
        wf = pkg.joinpath("transnetv2-pytorch-weights.pth")
        data = wf.read_bytes() if hasattr(wf, "read_bytes") else open(wf, "rb").read()
    except Exception as e:
        raise SceneCutError(
            f"neural weights unreadable: {e}. Reinstall transnetv2-pytorch==1.0.5",
            code=3) from e
    digest = hashlib.sha256(data).hexdigest()
    if digest != _NEURAL_WEIGHTS_SHA256:
        raise SceneCutError(
            "neural weights sha256 mismatch (expected "
            f"{_NEURAL_WEIGHTS_SHA256[:16]}..., got {digest[:16]}...) — refusing to "
            "run with corrupted weights (A24 fail-closed). Reinstall the package.",
            code=3)


def _pass_neural(video_path: str,
                 progress_cb=None) -> tuple[list[int], np.ndarray, np.ndarray]:
    """TransNetV2 pass (D13): chunked streaming CPU inference over the
    detect-path timeline.

    Returns (cut_frames, abrupt_probs, gradual_probs) where the probability
    arrays align with the GLOBAL frame timeline (overlap-trimmed per A21).
    Raises SceneCutError (code 3) when the extra is missing or weights fail.

    RNG isolation (A22): TransNetV2.__init__ reseeds torch/np/random to 42 —
    snapshot & restore all three around construction.
    """
    import random as _random

    try:
        import torch
        from transnetv2_pytorch import TransNetV2
    except ImportError as e:
        raise SceneCutError(
            "neural pass needs the optional extra: "
            "pip install 'scenecut[neural]' (transnetv2-pytorch + torch>=2.4)",
            code=3) from e

    _neural_check_weights()

    # --- RNG state isolation (A22)
    t_state = torch.get_rng_state()
    np_state = np.random.get_state()
    py_state = _random.getstate()
    try:
        model = TransNetV2(device="cpu")   # auto-loads bundled weights, eval()
    finally:
        torch.set_rng_state(t_state)
        np.random.set_state(np_state)
        _random.setstate(py_state)

    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise SceneCutError(f"neural pass cannot open {video_path!r}", code=3)

    abrupt_parts: list[np.ndarray] = []
    gradual_parts: list[np.ndarray] = []
    n_total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT)) or 0
    buf: list[np.ndarray] = []
    pos = 0
    emitted = 0

    def _flush(chunk: list[np.ndarray], is_first: bool) -> None:
        nonlocal emitted
        if not chunk:
            return
        x = torch.from_numpy(np.stack(chunk))       # [N, 27, 48, 3] uint8 RGB
        abrupt, gradual = model.predict_frames(x, quiet=True)
        a = abrupt.numpy() if hasattr(abrupt, "numpy") else np.asarray(abrupt)
        g = gradual.numpy() if hasattr(gradual, "numpy") else np.asarray(gradual)
        if not is_first:
            # Discard overlap-region predictions from the earlier chunk (A21):
            # this chunk's first _OVERLAP predictions duplicate the tail of the
            # previous chunk (already emitted); drop them, keep the rest.
            keep_from = _NEURAL_CHUNK_OVERLAP
            a, g = a[keep_from:], g[keep_from:]
        abrupt_parts.append(a)
        gradual_parts.append(g)
        emitted += len(a)

    try:
        while True:
            ok, frame = cap.read()
            if not ok:
                break
            if frame is None:
                continue
            small = cv2.resize(frame, (_NEURAL_INPUT_W, _NEURAL_INPUT_H),
                               interpolation=cv2.INTER_AREA)
            rgb = cv2.cvtColor(small, cv2.COLOR_BGR2RGB)
            buf.append(rgb)
            pos += 1
            if len(buf) >= _NEURAL_CHUNK + _NEURAL_CHUNK_OVERLAP:
                _flush(buf[: _NEURAL_CHUNK + _NEURAL_CHUNK_OVERLAP], emitted == 0)
                buf = buf[_NEURAL_CHUNK:]          # keep the overlap tail
            if progress_cb and n_total and pos % 500 == 0:
                _safe_progress(progress_cb, 0.42 + 0.06 * min(1.0, pos / n_total),
                               "neural pass")
        _flush(buf, emitted == 0)
    finally:
        cap.release()

    if not abrupt_parts:
        return [], np.zeros(0, dtype=np.float32), np.zeros(0, dtype=np.float32)
    abrupt = np.concatenate(abrupt_parts).astype(np.float32)
    gradual = np.concatenate(gradual_parts).astype(np.float32)

    # Emission (measured S3 on the fixture corpus):
    # 1. TransNetV2's abrupt head fires at the LAST frame of the previous
    #    shot (hard cuts: 119/239 vs GT 120/240) -> +1 maps to our
    #    first-frame-of-new-shot convention.
    # 2. Abrupt spikes INSIDE a sustained gradual run are mid-transition
    #    artifacts, not cuts (fade fixture: abrupt fires twice inside a
    #    57-frame fade). Runs >= _NEURAL_TRANSITION_MIN frames are transition
    #    zones: their abrupt cuts are dropped; the gradual head corroborates
    #    the heuristic fade/dissolve detectors instead (arbiter A20-a).
    #    Hard cuts' 2-frame gradual spikes are NOT runs -> kept.
    trans_zones = _gradual_runs(gradual, _NEURAL_TRANSITION_MIN)
    cut_frames = []
    for i in np.nonzero(abrupt > _NEURAL_THRESHOLD)[0]:
        i = int(i)
        if any(lo <= i < hi for lo, hi in trans_zones):
            continue
        cut_frames.append(i + 1)
    return cut_frames, abrupt, gradual


_NEURAL_TRANSITION_MIN = 8  # frames of sustained gradual-prob = transition zone


def _gradual_runs(gradual: np.ndarray, min_len: int) -> list[tuple[int, int]]:
    """[lo, hi) spans of consecutive gradual-prob > 0.5, length >= min_len."""
    if len(gradual) == 0:
        return []
    hot = gradual > 0.5
    runs = []
    start = None
    for i, h in enumerate(hot):
        if h and start is None:
            start = i
        elif not h and start is not None:
            if i - start >= min_len:
                runs.append((start, i))
            start = None
    if start is not None and len(hot) - start >= min_len:
        runs.append((start, len(hot)))
    return runs


def _neural_arbiter(cuts: list[dict], abrupt: np.ndarray, gradual: np.ndarray,
                    fps: float) -> tuple[list[dict], int, int]:
    """D13 v3.1 A20 arbiter. ORDER IS CRITICAL: gradual corroboration runs
    FIRST and exempts genuine gradual transitions from suppression; only then
    does abrupt-based suppression act on remaining single-weak-source cuts.

    Weak = sources list of exactly one entry, that source in
    {threshold, threshold_ceiling, dissolve} (A20-b).
      - dissolve-sourced weak cuts: suppress iff max gradual in +-16 < 0.1
        (gradual head only — dissolve cuts fire at window midpoint, the
        abrupt head fires at the boundary up to 16 frames away).
      - threshold/ceiling weak cuts: suppress iff gradual +-16 < 0.1 AND
        abrupt +-4 < 0.1 (both heads blind).

    Returns (surviving cuts, n_suppressed, n_corroborated).
    """
    if not cuts or len(abrupt) == 0:
        return cuts, 0, 0

    def _win(arr: np.ndarray, frame: int, radius: int) -> float:
        if len(arr) == 0:
            return 0.0
        lo = max(0, frame - radius)
        hi = min(len(arr), frame + radius + 1)
        if hi <= lo:
            return 0.0
        return float(np.max(arr[lo:hi]))

    WEAK = {"threshold", "threshold_ceiling", "dissolve"}
    corroborated = 0
    suppressed = 0
    out: list[dict] = []

    for c in cuts:
        src = c.get("source", "")
        srcs = c.get("sources") or [src]
        if c.get("frame_num", 0) == 0 or src == "implicit":
            out.append(c)
            continue
        f = int(c["frame_num"])
        # --- 1. Gradual corroboration FIRST (A20-a): any gradual-head support
        # within +-4 frames exempts + boosts (capped).
        if _win(gradual, f, 4) > 0.5 and c.get("type") in ("dissolve", "fade"):
            c = dict(c)
            c["confidence"] = round(min(0.95, float(c.get("confidence", 0.5)) + 0.1), 3)
            c["neural_corroborated"] = True
            corroborated += 1
            out.append(c)
            continue
        # --- 2. Abrupt suppression on single-weak-source cuts.
        if len(srcs) == 1 and src in WEAK:
            if src == "dissolve":
                blind = _win(gradual, f, 16) < 0.1
            else:
                blind = _win(gradual, f, 16) < 0.1 and _win(abrupt, f, 4) < 0.1
            if blind:
                suppressed += 1
                continue
        out.append(c)
    return out, suppressed, corroborated
