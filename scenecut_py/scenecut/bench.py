"""SceneCut benchmark harness — evaluation engine (DECISIONS.md D12, v3.1).

Frozen protocol (see .agents/DECISIONS.md D12 + research/bench-datasets.md):

* Cut-level (primary): greedy 1-to-1 nearest matching within tolerance τ
  (reported for τ = 0 and τ = 1). A19 determinism: predictions sorted
  frame-ascending; each takes the nearest UNMATCHED GT; the earlier GT wins
  equidistant ties. Greedy (TRECVID convention), NOT Hungarian.
* Fade-first ordering: predictions inside a GT gradual interval [start, end)
  are consumed by the gradual scorer BEFORE cut matching (a gradual match =
  prediction in interval; extra predictions in the same interval = gradual
  FPs; a GT gradual interval with no prediction = a gradual FN).
* Frame-level (secondary): binary per-frame labels over frames EXCLUDING GT
  gradual intervals; reports frames_scored.
* Micro-aggregation: TP/FP/FN are summed across videos and THEN ratios are
  taken — never a mean of per-video F1s.
* A16: proxy detection refused unless allow_proxy=True; per-video offset
  histogram (mean |p−g|) reported as a drift tripwire.
* A18: gradual-adjacent offset diagnostic (mean_abs_offset restricted to
  matches adjacent to gradual intervals — dissolve midpoint bias visibility).
* A14: ClipShots per-video best-offset verification (see bench_datasets).
* A15: regression gate against bench/golden.json floors (run_gate).

D14 (v4.3, additive — gradual-aware dual reporting; frozen D12 stays primary):
* evaluate_transitions: unified point/interval event matching. An interval
  [s, e) matches GT point g at τ iff its covered frames [s, e−1] intersect
  [g−τ, g+τ] — for points this IS the D12 rule (|p−g| ≤ τ), so the rule
  generalizes and never forks. Fade-first gradual consumption is extended to
  intervals (ANY event whose covered frames overlap a GT gradual interval is
  consumed by the gradual scorer first). Hard matching is greedy 1-1 with gap
  distances (0 when g is covered, else the distance to the nearest covered
  frame). Width discipline: interval MATCH windows are clipped to ≤64 frames
  anchored at the emission frame (n_width_capped + width_histogram).
* Diagnostics in every run_bench report: fp_near_miss_histogram,
  fp_by_source / fn_by_source; transitions@tolN + transitions_gradual
  families reported alongside hard_cuts@tolN / gradual.
* evaluate_clipshots_official: Tang et al. script compat block (type-
  restricted, union-count TP, labeled — never mixed with the native families).
* Optional τ=5 cell: run_bench(tolerances=...) / external_bench --tolerances
  (TRECVid-convention comparability; never gated anywhere).
"""
from __future__ import annotations

import json
import logging
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable, Sequence

from . import __version__, bench_datasets

log = logging.getLogger("scenecut.bench")

CONFIGS: dict[str, dict[str, Any]] = {
    "default": {},
    "conservative": {"threshold": 35, "dissolve_dissim": 0.18, "merge_short_scenes": 2.0},
    "sensitive": {"threshold": 15, "dissolve_dissim": 0.08},
}
TOLERANCES: tuple[int, ...] = (0, 1)

Gradual = Sequence[tuple[int, int]]


# ------------------------------------------------------------------ metrics


def _prf(tp: int, fp: int, fn: int) -> dict[str, Any]:
    """Precision/recall/F1 from raw counts (micro-aggregation cell helper)."""
    precision = tp / (tp + fp) if (tp + fp) else 0.0
    recall = tp / (tp + fn) if (tp + fn) else 0.0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) else 0.0
    return {"tp": tp, "fp": fp, "fn": fn,
            "precision": round(precision, 6), "recall": round(recall, 6),
            "f1": round(f1, 6)}


def evaluate_cuts(pred_cuts: Sequence[int], gt_cuts: Sequence[int],
                  gradual: Gradual | None = None, tolerance: int = 0) -> dict[str, Any]:
    """Greedy 1-to-1 cut matching with fade-first gradual consumption (D12).

    A19: predictions are sorted frame-ascending; each prediction takes the
    nearest unmatched GT within ``tolerance``; the earlier GT wins equidistant
    ties. Predictions inside a GT gradual interval [start, end) are consumed
    by the gradual scorer FIRST (A17 half-open semantics: a prediction at the
    interval's end frame is NOT gradual — it goes to cut matching).

    Returns ``{"hard_cuts": {tp, fp, fn}, "gradual": {tp, fp, fn} | None,
    "mean_abs_offset": float, "gradual_consumed": int,
    "gradual_adjacent_offset": float | None, "offsets": [int],
    "adjacent_offsets": [int]}`` where
    ``gradual`` is None when no GT gradual intervals exist,
    ``mean_abs_offset`` is Σ|p−g| / #matches over hard-cut matches (0.0 when
    no matches), and ``gradual_adjacent_offset`` (A18) restricts that mean to
    matches whose GT frame lies adjacent to a gradual interval (s−1 ≤ g ≤ e).
    """
    preds = sorted(int(p) for p in pred_cuts)
    gts = sorted(int(g) for g in gt_cuts)
    intervals = [(int(s), int(e)) for (s, e) in (gradual or [])]

    # --- fade-first: predictions inside GT gradual intervals are consumed by
    # the gradual scorer BEFORE cut matching.
    in_interval = [any(s <= p < e for (s, e) in intervals) for p in preds]
    remaining = [p for p, flag in zip(preds, in_interval) if not flag]
    gradual_counts = [0] * len(intervals)
    for p, flag in zip(preds, in_interval):
        if flag:
            for i, (s, e) in enumerate(intervals):
                if s <= p < e:
                    gradual_counts[i] += 1
                    break
    gradual_block = None
    if intervals:
        gradual_block = {
            "tp": sum(1 for c in gradual_counts if c >= 1),   # intervals matched
            "fp": sum(max(c - 1, 0) for c in gradual_counts),  # extra predictions
            "fn": sum(1 for c in gradual_counts if c == 0),    # empty intervals
        }

    # --- hard-cut greedy matching (A19).
    unmatched = list(gts)
    offsets: list[int] = []
    adjacent_offsets: list[int] = []
    fp = 0
    for p in remaining:
        best: int | None = None
        best_d = -1
        for g in unmatched:  # ascending: strict '<' keeps the earlier GT on ties
            d = abs(p - g)
            if d <= tolerance and (best is None or d < best_d):
                best, best_d = g, d
        if best is None:
            fp += 1
            continue
        unmatched.remove(best)
        offsets.append(best_d)
        if any(s - 1 <= best <= e for (s, e) in intervals):  # A18 adjacency
            adjacent_offsets.append(best_d)
    tp = len(offsets)
    fn = len(unmatched)

    return {
        "hard_cuts": {"tp": tp, "fp": fp, "fn": fn},
        "gradual": gradual_block,
        "mean_abs_offset": round(sum(offsets) / len(offsets), 6) if offsets else 0.0,
        "gradual_consumed": sum(1 for flag in in_interval if flag),
        "gradual_adjacent_offset": (round(sum(adjacent_offsets) / len(adjacent_offsets), 6)
                                    if adjacent_offsets else None),
        "offsets": offsets,
        "adjacent_offsets": adjacent_offsets,
    }


def evaluate_frames(pred_cuts: Sequence[int], gt_cuts: Sequence[int],
                    gradual: Gradual | None, n_frames: int) -> dict[str, int]:
    """Frame-level secondary metric: binary labels, GT gradual frames excluded.

    A frame is positive iff a cut event lands exactly on it (tolerance-0
    semantics). Frames inside GT gradual intervals [start, end) are EXCLUDED
    from scoring on both sides. Returns ``{tp, fp, fn, frames_scored}``.
    """
    excluded: set[int] = set()
    for s, e in (gradual or []):
        excluded.update(range(max(0, int(s)), min(int(e), n_frames)))
    gt_set = {int(g) for g in gt_cuts if 0 <= int(g) < n_frames}
    pred_set = {int(p) for p in pred_cuts if 0 <= int(p) < n_frames}
    return {
        "tp": len((gt_set & pred_set) - excluded),
        "fp": len((pred_set - gt_set) - excluded),
        "fn": len((gt_set - pred_set) - excluded),
        "frames_scored": n_frames - len(excluded),
    }


def _offset_histogram(offsets: list[int]) -> dict[str, Any]:
    """A16 drift tripwire: bucketed |p−g| distribution for one video."""
    buckets = {"0": 0, "1": 0, "2-4": 0, ">=5": 0}
    for o in offsets:
        if o == 0:
            buckets["0"] += 1
        elif o == 1:
            buckets["1"] += 1
        elif o <= 4:
            buckets["2-4"] += 1
        else:
            buckets[">=5"] += 1
    return {"matches": len(offsets),
            "mean": round(sum(offsets) / len(offsets), 4) if offsets else 0.0,
            "max": max(offsets) if offsets else 0,
            "buckets": buckets}


# ------------------------------------------------------------------ D14 gradual-aware


WIDTH_CAP = 64            # D14 width discipline: max interval match-window width
WIDTH_BUCKETS = ("1-8", "9-16", "17-32", "33-64", ">64")
NEAR_MISS_BUCKETS = ("0-1", "2-4", "5-10", "11-30", ">30")


def _normalize_events(pred_events: Sequence[Any]) -> list[tuple[int, tuple[int, int] | None]]:
    """Normalize mixed prediction events to ``(frame, interval | None)`` (D14).

    Accepts ``{"frame": int, "interval": [s, e) | None}`` dicts (the optional
    interval field may be absent or malformed — degrade to a point, never
    raise) and plain ints (point events). Output is sorted
    emission-ascending (points before intervals at equal frames).
    """
    out: list[tuple[int, tuple[int, int] | None]] = []
    for ev in pred_events:
        if not isinstance(ev, dict):
            out.append((int(ev), None))
            continue
        frame = ev.get("frame", ev.get("frame_num"))
        try:
            iv = ev.get("interval")
            if iv is not None:
                s, e = int(iv[0]), int(iv[1])
                if e > s:
                    out.append((int(frame if frame is not None else s), (s, e)))
                    continue
        except (TypeError, ValueError, IndexError):
            pass  # malformed interval field -> treat as a point event
        out.append((int(frame if frame is not None else 0), None))
    out.sort(key=lambda t: (t[0], t[1][0] if t[1] is not None else -1))
    return out


def _clip_interval_window(s: int, e: int, frame: int) -> tuple[int, int]:
    """D14 width discipline: the MATCH window of interval ``[s, e)`` emitted
    at ``frame`` — an inclusive frame range of at most :data:`WIDTH_CAP`
    frames, a subrange of the true interval, anchored at the emission frame
    (F4): symmetric ``[frame−31, frame+32]`` for interior emissions (dissolve
    center), forward ``[frame, frame+63]`` for onset emissions (fades extend
    forward), backward ``[e−64, e−1]`` for end-anchored emissions. No clip
    when the interval itself is ≤ 64 frames.
    """
    if e - s <= WIDTH_CAP:
        return s, e - 1
    if frame <= s:                      # onset emission (fade): forward clip
        lo, hi = s, s + WIDTH_CAP - 1
    elif frame >= e - 1:                # end-anchored emission: backward clip
        lo, hi = e - WIDTH_CAP, e - 1
    else:                               # interior emission (dissolve center)
        lo, hi = frame - 31, frame + 32
    return max(lo, s), min(hi, e - 1)


def _gap_distance(lo: int, hi: int, g: int) -> int:
    """Distance from GT frame ``g`` to the covered window ``[lo, hi]``:
    0 when covered, else the gap to the nearest covered frame (points: |p−g|)."""
    if lo <= g <= hi:
        return 0
    return max(lo - g, g - hi)


def evaluate_transitions(pred_events: Sequence[Any],
                         gt_cuts: Sequence[int],
                         gradual: Gradual | None = None,
                         tolerance: int = 0) -> dict[str, Any]:
    """D14 gradual-aware transition scoring — unified point/interval events.

    Match rule (generalizes D12, never forks it): an event matches GT point
    ``g`` at τ iff the event's MATCH frames intersect ``[g−τ, g+τ]`` — a
    point is ``|p−g| ≤ τ`` (byte-identical to :func:`evaluate_cuts`); an
    interval ``[s, e)`` (covered frames ``[s, e−1]``) is ``s ≤ g+τ`` and
    ``e−1 ≥ g−τ``. Fade-first gradual consumption is preserved from D12 and
    extended to intervals: ANY event whose COVERED frames (the FULL interval
    — the width clip below applies to hard matching only) overlap a GT
    gradual interval ``[gs, ge)`` is consumed by the gradual scorer BEFORE
    cut matching. Gradual block: greedy 1-1, events ascending, each takes
    the earliest unmatched overlapping GT interval (every overlap sits at
    gap 0, so the earlier GT wins ties — the D12 convention); extras are
    gradual FPs, unmatched GT intervals are gradual FNs.

    Hard matching: greedy 1-1 on the remaining events (emission ascending,
    strict ``<`` tie semantics), each takes the nearest unmatched GT within
    ``tolerance`` by gap distance. Width discipline: interval match windows
    are CLIPPED to ≤64 frames anchored at the emission frame
    (:func:`_clip_interval_window`); clipped events are counted in
    ``n_width_capped`` and pre-clip widths histogrammed.

    Returns the :func:`evaluate_cuts` shape — ``hard_cuts`` mirrors the
    interval-aware ``transitions`` block for shape compatibility; the frozen
    D12 point metric remains :func:`evaluate_cuts` — plus ``transitions``,
    ``n_width_capped`` and ``width_histogram``.
    """
    events = _normalize_events(pred_events)
    gts = sorted(int(g) for g in gt_cuts)
    intervals = [(int(s), int(e)) for (s, e) in (gradual or [])]

    n_width_capped = 0
    width_histogram = {k: 0 for k in WIDTH_BUCKETS}
    consumed: list[tuple[int, int, int, int]] = []    # (match_lo, hi, full_lo, hi)
    remaining: list[tuple[int, int, int, int]] = []

    for frame, iv in events:
        if iv is None:
            full = (frame, frame)
            match = full
        else:
            s, e = iv
            full = (s, e - 1)
            match = _clip_interval_window(s, e, frame)
            w = e - s
            if w > WIDTH_CAP:
                n_width_capped += 1
            if w <= 8:
                width_histogram["1-8"] += 1
            elif w <= 16:
                width_histogram["9-16"] += 1
            elif w <= 32:
                width_histogram["17-32"] += 1
            elif w <= WIDTH_CAP:
                width_histogram["33-64"] += 1
            else:
                width_histogram[">64"] += 1
        flo, fhi = full
        if any(flo <= ge - 1 and fhi >= gs for (gs, ge) in intervals):
            consumed.append((*match, flo, fhi))
        else:
            remaining.append((*match, flo, fhi))

    # --- fade-first gradual scorer: greedy 1-1, events ascending.
    matched_gt = [False] * len(intervals)
    gradual_fp = 0
    for _mlo, _mhi, flo, fhi in consumed:
        for i, (gs, ge) in enumerate(intervals):
            if matched_gt[i]:
                continue
            if flo <= ge - 1 and fhi >= gs:   # >= 1 shared covered frame
                matched_gt[i] = True
                break
        else:
            gradual_fp += 1                   # extra event in a taken interval
    gradual_block = None
    if intervals:
        gradual_block = {
            "tp": sum(1 for m in matched_gt if m),
            "fp": gradual_fp,
            "fn": sum(1 for m in matched_gt if not m),
        }

    # --- greedy 1-1 hard matching on the remaining events (A19 semantics).
    unmatched = list(gts)
    offsets: list[int] = []
    adjacent_offsets: list[int] = []
    fp = 0
    for mlo, mhi, _flo, _fhi in remaining:
        best: int | None = None
        best_d = -1
        for g in unmatched:  # ascending: strict '<' keeps the earlier GT on ties
            d = _gap_distance(mlo, mhi, g)
            if d <= tolerance and (best is None or d < best_d):
                best, best_d = g, d
        if best is None:
            fp += 1
            continue
        unmatched.remove(best)
        offsets.append(best_d)
        if any(s - 1 <= best <= e for (s, e) in intervals):  # A18 adjacency
            adjacent_offsets.append(best_d)
    tp = len(offsets)
    fn = len(unmatched)

    transitions = {"tp": tp, "fp": fp, "fn": fn}
    return {
        "hard_cuts": dict(transitions),   # evaluate_cuts-shape compatibility
        "transitions": transitions,
        "gradual": gradual_block,
        "mean_abs_offset": round(sum(offsets) / len(offsets), 6) if offsets else 0.0,
        "gradual_consumed": len(consumed),
        "gradual_adjacent_offset": (round(sum(adjacent_offsets) / len(adjacent_offsets), 6)
                                    if adjacent_offsets else None),
        "offsets": offsets,
        "adjacent_offsets": adjacent_offsets,
        "n_width_capped": n_width_capped,
        "width_histogram": width_histogram,
    }


def _official_block(tp: int, n_pred: int, n_gt: int) -> dict[str, Any]:
    """clipshots_official family cell: raw counts + ratios (micro-summable)."""
    return {"tp": tp, "preds": n_pred, "gts": n_gt,
            "precision": round(tp / n_pred, 6) if n_pred else 0.0,
            "recall": round(tp / n_gt, 6) if n_gt else 0.0}


def evaluate_clipshots_official(pred_events: Sequence[Any],
                                gt_cuts: Sequence[int],
                                gt_gradual: Gradual | None) -> dict[str, Any]:
    """D14 §5 ``clipshots_official`` compat block — Tang et al. script
    semantics, type-restricted, reported per ClipShots bench and NEVER mixed
    with the scenecut-native families in one table.

    Pred classification: point events are cut-class; interval events with
    width ≤ 2 frames are cut-class, wider intervals are gradual-class.
    Cuts: a cut-class pred matches a cut GT ``g`` iff its covered frames
    intersect ``[g−1, g+1]`` (points: ``|p−g| ≤ 1``); TP = # GT cuts matched
    by ≥1 cut-class pred (union count); FP = #cut-class preds − TP
    (precision by subtraction). Graduals: a gradual-class event matches a GT
    gradual interval ``[gs, ge)`` iff its covered frames ``[s, e−1]``
    intersect the one-frame-looser ``[gs−1, ge]`` (the official inclusive-edge
    rule mapped onto our half-open GT); TP = # GT gradual intervals
    overlapped; FP = #gradual-class events − TP. Zero confusion forgiveness:
    cross-type matches count nowhere.

    Returns ``{"cuts": {tp, preds, gts, precision, recall}, "graduals": …}``
    with raw counts so the caller can micro-sum across videos before taking
    ratios.
    """
    events = _normalize_events(pred_events)
    gts = sorted(int(g) for g in gt_cuts)
    gt_intervals = [(int(s), int(e)) for (s, e) in (gt_gradual or [])]
    cut_windows: list[tuple[int, int]] = []      # covered (lo, hi) inclusive
    gradual_intervals: list[tuple[int, int]] = []  # (s, e) half-open
    for frame, iv in events:
        if iv is None:
            cut_windows.append((frame, frame))
        elif iv[1] - iv[0] <= 2:                 # width <= 2 -> cut-class
            cut_windows.append((iv[0], iv[1] - 1))
        else:
            gradual_intervals.append(iv)

    tp_cuts = sum(1 for g in gts
                  if any(lo <= g + 1 and hi >= g - 1 for (lo, hi) in cut_windows))
    tp_graduals = sum(1 for (gs, ge) in gt_intervals
                      if any(s <= ge and e - 1 >= gs - 1
                             for (s, e) in gradual_intervals))
    return {"cuts": _official_block(tp_cuts, len(cut_windows), len(gts)),
            "graduals": _official_block(tp_graduals, len(gradual_intervals),
                                        len(gt_intervals))}


def _match_identities(pred_frames: Sequence[int], gt_cuts: Sequence[int],
                      gradual: Gradual | None, tolerance: int,
                      fade_first: bool = True) -> tuple[list[int], list[int]]:
    """Replicate :func:`evaluate_cuts` matching, returning identities (not
    just counts): ``(fp_frames, fn_gt_frames)``. With ``fade_first=False``
    gradual-region predictions stay in the hard-matching population — the
    near-miss diagnostic must SEE them to attribute them (the metric itself
    consumes them first). Kept in exact lockstep with ``evaluate_cuts``
    (same sort, consumption rule, greedy + tie semantics) so the diagnostics
    attribute the very TP/FP/FN partition the frozen metric reports.
    """
    preds = sorted(int(p) for p in pred_frames)
    gts = sorted(int(g) for g in gt_cuts)
    intervals = [(int(s), int(e)) for (s, e) in (gradual or [])]
    if fade_first:
        preds = [p for p in preds if not any(s <= p < e for (s, e) in intervals)]
    unmatched = list(gts)
    fp_frames: list[int] = []
    for p in preds:
        best: int | None = None
        best_d = -1
        for g in unmatched:  # ascending: strict '<' keeps the earlier GT on ties
            d = abs(p - g)
            if d <= tolerance and (best is None or d < best_d):
                best, best_d = g, d
        if best is None:
            fp_frames.append(p)
        else:
            unmatched.remove(best)
    return fp_frames, unmatched


def _near_miss_histogram(pred_frames: Sequence[int], gt_cuts: Sequence[int],
                         gradual: Gradual | None, tolerance: int) -> dict[str, Any]:
    """D14 §4 ``fp_near_miss_histogram`` — decompose the emission mass that
    hard matching rejects.

    Population: predictions with NO GT-cut match at ``tolerance`` under plain
    greedy matching (fade-first NOT applied — gradual-region emissions are
    exactly the population this diagnostic must see; the metric routes them
    to the gradual scorer). Each such emission is either inside a GT gradual
    interval ``[s, e)`` (counted in ``in_gt_gradual`` — attribution of
    gradual-region fires, not an error verdict) or bucketed by distance to
    the nearest GT cut. The ``0-1`` bucket (an addition over the spec's
    2-4/5-10/11-30/>30 set) catches greedy dupe extras — two emissions on
    one GT — so no FP vanishes from the decomposition; videos with no GT
    cuts land in ``>30``.
    """
    fp_frames, _fn = _match_identities(pred_frames, gt_cuts, gradual, tolerance,
                                       fade_first=False)
    buckets = {k: 0 for k in NEAR_MISS_BUCKETS}
    in_gradual = 0
    intervals = [(int(s), int(e)) for (s, e) in (gradual or [])]
    gts = [int(g) for g in gt_cuts]
    for p in fp_frames:
        if any(s <= p < e for (s, e) in intervals):
            in_gradual += 1
            continue
        d = min((abs(p - g) for g in gts), default=None)
        if d is None or d > 30:
            buckets[">30"] += 1
        elif d <= 1:
            buckets["0-1"] += 1
        elif d <= 4:
            buckets["2-4"] += 1
        elif d <= 10:
            buckets["5-10"] += 1
        else:
            buckets["11-30"] += 1
    return {"n": len(fp_frames), "buckets": buckets, "in_gt_gradual": in_gradual}


# ------------------------------------------------------------------ runner


def default_golden_path() -> Path:
    """Default location of the A15 golden floors: ``bench/golden.json``."""
    return Path(__file__).resolve().parents[2] / "bench" / "golden.json"


def run_bench(dataset: str,
              config_name: str = "default",
              params: dict[str, Any] | None = None,
              detect_fn: Callable[..., dict] | None = None,
              progress_cb: Callable[[int, int, str], None] | None = None,
              quick: bool = False,
              allow_proxy: bool = False,
              tolerances: Sequence[int] = TOLERANCES,
              entries: list[dict[str, Any]] | None = None,
              dataset_dir: str | Path | None = None) -> dict[str, Any]:
    """Run one (dataset × config) benchmark cell and return the report (D12).

    ``detect_fn`` defaults to :func:`scenecut.detect.detect_scenes` (imported
    lazily); ``params`` are merged over ``CONFIGS[config_name]` and passed
    through as kwargs. ``progress_cb(done_videos, total_videos, name)`` is
    called before each video. ``entries`` bypasses the dataset loader (unit
    tests / custom corpora). ``tolerances`` defaults to :data:`TOLERANCES`
    ``(0, 1)`` — external runs may request e.g. ``(0, 1, 5)`` for the optional
    TRECVid-convention τ=5 cell (D14 §6, never gated). Micro-aggregates
    TP/FP/FN across videos and only then computes ratios. The report schema is
    ``{scenecut_version, generated_at, config: {name, params}, tolerances,
    quick, datasets: {<name>: {meta, aggregate, per_video, evaluated_manifest}}}``
    plus the D14 additions: per-video ``transitions`` tol cells /
    ``transitions_gradual`` / ``width_histogram`` / ``n_width_capped`` /
    ``fp_by_source`` / ``fn_by_source`` / ``fp_near_miss_histogram`` (and
    ``clipshots_official`` on clipshots datasets), aggregate
    ``transitions@tolN`` / ``transitions_gradual`` / width + source +
    near-miss summed versions and ``clipshots_official``.
    """
    base = CONFIGS.get(config_name, {})
    merged: dict[str, Any] = {**base, **(dict(params) if params else {})}

    # A16: refuse proxy detection — proxy timelines shift cut positions.
    if merged.get("proxy") and not allow_proxy:
        raise ValueError(
            "bench refuses proxy detection paths (A16): proxies can shift cut "
            "positions; pass allow_proxy=True only after verifying proxy frame "
            "count and fps are identical to the source")

    if entries is None:
        entries = bench_datasets.load_dataset(dataset, dataset_dir=dataset_dir, quick=quick)
    if detect_fn is None:
        from .detect import detect_scenes  # lazy: keeps metric tests import-light
        detect_fn = detect_scenes
    tols = tuple(sorted({int(t) for t in tolerances}))
    max_tol = max(tols) if tols else 0

    per_video: list[dict[str, Any]] = []
    skipped: list[dict[str, str]] = []
    evaluated: list[str] = []
    # micro accumulators
    sums = {t: {"tp": 0, "fp": 0, "fn": 0} for t in tols}
    agg_offsets: dict[int, list[int]] = {t: [] for t in tols}
    agg_adjacent: dict[int, list[int]] = {t: [] for t in tols}
    frame_sums = {"tp": 0, "fp": 0, "fn": 0, "frames_scored": 0}
    gradual_sums = {"tp": 0, "fp": 0, "fn": 0}
    any_gradual = False
    # D14 accumulators: transitions family + diagnostics
    trans_sums = {t: {"tp": 0, "fp": 0, "fn": 0} for t in tols}
    agg_trans_offsets: dict[int, list[int]] = {t: [] for t in tols}
    trans_gradual_sums = {"tp": 0, "fp": 0, "fn": 0}
    any_trans_gradual = False
    fp_source_sums: dict[str, int] = {}
    fn_source_sums: dict[str, int] = {}
    near_miss_sums = {"n": 0, "in_gt_gradual": 0,
                      "buckets": {k: 0 for k in NEAR_MISS_BUCKETS}}
    official_sums = {"cuts": {"tp": 0, "preds": 0, "gts": 0},
                     "graduals": {"tp": 0, "preds": 0, "gts": 0}}
    any_official = False

    for i, entry in enumerate(entries):
        name = Path(str(entry["video_path"])).name
        if progress_cb is not None:
            progress_cb(i, len(entries), name)
        try:
            result = detect_fn(str(entry["video_path"]), **merged)
        except Exception as exc:  # A14: errors exclude the video, never abort
            skipped.append({"video": name, "reason": f"detect_error: {exc}"})
            log.warning("bench %s: skipped %s (detect error: %s)", dataset, name, exc)
            continue
        cuts = result.get("cuts", []) if isinstance(result, dict) else []
        # frame-0 events are never cut events (D12)
        pred_frames = sorted(int(c["frame_num"]) for c in cuts
                             if int(c.get("frame_num", 0) or 0) > 0)
        # D14: thread the FULL cut dicts through — optional `interval` fields
        # become transition events (points when absent); `source` fields feed
        # the FP/FN attribution diagnostics (pred_frames stays for the frozen
        # D12 metric call: dual reporting, never a replacement).
        pred_events = [{"frame": int(c["frame_num"]), "interval": c.get("interval")}
                       for c in cuts if int(c.get("frame_num", 0) or 0) > 0]
        source_of: dict[int, str] = {}
        for c in cuts:
            f = int(c.get("frame_num", 0) or 0)
            if f > 0:
                source_of.setdefault(f, str(c.get("source") or "unknown"))
        gt_cuts = [int(g) for g in entry.get("gt_cuts", [])]
        gradual = [(int(s), int(e)) for (s, e) in (entry.get("gt_gradual") or [])]
        n_frames = int((result.get("source") or {}).get("n_frames")
                       or (entry.get("meta") or {}).get("frames") or 0)

        # A14: ClipShots per-video best-offset verification — needs the
        # predictions, so it runs here (post-detection), applied to the GT.
        # Runs only when CUT GT exists: pure-negative videos (no GT at all)
        # and gradual-ONLY videos (gt_cuts == [], gradual spans only — 36/150
        # of only_gradual) have nothing to offset-resolve and MUST stay in the
        # bench (RV4 B3: excluding them biased the gradual corpus to cut-bearing
        # survivors). Gradual intervals are wide — a ±1 timeline offset is
        # immaterial for interval overlap.
        applied_offset = 0
        if dataset.startswith("clipshots") and gt_cuts:
            offset = bench_datasets.resolve_clipshots_offset(pred_frames, gt_cuts)
            if offset is None:
                skipped.append({"video": name,
                                "reason": "offset_ambiguous (A14: tie among {-1,0,+1})"})
                log.warning("bench clipshots: skipped %s (A14 offset ambiguous)", name)
                continue
            if offset != 0:
                applied_offset = offset
                gt_cuts = [g + offset for g in gt_cuts]
                gradual = [(s + offset, e + offset) for (s, e) in gradual]

        row: dict[str, Any] = {"video": name, "n_frames": n_frames,
                               "gt_cuts": len(gt_cuts), "pred_cuts": len(pred_frames),
                               "hard_cuts": {}, "transitions": {}}
        if applied_offset:
            row["clipshots_offset"] = applied_offset

        if n_frames > 0:
            fl = evaluate_frames(pred_frames, gt_cuts, gradual, n_frames)
            row["frame_level"] = dict(fl)
            for k in frame_sums:
                frame_sums[k] += fl[k]

        for t in tols:
            ev = evaluate_cuts(pred_frames, gt_cuts, gradual, tolerance=t)
            cell = _prf(ev["hard_cuts"]["tp"], ev["hard_cuts"]["fp"], ev["hard_cuts"]["fn"])
            cell["mean_abs_offset"] = round(ev["mean_abs_offset"], 4)
            row["hard_cuts"][f"tol{t}"] = cell
            for k in sums[t]:
                sums[t][k] += ev["hard_cuts"][k]
            agg_offsets[t].extend(ev["offsets"])
            agg_adjacent[t].extend(ev["adjacent_offsets"])
            if t == max_tol:
                row["mean_abs_offset"] = round(ev["mean_abs_offset"], 4)
                row["offset_histogram"] = _offset_histogram(ev["offsets"])
                row["gradual_adjacent_offset"] = (round(ev["gradual_adjacent_offset"], 4)
                                                  if ev["gradual_adjacent_offset"] is not None
                                                  else None)

            # D14 transitions family: interval-aware hard matching with the
            # width discipline; reported alongside hard_cuts (dual reporting).
            evt = evaluate_transitions(pred_events, gt_cuts, gradual, tolerance=t)
            tcell = _prf(evt["transitions"]["tp"], evt["transitions"]["fp"],
                         evt["transitions"]["fn"])
            tcell["mean_abs_offset"] = round(evt["mean_abs_offset"], 4)
            row["transitions"][f"tol{t}"] = tcell
            for k in trans_sums[t]:
                trans_sums[t][k] += evt["transitions"][k]
            agg_trans_offsets[t].extend(evt["offsets"])
            if t == max_tol:
                # width stats are tolerance-independent (clip is tau-free)
                row["n_width_capped"] = evt["n_width_capped"]
                row["width_histogram"] = dict(evt["width_histogram"])
                # gradual-aware gradual block (tolerance-independent: overlap)
                if gradual and evt["gradual"] is not None:
                    any_trans_gradual = True
                    for k in trans_gradual_sums:
                        trans_gradual_sums[k] += evt["gradual"][k]
                    row["transitions_gradual"] = dict(evt["gradual"])

        # CR2-4: gradual metrics are tolerance-INDEPENDENT (point-in-interval)
        # — accumulate ONCE per video, not once per tolerance (was doubled).
        if gradual:
            ev_g = evaluate_cuts(pred_frames, gt_cuts, gradual, tolerance=max_tol)
            if ev_g["gradual"] is not None:
                any_gradual = True
                for k in gradual_sums:
                    gradual_sums[k] += ev_g["gradual"][k]
                row["gradual"] = dict(ev_g["gradual"])
                row["gradual_consumed"] = ev_g["gradual_consumed"]

        # D14 §4 diagnostics (tol = max tol; primary-metric FP semantics).
        fp_frames, fn_gt_frames = _match_identities(pred_frames, gt_cuts, gradual,
                                                    tolerance=max_tol)
        fp_by_source: dict[str, int] = {}
        for p in fp_frames:
            src = source_of.get(p, "unknown")
            fp_by_source[src] = fp_by_source.get(src, 0) + 1
        fn_by_source: dict[str, int] = {}
        for g in fn_gt_frames:
            # a FN GT is attributed to the source of its NEAREST emission —
            # the detector family that fired closest and still missed
            if pred_frames:
                nearest = min(pred_frames, key=lambda p: (abs(p - g), p))
                src = source_of.get(nearest, "unknown")
            else:
                src = "none"
            fn_by_source[src] = fn_by_source.get(src, 0) + 1
        row["fp_by_source"] = fp_by_source
        row["fn_by_source"] = fn_by_source
        for k, v in fp_by_source.items():
            fp_source_sums[k] = fp_source_sums.get(k, 0) + v
        for k, v in fn_by_source.items():
            fn_source_sums[k] = fn_source_sums.get(k, 0) + v
        nmh = _near_miss_histogram(pred_frames, gt_cuts, gradual, tolerance=max_tol)
        row["fp_near_miss_histogram"] = nmh
        near_miss_sums["n"] += nmh["n"]
        near_miss_sums["in_gt_gradual"] += nmh["in_gt_gradual"]
        for k in near_miss_sums["buckets"]:
            near_miss_sums["buckets"][k] += nmh["buckets"][k]

        # D14 §5 clipshots_official compat block — ONLY on gradual-GT corpora
        # (clipshots splits): keep local/bbc/rai reports free of the compat
        # metric so it is never mixed with the scenecut-native families.
        if dataset.startswith("clipshots"):
            off = evaluate_clipshots_official(pred_events, gt_cuts, gradual)
            row["clipshots_official"] = off
            any_official = True
            for family in ("cuts", "graduals"):
                for k in ("tp", "preds", "gts"):
                    official_sums[family][k] += off[family][k]

        per_video.append(row)
        evaluated.append(name)

    if progress_cb is not None and entries:
        progress_cb(len(entries), len(entries), "")

    notes: list[str] = []
    if dataset == "local":
        notes.append("hard-cut-only scoring: dissolve/fade positions treated as "
                     "point cuts; gt_gradual=[] (local corpus)")
    if dataset.startswith("clipshots"):
        notes.append("A14 per-video best-offset {-1,0,+1} verification applied to GT")
        notes.append("clipshots_official (Tang et al. compat) reported separately "
                     "— never compare with scenecut-native families")
    if dataset.startswith("rai"):
        notes.append("scene-level GT: recall comparable, precision is a LOWER "
                     "BOUND (intra-scene shot cuts are unannotated)")

    aggregate: dict[str, Any] = {}
    for t in tols:
        cell = _prf(sums[t]["tp"], sums[t]["fp"], sums[t]["fn"])
        offs = agg_offsets[t]
        cell["mean_abs_offset"] = round(sum(offs) / len(offs), 4) if offs else 0.0
        aggregate[f"hard_cuts@tol{t}"] = cell
        tcell = _prf(trans_sums[t]["tp"], trans_sums[t]["fp"], trans_sums[t]["fn"])
        toffs = agg_trans_offsets[t]
        tcell["mean_abs_offset"] = round(sum(toffs) / len(toffs), 4) if toffs else 0.0
        aggregate[f"transitions@tol{t}"] = tcell
    aggregate["frame_level"] = dict(frame_sums)
    aggregate["mean_abs_offset"] = aggregate[f"hard_cuts@tol{max_tol}"]["mean_abs_offset"]
    if any_gradual:
        aggregate["gradual"] = _prf(gradual_sums["tp"], gradual_sums["fp"], gradual_sums["fn"])
    adj = agg_adjacent[max_tol] if max_tol in agg_adjacent else []
    aggregate["gradual_adjacent_offset"] = round(sum(adj) / len(adj), 4) if adj else None

    # D14 aggregate additions: gradual-aware family + diagnostics.
    if any_trans_gradual:
        aggregate["transitions_gradual"] = _prf(trans_gradual_sums["tp"],
                                                trans_gradual_sums["fp"],
                                                trans_gradual_sums["fn"])
    aggregate["n_width_capped"] = sum(int(r.get("n_width_capped", 0)) for r in per_video)
    width_agg = {k: 0 for k in WIDTH_BUCKETS}
    for r in per_video:
        for k, v in (r.get("width_histogram") or {}).items():
            width_agg[k] = width_agg.get(k, 0) + v
    aggregate["width_histogram"] = width_agg
    aggregate["fp_by_source"] = dict(fp_source_sums)
    aggregate["fn_by_source"] = dict(fn_source_sums)
    aggregate["fp_near_miss_histogram"] = {"n": near_miss_sums["n"],
                                            "buckets": dict(near_miss_sums["buckets"]),
                                            "in_gt_gradual": near_miss_sums["in_gt_gradual"]}
    if any_official:
        aggregate["clipshots_official"] = {
            family: _official_block(official_sums[family]["tp"],
                                    official_sums[family]["preds"],
                                    official_sums[family]["gts"])
            for family in ("cuts", "graduals")
        }

    meta = {
        "dataset": dataset,
        "n_videos_total": len(entries),
        "n_videos_evaluated": len(evaluated),
        "n_videos_skipped": len(skipped),
        "quick": bool(quick),
        "tolerances": list(tols),
        "notes": notes,
    }

    manifest = {"dataset": dataset, "videos": sorted(evaluated), "skipped": skipped}

    return {
        "scenecut_version": __version__,
        "generated_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "config": {"name": config_name, "params": merged},
        "tolerances": list(tols),
        "quick": bool(quick),
        "datasets": {
            dataset: {
                "meta": meta,
                "aggregate": aggregate,
                "per_video": per_video,
                "evaluated_manifest": manifest,
            },
        },
    }


# ------------------------------------------------------------------ gate (A15)


def run_gate(report: dict, golden_path: str | Path,
             quick: bool | None = None) -> dict[str, Any]:
    """A15 regression gate: compare the local corpus against golden floors.

    Diffs the report's ``evaluated_manifest`` against the golden manifest —
    any floored video missing or skipped → ``{"gate": "fail_missing"}``
    (exit-code 3 signal). Aggregate or per-fixture F1@tol1 below a golden
    floor → ``{"gate": "fail_floor"}`` (exit-code 2 signal). Otherwise
    ``{"gate": "pass"}`` (exit code 0). ``--quick`` uses the separate
    ``local_quick`` golden key (A18). Returns a dict with ``gate``,
    ``exit_code`` and diagnostic details.
    """
    if quick is None:
        quick = bool(report.get("quick", False))
    key = "local_quick" if quick else "local"
    golden = json.loads(Path(golden_path).read_text())
    gblock = golden.get(key)
    if gblock is None:
        return {"gate": "fail_missing", "exit_code": 3, "golden_key": key,
                "reason": f"golden.json has no '{key}' block",
                "missing_videos": [], "fixture_failures": []}

    local = (report.get("datasets") or {}).get("local")
    if not local:
        return {"gate": "fail_missing", "exit_code": 3, "golden_key": key,
                "reason": "report has no 'local' dataset block",
                "missing_videos": [], "fixture_failures": []}

    # --- manifest integrity: floored videos must all be evaluated.
    golden_videos = set((gblock.get("manifest") or {}).get("videos", []))
    evaluated = set((local.get("evaluated_manifest") or {}).get("videos", []))
    missing = sorted(golden_videos - evaluated)
    if missing:
        return {"gate": "fail_missing", "exit_code": 3, "golden_key": key,
                "reason": "floored videos missing/skipped from the evaluated manifest",
                "missing_videos": missing, "fixture_failures": []}

    tols = report.get("tolerances") or [1]
    if not tols:          # CR3: empty tolerances list → max() KeyError
        tols = [1]
    tol_key = "tol1" if 1 in tols else f"tol{max(tols)}"

    # --- aggregate floor.
    failures: list[dict[str, Any]] = []
    agg_f1 = ((local.get("aggregate") or {}).get(f"hard_cuts@{tol_key}") or {}).get("f1")
    agg_floor = (gblock.get("aggregate") or {}).get("floor")
    if agg_f1 is None or agg_floor is None:
        return {"gate": "fail_missing", "exit_code": 3, "golden_key": key,
                "reason": "aggregate F1 or floor missing (golden/report mismatch)",
                "missing_videos": [], "fixture_failures": []}
    if agg_f1 < agg_floor:
        failures.append({"scope": "aggregate", "f1": agg_f1, "floor": agg_floor})

    # --- per-fixture floors.
    rows = {r.get("video"): r for r in local.get("per_video", [])}
    for fname, fblock in sorted((gblock.get("fixtures") or {}).items()):
        floor = fblock.get("floor", 0.0)
        row = rows.get(fname)
        f1 = None
        if row is not None:
            f1 = ((row.get("hard_cuts") or {}).get(tol_key) or {}).get("f1")
        if f1 is None or f1 < floor:
            failures.append({"scope": "fixture", "video": fname, "f1": f1, "floor": floor})

    if failures:
        return {"gate": "fail_floor", "exit_code": 2, "golden_key": key,
                "aggregate_f1": agg_f1, "aggregate_floor": agg_floor,
                "fixture_failures": failures, "missing_videos": []}
    return {"gate": "pass", "exit_code": 0, "golden_key": key,
            "aggregate_f1": agg_f1, "aggregate_floor": agg_floor,
            "fixture_failures": [], "missing_videos": []}
