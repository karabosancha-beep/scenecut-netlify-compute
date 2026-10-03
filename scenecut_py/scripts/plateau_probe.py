#!/usr/bin/env python3
"""Prototype: plateau-refined dissolve emission (D16§3 v2 — blend-end
estimation). Measures the spec's predictions BEFORE implementation:

  itest_dissolve    emission 75 -> plateau_end+1 == 80 (GT exact)
  itest_match_dissolve (xfail case) — does plateau refinement fire?
  itest_fade / fade_white — plateau semantics on fades (should NOT fire;
                    fades are detected by the luma scan, not this path)
  ep01 chain sites  multi-emission chains collapse to one refined emission

Rule (v4.1): given a dissolve candidate at window [i, i+w), mid m:
  plateau_end   = last t in [m-w_max, m+w_max] with consec_js[t] >= eps
                  (scanning outward from m; sustained = the contiguous run)
  plateau_start = first t of that contiguous elevated run
  emission      = plateau_end + 1   (first clean frame)
  interval      = [plateau_start, plateau_end + 1)
  fallback      = window mid, interval [i, i+w) when no clean edges
"""
from __future__ import annotations

import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT / "scenecut_py"))

import numpy as np

EPS = 0.15          # consec_js_min — the plateau threshold (pinned constant)


def refine(feat, i: int, w: int, w_max: int) -> dict:
    m = i + w // 2
    lo = max(0, m - w_max)
    hi = min(feat.n - 1, m + w_max)
    js = feat.consec_js[lo:hi + 1]
    mid_idx = m - lo
    if mid_idx >= len(js) or js[mid_idx] < EPS:
        # midpoint not elevated: no measurable plateau at the candidate
        return {"emission": i + w // 2, "interval": [i, i + w],
                "fallback": "mid-not-elevated"}
    # expand the contiguous elevated run around the mid
    a = mid_idx
    while a > 0 and js[a - 1] >= EPS:
        a -= 1
    b = mid_idx
    while b < len(js) - 1 and js[b + 1] >= EPS:
        b += 1
    plateau_start = lo + a
    plateau_end = lo + b
    return {"emission": plateau_end + 1,
            "interval": [plateau_start, plateau_end + 1],
            "fallback": None}


def main() -> int:
    from scenecut.detect import _feature_pass, _detect_dissolves

    samples = REPO_ROOT / "scenecut_py" / "samples"
    for name in ("itest_dissolve.mp4", "itest_match_dissolve.mp4",
                 "itest_fade.mp4", "itest_fade_white.mp4",
                 "itest_pan.mp4", "itest_sunset.mp4"):
        feat = _feature_pass(str(samples / name), [8, 16, 32], fps=30.0)
        cands = _detect_dissolves(feat, [8, 16, 32], 0.12, 20.0, 0.15,
                                  0.6, 30.0)
        print(f"\n{name}: {len(cands)} dissolve candidate(s)")
        for c in cands:
            f = c["frame_num"]
            # recover the window(s) that produced it: try each w
            for w in (8, 16, 32):
                # the candidate came from SOME window; refine as if from
                # each to show convergence
                r = refine(feat, f - w // 2, w, 32)
                print(f"  cand@{f} (conf {c['confidence']:.2f}) from w={w}: "
                      f"emission {r['emission']} interval {r['interval']} "
                      f"{r['fallback'] or ''}")

    # ep01 chain site: 17078/17093/17108 (15-frame hash chains) + a real
    # gradual site — show consec_js shape
    print("\nep01 consec_js around chain site 17078-17108:")
    feat = _feature_pass(str(REPO_ROOT / "datasets/bbc/bbc_01.mp4"),
                         [8, 16, 32], fps=25.0)
    js = feat.consec_js
    for t in range(17060, 17130, 5):
        bar = "#" * int(min(js[t], 1.0) * 40)
        print(f"  t={t} js={js[t]:.3f} {bar}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
