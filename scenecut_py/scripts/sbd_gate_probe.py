#!/usr/bin/env python3
"""Diagnostic: why do SBD T-clips (gradual transitions) miss tier-1 emission?

Dumps the three tier-1 dissolve gates (endpoint JS, window MAD max, consec-JS
mean) over the center region of missed-T clips, per window size, so the
blocking gate is visible per clip. Also reports what tier-2/spatial sees.

Usage: python3 sbd_gate_probe.py [--n 8] [--label T] [--missed-only]
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT / "scenecut_py"))

import numpy as np

from scenecut.detect import _feature_pass

WINDOWS = [8, 16, 32, 64]
GATES = {"dissim_min": 0.12, "hard_cut_mad_min": 20.0, "consec_js_min": 0.15}


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=8)
    ap.add_argument("--label", default="T")
    ap.add_argument("--missed-only", action="store_true", default=True)
    ap.add_argument("--hit", action="store_true", help="probe gradual-HIT clips instead")
    args = ap.parse_args()

    state = json.loads((REPO_ROOT / "bench" / ".external_sbd_default.state.json").read_text())
    man = {s["clip"]: s for s in json.loads(
        (REPO_ROOT / "datasets" / "sbd" / "gt.json").read_text())["samples"]}

    pool = []
    for r in state["rows"]:
        name = r["video"]
        lab = name.rsplit("_", 1)[1][0]
        if lab != args.label:
            continue
        gr = r.get("transitions_gradual") or {}
        is_hit = bool(gr.get("tp"))
        if args.hit == is_hit:
            pool.append((name, man.get(name, {}).get("origin", "?")))
        if len(pool) >= args.n:
            break

    print(f"probing {len(pool)} label-{args.label} clips "
          f"({'HITS' if args.hit else 'MISSED'}), windows={WINDOWS}, gates={GATES}")
    for name, origin in pool:
        path = REPO_ROOT / "datasets" / "sbd" / "clips" / name
        feat = _feature_pass(str(path), WINDOWS, 30.0)
        n = min(feat.n, 61)
        center = 30
        print(f"\n=== {name} [{origin}] n={n}")
        for w in sorted(WINDOWS):
            if w >= n:
                continue
            lag = feat.lagged_js.get(w)
            if lag is None:
                print(f"  w={w}: (no lag array)")
                continue
            # best window centered near 30: i + w//2 ~= 30
            i_center = max(0, center - w // 2)
            rows = []
            for i in range(max(0, center - w - 4), min(n - w, center + w + 4)):
                ejs = float(lag[i])
                wmad = float(np.max(feat.mad[i:i + w])) if i + w <= n else float("nan")
                cjs = float(np.mean(feat.consec_js[i:i + w])) if i + w <= n else float("nan")
                mark = []
                if ejs >= GATES["dissim_min"]:
                    mark.append("E+")
                if wmad < GATES["hard_cut_mad_min"]:
                    mark.append("M+")
                if cjs >= GATES["consec_js_min"]:
                    mark.append("J+")
                if len(mark) == 3:
                    mark.append("*** CANDIDATE")
                rows.append(f"    i={i:3d} endJS={ejs:.3f} MADmax={wmad:6.1f} cJS={cjs:.3f} {' '.join(mark)}")
            print(f"  --- w={w} (center-aligned i={i_center}):")
            for line in rows:
                if "***" in line or args.n <= 3:
                    print(line)
        # spatial tier view: block-MAD endpoints + consec around center
        if hasattr(feat, "spatial_lag") and 8 in feat.spatial_lag:
            sl = feat.spatial_lag[8]
            print(f"  spatial w=8: lag[26..34] = "
                  f"{[round(float(sl[i]),1) for i in range(26, min(34, n-8))]}")
        if hasattr(feat, "spatial_consec"):
            print(f"  spatial_consec[24..36] = "
                  f"{[round(float(feat.spatial_consec[i]),1) for i in range(24, min(36, n))]}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
