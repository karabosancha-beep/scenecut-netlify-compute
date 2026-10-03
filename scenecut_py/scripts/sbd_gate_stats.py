#!/usr/bin/env python3
"""SBD gate-statistics collector — data for D19 (external gradual recalibration).

For every SBD clip (C/T/E) extracts center-region statistics of the three
tier-1 dissolve gates plus quiet-baseline references:

  endJS_best[w]  — max endpoint JS over center windows (per w in 8/16/32)
  mad_center     — max MAD over the center 30 frames [15,45)
  cjs_center     — mean consec-JS over [20,40)
  mad_base       — median MAD over quiet sides [0,10)+[51,61)
  cjs_base       — median consec-JS over quiet sides
  spatial_center — max spatial_consec over [20,40)

Output: bench/sbd_gate_stats.json (rows per clip, label/origin joined) +
printed per-label distributions for threshold separation analysis.
"""
from __future__ import annotations

import json
import sys
from collections import defaultdict
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT / "scenecut_py"))

import numpy as np

from scenecut.detect import _feature_pass

WINDOWS = [8, 16, 32]


def region_stats(feat, n: int) -> dict:
    out = {}
    for w in WINDOWS:
        if w >= n:
            continue
        lag = feat.lagged_js[w]
        i_lo, i_hi = max(0, 15), min(n - w, 45)
        if i_hi > i_lo:
            out[f"endjs_w{w}"] = round(float(np.max(lag[i_lo:i_hi])), 4)
    mad = feat.mad[:n]
    cjs = feat.consec_js[:n]
    out["mad_center"] = round(float(np.max(mad[15:min(45, n)])), 2)
    out["cjs_center"] = round(float(np.mean(cjs[20:min(40, n)])), 4)
    quiet = np.concatenate([mad[:min(10, n)], mad[min(51, n):n]])
    quietc = np.concatenate([cjs[:min(10, n)], cjs[min(51, n):n]])
    out["mad_base"] = round(float(np.median(quiet)), 2)
    out["cjs_base"] = round(float(np.median(quietc)), 4)
    sc = getattr(feat, "spatial_consec", None)
    if sc is not None:
        out["spatial_center"] = round(float(np.max(sc[20:min(40, n)])), 2)
        out["spatial_base"] = round(float(np.median(
            np.concatenate([sc[:min(10, n)], sc[min(51, n):n]]))), 2)
    return out


def main() -> int:
    man = json.loads((REPO_ROOT / "datasets" / "sbd" / "gt.json").read_text())
    rows = []
    for idx, s in enumerate(man["samples"]):
        if s["label"] not in ("C", "T", "E"):
            continue
        path = REPO_ROOT / "datasets" / "sbd" / "clips" / s["clip"]
        if not path.is_file():
            continue
        feat = _feature_pass(str(path), WINDOWS, 30.0)
        n = min(feat.n, 61)
        row = {"clip": s["clip"], "label": s["label"],
               "origin": s["origin"], "real": s["real"],
               **region_stats(feat, n)}
        rows.append(row)
        if (idx + 1) % 100 == 0:
            print(f"  ... {idx+1}/{len(man['samples'])}", flush=True)

    out_path = REPO_ROOT / "bench" / "sbd_gate_stats.json"
    out_path.write_text(json.dumps(rows, indent=1))

    # per-label distributions
    def q(vals, ps):
        v = sorted(vals)
        return {f"p{int(p*100)}": round(v[min(len(v)-1, int(p*len(v)))], 3)
                for p in ps}
    PS = (0.05, 0.25, 0.5, 0.75, 0.95)
    by_label = defaultdict(list)
    for r in rows:
        by_label[r["label"]].append(r)
    for lab, rs in sorted(by_label.items()):
        print(f"\n=== label {lab} (n={len(rs)})")
        for key in ("endjs_w16", "mad_center", "cjs_center", "mad_base",
                    "cjs_base", "spatial_center"):
            vals = [r[key] for r in rs if key in r]
            if vals:
                print(f"  {key:15} {q(vals, PS)}")
        # real-only for T
        if lab == "T":
            vals = [r["cjs_center"] for r in rs if r["real"] and "cjs_center" in r]
            print(f"  cjs_center REAL {q(vals, PS)}")
            vals = [r["mad_center"] for r in rs if r["real"] and "mad_center" in r]
            print(f"  mad_center REAL {q(vals, PS)}")
    print(f"\nwrote {out_path} ({len(rows)} rows)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
