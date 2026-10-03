#!/usr/bin/env python3
"""Characterize cut sites against the feature arrays — FP-vs-TP signatures.

For each queried frame site, extracts:
  spike_ratio   mad[p] / median(mad[p-30 .. p-5])   (outlier vs local baseline)
  luma_step     |mean_v[p] - mean_v[p-1]|
  js_step       consec_js[p-1]                       (hist change at the cut)
  motion_env    mean(mad[p-15 .. p+15])              (sustained elevation)
  edge_step     edge[p] / median(edge[p-30 .. p-5])

Usage: python3 fp_site_probe.py <video> --sites f1,f2,... --labels A,B,...
"""
from __future__ import annotations

import argparse
import json
import statistics
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT / "scenecut_py"))

import numpy as np


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("video")
    ap.add_argument("--sites", required=True, help="comma list of frames")
    ap.add_argument("--labels", required=True, help="comma list (TP/FP/...)")
    ap.add_argument("--json", dest="json_out", default=None)
    args = ap.parse_args()

    from scenecut.detect import _feature_pass

    feat = _feature_pass(args.video, [8, 16, 32], fps=25.0)
    n = feat.n

    def site_metrics(p: int) -> dict:
        lo = max(0, p - 30)
        base = feat.mad[lo:max(lo + 1, p - 4)]
        med = float(np.median(base)) if len(base) else 0.0
        spike = float(feat.mad[p]) if p < n else 0.0
        edge_base = feat.edge[lo:max(lo + 1, p - 4)]
        edge_med = float(np.median(edge_base)) if len(edge_base) else 0.0
        env = feat.mad[max(0, p - 15):min(n, p + 16)]
        return {
            "frame": p,
            "mad": round(spike, 1),
            "mad_base_med": round(med, 1),
            "spike_ratio": round(spike / med, 2) if med > 0.5 else None,
            "luma_step": round(abs(float(feat.mean_v[p]) - float(feat.mean_v[p - 1])), 1)
                         if 0 < p < n else None,
            "js_step": round(float(feat.consec_js[p - 1]), 3) if 0 < p < n else None,
            "motion_env": round(float(np.mean(env)), 1) if len(env) else None,
            "edge_ratio": round(float(feat.edge[p]) / edge_med, 2)
                          if edge_med > 1 and p < n else None,
        }

    sites = [int(s) for s in args.sites.split(",")]
    labels = args.labels.split(",")
    assert len(sites) == len(labels), "sites/labels length mismatch"
    rows = []
    for p, lab in zip(sites, labels):
        m = site_metrics(p)
        m["label"] = lab
        rows.append(m)

    def agg(lab: str) -> dict:
        sub = [r for r in rows if r["label"] == lab]
        if not sub:
            return {}
        def med(key):
            vals = [r[key] for r in sub if r[key] is not None]
            return round(statistics.median(vals), 2) if vals else None
        def p75(key):
            vals = sorted(r[key] for r in sub if r[key] is not None)
            return round(vals[3 * len(vals) // 4], 2) if vals else None
        return {"n": len(sub), "spike_ratio_med": med("spike_ratio"),
                "spike_ratio_p75": p75("spike_ratio"),
                "luma_step_med": med("luma_step"), "js_step_med": med("js_step"),
                "motion_env_med": med("motion_env"),
                "mad_med": med("mad"), "mad_base_med": med("mad_base_med")}

    summary = {lab: agg(lab) for lab in set(labels)}
    print(json.dumps(summary, indent=1))
    if args.json_out:
        Path(args.json_out).write_text(json.dumps(
            {"summary": summary, "rows": rows}, indent=1))
        print(f"written: {args.json_out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
