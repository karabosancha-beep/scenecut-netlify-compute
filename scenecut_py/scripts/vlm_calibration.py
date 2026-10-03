#!/usr/bin/env python3
"""D17 VLM seam-audit calibration driver — live experiment on the BBC ep01
snow-window census (300-420s subclip, 13 seams, GT false rate 1/13=0.077).

Protocol (R7 regression test):
  Stage 1: subset audit (--subset 6) — 4 GT-genuine + GT-false + a borderline
  Stage 3: full census (default) — target p_hat <= 0.23 (+-0.15 of GT 0.077)

Usage:
  python3 scripts/vlm_calibration.py [--subset N] [--out PATH]
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "scenecut_py"))

from scenecut.vlm import VLMClient  # noqa: E402

MANIFEST = Path("/home/z/calib/seams/manifest.json")
GT_CUT_FRAMES = {60, 269, 408, 887, 1041, 1244, 1394, 1528, 1880, 2314, 2545, 2814}
# detected census frames: 60,269,408,541,877,1041,1244,1394,1528,1880,2314,2545,2814
# (877 matches GT 887 within +-25; 541 is the sole GT-false)
CENSUS_FRAMES = [60, 269, 408, 541, 877, 1041, 1244, 1394, 1528, 1880, 2314, 2545, 2814]


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--subset", type=int, default=0, help="audit only first N seams")
    ap.add_argument("--out", type=str, default="/home/z/calib/calibration_result.json")
    ap.add_argument("--workers", type=int, default=3)
    args = ap.parse_args()

    manifest = json.load(open(MANIFEST))
    seams = manifest if isinstance(manifest, list) else manifest.get("seams", manifest)
    units = []
    for s in seams:
        frame = CENSUS_FRAMES[s["cut_index"]] if s["cut_index"] < len(CENSUS_FRAMES) else None
        # m7: match within +-25 frames (the bench convention) — exact set
        # membership mislabels displaced TPs (census f877 matches GT f887).
        matched = frame is not None and any(abs(frame - g) <= 25 for g in GT_CUT_FRAMES)
        gt = "GT-genuine" if matched else "GT-false"
        clip_path = str(MANIFEST.parent / s.get("file", s.get("path", "")))
        units.append({
            "key": clip_path,
            "cut_index": s["cut_index"],
            "path": clip_path,
            "multicent": s.get("multicent"),
            "gt": gt,
            "frame": frame,
            "cut_time": s.get("cut_time"),
        })
    if args.subset:
        units = units[:args.subset]

    client = VLMClient(workers=args.workers)
    verdicts = client.audit_seams_batch(units)

    n_false = n_genuine = n_unc = n_abs = 0
    rows = []
    for v, u in zip(verdicts, units):
        if v.verdict == "false":
            n_false += 1
        elif v.verdict == "genuine":
            n_genuine += 1
        elif v.agreement == "uncertain":
            n_unc += 1
        else:
            n_abs += 1
        rows.append({
            "cut_index": u["cut_index"], "frame": u["frame"], "gt": u["gt"],
            "verdict": v.verdict, "agreement": v.agreement,
            "confidence": v.confidence, "calls_used": v.calls_used,
            "reason": (v.raw[-1][:220] if v.raw else ""),
        })
        print(f"seam {u['cut_index']:>2} f={u['frame']:>4} {u['gt']:<12} "
              f"-> {str(v.verdict):<8} ({v.agreement}, conf={v.confidence})")
        if v.raw:
            print(f"         reason: {v.raw[-1][:200].replace(chr(10), ' ')}")

    n_aud = n_genuine + n_false
    p_hat = n_false / n_aud if n_aud else None
    n = len(units)
    print(f"\ncensus n={n}: genuine={n_genuine} false={n_false} "
          f"uncertain={n_unc} infra-abstain={n_abs}")
    print(f"p_hat = {p_hat} (target <= 0.23; GT false rate 0.077)")
    print(f"calls_used = {client.calls_used}")

    json.dump({"p_hat": p_hat, "n": n, "n_genuine": n_genuine, "n_false": n_false,
               "n_uncertain": n_unc, "n_abstain": n_abs, "calls_used": client.calls_used,
               "rows": rows}, open(args.out, "w"), indent=2)
    print(f"saved -> {args.out}")


if __name__ == "__main__":
    main()
