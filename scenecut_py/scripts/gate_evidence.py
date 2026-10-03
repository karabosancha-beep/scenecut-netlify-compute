#!/usr/bin/env python3
"""Full evidence base for the corroboration-gate design (D16, v4.1 spec).

RV4 B2/B4 corrections vs v1:
  * gate population = SOLO-hash cuts (``len(sources)==1 and sources[0]=="hash"``
    post-dedup) — NOT "hash-labeled" (SOURCE_PRIORITY lets corroborated
    hash+threshold/dissolve clusters wear the hash label).
  * spike_ratio uses the SPEC formula: mad[p] / max(median(mad[p-30 .. p-5]), 0.5)
    — the v1 `None`-when-quiet guard silently diverged from the spec.
  * sensitivity sweep over the three constants (spike, js, luma).
  * p < 6 (no baseline frames): gate skipped (video-head conservatism) — counted.
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

SPEC_RULE = ("spike3_or_js15_or_luma8")


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("video")
    ap.add_argument("--dataset", default="bbc")
    ap.add_argument("--tol", type=int, default=1)
    ap.add_argument("--out", default=None)
    args = ap.parse_args()

    from scenecut.bench_datasets import load_dataset
    from scenecut.detect import detect_scenes, _feature_pass

    entry = next(e for e in load_dataset(args.dataset)
                 if Path(e["video_path"]).name == Path(args.video).name)
    result = detect_scenes(entry["video_path"])
    cuts = result.get("cuts", [])
    preds = sorted(int(c["frame_num"]) for c in cuts
                   if int(c.get("frame_num", 0) or 0) > 0)
    by_frame: dict[int, dict] = {}
    for c in cuts:
        by_frame.setdefault(int(c["frame_num"]), c)
    gt = sorted(entry["gt_cuts"])

    unmatched = list(gt)
    pair_of: dict[int, int] = {}
    for p in preds:
        best, best_d = None, -1
        for g in unmatched:
            d = abs(p - g)
            if d <= args.tol and (best is None or d < best_d):
                best, best_d = g, d
        if best is not None:
            unmatched.remove(best)
            pair_of[p] = best

    feat = _feature_pass(entry["video_path"], [8, 16, 32], fps=25.0)
    n = feat.n

    def site(p: int) -> dict:
        if not (0 < p < n):
            return {"spike_ratio": None, "luma_step": None, "js_step": None,
                    "motion_env": None, "gate_skipped_head": True}
        lo = max(0, p - 30)
        base = feat.mad[lo:max(lo + 1, p - 4)]  # frames p-30 .. p-5
        med = float(np.median(base)) if len(base) else 0.0
        spike = float(feat.mad[p])
        env = feat.mad[max(0, p - 15):min(n, p + 16)]
        return {
            # SPEC formula (v4.1): floored baseline
            "spike_ratio": round(spike / max(med, 0.5), 2),
            "luma_step": round(abs(float(feat.mean_v[p]) - float(feat.mean_v[p - 1])), 1),
            "js_step": round(float(feat.consec_js[p - 1]), 3),
            "motion_env": round(float(np.mean(env)), 1) if len(env) else None,
            "gate_skipped_head": p < 6,
        }

    rows = []
    for p in preds:
        c = by_frame.get(p, {})
        srcs = c.get("sources") or [c.get("source")]
        r = {"frame": p, "source": c.get("source"),
             "sources": srcs,
             "solo_hash": len(srcs) == 1 and srcs[0] == "hash",
             "type": c.get("type"), "confidence": c.get("confidence"),
             "outcome": "TP" if p in pair_of else "FP"}
        r.update(site(p))
        rows.append(r)

    def summarize(sub: list[dict]) -> dict:
        def med(k):
            v = sorted(x[k] for x in sub if x.get(k) is not None)
            return round(statistics.median(v), 2) if v else None
        def p10(k):
            v = sorted(x[k] for x in sub if x.get(k) is not None)
            return round(v[max(0, len(v) // 10)], 2) if v else None
        return {"n": len(sub), "spike_ratio_med": med("spike_ratio"),
                "spike_ratio_p10": p10("spike_ratio"),
                "luma_step_med": med("luma_step"), "js_step_med": med("js_step"),
                "motion_env_med": med("motion_env")}

    solo_hash_tp = [r for r in rows if r["solo_hash"] and r["outcome"] == "TP"]
    solo_hash_fp = [r for r in rows if r["solo_hash"] and r["outcome"] == "FP"]
    corroborated = [r for r in rows if not r["solo_hash"]]

    def gate_sim(spike_min: float, js_min: float, luma_min: float):
        def keep(r):
            if r.get("gate_skipped_head"):
                return True
            return ((r["spike_ratio"] or 0) >= spike_min
                    or (r["js_step"] or 0) >= js_min
                    or (r["luma_step"] or 0) >= luma_min)
        tp_k = sum(1 for r in solo_hash_tp if keep(r))
        fp_k = sum(1 for r in solo_hash_fp if keep(r))
        return {"solo_hash_tp_kept": tp_k, "solo_hash_tp_total": len(solo_hash_tp),
                "solo_hash_fp_kept": fp_k, "solo_hash_fp_total": len(solo_hash_fp),
                "tp_dropped": len(solo_hash_tp) - tp_k,
                "fp_dropped": len(solo_hash_fp) - fp_k,
                "trade_ratio": round((len(solo_hash_fp) - fp_k)
                                     / max(1, len(solo_hash_tp) - tp_k), 1)}

    sweeps = {}
    for (s, j, l) in [(3, .15, 8), (2, .15, 8), (4, .15, 8), (3, .1, 8),
                      (3, .2, 8), (3, .15, 5), (3, .15, 12), (2, .1, 5),
                      (4, .2, 12), (1.5, .15, 8), (6, .15, 8)]:
        sweeps[f"spike{s}_js{j}_luma{l}"] = gate_sim(s, j, l)

    out = {
        "video": Path(args.video).name, "tol": args.tol,
        "spec_rule": SPEC_RULE,
        "n_pred": len(preds), "n_gt": len(gt),
        "summary": {
            "solo_hash/TP": summarize(solo_hash_tp),
            "solo_hash/FP": summarize(solo_hash_fp),
            "corroborated/TP": summarize([r for r in corroborated
                                          if r["outcome"] == "TP"]),
            "corroborated/FP": summarize([r for r in corroborated
                                          if r["outcome"] == "FP"]),
        },
        "gate_simulations": sweeps,
        "rows": rows,
    }
    print(json.dumps({"summary": out["summary"],
                      "gate_simulations": sweeps}, indent=1))
    if args.out:
        Path(args.out).write_text(json.dumps(out, indent=1))
        print(f"written: {args.out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
