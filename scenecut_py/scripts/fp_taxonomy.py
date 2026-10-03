#!/usr/bin/env python3
"""FP/FN taxonomy for a benched video — the D14/D16 design input.

Classifies every false positive and false negative from a detection run so
detector fixes are driven by measured error structure, not guesses:

  FP classes:
    near_miss     |p - nearest unmatched GT| in 2..4   (offset/emission)
    mid_range     |p - nearest GT| in 5..30            (gradual displacement
                                                       hypothesis zone)
    isolated      |p - nearest GT| > 30                (true FP: motion/flash)
  FP attributes:
    source/type   the cut's own `source` + `type` (adaptive/hash/dissolve/
                  fade_white/...; cut/dissolve/fade)
    clustered     another FP within +-CLUSTER_RADIUS frames (multi-emission)
  FN classes:
    near_miss     nearest pred in 2..4
    mid_range     nearest pred in 5..30
    missed        nearest pred > 30 (or none)

Usage:
  python3 fp_taxonomy.py <video> [--gt <dataset>:<key>] [--tol 1] [--json out]
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT / "scenecut_py"))

CLUSTER_RADIUS = 16  # the dissolve NMS radius — multi-emission chains


def classify_distance(d: int) -> str:
    if d <= 1:
        return "matched"
    if d <= 4:
        return "near_miss_2_4"
    if d <= 30:
        return "mid_range_5_30"
    return "isolated_gt30"


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("video")
    ap.add_argument("--dataset", default="bbc")
    ap.add_argument("--tol", type=int, default=1)
    ap.add_argument("--params", default="{}")
    ap.add_argument("--json", dest="json_out", default=None)
    args = ap.parse_args()

    from scenecut.bench import evaluate_cuts
    from scenecut.bench_datasets import load_dataset
    from scenecut.detect import detect_scenes

    entries = load_dataset(args.dataset)
    entry = next((e for e in entries
                  if Path(e["video_path"]).name == Path(args.video).name), None)
    if entry is None:
        raise SystemExit(f"video {args.video} not in dataset {args.dataset}")

    result = detect_scenes(entry["video_path"], **json.loads(args.params))
    cuts = result.get("cuts", [])
    preds = sorted(int(c["frame_num"]) for c in cuts
                   if int(c.get("frame_num", 0) or 0) > 0)
    by_frame = {int(c["frame_num"]): c for c in cuts}
    gt = sorted(entry["gt_cuts"])
    gradual = [(int(s), int(e)) for (s, e) in (entry.get("gt_gradual") or [])]

    ev = evaluate_cuts(preds, gt, gradual, tolerance=args.tol)
    matched_pairs = [(p, g) for (p, g) in
                     zip(preds, [None] * len(preds))]  # placeholder, not used

    # Re-do greedy matching here to KEEP the pairing (bench discards it).
    unmatched = list(gt)
    pairs: list[tuple[int, int, int]] = []  # (pred, gt, offset)
    for p in preds:
        best, best_d = None, -1
        for g in unmatched:
            d = abs(p - g)
            if d <= args.tol and (best is None or d < best_d):
                best, best_d = g, d
        if best is not None:
            unmatched.remove(best)
            pairs.append((p, best, best_d))
    matched_preds = {p for (p, _, _) in pairs}
    fp_frames = [p for p in preds if p not in matched_preds]

    def nearest_gt(p: int) -> int:
        return min((abs(p - g) for g in gt), default=10 ** 9)

    def nearest_pred(g: int) -> int:
        return min((abs(g - p) for p in preds), default=10 ** 9)

    fp_records = []
    for p in fp_frames:
        c = by_frame.get(p, {})
        d = nearest_gt(p)
        # cluster: another FP within radius
        cluster_n = sum(1 for q in fp_frames
                        if q != p and abs(q - p) <= CLUSTER_RADIUS)
        fp_records.append({
            "frame": p,
            "class": classify_distance(d),
            "dist_gt": d,
            "source": c.get("source"),
            "type": c.get("type"),
            "confidence": c.get("confidence"),
            "cluster_neighbors": cluster_n,
        })

    fn_records = []
    for g in unmatched:
        d = nearest_pred(g)
        fn_records.append({
            "frame": g,
            "class": classify_distance(d),
            "dist_pred": d,
        })

    def tally(records: list[dict], key: str = "class") -> dict:
        out: dict[str, int] = {}
        for r in records:
            out[r[key]] = out.get(r[key], 0) + 1
        return out

    from collections import Counter
    src_tally = Counter((r["source"], r["type"]) for r in fp_records)
    clustered = sum(1 for r in fp_records if r["cluster_neighbors"] > 0)

    summary = {
        "video": Path(args.video).name,
        "n_pred": len(preds),
        "n_gt": len(gt),
        "tp": len(pairs),
        "fp": len(fp_frames),
        "fn": len(fn_records),
        "tol": args.tol,
        "fp_class_tally": tally(fp_records),
        "fn_class_tally": tally(fn_records),
        "fp_source_tally": {f"{s}/{t}": n for (s, t), n in
                            sorted(src_tally.items(), key=lambda kv: -kv[1])},
        "fp_clustered": clustered,
        "fp_clustered_pct": round(100 * clustered / len(fp_frames), 1)
                            if fp_frames else 0.0,
        "fp_records": fp_records,
        "fn_records": fn_records,
    }
    print(json.dumps({k: v for k, v in summary.items()
                      if k not in ("fp_records", "fn_records")}, indent=1))
    if args.json_out:
        Path(args.json_out).write_text(json.dumps(summary, indent=1))
        print(f"written: {args.json_out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
