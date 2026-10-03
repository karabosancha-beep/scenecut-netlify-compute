#!/usr/bin/env python3
"""External bench runner — resumable, chunked, accumulate-merge (D15 + D14 F6).

The sandbox caps single commands at 10 min and reaps background processes, so
long benches run as REPEATED INVOCATIONS with durable state:

  state file:  bench/.external_<dataset>_<config>.state.json
               {rows: [...], cursor: N, skipped: [{video, reason}]} (per-video
               rows + resume cursor + F6 skip list merged across invocations)
  report:      bench/external_<dataset>_<config>.json (regenerated from rows
               on every invocation — micro-aggregated fresh, never a mean of
               per-video F1s; carries the F6 skip list so the 257-on-disk vs
               N-rows discrepancy is auditable)

Usage:
  python3 external_bench.py <dataset> [--config default] [--neural]
  python3 external_bench.py <dataset> --reset
  python3 external_bench.py <dataset> --report-only
  python3 external_bench.py <dataset> --tolerances 0,1,5   # D14 optional tau=5

Non-default tolerance sets fork a separate state/report file (suffix _t015)
so a tau=5 protocol never mixes rows with a (0,1) run; the D12 primary
tolerances stay (0, 1) by default and nothing is gated on the extra cell.
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT / "scenecut_py"))

BENCH_DIR = REPO_ROOT / "bench"
TIME_BUDGET = 500.0  # seconds of detection work per invocation


def _agg_family(rows: list[dict], key: str, tol: int) -> dict:
    tp = fp = fn = 0
    for r in rows:
        if key in ("hard_cuts", "transitions"):   # per-tolerance cells
            cell = (r.get(key) or {}).get(f"tol{tol}")
        else:  # "gradual" / "transitions_gradual" — tolerance-independent blocks
            cell = r.get(key)
        if not cell:
            continue
        tp += cell.get("tp", 0)
        fp += cell.get("fp", 0)
        fn += cell.get("fn", 0)
    p = tp / (tp + fp) if tp + fp else 0.0
    rc = tp / (tp + fn) if tp + fn else 0.0
    f1 = 2 * p * rc / (p + rc) if p + rc else 0.0
    return {"tp": tp, "fp": fp, "fn": fn,
            "precision": round(p, 4), "recall": round(rc, 4), "f1": round(f1, 4)}


def _merge_skipped(existing: list[dict], new: list[dict]) -> list[dict]:
    """F6: merge per-invocation skip lists, deduped by video name."""
    out = list(existing)
    seen = {s.get("video") for s in out}
    for s in new:
        if s.get("video") not in seen:
            out.append(s)
            seen.add(s.get("video"))
    return out


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("dataset")
    ap.add_argument("--config", default="default")
    ap.add_argument("--neural", action="store_true")
    ap.add_argument("--reset", action="store_true")
    ap.add_argument("--report-only", action="store_true")
    ap.add_argument("--limit", type=int, default=None,
                    help="cap entries (debugging)")
    ap.add_argument("--time-budget", type=float, default=TIME_BUDGET,
                    help="seconds of detection work per invocation (leave headroom "
                         "under any OUTER shell timeout: one in-flight entry at "
                         "neural pace can run 60-130s past the budget check)")
    ap.add_argument("--tolerances", default="0,1",
                    help="comma-separated tolerance list passed to run_bench "
                         "(default 0,1; e.g. 0,1,5 for the optional D14 "
                         "TRECVid-convention tau=5 cell — never gated)")
    args = ap.parse_args()
    tols = tuple(int(x) for x in str(args.tolerances).split(",") if x.strip())

    suffix = "neural" if args.neural else args.config
    if tols != (0, 1):  # non-default protocol -> separate state/report files
        suffix += "_t" + "".join(str(t) for t in tols)
    stem = f"{args.dataset.replace(':', '_')}_{suffix}"
    state_path = BENCH_DIR / f".external_{stem}.state.json"
    report_path = BENCH_DIR / f"external_{stem}.json"

    from scenecut.bench import run_bench
    from scenecut.bench_datasets import load_dataset

    entries = load_dataset(args.dataset)
    if args.limit:
        entries = entries[:args.limit]

    if args.reset and state_path.exists():
        state_path.unlink()
        print(f"state reset: {state_path}")

    # F6: `skipped` persists in the state file (merged across invocations).
    # Old state files without the key load with [] (backward compatible).
    state = {"rows": [], "cursor": 0, "skipped": []}
    if state_path.exists():
        loaded = json.loads(state_path.read_text())
        state = {"rows": loaded.get("rows", []),
                 "cursor": int(loaded.get("cursor", 0)),
                 "skipped": loaded.get("skipped", [])}
    rows: list[dict] = state["rows"]
    cursor: int = state["cursor"]
    skipped: list[dict] = list(state["skipped"])
    done_videos = {r["video"] for r in rows}
    print(f"[external] {args.dataset} ({args.config}"
          f"{' +neural' if args.neural else ''}): "
          f"{len(rows)} rows done, {len(skipped)} skipped, "
          f"cursor {cursor}/{len(entries)}")

    if not args.report_only:
        t0 = time.time()
        while cursor < len(entries) and time.time() - t0 < args.time_budget:
            entry = entries[cursor]
            name = Path(str(entry["video_path"])).name
            if name in done_videos:
                cursor += 1
                continue
            # one video at a time — run_bench with entries=[entry]
            try:
                rep = run_bench(args.dataset, args.config,
                                {"neural": True} if args.neural else {},
                                entries=[entry], tolerances=tols)
                pv = rep["datasets"][args.dataset]["per_video"]
                if not pv:  # skipped (A14 ambiguous / detect error)
                    new_skips = (((rep["datasets"][args.dataset] or {})
                                  .get("evaluated_manifest") or {}).get("skipped")
                                 or [])
                    skipped = _merge_skipped(skipped, new_skips)
                    why = new_skips[-1].get("reason", "?") if new_skips else "unknown"
                    print(f"  [{cursor+1}/{len(entries)}] {name}: skipped ({why})",
                          flush=True)
                else:
                    rows.append(pv[0])
            except Exception as exc:
                print(f"  [{cursor+1}/{len(entries)}] {name}: ERROR {exc}",
                      flush=True)
            cursor += 1
            if len(rows) % 20 == 0:
                print(f"  ... {len(rows)} videos benched "
                      f"({time.time()-t0:.0f}s)", flush=True)
        state = {"rows": rows, "cursor": cursor, "skipped": skipped}
        state_path.write_text(json.dumps(state))
        print(f"[external] chunk done: {len(rows)} rows, {len(skipped)} skipped, "
              f"cursor {cursor}/{len(entries)} — re-run to continue")

    # ---- report: micro-aggregate fresh from rows
    out = {
        "dataset": args.dataset,
        "config": args.config,
        "neural": bool(args.neural),
        "tolerances": list(tols),
        "n_videos": len(rows),
        "n_skipped": len(skipped),
        "skipped": skipped,  # F6: persisted + auditable
        "aggregate": {},
        "per_video": rows,
    }
    for t in tols:
        out["aggregate"][f"hard_cuts@tol{t}"] = _agg_family(rows, "hard_cuts", t)
        out["aggregate"][f"transitions@tol{t}"] = _agg_family(rows, "transitions", t)
    out["aggregate"]["gradual"] = _agg_family(rows, "gradual", 0)
    out["aggregate"]["transitions_gradual"] = _agg_family(rows, "transitions_gradual", 0)
    report_path.write_text(json.dumps(out, indent=1, default=str))
    for k in sorted(out["aggregate"]):
        c = out["aggregate"][k]
        print(f"  {k}: P={c['precision']} R={c['recall']} F1={c['f1']} "
              f"(tp {c['tp']} / fp {c['fp']} / fn {c['fn']})")
    print(f"report: {report_path} ({len(rows)} videos, {len(skipped)} skipped)")
    if cursor < len(entries):
        return 1  # signal: not finished — re-run
    return 0


if __name__ == "__main__":
    sys.exit(main())
