#!/usr/bin/env python3
"""Regenerate bench/golden.json — local-corpus regression floors (D12, A15).

Runs the DEFAULT config over the whole local fixture corpus, measures F1@tol1
per fixture, and sets floors:

* floor = round(max(f1@tol1 - 0.10, 0), 2)  — the 0.10 regression margin;
* fixtures that pass cleanly (f1@tol1 >= 0.99) are floored at a minimum of
  0.50 (a clean pass should never be allowed to regress to near-zero);
* documented xfails (itest_match_dissolve.mp4 — session-2 scorecard MISS,
  spatially blind match-dissolve, Phase 1.5) are floored at 0.0.

Also records the evaluated manifest (A15 gate integrity: the gate diffs it
against this golden manifest) and writes the A18 ``local_quick`` key (quick
mode does not subset the local corpus — D12 quick applies to external
datasets only — so the floors are identical under a separate key).

Usage: python3 scenecut_py/scripts/make_golden.py   (from the repo root)
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT / "scenecut_py"))

from scenecut import __version__                      # noqa: E402
from scenecut.bench import run_bench                  # noqa: E402

XFAIL_FIXTURES: set[str] = set()  # D18: match-dissolve detected (Phase 1.5 closed)
FLOOR_MARGIN = 0.10
CLEAN_PASS_F1 = 0.99
CLEAN_MIN_FLOOR = 0.5


def _floor(fixture: str, f1: float) -> float:
    """A15 floor rule — see module docstring."""
    if fixture in XFAIL_FIXTURES:
        return 0.0
    fl = round(max(f1 - FLOOR_MARGIN, 0.0), 2)
    if f1 >= CLEAN_PASS_F1:
        fl = max(fl, CLEAN_MIN_FLOOR)
    return fl


def _measure(local: dict, label: str) -> dict:
    """Measure one corpus run into a golden block (A25: neural separate)."""
    rows = {r["video"]: r for r in local["per_video"]}
    fixtures: dict[str, dict] = {}
    print(f"\n== {label} ==")
    print(f"{'fixture':28s} {'F1@tol1':>8s} {'floor':>6s}  xfail")
    for name in sorted(rows):
        f1 = rows[name]["hard_cuts"]["tol1"]["f1"]
        fl = _floor(name, f1)
        fixtures[name] = {"f1_tol1": f1, "floor": fl, "xfail": name in XFAIL_FIXTURES}
        print(f"{name:28s} {f1:8.3f} {fl:6.2f}  {'YES' if name in XFAIL_FIXTURES else ''}")

    agg_f1 = local["aggregate"]["hard_cuts@tol1"]["f1"]
    agg_floor = _floor("__aggregate__", agg_f1)
    print(f"{'<aggregate micro F1>':28s} {agg_f1:8.3f} {agg_floor:6.2f}")
    return {
        "aggregate": {"f1_tol1": agg_f1, "floor": agg_floor},
        "fixtures": fixtures,
        "manifest": local["evaluated_manifest"],
    }


def main() -> int:
    report = run_bench("local", "default")
    local = report["datasets"]["local"]
    if local["meta"]["n_videos_skipped"]:
        print("WARNING: skipped videos — golden manifest incomplete:",
              local["evaluated_manifest"]["skipped"])
    block = _measure(local, "heuristic (default)")
    golden = {
        "schema": "bench-golden/1.0",
        "description": (
            "Local-corpus regression floors for `scenecut bench --gate` (D12 A15). "
            "floor = round(max(F1@tol1 - 0.10, 0), 2) with the default config; "
            "cleanly-passing fixtures (F1 >= 0.99) floored at min 0.5; documented "
            "xfails floored at 0.0. NOTE: empty-GT negative fixtures (pan/static/"
            "sunset) vacuously measure F1=0.0 and carry floor 0.0 — their "
            "protection is the AGGREGATE micro-F1 floor (any new FP there drops "
            "it below 0.65) plus manifest integrity. itest_dissolve emits at the "
            "window midpoint 75 vs GT midpoint 80 (offset 5 > tol 1) so its "
            "tol-1 F1 is 0. Regenerate: python3 scenecut_py/scripts/make_golden.py"
        ),
        "scenecut_version": report["scenecut_version"],
        "generated_at": report["generated_at"],
        "config": report["config"],
        "xfail_fixtures": sorted(XFAIL_FIXTURES),
        "tolerance": 1,
        "local": block,
        "local_quick": {
            **block,
            "note": ("quick mode does not subset the local corpus (D12 quick applies "
                     "to external datasets only); floors identical to 'local' under a "
                     "separate key per A18"),
        },
    }

    # A25: --neural gets SEPARATE floors — never compared against heuristic
    # floors. Requires scenecut[neural] installed; skipped gracefully if not.
    try:
        neural_report = run_bench("local", "default", {"neural": True})
        neural_block = _measure(neural_report["datasets"]["local"], "neural (D13)")
        golden["local_neural"] = {
            **neural_block,
            "note": ("TransNetV2 pass + arbiter enabled (--neural). Separate floors "
                     "per A25; emission convention +1 and transition-zone "
                     "suppression are part of this baseline"),
        }
    except Exception as e:
        print(f"\nNOTE: neural floors not measured ({type(e).__name__}: {e}) — "
              "install scenecut[neural] and regenerate")
    out = REPO_ROOT / "bench" / "golden.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(golden, indent=2) + "\n")
    print(f"\nwrote {out} (scenecut {__version__}, "
          f"{len(block['fixtures'])} fixtures, "
          f"neural: {'yes' if 'local_neural' in golden else 'skipped'})")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
