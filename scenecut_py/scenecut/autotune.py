"""Auto-tune: VLM-audited parameter tuning loop (D9-impl, DECISIONS.md v3.1).

Loop: detect(params) -> stratified sample of the post-merge cut population ->
VLM 2-call vote audit -> HT estimate + Wilson-upper acceptance gate ->
source-routed monotone tightening -> re-detect. Max 3 iterations.

Statistical design (v3.1):
- Two-way stratification: confidence quartile x primary source -> cells.
  Cells with <= 3 members are censused. Census mode when population <= 24.
- Acceptance gate A2: Wilson 95% UPPER bound on audited falses (k of
  n_audited) <= 0.20 — uniform across census/sample modes, conservative.
- Abstain protocol A3: exclude the unit, reweight within cell
  (p_h = false_h / (n_h - a_h)); abort iteration when pooled abstention > 25%
  or any cell has a_h >= ceil(n_h / 2).
- Budget A5: every real bridge call counts (incl. retries + tie-breaks +
  recall guard); target n shrinks when remaining budget is short.
- Router A9/A11: dominant failing class (max HT-weighted FP mass, primary
  source attribution) -> ONE knob per iteration, monotone tightening only:
    adaptive       -> threshold +5 (cap 42)
    hash           -> hash_threshold +0.05 (cap 0.7)   [D16 v4.3: hash fires
                      on hash DISTANCE — the adaptive ratio is the wrong knob]
    dissolve       -> dissolve_dissim +0.03 (cap 0.30)
    threshold      -> fade_min_len +0.1 (cap 0.8)
    threshold_ceiling -> fade_ceiling +2 (cap 250)
  Neural FPs are advisory-only (A13 — no knob routes to neural).
- Terminal states A10: accepted | not_accepted | at_caps | not_adjustable |
  budget_exhausted | vlm_unavailable | recall_guard_stop | no_cuts |
  heuristic_converged | aborted.
- --apply semantics A10: writes a project ONLY from last-accepted params
  (or current params with --force when nothing was accepted).
"""
from __future__ import annotations

import json
import math
import os
import random
import signal
import threading
import time
from dataclasses import dataclass, field, asdict
from pathlib import Path
from typing import Callable, Optional

from .config import DEFAULTS
from .util import SceneCutError

# ------------------------------------------------------------------ params

_KNOBS = {
    # failing class -> (param attr, step, cap)   [A11: threshold step +5 cap 42]
    "adaptive": ("threshold", 5.0, 42.0),
    # D16 v4.3: hash FPs fire on hash DISTANCE — route to the HashDetector
    # firing distance, NOT the adaptive ratio threshold (RV4-S7 wrong-knob).
    # Raising hash_threshold -> fewer hash fires (monotone tightening).
    "hash": ("hash_threshold", 0.05, 0.7),
    "dissolve": ("dissolve_dissim", 0.03, 0.30),
    "threshold": ("fade_min_len", 0.1, 0.8),
    "threshold_ceiling": ("fade_ceiling", 2, 250),
}

_ROUTABLE = ("adaptive", "hash", "dissolve", "threshold", "threshold_ceiling")


@dataclass
class TunableParams:
    threshold: float = DEFAULTS["threshold"]
    dissolve_dissim: float = DEFAULTS["dissolve_dissim"]
    fade_min_len: float = DEFAULTS["fade_min_len"]
    fade_ceiling: int = DEFAULTS["fade_ceiling"]
    hash_threshold: float = DEFAULTS["hash_threshold"]
    hash_corroboration: bool = DEFAULTS.get("hash_corroboration", True)
    # CR4-m3: same carried-not-tuned class — a user-disabled tier-2 spatial
    # pass must stay disabled through the loop's re-detections.
    spatial_dissolve: bool = DEFAULTS.get("spatial_dissolve", True)
    # D19: the recalibrated tier-1 gates are carried-not-tuned (the autotune
    # loop tunes dissolve_dissim only; these ride along at user/default
    # values through every re-detection).
    hard_cut_mad_min: float = DEFAULTS.get("hard_cut_mad_min", 40.0)
    consec_js_min: float = DEFAULTS.get("consec_js_min", 0.025)
    consec_js_frame_min: float = DEFAULTS.get("consec_js_frame_min", 0.03)
    consec_js_sustain_frac: float = DEFAULTS.get("consec_js_sustain_frac", 0.35)
    consec_js_ratio_min: float = DEFAULTS.get("consec_js_ratio_min", 1.8)

    def as_dict(self) -> dict:
        return asdict(self)

    @classmethod
    def from_detect_params(cls, d: dict | None) -> "TunableParams":
        """Inherit the project's CURRENT detect params (A12) — fall back to
        DEFAULTS when absent (None means 'default' in detect_scenes)."""
        d = d or {}
        return cls(
            threshold=float(d.get("threshold") or DEFAULTS["threshold"]),
            dissolve_dissim=float(d.get("dissolve_dissim") or DEFAULTS["dissolve_dissim"]),
            fade_min_len=float(d.get("fade_min_len") or DEFAULTS["fade_min_len"]),
            fade_ceiling=int(d.get("fade_ceiling_threshold") or d.get("fade_ceiling")
                             or DEFAULTS["fade_ceiling"]),
            hash_threshold=float(d["hash_threshold"])
            if d.get("hash_threshold") is not None
            else DEFAULTS["hash_threshold"],
            hash_corroboration=bool(d.get("hash_corroboration",
                                          DEFAULTS.get("hash_corroboration", True))),
            spatial_dissolve=bool(d.get("spatial_dissolve",
                                        DEFAULTS.get("spatial_dissolve", True))),
            hard_cut_mad_min=float(d.get("hard_cut_mad_min")
                                   if d.get("hard_cut_mad_min") is not None
                                   else DEFAULTS.get("hard_cut_mad_min", 40.0)),
            consec_js_min=float(d.get("consec_js_min")
                                if d.get("consec_js_min") is not None
                                else DEFAULTS.get("consec_js_min", 0.025)),
            consec_js_frame_min=float(d.get("consec_js_frame_min")
                                      if d.get("consec_js_frame_min") is not None
                                      else DEFAULTS.get("consec_js_frame_min", 0.03)),
            consec_js_sustain_frac=float(d.get("consec_js_sustain_frac")
                                         if d.get("consec_js_sustain_frac") is not None
                                         else DEFAULTS.get("consec_js_sustain_frac", 0.35)),
            consec_js_ratio_min=float(d.get("consec_js_ratio_min")
                                      if d.get("consec_js_ratio_min") is not None
                                      else DEFAULTS.get("consec_js_ratio_min", 1.8)),
        )

    def tighten(self, klass: str) -> Optional[tuple[str, float, float]]:
        """Monotone tightening by failing class. Returns (attr, old, new) or
        None when the class's knob is already at cap."""
        if klass not in _KNOBS:
            return None
        attr, step, cap = _KNOBS[klass]
        cur = float(getattr(self, attr))
        if cur >= cap - 1e-9:
            return None
        new = min(cap, cur + step)
        setattr(self, attr, type(getattr(self, attr))(new))
        return (attr, cur, new)

    def all_at_caps(self) -> bool:
        return all(
            float(getattr(self, attr)) >= cap - 1e-9
            for attr, _s, cap in _KNOBS.values()
        )


# ------------------------------------------------------------------ sampling

def _build_cells(cuts: list[dict]) -> dict[tuple[int, str], list[int]]:
    """Two-way stratification: confidence quartile x primary source (D9-impl 2).

    Returns {(quartile 0..3, source): [cut positions in population list]}.
    Quartiles by confidence rank (dense, deterministic: sorted then split).
    """
    n = len(cuts)
    order = sorted(range(n), key=lambda i: (cuts[i].get("confidence") or 0.0, i))
    cells: dict[tuple[int, str], list[int]] = {}
    for rank, pos in enumerate(order):
        q = min(3, 4 * rank // max(1, n))
        src = str(cuts[pos].get("source") or "adaptive")
        cells.setdefault((q, src), []).append(pos)
    return cells


def draw_sample(cuts: list[dict], rng: random.Random,
                target_n: int = 24, small_cell: int = 3
                ) -> tuple[list[dict], bool]:
    """Stratified draw. Returns (sample, census_mode).

    sample items: {pos, cell, inclusion_prob, cut}. Census when population
    <= target_n. Small cells (<= small_cell) censused; proportional allocation
    (largest remainder) across the rest.

    CR2-2: when the number of big cells exceeds the remaining target
    (budget-constrained targets), the two-way (quartile x source) grid is
    COLLAPSED to quartile-only strata — otherwise zero-allocation strata
    silently vanish from the HT denominator and p_hat inflates up to 12x
    (measured). Coarser-but-complete beats fine-but-biased.
    """
    n = len(cuts)
    if n <= target_n:
        sample = [{"pos": i, "cell": "census", "inclusion_prob": 1.0, "cut": c}
                  for i, c in enumerate(cuts)]
        return sample, True

    cells = _build_cells(cuts)
    big_cells = {k: v for k, v in cells.items() if len(v) > small_cell}
    n_census = sum(len(v) for v in cells.values() if len(v) <= small_cell)
    remaining_target = max(0, target_n - n_census)
    if len(big_cells) > remaining_target:
        # CR2-2 collapse: merge source dimension within each quartile.
        merged: dict[int, list[int]] = {}
        for (q, _src), members in cells.items():
            merged.setdefault(q, []).extend(members)
        cells = {(q, "any"): sorted(m) for q, m in merged.items()}
        big_cells = {k: v for k, v in cells.items() if len(v) > small_cell}
        n_census = sum(len(v) for v in cells.values() if len(v) <= small_cell)
        remaining_target = max(0, target_n - n_census)

    sample: list[dict] = []
    for key, members in cells.items():
        if len(members) <= small_cell:
            for m in members:
                sample.append({"pos": m, "cell": key, "inclusion_prob": 1.0,
                               "cut": cuts[m]})

    big_pop = sum(len(v) for v in big_cells.values())
    if big_pop == 0 or remaining_target == 0:
        return sample, False

    # Proportional allocation via largest remainder, with a min-1 guarantee
    # per big cell (CR2-2: zero-allocation cells vanish from the HT
    # denominator). The collapse above bounds cell count <= remaining_target,
    # so min-1 can always be honored after trimming the largest cells.
    exact = {k: remaining_target * len(v) / big_pop for k, v in big_cells.items()}
    alloc = {k: max(1, int(math.floor(e))) for k, e in exact.items()}
    over = sum(alloc.values()) - remaining_target
    while over > 0:
        # Decrement the largest allocation that can shrink (deterministic).
        k_max = max((k for k in alloc if alloc[k] > 1),
                    key=lambda k: (alloc[k], len(big_cells[k])), default=None)
        if k_max is None:
            break
        alloc[k_max] -= 1
        over -= 1
    left = remaining_target - sum(alloc.values())
    if left > 0:
        for k in sorted(exact, key=lambda k: (-(exact[k] - alloc[k]), k))[:left]:
            alloc[k] += 1
    for k, members in big_cells.items():
        n_h = min(alloc[k], len(members))
        if n_h <= 0:
            continue
        chosen = rng.sample(members, n_h)
        for m in chosen:
            sample.append({"pos": m, "cell": k,
                           "inclusion_prob": n_h / len(members), "cut": cuts[m]})
    return sample, False


# ------------------------------------------------------------------ estimation

def _wilson_upper(k: int, n: int, z: float = 1.96) -> float:
    """Wilson score interval UPPER bound. k falses in n audited."""
    if n <= 0:
        return 1.0
    p = k / n
    denom = 1 + z * z / n
    centre = p + z * z / (2 * n)
    margin = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n))
    return min(1.0, (centre + margin) / denom)


def estimate_false_rate(sample: list[dict],
                        verdicts: dict[int, dict],
                        population_n: int,
                        census: bool) -> dict:
    """HT estimate + Wilson-upper acceptance (A2/A3).

    verdicts: {pos: {"verdict": "genuine"|"false"|None, ...}} keyed by
    population position. Abstains excluded with within-cell reweighting.
    """
    cells: dict[str, dict] = {}
    for s in sample:
        c = cells.setdefault(str(s["cell"]),
                              {"N_h": 0, "n_h": 0, "a_h": 0, "f_h": 0, "u_h": 0})
        c["n_h"] += 1
        v = verdicts.get(s["pos"], {})
        if v.get("verdict") is None and v.get("agreement") == "uncertain":
            # D17: model-uncertain (both votes honestly unsure / tie-break
            # refused). Excluded from the audited denominator like an
            # abstain, but NOT an infra failure — must not feed _abstain_abort
            # (an infrastructure lie would abort honest-but-ambivalent runs
            # as vlm_unavailable).
            c["u_h"] += 1
        elif v.get("verdict") is None:
            c["a_h"] += 1
        elif v["verdict"] == "false":
            c["f_h"] += 1
    if census:
        for c in cells.values():
            c["N_h"] = c["n_h"]
    else:
        # N_h = sampled weight-scaled size: n_h / inclusion_prob summed per cell
        # (cells are sampled at a single inclusion prob, so this is exact).
        per_cell_probs: dict[str, list[float]] = {}
        for s in sample:
            per_cell_probs.setdefault(str(s["cell"]), []).append(s["inclusion_prob"])
        for key, probs in per_cell_probs.items():
            pi = probs[0]
            cells[key]["N_h"] = (cells[key]["n_h"] / pi) if pi > 0 else cells[key]["n_h"]
    # Normalize N_h to the population total (guards rounding drift).
    total_n_hat = sum(c["N_h"] for c in cells.values()) or population_n or 1

    p_hat_num = 0.0
    k = a = u = n_aud = 0
    for c in cells.values():
        parsed = c["n_h"] - c["a_h"] - c["u_h"]
        k += c["f_h"]
        a += c["a_h"]
        u += c["u_h"]
        n_aud += parsed
        if parsed > 0:
            p_hat_num += (c["N_h"] / total_n_hat) * (c["f_h"] / parsed)

    wilson_u = _wilson_upper(k, n_aud)
    return {
        "p_hat": round(p_hat_num, 4),
        "k_falses": k,
        "n_audited": n_aud,
        "n_abstains": a,
        "n_uncertain": u,
        "abstain_rate": round(a / len(sample), 4) if sample else 0.0,
        "uncertain_rate": round(u / len(sample), 4) if sample else 0.0,
        "wilson_upper_95": round(wilson_u, 4),
        "ci95_wilson_pooled_conservative": [
            round(_wilson_lower(k, n_aud), 4), round(wilson_u, 4)] if n_aud else None,
        "accepted": wilson_u <= 0.20,
        "sample_mode": "census" if census else "stratified",
    }


def _wilson_lower(k: int, n: int, z: float = 1.96) -> float:
    if n <= 0:
        return 0.0
    p = k / n
    denom = 1 + z * z / n
    centre = p + z * z / (2 * n)
    margin = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n))
    return max(0.0, (centre - margin) / denom)


def _abstain_abort(estimate: dict, sample: list[dict],
                   verdicts: dict[int, dict]) -> bool:
    """A3 abort rules: pooled abstention > 25% OR any cell half-abstained.

    D17: only INFRA abstains count (verdict None with agreement !=
    "uncertain"). Model-uncertain units (agreement == "uncertain") are
    honest non-answers, not infrastructure failure — they shrink n_audited
    (the Wilson gate honestly widens) but must never abort as
    vlm_unavailable.
    """
    if not sample or estimate["n_abstains"] / len(sample) > 0.25:
        return True
    per_cell: dict[str, dict] = {}
    for s in sample:
        c = per_cell.setdefault(str(s["cell"]), {"n": 0, "a": 0})
        c["n"] += 1
        v = verdicts.get(s["pos"], {})
        if v.get("verdict") is None and v.get("agreement") != "uncertain":
            c["a"] += 1
    return any(c["a"] >= math.ceil(c["n"] / 2) and c["n"] > 0
               for c in per_cell.values())


# ------------------------------------------------------------------ routing

def route_adjustment(sample: list[dict], verdicts: dict[int, dict],
                     population_n: int) -> tuple[Optional[str], dict]:
    """Pick the failing class with max HT-weighted FP mass (A9).

    Neural FPs are advisory-only (A13). Returns (class|None, mass-by-class).
    """
    mass: dict[str, float] = {}
    for s in sample:
        v = verdicts.get(s["pos"], {})
        if v.get("verdict") != "false":
            continue
        src = str(s["cut"].get("source") or "adaptive")
        pi = s["inclusion_prob"]
        # HT contribution of this unit: 1/pi (estimated population false count).
        mass[src] = mass.get(src, 0.0) + (1.0 / pi if pi > 0 else 1.0)
    routable = {k: v for k, v in mass.items() if k in _ROUTABLE}
    best = None
    if routable:
        # tie-break by canonical priority order
        best = max(_ROUTABLE, key=lambda k: (routable.get(k, 0.0),
                                             -_ROUTABLE.index(k)))
        if routable.get(best, 0.0) <= 0:
            best = None
    return best, mass


def heuristic_verdicts(cuts: list[dict]) -> dict[int, dict]:
    """A4 offline audit: suspect iff motion_gated OR confidence < 0.20."""
    out = {}
    for i, c in enumerate(cuts):
        suspect = bool(c.get("motion_gated")) or (c.get("confidence") or 0) < 0.20
        out[i] = {"verdict": "false" if suspect else "genuine",
                  "confidence": c.get("confidence"), "heuristic": True}
    return out


# ------------------------------------------------------------------ events

def _ev(event: str, **kw) -> dict:
    return {"event": event, "ts": round(time.time(), 3), **kw}


# ------------------------------------------------------------------ main loop

def autotune(input_path: str,
             proxy: str | None = None,
             audit: str = "vlm",
             max_iterations: int = 3,
             max_calls: int = 200,
             seed: int = 42,
             start_params: dict | None = None,
             apply: bool = False,
             force: bool = False,
             output: str | Path | None = None,
             progress_cb: Callable[[dict], None] | None = None,
             workdir: str | Path | None = None,
             seam_radius: float = 5.0,
             dry_run_detect: Callable[..., dict] | None = None) -> dict:
    """Run the auto-tune loop. Returns the final report dict.

    dry_run_detect: injectable detect function (tests) — signature mirrors
    scenecut.detect.detect_scenes.
    """
    from .detect import detect_scenes
    from .project import save_project

    detect_fn = dry_run_detect or detect_scenes
    if audit not in ("vlm", "heuristic"):
        raise SceneCutError(f"audit must be 'vlm' or 'heuristic' (got {audit!r})", code=2)

    input_path = str(input_path)
    out_path = Path(output) if output else Path(input_path + ".scenes.json")
    workdir = Path(workdir) if workdir else out_path.parent / "autotune"
    # Absolute everything: the VLM subprocess runs with a different cwd than
    # the CLI (repo root) — relative paths break bun (live-found in S3).
    input_path = os.path.abspath(input_path)
    out_path = Path(os.path.abspath(str(out_path)))
    workdir = Path(os.path.abspath(str(workdir)))
    workdir.mkdir(parents=True, exist_ok=True)

    params = TunableParams.from_detect_params(start_params)
    rng = random.Random(seed)

    vlm = None
    if audit == "vlm":
        from .vlm import VLMClient
        vlm = VLMClient(budget=max_calls)
        progress_cb = progress_cb or (lambda e: None)
    else:
        progress_cb = progress_cb or (lambda e: None)

    # SIGTERM: stop accepting new work, write partial report atomically (A12).
    abort = threading.Event()

    def _on_term(signum, frame):
        abort.set()

    prev_handler = None
    try:
        prev_handler = signal.signal(signal.SIGTERM, _on_term)
    except ValueError:
        pass  # not main thread (tests)

    report: dict = {
        "schema": "1.0",
        "input": os.path.basename(input_path),
        "audit": audit if audit == "vlm" else "heuristic-only",
        "seed": seed,
        "iterations": [],
        "calls_used": 0,
        "started_at": time.time(),
    }

    def emit(e: dict) -> None:
        try:
            progress_cb(e)
        except Exception:
            pass

    def write_report(final: dict) -> None:
        """Atomic write (tmp + rename) — A12."""
        final["calls_used"] = vlm.calls_used if vlm else 0
        final["finished_at"] = time.time()
        tmp = workdir / "report.json.tmp"
        with open(tmp, "w") as f:
            json.dump(final, f, indent=2)
        os.replace(tmp, workdir / "report.json")

    try:
        status = None
        last_accepted: Optional[TunableParams] = None
        audited_params: Optional[TunableParams] = None
        orig_cut_count: Optional[int] = None
        prev_cut_count: Optional[int] = None
        best_estimate: Optional[dict] = None
        suspected_misses: list[dict] = []

        for it in range(1, max_iterations + 1):
            if abort.is_set():
                status = "aborted"
                break

            emit(_ev("iteration_start", iteration=it, params=params.as_dict()))

            project = detect_fn(input_path, proxy=proxy, **_detect_kwargs(params))
            # NOTE (CR2-1): audited_params is set ONLY after a completed audit
            # below — early exits (tripwire / no_cuts / budget) must NOT export
            # params that were never audited as "recommended".
            cuts = [c for c in project.get("cuts", []) if (c.get("frame_num") or 0) > 0]
            n_cuts = len(cuts)
            emit(_ev("detected", iteration=it, n_cuts=n_cuts,
                     n_scenes=len(project.get("scenes", []))))

            if orig_cut_count is None:
                orig_cut_count = n_cuts
            # Tripwire A10: drop vs BOTH previous iteration and original.
            if prev_cut_count is not None and prev_cut_count > 0:
                drop_prev = (prev_cut_count - n_cuts) / prev_cut_count
                if drop_prev > 0.25:
                    status = "recall_guard_stop"
                    emit(_ev("tripwire", iteration=it, drop_prev=round(drop_prev, 3),
                             prev=prev_cut_count, cur=n_cuts))
                    break
            if orig_cut_count > 0 and n_cuts < orig_cut_count:
                drop_orig = (orig_cut_count - n_cuts) / orig_cut_count
                if drop_orig > 0.25 and it > 1:
                    status = "recall_guard_stop"
                    emit(_ev("tripwire", iteration=it, drop_orig=round(drop_orig, 3),
                             original=orig_cut_count, cur=n_cuts))
                    break

            if n_cuts == 0:
                status = "no_cuts"
                emit(_ev("no_cuts", iteration=it))
                # Recall guard still runs (advisory) — handled below via break
                # after a guard pass on the whole-video scene list.
                _run_recall_guard(project, input_path, workdir, it, vlm, audit,
                                  emit, abort, suspected_misses)
                break

            # ---- sample
            budget_left = vlm.remaining_budget() if vlm else max_calls
            guard_reserve = 7 if audit == "vlm" else 0
            target_n = 24
            if audit == "vlm" and budget_left < 2 * target_n + guard_reserve:
                target_n = max(8, (budget_left - guard_reserve) // 2)
                if (budget_left - guard_reserve) // 2 < 8:
                    # CR2-9: emit + terminal BEFORE sampling (was dead code that
                    # still generated seam clips for an unaffordable audit).
                    status = "budget_exhausted"
                    emit(_ev("budget_exhausted", iteration=it, remaining=budget_left))
                    break
            sample, census = draw_sample(cuts, rng, target_n=target_n)
            emit(_ev("sampled", iteration=it, n=len(sample), target=target_n,
                     mode="census" if census else "stratified",
                     realized_population=n_cuts))

            # ---- audit
            if audit == "vlm":
                verdicts = _audit_with_vlm(
                    project, cuts, sample, input_path, workdir, it, vlm,
                    seam_radius, emit, abort)
                if verdicts is None:
                    status = "budget_exhausted"
                    break
            else:
                verdicts = heuristic_verdicts(cuts)

            est = estimate_false_rate(sample, verdicts, n_cuts, census)
            # CR2-8: mid-batch budget exhaustion visibility — outcomes with
            # "budget exhausted" errors degrade the sample; surface it.
            budget_shortfall = any(
                "budget exhausted" in str(v.get("errors") or v.get("error") or "")
                for v in verdicts.values())
            est["budget_shortfall"] = budget_shortfall
            est["single_rate"] = round(
                sum(1 for v in verdicts.values() if v.get("agreement") == "single")
                / len(sample), 4) if sample else 0.0
            best_estimate = est
            # CR2-1: a COMPLETE audit just happened — these params are audited.
            audited_params = TunableParams(**params.as_dict())
            emit(_ev("estimate", iteration=it, **{k: est[k] for k in
                                                  ("p_hat", "k_falses", "n_audited",
                                                   "wilson_upper_95", "accepted",
                                                   "single_rate", "n_uncertain",
                                                   "uncertain_rate")}))

            iter_rec = {
                "iteration": it,
                "params": params.as_dict(),
                "population": n_cuts,
                "sampled": len(sample),
                "sample_mode": est["sample_mode"],
                "sample": [{"pos": s["pos"],
                            "frame_num": s["cut"].get("frame_num"),
                            "source": s["cut"].get("source"),
                            "confidence": s["cut"].get("confidence"),
                            "inclusion_prob": round(s["inclusion_prob"], 4),
                            "verdict": verdicts.get(s["pos"], {}).get("verdict"),
                            "agreement": verdicts.get(s["pos"], {}).get("agreement")}
                           for s in sample],
                "estimate": est,
            }

            # ---- recall guard
            guard = _run_recall_guard(project, input_path, workdir, it, vlm, audit,
                                      emit, abort, suspected_misses)
            iter_rec["recall_guard"] = guard

            # ---- accept? (VLM mode only — A4: heuristic mode has NO Wilson
            # gate; it terminates via suspect-rate trend, not certification)
            if audit == "vlm" and est["accepted"]:
                status = "accepted"
                last_accepted = TunableParams(**params.as_dict())
                report["iterations"].append(iter_rec)
                emit(_ev("accepted", iteration=it, params=params.as_dict()))
                break

            if audit == "vlm" and _abstain_abort(est, sample, verdicts):
                status = "vlm_unavailable"
                report["iterations"].append(iter_rec)
                emit(_ev("vlm_unavailable", iteration=it,
                         abstain_rate=est["abstain_rate"]))
                break

            # CR2-8/CR2-9: budget ran out mid-batch — stop here (partial data
            # would route adjustments on an degraded sample).
            if audit == "vlm" and est.get("budget_shortfall"):
                report["iterations"].append(iter_rec)
                status = "budget_exhausted"
                emit(_ev("budget_exhausted", iteration=it,
                         remaining=vlm.remaining_budget()))
                break

            # ---- route adjustment
            if audit == "heuristic":
                # Zero suspects on any iteration = converged for this coarse
                # proxy (nothing left to fix without VLM evidence).
                if est["p_hat"] == 0.0:
                    report["iterations"].append(iter_rec)
                    status = "heuristic_converged"
                    emit(_ev("heuristic_converged", iteration=it))
                    break
                prev_rate = report["iterations"][-1]["estimate"]["p_hat"] \
                    if report["iterations"] else None
                report["iterations"].append(iter_rec)
                cur_rate = est["p_hat"]
                if prev_rate is not None and prev_rate > 0 and \
                        abs(prev_rate - cur_rate) / prev_rate < 0.20:
                    status = "heuristic_converged"
                    break
                if prev_rate is not None and cur_rate >= prev_rate:
                    status = "heuristic_converged"
                    break
            else:
                report["iterations"].append(iter_rec)

            klass, mass = route_adjustment(sample, verdicts, n_cuts)
            if klass is None:
                if not mass:
                    # CR2-11: zero FPs observed — honest state is not_accepted
                    # ("sample too small to certify, nothing found"), NOT
                    # not_adjustable (reserved for FPs that exist but have no
                    # routable knob, e.g. neural-only).
                    status = "not_accepted"
                elif params.all_at_caps():
                    status = "at_caps"
                else:
                    status = "not_adjustable"
                emit(_ev("no_adjustment", iteration=it, mass=mass, status=status))
                break

            adjustment = params.tighten(klass)
            if adjustment is None:
                status = "at_caps"
                emit(_ev("no_adjustment", iteration=it, klass=klass, status=status))
                break
            attr, old, new = adjustment
            emit(_ev("adjust", iteration=it, klass=klass, knob=attr,
                     from_=old, to=new, fp_mass=mass))
            prev_cut_count = n_cuts

        if status is None:
            status = "not_accepted"

        # ---- apply (A10)
        applied_path = None
        if apply:
            chosen = last_accepted
            if chosen is None and force:
                chosen = TunableParams(**params.as_dict())
            if chosen is not None:
                final_project = detect_fn(input_path, proxy=proxy,
                                          **_detect_kwargs(chosen))
                save_project(final_project, out_path)
                applied_path = str(out_path)
                emit(_ev("apply", path=applied_path, params=chosen.as_dict(),
                         source="last_accepted" if last_accepted else "force"))
            else:
                emit(_ev("apply_skipped", reason="nothing accepted; use --force"))

        report.update({
            "status": status,
            "final_params": (audited_params or params).as_dict(),
            "next_params_unaudited": params.as_dict(),
            "last_accepted_params": last_accepted.as_dict() if last_accepted else None,
            "recommended_params": (last_accepted or audited_params or params).as_dict(),
            "false_split_rate_estimate": best_estimate,
            "recall_guard": {"suspected_misses": suspected_misses},
            "applied": applied_path,
            "report_path": str(workdir / "report.json"),
        })
        write_report(report)
        emit(_ev("done", status=status, calls_used=report["calls_used"],
                 applied=applied_path))
        return report

    finally:
        if prev_handler is not None:
            try:
                signal.signal(signal.SIGTERM, prev_handler)
            except ValueError:
                pass
        # Ensure a partial report exists even on unexpected paths.
        if not (workdir / "report.json").exists():
            report.setdefault("status", "aborted")
            write_report(report)


def _detect_kwargs(p: TunableParams) -> dict:
    return {
        "threshold": p.threshold,
        "dissolve_dissim": p.dissolve_dissim,
        "fade_min_len": p.fade_min_len,
        "fade_ceiling": p.fade_ceiling,
        # RV4 finding: a knob missing here silently resets to DEFAULTS on
        # re-detection — the tuned value must ride along every detect call.
        "hash_threshold": p.hash_threshold,
        # CR3: preserve the user's gate setting through the loop
        "hash_corroboration": p.hash_corroboration,
        # CR4-m3: preserve the user's tier-2 spatial setting through the loop
        "spatial_dissolve": p.spatial_dissolve,
        # D19: preserve the recalibrated tier-1 gates through the loop
        "hard_cut_mad_min": p.hard_cut_mad_min,
        "consec_js_min": p.consec_js_min,
        "consec_js_frame_min": p.consec_js_frame_min,
        "consec_js_sustain_frac": p.consec_js_sustain_frac,
        "consec_js_ratio_min": p.consec_js_ratio_min,
    }


def _audit_with_vlm(project, cuts, sample, input_path, workdir, it, vlm,
                    seam_radius, emit, abort) -> Optional[dict[int, dict]]:
    """Generate seam clips for the sampled subset + run the VLM vote batch.

    Returns verdicts keyed by population position, or None when the budget
    ran out mid-audit (budget_exhausted).
    """
    from .seam import generate_seam_clips

    seams_dir = workdir / f"iter_{it}" / "seams"
    # Project cut index c (>=1) = seam cut_index c-1 (verified mapping).
    wanted = {}
    for s in sample:
        c = s["cut"].get("index")
        if c is None:
            continue
        wanted[int(c) - 1] = s["pos"]
    if not wanted:
        return {}

    generated = generate_seam_clips(
        input_path, project, outdir=seams_dir, max_seam_radius=seam_radius,
        cut_indices=sorted(wanted.keys()))

    # A7: flag seam clips that contain other cuts (timestamp injection).
    cut_secs = [c.get("seconds", 0.0) for c in cuts]
    audit_items = []
    for g in generated:
        pos = wanted.get(g["cut_index"])
        if pos is None:
            continue
        others = [t for t in cut_secs
                  if g["seam_start"] - 0.01 < t < g["seam_end"] - 0.01
                  and abs(t - g["cut_time"]) > 0.35]
        multicent = None
        if others:
            multicent = max(0.0, g["cut_time"] - g["seam_start"])
        audit_items.append({
            "key": f"cut_{pos}", "cut_index": int(g["cut_index"]) + 1,
            "path": g["path"], "multicent": multicent,
        })

    budget_needed = 2 * len(audit_items)
    if vlm.remaining_budget() < budget_needed:
        return None

    results = vlm.audit_seams_batch(audit_items)
    verdicts = {}
    for r, item in zip(results, audit_items):
        pos = wanted.get(item["cut_index"] - 1)
        verdicts[pos] = {"verdict": r.verdict, "confidence": r.confidence,
                         "agreement": r.agreement, "calls_used": r.calls_used,
                         "errors": r.errors}
        emit(_ev("seam_verdict", cut=item["cut_index"], verdict=r.verdict,
                 agreement=r.agreement, confidence=r.confidence,
                 errors=r.errors))
    return verdicts


def _run_recall_guard(project, input_path, workdir, it, vlm, audit,
                      emit, abort, suspected_misses) -> dict:
    """4 random interior windows + 2 longest scenes -> shotchange audit (A16)."""
    if audit != "vlm" or vlm is None:
        return {"skipped": "heuristic mode"}
    try:
        from .seam import generate_interior_clips
        guard_rng = random.Random(1000 + it)
        clips = generate_interior_clips(
            project, workdir / f"iter_{it}" / "interiors",
            n_windows=4, window=4.0, cap=6.0, longest=2, rng=guard_rng,
            source_path=input_path)
    except SceneCutError as e:
        return {"skipped": str(e)}

    if not clips:
        return {"sampled": 0, "suspected_misses": []}
    if vlm.remaining_budget() < len(clips):
        return {"sampled": 0, "skipped": "budget"}

    items = [{"key": f"int_{c['scene_index']}_{k}", "scene_index": c["scene_index"],
              "path": c["path"]} for k, c in enumerate(clips)]
    results = vlm.audit_interiors_batch(items)
    misses = []
    for r in results:
        emit(_ev("guard_verdict", scene=r.scene_index, shot_change=r.shot_change,
                 confidence=r.confidence, error=r.error))
        if r.shot_change == "yes":
            misses.append({"scene_index": r.scene_index,
                           "confidence": r.confidence,
                           "iteration": it})
    suspected_misses.extend(misses)
    return {"sampled": len(results), "suspected_misses": misses}
