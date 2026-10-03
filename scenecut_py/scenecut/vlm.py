"""VLM audit client — wraps the bun vlm_analyze.ts bridge for autotune (D9-impl).

Protocol (DECISIONS.md v3.1):
- 2-call vote per sampled seam with DECORRELATED prompts (A6): call 1 uses the
  built-in falsesplit prompt; call 2 uses a rephrased variant via VLM_PROMPT.
- Vote ties -> exactly 1 tie-break call (A3). Unparseable tie-break -> ABSTAIN
  (never FALSE — asymmetric conservatism poisons the acceptance gate).
- Infrastructure failures (timeout / non-zero exit / ok:false) retry once, then
  ABSTAIN (A8: per-call timeout 75s).
- Flattened two-phase batch execution (no nested pool deadlock): phase 1
  submits both vote calls for every seam; phase 2 submits tie-breaks only where
  needed.
- Budget accounting counts every real subprocess call (retries + tie-breaks).
"""
from __future__ import annotations

import json
import os
import re
import subprocess
import tempfile
import threading
import time
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

# Decorrelated vote variant (A6) — same semantics, different phrasing so
# prompt-anchored failure modes don't correlate across the two calls.
# D17 calibration (S4 measured: shared blind spot over-judged FALSE on
# low-contrast similar-content shots 8/14 vs GT ~2/17): both variants now
# (a) define "shot" operationally (one continuous camera recording),
# (b) enumerate the GENUINE classes that look continuous (same-subject
# re-frame / angle change / cutaway / match cut) — not just FALSE causes,
# (c) demand evidence BEFORE verdict, (d) allow UNCERTAIN so ambiguity
# abstains instead of coin-flipping FALSE.
VARIANT_B_PROMPT = (
    "Act as a film editor reviewing one flagged cut point (near the middle of "
    "the clip) from an automated editor. Question: at that point, did the "
    "recording switch from one camera setup to another camera setup?\n\n"
    "A switch between camera setups is REAL - answer GENUINE. Real switches "
    "include angle changes on the same subject, re-framings (close-up vs "
    "wide) of the same subject, cutaways within the same location, and any "
    "hard cut / dissolve / fade / wipe - including switches where both sides "
    "look very similar (same animal, same snow, same colors).\n"
    "NO switch means the detector erred - answer FALSE: one continuous "
    "recording in which the camera pans/tilts/zooms/tracks, focus pulls, "
    "exposure drifts, or a large object moves: the picture changes gradually "
    "and coherently over several frames, never all at once.\n\n"
    "Before deciding, list what you observe at the cut point: framing (shot "
    "size/angle) before vs after; subject position and scale before vs after; "
    "background before vs after; whether the change happens within a single "
    "frame or spreads over many frames.\n"
    "Answer:\n1. VERDICT: GENUINE / FALSE / UNCERTAIN\n2. CONFIDENCE: 0-100%\n"
    "3. REASON: one short sentence naming your strongest evidence.\n"
    "Guidance: similar-looking content is not evidence of an error. Say "
    "UNCERTAIN only if the observables genuinely conflict."
)

TIE_BREAK_PROMPT = (
    "Final decision needed. This clip contains a candidate cut near the "
    "midpoint. A shot = one continuous camera recording. Answer strictly:\n"
    "1. VERDICT: GENUINE / FALSE / UNCERTAIN - GENUINE if two different "
    "camera recordings are joined (hard cut, dissolve, fade, wipe, or match "
    "cut - even between visually similar shots of the same subject or scene); "
    "FALSE if it is one continuous recording (coherent multi-frame pan/tilt/"
    "zoom/tracking, focus pull, gradual exposure drift, or large subject "
    "motion - no single frame where the whole picture changes); UNCERTAIN "
    "only if the evidence genuinely conflicts.\n2. CONFIDENCE: 0-100%"
)

# Prompt with the audited cut's timestamp injected (A7 — multi-cut seam clips).
_MULTICUT_TEMPLATE = (
    "You are a professional film editor. This clip may contain MORE THAN ONE "
    "cut; judge ONLY the cut at approximately {t:.1f} seconds from the clip "
    "start. A shot = one continuous camera recording; every edit joining two "
    "recordings is a GENUINE cut - including same-subject re-frames, angle "
    "changes, cutaways, and match cuts between visually similar shots. The "
    "detector erred (FALSE) only if that point sits inside ONE continuous "
    "recording: coherent multi-frame pan/tilt/zoom, focus pull, gradual "
    "exposure drift, or large subject motion.\n\n"
    "Answer:\n1. VERDICT: GENUINE / FALSE / UNCERTAIN\n2. CONFIDENCE: 0-100%\n"
    "3. REASON: one short sentence naming your strongest evidence."
)


def _multicut_clause(t: float) -> str:
    """CR2-3: the same timestamp instruction appended to variant_b and
    tie-break prompts on multi-cut seam clips."""
    return (f"\n\nIMPORTANT: This clip contains multiple cuts; judge ONLY the "
            f"cut at approximately {t:.1f} seconds from the clip start.")


@dataclass
class CallOutcome:
    """One bridge subprocess call (post retry)."""
    key: str
    prompt: str           # "falsesplit" | "variant_b" | "tiebreak" | "shotchange" | "multicut"
    ok: bool              # infrastructure success (call completed + parseable file)
    verdict: Optional[str]   # "genuine" | "false" | "yes" | "no" | None
    confidence: Optional[float]
    content: str = ""
    error: Optional[str] = None


@dataclass
class SeamVerdict:
    """Aggregated verdict for one audited seam (2-call vote + optional tie-break)."""
    key: str
    cut_index: int                     # project cut list index (NOT seam cut_index)
    verdict: Optional[str]             # "genuine" | "false" | None (abstain)
    confidence: Optional[float]
    agreement: str                     # agree | tie_broken | single | abstain
    calls_used: int
    raw: list[str] = field(default_factory=list)
    errors: list[str] = field(default_factory=list)  # infra failure reasons (diagnostics)


@dataclass
class InteriorVerdict:
    key: str
    scene_index: int
    shot_change: Optional[str]         # "yes" | "no" | None (abstain)
    confidence: Optional[float]
    calls_used: int
    error: Optional[str] = None


# ------------------------------------------------------------------ parsing

_VERDICT_PATTERNS = [
    # "GENUINE OR FALSE: **GENUINE**", "VERDICT: GENUINE", "VERDICT: UNCERTAIN".
    # (?![-\w]) rejects hyphenated qualifiers ("false-ish", "genuine-ish") —
    # CR4-m1: they fall to the fallback which sees the fuller context.
    re.compile(r"(?:genuine\s+or\s+false|verdict)\s*[：:.]?\s*\**\s*(genuine|false|uncertain)(?![-\w])", re.I),
    # "SHOT CHANGE: YES"
    re.compile(r"shot\s+change\s*[：:.]?\s*\**\s*(yes|no)\b", re.I),
]
_CONF_PATTERN = re.compile(r"confidence\s*[：:.]?\s*\**\s*(\d{1,3})\s*%?", re.I)
# Lines that are template labels, not verdicts ("IF FALSE: N/A" echoes).
_LABEL_LINE = re.compile(r"^\s*\d?[.)]?\s*\**\s*if\s+(false|genuine|yes|no)\b", re.I)


def parse_verdict(text: str) -> tuple[Optional[str], Optional[float]]:
    """Parse a 3-way verdict (genuine/false/uncertain) + confidence from free
    VLM text.

    Returns (verdict, confidence) — verdict None means unparseable (-> abstain
    per A3; never guess FALSE from ambiguity). D17: "uncertain" is first-class;
    negated or qualified class words ("not false", "false-ish") never coerce
    to the class word they contain.
    """
    if not text:
        return None, None
    for pat in _VERDICT_PATTERNS:
        m = pat.search(text)
        if m:
            v = m.group(1).lower()
            break
    else:
        # Fallback: majority of standalone occurrences outside label lines.
        # D17: "uncertain" is a first-class fallback verdict — mixed
        # uncertain+class text must NOT coerce to a class (S4 bug class:
        # "VERDICT: UNCERTAIN - could be genuine" leaked the word "genuine"
        # and silently flipped the unit).
        body = "\n".join(ln for ln in text.splitlines() if not _LABEL_LINE.match(ln))
        # CR4-m1: a single alternation lookbehind (both branches 4 chars —
        # fixed-width holds) excludes negated mentions ("not false",
        # "wasn't genuine"): negation makes the mention evidence of nothing;
        # a text of ONLY negations lands on None (abstain) — the conservative
        # direction, never a class word.
        g = len(re.findall(r"(?<!not |n't )\bgenuine\b", body, re.I))
        f = len(re.findall(r"(?<!not |n't )\bfalse\b", body, re.I))
        u = len(re.findall(r"\buncertain\b", body, re.I))
        if g and not f and not u:
            v = "genuine"
        elif f and not g and not u:
            v = "false"
        elif u and not g and not f:
            v = "uncertain"
        elif g and u and not f:
            v = "uncertain"  # leaning-genuine ambiguity: not a confident genuine
        elif f and u and not g:
            v = "uncertain"  # leaning-false ambiguity: not a confident false
        else:
            y = len(re.findall(r"\byes\b", body, re.I))
            n = len(re.findall(r"\bno\b", body, re.I))
            if y and not n:
                v = "yes"
            elif n and not y:
                v = "no"
            else:
                v = None
    mc = _CONF_PATTERN.search(text)
    conf = min(1.0, max(0.0, int(mc.group(1)) / 100.0)) if mc else None
    return v, conf


# ------------------------------------------------------------------ client

def find_vlm_script() -> Path:
    """Resolve vlm_analyze.ts package-relative ONLY (A29 — cwd-dependent
    resolution broke when the Next.js route spawned the CLI from its own cwd)."""
    here = Path(__file__).resolve().parent
    candidates = [
        here.parent / "scripts" / "vlm_analyze.ts",
    ]
    for c in candidates:
        if c.exists():
            return c
    raise FileNotFoundError(
        "vlm_analyze.ts not found next to the scenecut package "
        f"(looked at {candidates[0]}); pass script_path explicitly")


class VLMClient:
    """Thread-safe batch audit client over the bun bridge."""

    def __init__(self,
                 script_path: str | Path | None = None,
                 timeout_s: float = 75.0,
                 workers: int = 3,
                 budget: int = 200):
        """workers=3 (not the spec's 4-6): the live endpoint returns 429 above
        ~3 concurrent video calls (measured S3) — backoff handles the rest."""
        self.script = Path(script_path) if script_path else find_vlm_script()
        self.timeout_s = timeout_s
        self.workers = workers
        self.budget = budget
        self._calls_used = 0
        self._lock = threading.Lock()

    @property
    def calls_used(self) -> int:
        return self._calls_used

    def remaining_budget(self) -> int:
        return self.budget - self._calls_used

    def _spend(self, n: int = 1) -> bool:
        with self._lock:
            if self._calls_used + n > self.budget:
                return False
            self._calls_used += n
            return True

    def _run_once(self, clip_path: str, prompt_type: str,
                  prompt_override: str | None) -> tuple[bool, str, Optional[str]]:
        """One bridge subprocess. Returns (ok, content, error)."""
        # ABSOLUTE clip path: the autotune CLI may run from a different cwd
        # than the VLM subprocess (repo root) — relative paths break bun.
        clip_abs = os.path.abspath(str(clip_path))
        fd, tmp = tempfile.mkstemp(suffix=".json", prefix="vlm_")
        os.close(fd)
        try:
            env = dict(os.environ)
            if prompt_override:
                env["VLM_PROMPT"] = prompt_override
            else:
                # CR2-6: ambient VLM_PROMPT would silently destroy the A6
                # decorrelation (both vote calls using the same prompt).
                env.pop("VLM_PROMPT", None)
            proc = subprocess.run(
                ["bun", "run", str(self.script), clip_abs, prompt_type, tmp],
                capture_output=True, text=True, timeout=self.timeout_s,
                cwd=str(self.script.parent.parent.parent),
            )
            if proc.returncode != 0:
                # The bridge prints {ok:false, error} to stdout on failure —
                # surface it for diagnosis (S3 live debugging).
                detail = ""
                try:
                    last = proc.stdout.strip().split("\n")[-1]
                    d = json.loads(last)
                    detail = f" ({d.get('error', '')[:120]})" if isinstance(d, dict) else ""
                except Exception:
                    pass
                return False, "", f"exit {proc.returncode}{detail}"
            try:
                with open(tmp) as f:
                    data = json.load(f)
                if not data.get("ok"):
                    return False, "", str(data.get("error", "vlm error"))[:200]
                return True, str(data.get("content", "")), None
            except (json.JSONDecodeError, OSError) as e:
                return False, "", f"unreadable output: {e}"
        except subprocess.TimeoutExpired:
            return False, "", "timeout"
        except FileNotFoundError:
            return False, "", "bun not found"
        except OSError as e:
            return False, "", f"spawn: {e}"
        finally:
            try:
                os.unlink(tmp)
            except OSError:
                pass

    def _call(self, key: str, clip_path: str, prompt_type: str,
              prompt_override: str | None = None) -> CallOutcome:
        """One logical call: retry on infrastructure failure (A8), with
        429-aware exponential backoff (measured S3: the endpoint throttles
        concurrent video calls — immediate retries just re-hit the limit)."""
        last_err: str | None = None
        for attempt in range(1, 4):  # up to 3 attempts when rate-limited
            if not self._spend():
                return CallOutcome(key, prompt_type, False, None, None,
                                   error="budget exhausted")
            ok, content, err = self._run_once(clip_path, prompt_type, prompt_override)
            if ok:
                verdict, conf = parse_verdict(content)
                return CallOutcome(key, prompt_type, True, verdict, conf, content)
            last_err = err
            # CR2-7: back off only when another attempt follows (no dead sleep
            # after the final failure).
            if err and "429" in err and attempt < 3:
                time.sleep(5.0 + 10.0 * (attempt - 1))  # 5s, 15s
                continue
            if attempt >= 2:  # non-429 failures: single retry only (A8)
                break
        return CallOutcome(key, prompt_type, False, None, None, error=last_err)

    # -------------------------------------------------------------- batches

    def audit_seams_batch(self, seams: list[dict]) -> list[SeamVerdict]:
        """Audit sampled seams: 2-call decorrelated vote + tie-break (A3/A6).

        Each seam dict: {key, cut_index, path, multicent?: float}
        (`multicent` = seconds of the audited cut within the clip, when the
        clip contains other cuts — A7 prompt injection).
        """
        if not seams:
            return []
        outcomes: dict[tuple[str, str], CallOutcome] = {}

        def submit(pool: ThreadPoolExecutor, seam: dict, prompt: str,
                   override: str | None):
            path = seam["path"]
            return pool.submit(self._call, seam["key"], path, prompt, override)

        # Phase 1: both vote calls for every seam, flattened (no nested pools).
        # CR2-3: the multicut timestamp clause applies to ALL prompts (call 1,
        # variant_b AND later tie-breaks) — a prompt that judges "the cut near
        # the middle" on a multi-cut clip audits the wrong cut.
        with ThreadPoolExecutor(max_workers=self.workers) as pool:
            futures = []
            for s in seams:
                if s.get("multicent") is not None:
                    override = _MULTICUT_TEMPLATE.format(t=s["multicent"])
                    futures.append((s["key"], "multicut", submit(pool, s, "falsesplit", override)))
                    vb = VARIANT_B_PROMPT + _multicut_clause(s["multicent"])
                    futures.append((s["key"], "variant_b", submit(pool, s, "falsesplit", vb)))
                else:
                    futures.append((s["key"], "falsesplit", submit(pool, s, "falsesplit", None)))
                    futures.append((s["key"], "variant_b", submit(pool, s, "falsesplit", VARIANT_B_PROMPT)))
            for key, prompt, fut in futures:
                outcomes[(key, prompt)] = fut.result()

        # Phase 2: tie-breaks where the two parsed verdicts disagree.
        need_tb = []
        for s in seams:
            a = outcomes.get((s["key"], "falsesplit")) or outcomes.get((s["key"], "multicut"))
            b = outcomes.get((s["key"], "variant_b"))
            if (a and a.ok and a.verdict in ("genuine", "false")
                    and b and b.ok and b.verdict in ("genuine", "false")
                    and a.verdict != b.verdict):
                need_tb.append(s)
        if need_tb:
            with ThreadPoolExecutor(max_workers=self.workers) as pool:
                futs = []
                for s in need_tb:
                    tb = TIE_BREAK_PROMPT
                    if s.get("multicent") is not None:  # CR2-3
                        tb = TIE_BREAK_PROMPT + _multicut_clause(s["multicent"])
                    futs.append((s["key"], pool.submit(self._call, s["key"], s["path"],
                                                       "falsesplit", tb)))
                for key, fut in futs:
                    outcomes[(key, "tiebreak")] = fut.result()

        # Assemble.
        results = []
        for s in seams:
            a = outcomes.get((s["key"], "falsesplit")) or outcomes.get((s["key"], "multicut"))
            b = outcomes.get((s["key"], "variant_b"))
            tb = outcomes.get((s["key"], "tiebreak"))
            results.append(VLMClient._assemble(a, b, tb, s["key"], s["cut_index"]))
        return results

    @staticmethod
    def _assemble(a: Optional[CallOutcome], b: Optional[CallOutcome],
                  tb: Optional[CallOutcome], key: str,
                  cut_index: int) -> SeamVerdict:
        """Fold two decorrelated votes (+ optional tie-break) into one verdict.

        D17 3-way semantics:
        - (confident, confident) same class -> agree; disagree -> tie-break
          (unparseable tie-break -> infra ABSTAIN per A3; an UNCERTAIN
          tie-break -> model-uncertain, agreement="uncertain").
        - (confident, uncertain) -> the confident vote stands, agreement
          "single" (decorrelation result preserved; the estimate event
          exposes single_rate + n_uncertain for downstream observability —
          NOTE: the HT estimator does NOT currently down-weight single
          votes; tightening on them is a documented calibration tradeoff).
        - (uncertain, uncertain) -> verdict None, agreement "uncertain"
          (model-level abstain — tracked separately from infra abstains in
          autotune.estimate_false_rate / _abstain_abort).
        """
        used = sum(1 for o in (a, b, tb) if o is not None)
        raw = [o.content for o in (a, b, tb) if o is not None and o.content]
        errs = [o.error for o in (a, b, tb) if o is not None and o.error]
        va = a.verdict if (a and a.ok) else None
        vb = b.verdict if (b and b.ok) else None
        if va in ("genuine", "false") and vb in ("genuine", "false"):
            if va == vb:
                verdict, agreement = va, "agree"
            elif tb and tb.ok and tb.verdict in ("genuine", "false"):
                verdict, agreement = tb.verdict, "tie_broken"
            elif tb and tb.ok and tb.verdict == "uncertain":
                verdict, agreement = None, "uncertain"
            else:
                # A3: unparseable tie-break -> ABSTAIN (never FALSE).
                verdict, agreement = None, "abstain"
        elif va == "uncertain" and vb == "uncertain":
            # Both votes honestly unsure: model-uncertain abstain.
            verdict, agreement = None, "uncertain"
        elif va == "uncertain" and vb in ("genuine", "false"):
            verdict, agreement = vb, "single"
        elif vb == "uncertain" and va in ("genuine", "false"):
            verdict, agreement = va, "single"
        elif va in ("genuine", "false"):
            verdict, agreement = va, "single"
        elif vb in ("genuine", "false"):
            verdict, agreement = vb, "single"
        else:
            verdict, agreement = None, "abstain"
        confs = [o.confidence for o in (a, b, tb)
                 if o and o.ok and o.confidence is not None]
        conf = round(sum(confs) / len(confs), 3) if confs else None
        return SeamVerdict(
            key=key, cut_index=cut_index, verdict=verdict,
            confidence=conf, agreement=agreement, calls_used=used, raw=raw,
            errors=errs)

    def audit_interiors_batch(self, clips: list[dict]) -> list[InteriorVerdict]:
        """Recall-guard windows: single shotchange call each."""
        if not clips:
            return []
        results = []
        with ThreadPoolExecutor(max_workers=self.workers) as pool:
            futs = {c["key"]: pool.submit(self._call, c["key"], c["path"], "shotchange", None)
                    for c in clips}
            for c in clips:
                o = futs[c["key"]].result()
                results.append(InteriorVerdict(
                    key=c["key"], scene_index=c["scene_index"],
                    shot_change=o.verdict if o.ok else None,
                    confidence=o.confidence, calls_used=1, error=o.error))
        return results
