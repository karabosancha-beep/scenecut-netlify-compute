"""Markdown cinematography report — PLAN Phase 2.3 (analysis export).

Generates a structured Markdown report from a project's stored VLM scene
analyses. Honest-by-construction: every statement in the report is sourced
from stored data (project JSON + analysis files); nothing is invented.

Where analyses live (repo reality, 2026-10):
  - Per-scene VLM analyses are written by `scenecut analyze` / the Next.js
    analyze/scenes route as FILES next to the project JSON:
        <project_dir>/analysis/scene_NNN_vlm.json
    with the bridge shape {ok, model, content, content_length, usage} where
    `content` is the raw free-form answer to the 14-point cinematography
    prompt (scripts/vlm_analyze.ts):
        1 FILM / 2 SHOT SIZE / 3 ANGLE / 4 MOVEMENT / 5 LENS / 6 COMPOSITION /
        7 LIGHTING / 8 COLOR / 9 MISE-EN-SCENE / 10 CONTENT / 11 MOOD /
        12 WHY BEAUTIFUL / 13 INFLUENCES / 14 STUDENT LESSON
    Older files may be raw API responses ({choices: [{message: {content}}]}).
  - The project JSON (project.py `empty_project`) has no analysis field yet;
    this module ALSO accepts a structured `project["analysis"]` list of
    per-scene entries ({scene_index, <field>: <value>, ...} and/or a raw
    `content` string), so a future structured store drops in without changes.

API:
    generate_report(project: dict, analyses: dict[int, dict] | None = None) -> str
    parse_scene_analysis(text: str) -> dict[str, str]
    load_scene_analyses(project_path) -> dict[int, dict]
    collect_analysis(project: dict, analyses: dict[int, dict] | None) -> dict[int, dict]
"""
from __future__ import annotations

import json
import re
import statistics
import time
from pathlib import Path

from . import __version__
from .project import apply_overrides
from .timecode import seconds_to_hmsms

# The 14-point cinematography prompt fields, in prompt order (canonical).
SCENE_FIELDS = [
    "FILM", "SHOT SIZE", "ANGLE", "MOVEMENT", "LENS", "COMPOSITION",
    "LIGHTING", "COLOR", "MISE-EN-SCENE", "CONTENT", "MOOD",
    "WHY BEAUTIFUL", "INFLUENCES", "STUDENT LESSON",
]

# Columns tried (in priority order) for the scene-overview table; the first
# three that exist in at least one analysis entry become table columns.
_DOMINANT_COLUMNS = ["SHOT SIZE", "ANGLE", "MOVEMENT", "LIGHTING", "COLOR", "MOOD"]

# Normalized (lowercase, spacing-collapsed) key aliases -> canonical field.
FIELD_ALIASES = {
    "film": "FILM",
    "shot size": "SHOT SIZE", "shotsize": "SHOT SIZE", "shot": "SHOT SIZE",
    "angle": "ANGLE", "camera angle": "ANGLE",
    "movement": "MOVEMENT", "camera movement": "MOVEMENT",
    "lens": "LENS", "focal length": "LENS",
    "composition": "COMPOSITION", "framing": "COMPOSITION",
    "lighting": "LIGHTING", "light": "LIGHTING",
    "color": "COLOR", "colour": "COLOR", "palette": "COLOR",
    "mise en scene": "MISE-EN-SCENE", "miseenscene": "MISE-EN-SCENE",
    "mise en scène": "MISE-EN-SCENE",
    "content": "CONTENT", "description": "CONTENT", "action": "CONTENT",
    "mood": "MOOD", "atmosphere": "MOOD",
    "why beautiful": "WHY BEAUTIFUL",
    "influences": "INFLUENCES", "references": "INFLUENCES",
    "student lesson": "STUDENT LESSON", "lesson": "STUDENT LESSON",
}

# Entry keys that are metadata, not cinematography fields.
_META_KEYS = {
    "scene_index", "index", "scene", "file", "result", "cached", "ok",
    "error", "model", "content", "content_length", "usage", "analyzed_at",
    "path", "clip", "start", "end", "duration", "thumbnail", "label", "tags",
    "frame", "frames",
}

# Vocabulary for simple frequency observations. Only reported when the stored
# analysis text actually contains the token (earliest match wins) — no
# invented categories.
_VOCAB: dict[str, list[str]] = {
    "SHOT SIZE": [
        "extreme close-up", "medium close-up", "medium long shot",
        "extreme long shot", "establishing shot", "over-the-shoulder",
        "over the shoulder", "close-up", "close up", "medium shot",
        "full shot", "long shot", "wide shot", "two shot", "macro", "insert",
    ],
    "ANGLE": [
        "eye-level", "eye level", "low angle", "high angle", "dutch angle",
        "canted", "overhead", "top-down", "bird's-eye", "birds eye",
        "worm's-eye", "subjective", "pov",
    ],
    "MOVEMENT": [
        "whip pan", "locked down", "pull back", "pull out", "push in",
        "static", "tripod", "pan", "tilt", "zoom", "dolly", "tracking",
        "truck", "pedestal", "crane", "jib", "handheld", "steadicam",
        "gimbal",
    ],
    "LIGHTING": [
        "high-key", "high key", "low-key", "low key", "natural light",
        "available light", "chiaroscuro", "side-lit", "side lit", "backlit",
        "back-lit", "back lit", "front-lit", "silhouette", "practical",
        "three-point", "three point", "hard light", "soft light",
        "golden hour", "daylight", "moonlight",
    ],
}
# Token variants normalized to one spelling for counting.
_TOKEN_CANON = {
    "close up": "close-up", "eye level": "eye-level", "high key": "high-key",
    "low key": "low-key", "side lit": "side-lit", "back-lit": "backlit",
    "back lit": "backlit", "three point": "three-point",
    "birds eye": "bird's-eye",
}

# Field-heading line of the 14-point VLM answer, tolerant of numbering
# ("2." / "3)" / bare "7 "), bold ("**ANGLE**"), and ":" / "-" / "—" separators.
_FIELD_LINE = re.compile(
    r"^\s*(?:\d{1,2}[.)\s]+)?\**\s*"
    r"(FILM|SHOT\s*SIZE|ANGLE|MOVEMENT|LENS|COMPOSITION|LIGHTING|COLOR|COLOUR"
    r"|MISE[\s\-]*EN[\s\-]*SC[ÈE]NE|CONTENT|MOOD|WHY\s*BEAUTIFUL|INFLUENCES"
    r"|STUDENT\s*LESSON)"
    r"\s*\**\s*[：:\-–—]?\s*(.*)$",
    re.IGNORECASE,
)
_SCENE_FILE = re.compile(r"^scene_(\d+)_vlm\.json$")


# --------------------------------------------------------------- text parsing

def _clean(text: str) -> str:
    """Collapse whitespace/newlines to single spaces."""
    return re.sub(r"\s+", " ", str(text)).strip()


def _canonical_field(name) -> str:
    """Map an arbitrary stored key to a canonical display field name."""
    norm = re.sub(r"[\s_\-]+", " ", str(name).strip().lower()).strip()
    if norm in FIELD_ALIASES:
        return FIELD_ALIASES[norm]
    return " ".join(w[:1].upper() + w[1:] for w in norm.split()) or str(name)


def parse_scene_analysis(text: str) -> dict[str, str]:
    """Parse the raw VLM 14-point cinematography answer into {FIELD: text}.

    Continuation lines (no new field heading) are appended to the current
    field. Fields absent from the answer are absent from the result — no
    placeholder values.
    """
    fields: dict[str, str] = {}
    current: str | None = None
    for line in (text or "").splitlines():
        m = _FIELD_LINE.match(line)
        if m:
            current = _canonical_field(m.group(1))
            fields.setdefault(current, "")
            val = m.group(2).strip().strip("*_").strip()
            if val:
                fields[current] = (fields[current] + " " + val).strip()
        elif current is not None:
            val = line.strip().strip("*_").strip()
            if val:
                fields[current] = (fields[current] + " " + val).strip()
    return {k: _clean(v) for k, v in fields.items() if _clean(v)}


def _value_str(v) -> str:
    """Render an arbitrary stored value as report text."""
    if isinstance(v, bool):
        return "yes" if v else "no"
    if isinstance(v, (int, float)):
        return str(v)
    if isinstance(v, (list, tuple)):
        return ", ".join(_value_str(x) for x in v)
    if isinstance(v, dict):
        return json.dumps(v, default=str, ensure_ascii=False)
    return _clean(v)


def _normalize_entry(entry: dict) -> dict:
    """One analysis entry -> flat {FIELD: text, _model, _source_file} dict."""
    fields: dict[str, str] = {}
    content = entry.get("content")
    if isinstance(content, str) and content.strip():
        fields.update(parse_scene_analysis(content))
    for k, v in entry.items():
        if k in _META_KEYS or k.startswith("_"):
            continue
        if v is None or v == "" or v == []:
            continue
        fields[_canonical_field(k)] = _value_str(v)
    out = dict(fields)
    if entry.get("model"):
        out["_model"] = str(entry["model"])
    if entry.get("file"):
        out["_source_file"] = str(entry["file"])
    return out


def _project_entries(project: dict) -> dict[int, dict]:
    """Extract {scene_index: normalized entry} from project["analysis"].

    Accepts a list of entries, a {"scenes": [...]} wrapper (the consolidated
    report.json shape), or a {index: entry} mapping. Each entry identifies its
    scene via scene_index / index / scene (int or numeric string).
    """
    raw = project.get("analysis")
    items: list[dict] = []
    if isinstance(raw, dict):
        scenes = raw.get("scenes")
        if isinstance(scenes, list):
            items = [e for e in scenes if isinstance(e, dict)]
        else:
            for k, v in raw.items():
                if isinstance(v, dict):
                    try:
                        items.append({**v, "scene_index": int(k)})
                    except (TypeError, ValueError):
                        continue
    elif isinstance(raw, list):
        items = [e for e in raw if isinstance(e, dict)]
    out: dict[int, dict] = {}
    for e in items:
        idx = e.get("scene_index", e.get("index", e.get("scene")))
        try:
            idx = int(idx)
        except (TypeError, ValueError):
            continue
        entry = _normalize_entry(e)
        if entry:
            out[idx] = entry
    return out


def collect_analysis(project: dict,
                     analyses: dict[int, dict] | None = None) -> dict[int, dict]:
    """Merge analysis sources: explicit `analyses` (e.g. disk-loaded) first,
    then project["analysis"] entries (explicit storage wins on conflict)."""
    entries: dict[int, dict] = {}
    if analyses:
        for idx, e in analyses.items():
            if e:
                entries[idx] = e
    for idx, e in _project_entries(project).items():
        entries[idx] = e
    return entries


# ---------------------------------------------------------------- disk loader

def load_scene_analyses(project_path) -> dict[int, dict]:
    """Read <project_dir>/analysis/scene_NNN_vlm.json files.

    Accepts a project DIRECTORY or the project JSON path (the analysis dir is
    its sibling). Supports the bridge shape ({ok, model, content, ...}) and
    legacy raw API responses ({choices: [{message: {content}}]}). Failed
    (ok:false / unreadable / contentless) files are skipped.
    """
    p = Path(project_path)
    analysis_dir = (p / "analysis") if p.is_dir() else (p.parent / "analysis")
    out: dict[int, dict] = {}
    if not analysis_dir.is_dir():
        return out
    for f in sorted(analysis_dir.iterdir()):
        m = _SCENE_FILE.match(f.name)
        if not m or not f.is_file():
            continue
        try:
            data = json.loads(f.read_text(encoding="utf-8"))
        except (json.JSONDecodeError, OSError):
            continue
        if not isinstance(data, dict):
            continue
        content = data.get("content")
        if not content:
            choices = data.get("choices")
            if isinstance(choices, list) and choices:
                try:
                    content = choices[0]["message"]["content"]
                except (KeyError, IndexError, TypeError):
                    content = None
        if not isinstance(content, str) or not content.strip():
            continue
        entry = parse_scene_analysis(content)
        if data.get("model"):
            entry["_model"] = str(data["model"])
        entry["_source_file"] = f.name
        if entry:
            out[int(m.group(1))] = entry
    return out


# ------------------------------------------------------------ md formatting

def _md_escape(text: str) -> str:
    """Make free text safe inside a Markdown table cell / inline context."""
    return _clean(text).replace("|", "\\|")


def _shorten(text: str, limit: int = 48) -> str:
    text = _clean(text)
    if len(text) <= limit:
        return text
    cut = text[:limit]
    if " " in cut:
        cut = cut.rsplit(" ", 1)[0]
    return cut.rstrip(" ,;:-–—") + "…"


def _categorize(field: str, value: str) -> str | None:
    """Earliest vocabulary token found in the stored text, else None."""
    tokens = _VOCAB.get(field)
    if not tokens or not value:
        return None
    low = value.lower()
    best_start, best_tok = None, None
    for tok in tokens:
        m = re.search(rf"\b{re.escape(tok)}\b", low)
        if m and (best_start is None or m.start() < best_start
                  or (m.start() == best_start and len(tok) > len(best_tok or ""))):
            best_start, best_tok = m.start(), tok
    if best_tok is None:
        return None
    return _TOKEN_CANON.get(best_tok, best_tok)


def _entry_fields(entry: dict) -> dict[str, str]:
    """Cinematography fields of an entry, canonical order first."""
    known = {f: entry[f] for f in SCENE_FIELDS if f in entry}
    extras = {k: v for k, v in entry.items()
              if not k.startswith("_") and k not in known}
    known.update(extras)
    return known


def _frequency_line(field: str, entries: dict[int, dict]) -> str | None:
    """'- Shot size: medium shot ×2, close-up ×1 (3/3 scenes classified)'."""
    counts: dict[str, int] = {}
    classified = 0
    for entry in entries.values():
        cat = _categorize(field, entry.get(field, ""))
        if cat:
            counts[cat] = counts.get(cat, 0) + 1
            classified += 1
    if not counts:
        return None
    ranked = sorted(counts.items(), key=lambda kv: (-kv[1], kv[0]))
    pretty = ", ".join(f"{cat} ×{n}" for cat, n in ranked)
    label = field.lower()
    return f"- {label}: {pretty} ({classified}/{len(entries)} scenes classified)"


def _scene_header(i: int, scene: dict) -> str:
    start = seconds_to_hmsms(scene.get("start", 0))
    end = seconds_to_hmsms(scene.get("end", 0))
    dur = scene.get("duration", 0)
    return f"### Scene {i + 1} — {start} → {end} ({dur:.2f}s)"


# ---------------------------------------------------------------- the report

def generate_report(project: dict, analyses: dict[int, dict] | None = None) -> str:
    """Generate the Markdown cinematography report for a project dict.

    `analyses` optionally supplies {scene_index: entry} entries (e.g. loaded
    from the analysis/ dir by `load_scene_analyses`); entries stored inside
    the project dict itself (project["analysis"]) are always honored and win
    on conflict.
    """
    try:
        applied = apply_overrides(project)
    except Exception:
        applied = project  # defensive: report on raw data rather than fail
    scenes = applied.get("scenes", []) or []
    src = applied.get("source", {}) or {}
    entries = collect_analysis(applied, analyses)

    n_scenes = len(scenes)
    n_analyzed = sum(1 for i in range(n_scenes) if i in entries)
    unanalyzed = [i for i in range(n_scenes) if i not in entries]
    orphans = sorted(i for i in entries if i >= n_scenes)

    name = (project.get("name") or project.get("id")
            or Path(str(src.get("path", ""))).stem or "Untitled")
    duration = src.get("duration", 0) or 0
    fps = src.get("fps", 0) or 0

    lines: list[str] = []
    lines.append(f"# Cinematography Report — {name}")
    lines.append("")
    lines.append(f"_Generated by SceneCut v{__version__} on "
                 f"{time.strftime('%Y-%m-%d %H:%M UTC', time.gmtime())}_")
    lines.append("")

    # -- Summary ------------------------------------------------------------
    lines.append("## Summary")
    lines.append("")
    lines.append(f"- **Project**: {name}"
                 + (f" (id `{project.get('id')}`)" if project.get("id") else ""))
    src_name = Path(str(src.get("path", ""))).name or "unknown source"
    res = ""
    if src.get("width") and src.get("height"):
        res = f", {src['width']}×{src['height']}"
    lines.append(f"- **Source**: {src_name} — {seconds_to_hmsms(duration)} "
                 f"({duration:.2f}s @ {fps:g}fps{res})")
    lines.append(f"- **Scenes**: {n_scenes} — {n_analyzed} analyzed, "
                 f"{len(unanalyzed)} without analysis")
    if n_scenes:
        pct = round(100.0 * n_analyzed / n_scenes)
        lines.append(f"- **Analysis coverage**: {n_analyzed}/{n_scenes} scenes ({pct}%)")
    else:
        lines.append("- **Analysis coverage**: no scenes in project")
    lines.append("")

    # -- Scene overview table -------------------------------------------------
    lines.append("## Scene Overview")
    lines.append("")
    cols = [c for c in _DOMINANT_COLUMNS if any(c in e for e in entries.values())][:3]
    header = ["#", "Start", "End", "Dur"] + cols
    lines.append("| " + " | ".join(header) + " |")
    lines.append("|" + "|".join("---" for _ in header) + "|")
    for i, scene in enumerate(scenes):
        entry = entries.get(i)
        row = [
            str(i + 1),
            seconds_to_hmsms(scene.get("start", 0)),
            seconds_to_hmsms(scene.get("end", 0)),
            f"{scene.get('duration', 0):.2f}s",
        ]
        for c in cols:
            row.append(_shorten(_md_escape(entry[c])) if entry and entry.get(c) else "—")
        lines.append("| " + " | ".join(row) + " |")
    lines.append("")

    # -- Observations ---------------------------------------------------------
    lines.append("## Observations")
    lines.append("")
    obs: list[str] = []
    if n_scenes > 1:
        durs = [s.get("duration", 0) for s in scenes]
        short_i = min(range(n_scenes), key=lambda i: durs[i])
        long_i = max(range(n_scenes), key=lambda i: durs[i])
        obs.append(f"- Durations: mean {statistics.mean(durs):.2f}s · "
                   f"median {statistics.median(durs):.2f}s · "
                   f"shortest scene {short_i + 1} ({durs[short_i]:.2f}s) · "
                   f"longest scene {long_i + 1} ({durs[long_i]:.2f}s)")
    if entries:
        # CR4-m4: frequency stats count only analyses bound to an EXISTING
        # scene — orphan (stale) entries mislead the vocabulary counts and
        # the coverage denominator (probed: 2-scene project reporting
        # "3/3 scenes classified").
        valid_entries = {i: e for i, e in entries.items() if i < n_scenes}
        for field in ("SHOT SIZE", "ANGLE", "MOVEMENT", "LIGHTING"):
            line = _frequency_line(field, valid_entries)
            if line:
                obs.append(line)
    if obs:
        lines.extend(obs)
        lines.append("")
        lines.append("_Frequency stats are computed only from tokens present in "
                     "the stored analyses — no data is invented._")
    else:
        lines.append("_No aggregate observations available "
                     f"({n_analyzed} analyzed scene"
                     + ("s" if n_analyzed != 1 else "") + ")._")
    lines.append("")

    # -- Per-scene details ----------------------------------------------------
    lines.append("## Scene Details")
    lines.append("")
    if n_analyzed:
        for i, scene in enumerate(scenes):
            entry = entries.get(i)
            if entry is None:
                continue
            lines.append(_scene_header(i, scene))
            lines.append("")
            prov = []
            if entry.get("_model"):
                prov.append(f"model: {entry['_model']}")
            if entry.get("_source_file"):
                prov.append(f"source: {entry['_source_file']}")
            if prov:
                lines.append(f"*{' · '.join(prov)}*")
                lines.append("")
            if scene.get("label"):
                lines.append(f"**Label** — {_md_escape(scene['label'])}")
                lines.append("")
            if scene.get("tags"):
                lines.append(f"**Tags** — {_md_escape(', '.join(scene['tags']))}")
                lines.append("")
            for field, value in _entry_fields(entry).items():
                lines.append(f"**{field}** — {_md_escape(value)}")
                lines.append("")
    else:
        lines.append("_No scenes have stored VLM analysis._")
        lines.append("")

    # -- Unanalyzed scenes ------------------------------------------------------
    if unanalyzed:
        lines.append("## Unanalyzed Scenes")
        lines.append("")
        listing = ", ".join(
            f"Scene {i + 1} ({seconds_to_hmsms(scenes[i].get('start', 0))} → "
            f"{seconds_to_hmsms(scenes[i].get('end', 0))})" for i in unanalyzed)
        lines.append(f"{listing} lack stored VLM analysis — run "
                     "`scenecut cut --for-vlm` then `scenecut analyze --what scenes` "
                     "to generate them.")
        lines.append("")
    if orphans:
        lines.append(f"> Note: {len(orphans)} stored analysis "
                     f"entr{'y' if len(orphans) == 1 else 'ies'} reference scene "
                     f"index{'es' if len(orphans) > 1 else ''} beyond the current "
                     f"scene list ({', '.join(str(i) for i in orphans)}) — "
                     "stale after cut edits; not shown above.")
        lines.append("")

    lines.append("---")
    lines.append(f"*Generated by SceneCut v{__version__} from stored project data "
                 "and VLM analyses.*")
    lines.append("")
    return "\n".join(lines)
