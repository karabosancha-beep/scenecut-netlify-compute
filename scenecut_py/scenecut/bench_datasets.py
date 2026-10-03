"""Benchmark dataset loaders for `scenecut bench` (DECISIONS.md D12, v3.1).

Internal ground-truth convention — 0-based "first frame of the new shot".
Per-dataset conversion table (A17):

==================  =======================  =====================================
dataset             source format            conversion
==================  =======================  =====================================
local               samples/                 already internal; dissolve/fade
                    ground_truth.json        positions are measured midpoints/
                                             onsets treated as POINT cuts
                                             (gt_gradual == [] — local is
                                             hard-cut-only scoring).
bbc                 fixed/<NN>-scenes.txt    tab-separated contiguous inclusive
                                             0-based [start, end] shot spans;
                                             cut = end + 1 (trailing span's
                                             cut is the video end, not a cut).
clipshots           annotations/*.json       transitions [[start, end]]: hard
                                             cut iff end == start + 1 ->
                                             cut = end; wider spans ->
                                             gradual interval, half-open
                                             [start, end).
rai                 RAIDataset/scenes_N.txt  1-based inclusive [start, end]
                                             SCENE spans (a scene = several
                                             shots); cut_0based = next start
                                             - 1. Scene-level: intra-scene
                                             shot cuts are unannotated (they
                                             are NOT errors — see loader
                                             notes; precision is a lower
                                             bound, recall is comparable).
sbd                 gt.json + clips/         61-frame snippets, boundary
                                             centered on 0-indexed frame 30:
                                             C -> cut@30; T -> gradual
                                             [20,40) (center±10 convention);
                                             E -> negative (FP counts).
==================  =======================  =====================================

Boundary events at frame 0 are never cut events and are dropped at load.

ClipShots offset verification (A14): annotation and decoder timelines can
disagree by one frame, so `resolve_clipshots_offset` searches the per-video
best offset in {-1, 0, +1} maximizing τ=0 matches against the PREDICTIONS.
bench.py calls it after detection (predictions are needed for resolution) and
applies the winning offset to the ground truth. An ambiguous maximum (a tie —
including the all-zero case) or a detection error excludes the video
(fail-closed) and is logged + recorded in `evaluated_manifest.skipped`.
"""
from __future__ import annotations

import json
import logging
import shutil
import subprocess
import zipfile
from collections import Counter
from pathlib import Path
from typing import Any, Callable

log = logging.getLogger("scenecut.bench")

# Entry shape (all loaders): {"video_path": str, "gt_cuts": [int],
# "gt_gradual": [(start, end), ...] half-open, "meta": {...}}.
VideoEntry = dict[str, Any]

# Zenodo record 14873790 "BBC Dataset (Annotation Fixed)" — CC-BY-4.0.
# NOTE: record 14865504 carries the same videos with BROKEN annotations
# (first shot "0 632" vs fixed "0 649") — never use it.
BBC_ZENODO_RECORD = "14873790"
BBC_URLS: dict[str, str] = {
    "fixed.zip": f"https://zenodo.org/records/{BBC_ZENODO_RECORD}/files/fixed.zip",
    "videos.zip": f"https://zenodo.org/records/{BBC_ZENODO_RECORD}/files/videos.zip",
}

_VIDEO_SUFFIXES = {".mp4", ".mkv", ".mov", ".mxf", ".avi", ".webm"}


# ------------------------------------------------------------------ local


def load_local(samples_dir: str | Path) -> list[VideoEntry]:
    """Load the in-repo fixture corpus from ``samples/ground_truth.json`` (D12).

    type_hint "none" (encoded as an empty ``expected_cuts`` list) means the
    fixture must produce ZERO cuts. Dissolve/fade hints contribute their
    measured position as a point cut; ``gt_gradual`` is always ``[]`` for the
    local corpus — it is scored hard-cut-only, with dissolve positions
    treated as point cuts (fixture ground truth has measured positions only,
    not derivable transition spans).
    """
    d = Path(samples_dir)
    gt_path = d / "ground_truth.json"
    if not gt_path.is_file():
        raise FileNotFoundError(f"ground_truth.json not found at {gt_path}")
    gt = json.loads(gt_path.read_text())
    entries: list[VideoEntry] = []
    for fx in gt.get("fixtures", []):
        name = fx.get("fixture", "")
        path = d / name
        if not path.is_file():
            log.warning("bench local: fixture %s missing on disk — skipped", name)
            continue
        gt_cuts = [
            int(c["frame"]) for c in fx.get("expected_cuts", [])
            if str(c.get("type_hint", "cut")).strip().lower() != "none"
            and int(c["frame"]) > 0
        ]
        entries.append({
            "video_path": str(path),
            "gt_cuts": sorted(gt_cuts),
            "gt_gradual": [],
            "meta": {
                "dataset": "local",
                "purpose": fx.get("purpose", ""),
                "type_hints": [c.get("type_hint") for c in fx.get("expected_cuts", [])],
                "frames": fx.get("frames"),
                "fps": fx.get("fps"),
                "duration": fx.get("duration"),
            },
        })
    entries.sort(key=lambda e: Path(e["video_path"]).name)
    return entries


# ------------------------------------------------------------------ bbc


def _bbc_annotation_files(bbc_dir: Path) -> list[Path]:
    """Find ``*-scenes.txt`` annotation files under a BBC dataset dir."""
    hits = list(bbc_dir.glob("fixed/*-scenes.txt")) + list(bbc_dir.glob("*-scenes.txt"))
    return sorted(set(hits))


def _find_bbc_video(bbc_dir: Path, stem: str) -> Path | None:
    """Locate ``bbc_<stem>.mp4`` — direct, under videos/, or by recursive glob."""
    for cand in (bbc_dir / f"bbc_{stem}.mp4",
                 bbc_dir / "videos" / f"bbc_{stem}.mp4",
                 bbc_dir / f"bbc_{stem}.MP4"):
        if cand.is_file():
            return cand
    hits = [p for p in bbc_dir.rglob(f"bbc_{stem}.*")
            if p.is_file() and p.suffix.lower() in _VIDEO_SUFFIXES]
    return sorted(hits)[0] if hits else None


def load_bbc(bbc_dir: str | Path) -> list[VideoEntry]:
    """Load BBC Planet Earth (Zenodo 14873790, CC-BY-4.0; D12).

    Reads ``fixed/<NN>-scenes.txt`` (tab-separated contiguous inclusive
    0-based [start, end] shot spans per line) plus the matching video
    ``bbc_NN.mp4``. Cut = end + 1; the trailing span's cut is the video end
    (not a cut event) and is dropped, as are cut events at frame 0.
    ``gt_gradual`` is always ``[]`` (BBC PE annotations are shot-level only).
    Videos missing on disk are skipped + logged.
    """
    d = Path(bbc_dir)
    anns = _bbc_annotation_files(d)
    if not anns:
        raise FileNotFoundError(
            f"no fixed/*-scenes.txt annotations under {d} — see datasets/README.md "
            "(download_bbc or `scenecut bench --download bbc`)")
    entries: list[VideoEntry] = []
    for txt in anns:
        stem = txt.name[: -len("-scenes.txt")]
        video = _find_bbc_video(d, stem)
        if video is None:
            log.warning("bench bbc: video for %s not found — skipped", txt.name)
            continue
        cuts: list[int] = []
        spans: list[tuple[int, int]] = []
        bad = 0
        for line in txt.read_text().splitlines():
            fields = line.strip().split()
            if not fields:
                continue
            try:
                start, end = int(fields[0]), int(fields[1])
            except (ValueError, IndexError):
                bad += 1
                continue
            if start < 0 or end < start:
                bad += 1
                continue
            spans.append((start, end))
        # cut = end + 1 for every span EXCEPT the last (its end+1 is the video
        # end — the detector drops end-of-video cuts, so keeping it would be a
        # guaranteed FN). Frame-0 events are skipped defensively.
        for start, end in spans[:-1]:
            cut = end + 1
            if cut > 0:
                cuts.append(cut)
        if bad:
            log.warning("bench bbc: %d malformed lines in %s", bad, txt.name)
        entries.append({
            "video_path": str(video),
            "gt_cuts": cuts,
            "gt_gradual": [],
            "meta": {
                "dataset": "bbc_planet_earth",
                "episode": stem,
                "n_shots": len(spans),
                "annotation_file": txt.name,
            },
        })
    return entries


def download_bbc(dest: str | Path) -> Path:
    """Download + extract BBC Planet Earth into ``dest``; returns ``dest``.

    ``wget -c`` (resumable) of fixed.zip + videos.zip (≈4.7 GB) from Zenodo
    record 14873790, then extraction, then verification: at least one
    annotation file must exist AND parse into integer shot spans before
    success is returned. Raises RuntimeError on wget/extraction/verification
    failure; FileNotFoundError-style early return if annotations are already
    present and valid (idempotent re-run).
    """
    dest = Path(dest)
    dest.mkdir(parents=True, exist_ok=True)
    if not _bbc_annotation_files(dest):
        wget = shutil.which("wget")
        if wget is None:
            raise RuntimeError("wget not found on PATH (required for resumable download)")
        for fname, url in BBC_URLS.items():
            zpath = dest / fname
            log.info("bench bbc: downloading %s", url)
            subprocess.run([wget, "-c", "-O", str(zpath), url], check=True)
            with zipfile.ZipFile(zpath) as zf:
                zf.extractall(dest)
    anns = _bbc_annotation_files(dest)
    if not anns:
        raise RuntimeError(f"BBC download verification failed: no *-scenes.txt under {dest}")
    txt = anns[0]
    n_spans = 0
    for line in txt.read_text().splitlines():
        fields = line.strip().split()
        if len(fields) >= 2:
            try:
                int(fields[0]), int(fields[1])
                n_spans += 1
            except ValueError:
                continue
    if n_spans == 0:
        raise RuntimeError(f"BBC download verification failed: {txt} has no parseable spans")
    log.info("bench bbc: %d annotation files ready under %s", len(anns), dest)
    return dest


# ------------------------------------------------------------------ clipshots


def _find_clipshots_video(clipshots_dir: Path, name: str) -> Path | None:
    """Locate a ClipShots video by annotation key (with/without extension)."""
    stem = Path(name).stem
    for cand in (clipshots_dir / "videos" / name, clipshots_dir / name,
                 clipshots_dir / "videos" / stem, clipshots_dir / stem):
        if cand.is_file():
            return cand
    return None


def load_clipshots(clipshots_dir: str | Path, split: str = "test") -> list[VideoEntry]:
    """Load ClipShots annotations (D12; opt-in stress corpus, MIT).

    Annotation JSON shape ``{<video>: {"transitions": [[start, end], ...],
    "frame_num": N}}``: hard cut ⇔ ``end == start + 1`` → ``cut = end``;
    wider spans → ``gt_gradual = (start, end)`` half-open [start, end)
    (transition begins at ``start``, next shot begins at ``end``).
    Videos missing on disk are skipped + logged.
    """
    d = Path(clipshots_dir)
    ann_path = None
    for cand in (d / "annotations" / f"{split}.json", d / f"{split}.json"):
        if cand.is_file():
            ann_path = cand
            break
    if ann_path is None:
        raise FileNotFoundError(
            f"ClipShots annotations for split '{split}' not found under {d} — see "
            "datasets/README.md (annotations from github.com/Tangshitao/ClipShots)")
    data = json.loads(ann_path.read_text())
    entries: list[VideoEntry] = []
    for video_name in sorted(data):
        info = data[video_name] or {}
        transitions = info.get("transitions") or []
        path = _find_clipshots_video(d, video_name)
        if path is None:
            log.warning("bench clipshots: video %s missing on disk — skipped", video_name)
            continue
        gt_cuts: list[int] = []
        gt_gradual: list[tuple[int, int]] = []
        for tr in transitions:
            try:
                s, e = int(tr[0]), int(tr[1])
            except (ValueError, IndexError, TypeError):
                continue
            if e == s + 1:
                if e > 0:  # skip boundary events at frame 0
                    gt_cuts.append(e)
            else:
                gt_gradual.append((s, e))
        entries.append({
            "video_path": str(path),
            "gt_cuts": sorted(gt_cuts),
            "gt_gradual": gt_gradual,
            "meta": {
                "dataset": "clipshots",
                "split": split,
                "frame_num": info.get("frame_num"),
                "n_transitions": len(transitions),
            },
        })
    return entries


def resolve_clipshots_offset(pred_frames: list[int], gt_cuts: list[int],
                              offsets: tuple[int, ...] = (-1, 0, 1)) -> int | None:
    """A14: per-video best-offset search over ``offsets`` (default {-1, 0, +1}).

    Each candidate offset is scored by the τ=0 exact-match count (multiset
    1-to-1) between the predictions and the GT shifted by the offset.
    Returns the unique argmax, or ``None`` when the maximum is ambiguous (a
    tie between two or more offsets — including the all-zero case that arises
    when there are no GT cuts or the detector missed everything): the caller
    must then EXCLUDE the video (fail-closed) and log it.
    """
    pred_counts = Counter(int(p) for p in pred_frames)
    scores: dict[int, int] = {}
    for off in offsets:
        gt_counts = Counter(int(g) + off for g in gt_cuts)
        scores[off] = sum(min(c, gt_counts.get(f, 0)) for f, c in pred_counts.items())
    best = max(scores.values())
    winners = [off for off, score in scores.items() if score == best]
    return winners[0] if len(winners) == 1 else None


# ------------------------------------------------------------------ rai


def load_rai(rai_dir: str | Path) -> list[VideoEntry]:
    """Load RAI scene-detection corpus (HF mirror RorooroR/RaiSceneDetection;
    research-use license, 10 broadcast videos ~hours total).

    ``RAIDataset/scenes_N.txt`` holds one 1-based inclusive [start, end]
    SCENE span per line, contiguous. Boundary between scene k and k+1:
    0-based first-frame-of-next-scene = ``start_{k+1} - 1`` (== ``end_k``
    when contiguous — both 1-based). The trailing span's boundary is the
    video end (not a cut event) and is dropped, as are frame-0 events.

    **Scene-level GT caveat (recorded in ``meta`` and bench notes)**: a RAI
    scene groups several shots; intra-scene shot cuts are unannotated and
    are NOT detector errors. Against this GT, RECALL is directly comparable
    (scene boundaries are real shot/transition boundaries) but PRECISION is
    a LOWER BOUND only — use it for regression tracking, never for headline
    accuracy claims.
    """
    d = Path(rai_dir)
    root = d / "RAIDataset" if (d / "RAIDataset").is_dir() else d
    anns = sorted(root.glob("scenes_*.txt"))
    if not anns:
        raise FileNotFoundError(
            f"no scenes_*.txt under {d} — see datasets/README.md (RAI via HF "
            "mirror RorooroR/RaiSceneDetection)")
    entries: list[VideoEntry] = []
    for txt in anns:
        num = txt.stem[len("scenes_") :]
        video = root / "videos" / f"{num}.mp4"
        if not video.is_file():
            log.warning("bench rai: video for %s not found — skipped", txt.name)
            continue
        spans: list[tuple[int, int]] = []
        bad = 0
        for line in txt.read_text().splitlines():
            fields = line.strip().split()
            if not fields:
                continue
            try:
                start, end = int(fields[0]), int(fields[1])
            except (ValueError, IndexError):
                bad += 1
                continue
            if start < 1 or end < start:
                bad += 1
                continue
            spans.append((start, end))
        # boundary k (between span k and k+1): 0-based = start_{k+1} - 1.
        cuts: list[int] = []
        for cur, nxt in zip(spans, spans[1:]):
            cut = nxt[0] - 1
            if cut > 0:
                cuts.append(cut)
        if bad:
            log.warning("bench rai: %d malformed lines in %s", bad, txt.name)
        entries.append({
            "video_path": str(video),
            "gt_cuts": cuts,
            "gt_gradual": [],
            "meta": {
                "dataset": "rai",
                "scene_level_gt": True,
                "video": num,
                "n_scenes": len(spans),
            },
        })
    return entries


# ------------------------------------------------------------------ sbd


# SBD snippet convention (it-just-works/shot-boundary-detection, MIT):
# every clip is exactly 61 frames with the boundary event CENTERED on the
# 31st frame (1-indexed) == 0-indexed 30. GT conversion:
#   C -> hard cut at frame 30 (internal convention: first frame of the new
#        shot, 0-based)
#   T -> gradual interval [30-10, 30+10) — the dataset documents only the
#        CENTER, not the span; center±10 is the fixed scoring convention
#        (recorded here + in the fetch script; the transitions_gradual
#        family scores interval overlap, so a true span narrower/wider than
#        ±10 still matches as long as it overlaps)
#   E -> no events (pure negative: any emission is a FP)
SBD_CENTER = 30
SBD_HALF_WIDTH = 10


def load_sbd(sbd_dir: str | Path) -> list[VideoEntry]:
    """Load the SBD 61-frame snippet corpus (see ``scripts/sbd_fetch.py``).

    Manifest ``gt.json`` rows carry {clip, label C/T/E, origin, real,
    video_id, frames}. Clips whose decode failed at fetch time were never
    admitted; frame counts outside 58-64 are skipped here defensively.
    Stratification lives in ``meta`` (label/origin/real) so reports can
    split real vs synthetic sources.
    """
    d = Path(sbd_dir)
    gt_path = d / "gt.json"
    if not gt_path.is_file():
        raise FileNotFoundError(
            f"gt.json not found at {gt_path} — see scripts/sbd_fetch.py "
            "(HF: it-just-works/shot-boundary-detection)")
    manifest = json.loads(gt_path.read_text())
    entries: list[VideoEntry] = []
    for s in manifest.get("samples", []):
        clip = str(s.get("clip", ""))
        path = d / "clips" / clip
        if not path.is_file():
            log.warning("bench sbd: clip %s missing on disk — skipped", clip)
            continue
        n_frames = int(s.get("frames", 61) or 61)
        if not 58 <= n_frames <= 64:
            log.warning("bench sbd: clip %s has %d frames (outside 58-64) — "
                        "skipped", clip, n_frames)
            continue
        label = str(s.get("label", "")).strip().upper()
        center = min(SBD_CENTER, n_frames - 2)
        if label == "C":
            gt_cuts, gt_gradual = [center], []
        elif label == "T":
            gt_cuts = []
            gt_gradual = [(max(0, center - SBD_HALF_WIDTH),
                           min(n_frames, center + SBD_HALF_WIDTH))]
        elif label == "E":
            gt_cuts, gt_gradual = [], []
        else:
            log.warning("bench sbd: clip %s label %r invalid — skipped",
                        clip, label)
            continue
        entries.append({
            "video_path": str(path),
            "gt_cuts": gt_cuts,
            "gt_gradual": gt_gradual,
            "meta": {
                "dataset": "sbd",
                "label": label,
                "origin": s.get("origin", ""),
                "real": bool(s.get("real")),
                "video_id": s.get("video_id", ""),
                "frames": n_frames,
            },
        })
    entries.sort(key=lambda e: Path(e["video_path"]).name)
    return entries


# ------------------------------------------------------------------ registry


# ``clipshots:<split>`` resolves the split via the registry alias table.
DATASETS: dict[str, Callable[..., list[VideoEntry]]] = {
    "local": load_local,
    "bbc": load_bbc,
    "clipshots": load_clipshots,
    "clipshots:only_gradual": lambda d, split="only_gradual": load_clipshots(d, split),
    "clipshots:test": lambda d, split="test": load_clipshots(d, split),
    "clipshots:train": lambda d, split="train": load_clipshots(d, split),
    "rai": load_rai,
    "sbd": load_sbd,
}


def default_dataset_dir(name: str) -> Path:
    """Default on-disk location for a dataset (D12: ``datasets/`` at repo root)."""
    pkg_parent = Path(__file__).resolve().parents[1]   # scenecut_py/
    repo_root = pkg_parent.parent                       # repo root
    base = name.split(":", 1)[0]                        # clipshots:<split>
    if base == "local":
        return pkg_parent / "samples"
    if base == "bbc":
        return repo_root / "datasets" / "bbc"
    if base == "clipshots":
        return repo_root / "datasets" / "clipshots"
    if base == "rai":
        return repo_root / "datasets" / "rai"
    if base == "sbd":
        return repo_root / "datasets" / "sbd"
    raise KeyError(f"unknown dataset '{name}' (available: {sorted(DATASETS)})")


def load_dataset(name: str, dataset_dir: str | Path | None = None,
                 quick: bool = False) -> list[VideoEntry]:
    """Dispatch to a registered loader and apply ``--quick`` subsetting (D12).

    Quick: BBC first 3 episodes; ClipShots (any split) first 100 sorted-name;
    RAI first 3 videos. The local corpus is NOT subset (it is the gate
    dataset and already small) — quick mode still uses a separate golden
    key per A18.
    """
    if name not in DATASETS:
        raise KeyError(f"unknown dataset '{name}' (available: {sorted(DATASETS)})")
    d = Path(dataset_dir) if dataset_dir is not None else default_dataset_dir(name)
    entries = DATASETS[name](d)  # clipshots:<split> aliases bake the split in
    if quick:
        entries = _quick_subset(name, entries)
    return entries


def _quick_subset(name: str, entries: list[VideoEntry]) -> list[VideoEntry]:
    """--quick subset (D12): bbc → 3 eps; clipshots* → 100; rai → 3 videos;
    sbd → 150 (clips are 61-frame snippets — 150 spans all three labels)."""
    base = name.split(":", 1)[0]
    if base == "bbc":
        return entries[:3]
    if base == "clipshots":
        return entries[:100]
    if base == "rai":
        return entries[:3]
    if base == "sbd":
        return entries[:150]
    return entries
