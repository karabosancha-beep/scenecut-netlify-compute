#!/usr/bin/env python3
"""Fetch a stratified sample of the SBD 61-frame snippet corpus (HF).

Dataset: ``it-just-works/shot-boundary-detection`` (MIT) — 3.4M uniformly
structured 61-frame clips centered on frame 31 (1-indexed), labeled:

  C — hard Cut at the center
  T — gradual Transition centered at the center
  E — Empty (no boundary)

Sources: AutoShot / ClipShots / crawled Pexels, real + synthetic. The test
split alone is ~51 GB across 117 webdataset tars, so this fetch STREAMS tars
(hf-mirror.com first, huggingface.co fallback) and extracts only a stratified
sample, real-origin first:

  quotas (defaults): 120 real per label (C/T/E) + 80 synthetic for C/T
  only = 560 clips. Synthetic-E does not exist in the corpus (verified:
  12 full tars / ~59K samples, zero synthetic-E — synthetic generation
  produces cut/transition effects only; E clips are real footage), so the
  synthetic-E quota is 0 by design.

Output: ``datasets/sbd/clips/<name>.mp4`` + ``datasets/sbd/gt.json`` manifest
({clip, label, origin, real, video_id, frames} — frames verified by decode at
fetch time; clips that fail decode are not admitted).

Usage:
  python3 sbd_fetch.py                     # defaults, resumable (skips met quotas)
  python3 sbd_fetch.py --per-label-real 200 --per-label-syn 100
  python3 sbd_fetch.py --reset             # wipe datasets/sbd and refetch
"""
from __future__ import annotations

import argparse
import json
import shutil
import sys
import tarfile
import time
from collections import Counter
from pathlib import Path
from urllib.request import Request, urlopen

REPO_ROOT = Path(__file__).resolve().parents[2]
SBD_DIR = REPO_ROOT / "datasets" / "sbd"
CLIPS_DIR = SBD_DIR / "clips"
GT_PATH = SBD_DIR / "gt.json"

DATASET_BASE = "it-just-works/shot-boundary-detection"
MIRRORS = [
    "https://hf-mirror.com",
    "https://huggingface.co",
]
N_TEST_TARS = 117


def _open_stream(url: str):
    req = Request(url, headers={"User-Agent": "scenecut-sbd-fetch/1.0"})
    return urlopen(req, timeout=60)


def _tar_url(mirror: str, idx: int) -> str:
    return f"{mirror}/datasets/{DATASET_BASE}/resolve/main/test/test-{idx:06d}.tar"


def _label_ok(sample: dict) -> str | None:
    lab = str(sample.get("label", "")).strip().upper()
    return lab if lab in ("C", "T", "E") else None


def _is_real(origin: str) -> bool:
    return "synthetic" not in str(origin).lower()


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--per-label-real", type=int, default=120)
    ap.add_argument("--per-label-syn", type=int, default=80,
                    help="synthetic quota for C and T (synthetic-E does not "
                         "exist in this corpus — fixed at 0)")
    ap.add_argument("--max-tars", type=int, default=N_TEST_TARS)
    ap.add_argument("--reset", action="store_true")
    ap.add_argument("--rescan", action="store_true",
                    help="scan tars already recorded as scanned in gt.json "
                         "(default: resume after the last scanned tar)")
    args = ap.parse_args()

    if args.reset and SBD_DIR.exists():
        shutil.rmtree(SBD_DIR)
        print(f"reset: removed {SBD_DIR}")
    CLIPS_DIR.mkdir(parents=True, exist_ok=True)

    # resumable: count what's already on disk
    manifest: dict = {"fetched_at": None, "tars_scanned": 0, "samples": []}
    if GT_PATH.is_file():
        manifest = json.loads(GT_PATH.read_text())
    have = Counter()
    for s in manifest.get("samples", []):
        have[(s["label"], bool(s.get("real")))] += 1
    quota = {(lab, True): args.per_label_real for lab in ("C", "T", "E")}
    quota.update({("C", False): args.per_label_syn,
                  ("T", False): args.per_label_syn})
    quota[("E", False)] = 0  # synthetic-E does not exist (see module doc)
    remaining = {k: max(0, v - have.get(k, 0)) for k, v in quota.items()}
    if all(v == 0 for v in remaining.values()):
        print(f"quotas already met ({dict(have)}) — nothing to do "
              f"(use --reset to refetch)")
        return 0

    import cv2  # decode verification at fetch time

    t0 = time.time()
    fetched = 0
    bytes_read = 0
    def _save_manifest(tars_scanned: int) -> None:
        manifest["tars_scanned"] = tars_scanned
        manifest["fetched_at"] = time.strftime("%Y-%m-%dT%H:%M:%SZ",
                                                time.gmtime())
        GT_PATH.write_text(json.dumps(manifest, indent=1))
    for tar_idx in range(args.max_tars):
        if all(v == 0 for v in remaining.values()):
            break
        # resume: skip tars already fully scanned (unless --rescan)
        if not args.rescan and tar_idx < int(manifest.get("tars_scanned", 0)):
            continue
        stream = None
        used_mirror = None
        for mirror in MIRRORS:
            url = _tar_url(mirror, tar_idx)
            try:
                stream = _open_stream(url)
                used_mirror = mirror
                break
            except Exception as exc:
                print(f"  mirror miss {mirror}: {exc}", flush=True)
        if stream is None:
            print(f"  tar {tar_idx}: no mirror reachable — skipping", flush=True)
            continue
        print(f"[sbd] streaming tar {tar_idx:06d} via {used_mirror} "
              f"({fetched} clips fetched so far)...", flush=True)
        try:
            tf = tarfile.open(fileobj=stream, mode="r|")
            # webdataset tars store mp4+json pairs ADJACENT but in EITHER
            # order (this dataset: mp4 first, then json) — buffer the side
            # that arrives first and emit on the pair's completion. Cap keeps
            # the buffer bounded (~8 clips ≈ 1 MB) even for odd layouts.
            pending: dict[str, bytes] = {}
            for member in tf:
                key, _, ext = member.name.rpartition(".")
                if ext == "mp4":
                    if len(pending) >= 8:
                        pending.pop(next(iter(pending)))
                    data = tf.extractfile(member).read()
                    pending[key] = data
                    bytes_read += len(data)
                elif ext == "json":
                    try:
                        meta = json.loads(tf.extractfile(member).read())
                    except Exception:
                        pending.pop(key, None)
                        continue
                    lab = _label_ok(meta)
                    real = _is_real(meta.get("origin", ""))
                    data = pending.pop(key, None)
                    if lab is None or data is None:
                        continue
                    if remaining.get((lab, real), 0) <= 0:
                        continue
                    name = f"t{tar_idx:02d}_{key}_{lab}.mp4"
                    path = CLIPS_DIR / name
                    path.write_bytes(data)
                    # decode verification: 61 frames expected; admit 58-64
                    cap = cv2.VideoCapture(str(path))
                    n = 0
                    while True:
                        ok, frame = cap.read()
                        if not ok:
                            break
                        n += 1
                    cap.release()
                    if not 58 <= n <= 64:
                        path.unlink()
                        continue
                    meta.update({"clip": name, "frames": n, "label": lab,
                                 "real": real})
                    manifest["samples"].append(meta)
                    remaining[(lab, real)] -= 1
                    fetched += 1
                    if fetched % 50 == 0:
                        print(f"  ... {fetched} fetched "
                              f"({time.time()-t0:.0f}s, "
                              f"{bytes_read/1e6:.0f} MB streamed)", flush=True)
                    if all(v == 0 for v in remaining.values()):
                        break
        except Exception as exc:
            print(f"  tar {tar_idx}: stream ended ({exc})", flush=True)
        finally:
            try:
                stream.close()
            except Exception:
                pass
        # incremental persistence: a killed run keeps every completed tar
        _save_manifest(tar_idx + 1)
        counts = Counter((s["label"], s["real"]) for s in manifest["samples"])
        print(f"  tar {tar_idx}: cumulative {dict(counts)} "
              f"({bytes_read/1e6:.0f} MB, {time.time()-t0:.0f}s)", flush=True)

    _save_manifest(manifest.get("tars_scanned", 0))
    counts = Counter((s["label"], s["real"]) for s in manifest["samples"])
    print(f"[sbd] done: {len(manifest['samples'])} clips on disk, "
          f"{dict(counts)}, {bytes_read/1e6:.0f} MB streamed in "
          f"{time.time()-t0:.0f}s -> {GT_PATH}")
    missing = {k: v for k, v in remaining.items() if v > 0}
    if missing:
        print(f"[sbd] NOTE: quotas not fully met after {args.max_tars} tars: "
              f"{missing} (manifest still written; bench works with what's there)")
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
