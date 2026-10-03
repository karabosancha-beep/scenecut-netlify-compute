#!/usr/bin/env python3
"""ClipShots gradual-corpus fetcher — streams the HF-mirror gzip-tar and
extracts a filtered subset (R5 Stage A recipe, verified live 2026-10-01).

Mechanics (see .agents/research/r5-datasets.md):
  * HuggingFace mirror ``RorooroR/ClipShots`` hosts the original 3-part
    archive as plain HTTP-range-able files (verified 206, ~10 MB/s).
  * ``ClipShots-a`` is a single gzip stream; tar member order is
    case-insensitive alphabetical: ``annotations/`` < ``tools`` < ``videos/``,
    and within videos: ``only_gradual/`` < ``test/`` < ``train/``.
  * Therefore a ``Range: bytes=0-<N>`` prefix stream yields the annotation
    JSONs FIRST (collect the GT-bearing name set on the fly), then the
    only_gradual videos in order — a single filtered pass extracts the
    gradual stress subset without storing the 46 GB archive.

Resumability: tar streams cannot be byte-resumed, but extracted files are
skipped on re-run (idempotent). Raise ``--end`` to pull a deeper prefix.

Usage:
  python3 clipshots_fetch.py [--end 2147483647] [--start 0]
                              [--dest ../../datasets/clipshots]
                              [--keep-negatives]
"""
from __future__ import annotations

import argparse
import json
import subprocess
import sys
import tarfile
from pathlib import Path

MIRROR = "https://huggingface.co/datasets/RorooroR/ClipShots/resolve/main/ClipShots-a"

# The three annotation JSONs (also present inside the tar stream, but
# fetching them from the tar keeps the whole acquisition single-source).
ANN_NAMES = ("only_gradual.json", "test.json", "train.json")


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--start", type=int, default=0)
    ap.add_argument("--end", type=int, default=2147483647,
                    help="range end (exclusive-ish) in bytes [default 2 GiB]")
    ap.add_argument("--dest", default=None, help="destination dir "
                    "(default: ../../datasets/clipshots relative to this file)")
    ap.add_argument("--keep-negatives", action="store_true",
                    help="also extract only_gradual videos with EMPTY "
                    "transition lists (hard negatives for precision stress)")
    args = ap.parse_args()

    dest = Path(args.dest).resolve() if args.dest else \
        Path(__file__).resolve().parents[2] / "datasets" / "clipshots"
    videos_dir = dest / "videos"
    ann_dir = dest / "annotations"
    videos_dir.mkdir(parents=True, exist_ok=True)
    ann_dir.mkdir(parents=True, exist_ok=True)

    # ---- phase 0: annotation files already on disk? load the GT set.
    gt_set: set[str] = set()
    neg_set: set[str] = set()
    ann_local = ann_dir / "only_gradual.json"
    if ann_local.is_file():
        data = json.loads(ann_local.read_text())
        for name, info in data.items():
            base = Path(name).name
            if (info or {}).get("transitions"):
                gt_set.add(base)
            else:
                neg_set.add(base)
        print(f"[0] annotations present: {len(gt_set)} GT-bearing, "
              f"{len(neg_set)} negatives from {len(data)} keys")

    want_ann = {a for a in ANN_NAMES if not (ann_dir / a).is_file()}

    # ---- phase 1: stream the ranged prefix through tarfile.
    cmd = ["curl", "-sL", "--retry", "3", "--retry-delay", "2",
           "-H", f"Range: bytes={args.start}-{args.end}", MIRROR]
    print(f"[1] streaming {MIRROR} bytes {args.start}-{args.end} ...",
          flush=True)
    proc = subprocess.Popen(cmd, stdout=subprocess.PIPE)
    assert proc.stdout is not None

    n_ann = n_vid = n_skip_existing = 0
    bytes_seen = 0
    try:
        tf = tarfile.open(fileobj=proc.stdout, mode="r|gz")
        for member in tf:
            name = member.name
            base = Path(name).name
            if not member.isfile():
                continue
            if "annotations/" in name and base in ANN_NAMES:
                if base in want_ann or base == "only_gradual.json":
                    f = tf.extractfile(member)
                    if f is not None:
                        (ann_dir / base).write_bytes(f.read())
                        n_ann += 1
                        print(f"[1] annotation saved: {base}", flush=True)
                    if base == "only_gradual.json":
                        data = json.loads((ann_dir / base).read_text())
                        for k, info in data.items():
                            b = Path(k).name
                            if (info or {}).get("transitions"):
                                gt_set.add(b)
                            else:
                                neg_set.add(b)
                        print(f"[1] GT set built: {len(gt_set)} bearing, "
                              f"{len(neg_set)} negatives", flush=True)
                continue
            if "/videos/only_gradual/" in name and name.endswith(".mp4"):
                keep = base in gt_set or (args.keep_negatives and base in neg_set)
                if not keep:
                    continue
                out = videos_dir / base
                if out.is_file() and out.stat().st_size == member.size:
                    n_skip_existing += 1
                    continue
                f = tf.extractfile(member)
                if f is not None:
                    out.write_bytes(f.read())
                    n_vid += 1
                    if n_vid % 20 == 0:
                        print(f"[1] extracted {n_vid} videos ...", flush=True)
            # everything else (tools/, test/, train/) is skipped
    except tarfile.ReadError as exc:
        # expected at range truncation — the stream simply ends mid-member
        print(f"[1] stream ended (tar ReadError at truncation): {exc}",
              flush=True)
    finally:
        proc.stdout.close()
        rc = proc.wait()
        print(f"[1] curl exit: {rc}", flush=True)

    # ---- phase 2: report
    on_disk = sorted(p.name for p in videos_dir.glob("*.mp4"))
    bearing = [v for v in on_disk if v in gt_set]
    negatives = [v for v in on_disk if v in neg_set]
    print(f"[2] done: annotations={n_ann} extracted={n_vid} "
          f"skipped_existing={n_skip_existing}")
    print(f"[2] on disk: {len(on_disk)} only_gradual videos "
          f"({len(bearing)} GT-bearing gradual corpus, "
          f"{len(negatives)} hard negatives)")
    (dest / "fetch_state.json").write_text(json.dumps({
        "mirror": MIRROR, "range": [args.start, args.end],
        "gt_bearing": bearing, "negatives": negatives,
    }, indent=1))
    return 0


if __name__ == "__main__":
    sys.exit(main())
